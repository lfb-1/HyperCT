import random
import mlflow
import numpy as np
import torch
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
import os
import pandas as pd
import plotly.express as px
from sklearn.metrics import (
    roc_curve,
    auc,
)


def seed_all(config):
    training_cfg = getattr(config, "training", None)
    seed = getattr(training_cfg, "seed", None) if training_cfg is not None else None
    if seed is None:
        seed = getattr(config, "seed", None)
    if seed is None:
        raise AttributeError("seed_all expected config.training.seed or config.seed to be defined")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_azure():
    account_url = os.getenv(
        "AZURE_STORAGE_ACCOUNT_URL",
        "https://eus2prdcornellaiechosa.blob.core.windows.net"
    )
    client_id = os.getenv("AZURE_MANAGED_IDENTITY_CLIENT_ID")

    if client_id:
        credential = DefaultAzureCredential(managed_identity_client_id=client_id)
    else:
        credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(account_url, credential=credential)
    return blob_service_client


def filter_empty_input(data, labels, weights, lvef):
    non_empty_input = (torch.amax(data, dim=(1, 2, 3, 4)) > 0).nonzero().squeeze()
    data_resampled = data[non_empty_input]
    label_resampled = labels[non_empty_input]
    weights_resampled = weights[non_empty_input]
    lvef_resampled = lvef[non_empty_input]
    if data_resampled.dim() == 4:
        data_resampled, label_resampled, lvef_resampled = [i.unsqueeze(0) for i in [data_resampled, label_resampled, lvef_resampled]]
    return data_resampled, label_resampled, weights_resampled, lvef_resampled


def save_pred(epoch, total_train_preds, total_train_labels, total_train_lvef, prefix):
    np.save(f"preds/{prefix}_{epoch}.npy", total_train_preds.numpy())
    mlflow.log_artifact(f"preds/{prefix}_{epoch}.npy")
    if prefix == "train" or ("val" in prefix and epoch == 0) or ("test" in prefix and epoch == 0):
        np.save(f"{prefix}_targets{epoch}.npy", total_train_labels.numpy())
        np.save(f"{prefix}_lvef{epoch}.npy", total_train_lvef.numpy())
        mlflow.log_artifact(f"{prefix}_targets{epoch}.npy")
        mlflow.log_artifact(f"{prefix}_lvef{epoch}.npy")


def plot_roc(all_preds, all_labels):
    fpr, tpr, threshold = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
    random_guess_data = pd.DataFrame({"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]})

    # Create a Plotly Express figure
    fig = px.line(
        roc_data,
        x="False Positive Rate",
        y="True Positive Rate",
        # title=f"ROC Curve (AUC = {roc_auc:.2f})",
        width=700,
        height=500,
    )
    fig.add_scatter(
        x=random_guess_data["False Positive Rate"],
        y=random_guess_data["True Positive Rate"],
        mode="lines",
        line=dict(color="gray", dash="dash"),
        name="Random Guess",
    )
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", showlegend=True)
    return fig, roc_auc


def log_metrics(site_str, train_str, epoch, train_auc, train_metrics, train_loss, train_pearson, train_roc, time_str="all"):
    mlflow.log_metrics(
        {
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} loss": train_loss,
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} auc": train_auc,
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} f1": train_metrics[0],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} precision": train_metrics[1],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} recall": train_metrics[2],
            f"{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} pearson": train_pearson,
        }
    )
    mlflow.log_figure(
        train_roc,
        f"plots/{site_mapping[site_str]}-{task_mapping[train_str]}-{task_mapping[time_str]} {epoch} ROC.html",
    )


site_mapping = {"c": "Columbia", "w": "Cornell"}

task_mapping = {
    "train": "Train",
    "val": "Val",
    "test": "Test",
    # "wsm": "Cornell smooth",
    # "wsh": "Cornell sharp",
    "raa": "Race Asian",
    "rw": "Race White",
    "rb": "Race Black",
    "rai": "Race indian/Alaska",
    "lvdnm": "Heart LVD Normal Male",
    "lvdnf": "Heart LVD Normal Female",
    "lvdam": "Heart LVD Abnormal Male",
    "lvdaf": "Heart LVD Abnormal FeMale",
    "lvsnm": "Heart LVS Normal Male",
    "lvsnf": "Heart LVS Normal Female",
    "lvsam": "Heart LVS Abnormal Male",
    "lvsaf": "Heart LVS Abnormal FeMale",
    "sm": "Gender M",
    "sf": "Gender F",
    "so": "Gender O",
    "ay": "Young Age",
    "am": "Middle Age",
    "ao": "Old Age",
    "1d": "CT-ECHO in 1 day",
    "1w": "CT-ECHO in 1 week",
    "1m": "CT-ECHO in 1 month",
    "all": "",
}

race_mapping = {
    "aapi": ["asian", "nat.hawaiian/oth.pacific island"],
    "white": ["white"],
    "black": ["black or african american"],
    "aian": ["american indian or alaska nation"],
}

query_func = {
    "raa": f"race_1 in {race_mapping['aapi']}",
    "rw": f"race_1 in {race_mapping['white']}",
    "rb": f"race_1 in {race_mapping['black']}",
    "ay": "Patients_Age < 40",
    "am": "40 <= Patients_Age < 65",
    "ao": "65 <= Patients_Age < 150",
    "sm": 'Patients_Sex == "M" ',
    "sf": 'Patients_Sex == "F" ',
    "so": 'Patients_Sex == "O" ',
    "lvdnm": "lvd_classification == 'Normal' & Patients_Sex == 'M' ",
    "lvdnf": "lvd_classification == 'Normal' & Patients_Sex == 'F' ",
    "lvdam": "lvd_classification == 'Abnormal' & Patients_Sex == 'M'  ",
    "lvdaf": "lvd_classification == 'Abnormal' & Patients_Sex == 'F'  ",
    "lvsnm": "lvs_classification == 'Normal' & Patients_Sex == 'M' ",
    "lvsnf": "lvs_classification == 'Normal' & Patients_Sex == 'F' ",
    "lvsam": "lvs_classification == 'Abnormal' & Patients_Sex == 'M' ",
    "lvsaf": "lvs_classification == 'Abnormal' & Patients_Sex == 'F' ",
}
