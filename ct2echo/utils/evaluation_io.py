"""
Evaluation results I/O utilities.

Contains:
- save_evaluation_results: Save evaluation predictions and metrics to files
"""

import os
import pickle

import torch
import pandas as pd
import numpy as np
from loguru import logger


def save_evaluation_results(results, epoch, output_dir, dataset_name):
    """Save evaluation predictions and metrics to files"""
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Save detailed results (includes all tensors)
    results_file = os.path.join(epoch_dir, f"{dataset_name}_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    # Save predictions as CSV files for each task (more accessible format)
    predictions_dir = os.path.join(epoch_dir, f"{dataset_name}_predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    for task_name, metrics in results.items():
        if task_name == "overall":
            continue  # Skip overall for individual task CSVs

        if "predictions" in metrics and "targets" in metrics:
            # Convert tensors to numpy for CSV saving
            predictions = metrics["predictions"].numpy() if torch.is_tensor(metrics["predictions"]) else np.array(metrics["predictions"])
            targets = metrics["targets"].numpy() if torch.is_tensor(metrics["targets"]) else np.array(metrics["targets"])
            outputs = metrics["outputs"].numpy() if torch.is_tensor(metrics["outputs"]) else np.array(metrics["outputs"])

            # Create DataFrame with predictions and targets
            df = pd.DataFrame(
                {
                    "sample_id": range(len(predictions)),
                    "prediction_prob": predictions,
                    "target": targets,
                    "raw_output": outputs,
                    "binary_prediction": (predictions > 0.5).astype(int),
                }
            )

            # Save task-specific predictions
            task_csv_file = os.path.join(predictions_dir, f"{task_name}_predictions.csv")
            df.to_csv(task_csv_file, index=False)

    # Save overall predictions if available
    if "overall" in results and "predictions" in results["overall"]:
        overall_predictions = (
            results["overall"]["predictions"].numpy()
            if torch.is_tensor(results["overall"]["predictions"])
            else np.array(results["overall"]["predictions"])
        )
        overall_targets = (
            results["overall"]["targets"].numpy() if torch.is_tensor(results["overall"]["targets"]) else np.array(results["overall"]["targets"])
        )
        overall_outputs = (
            results["overall"]["outputs"].numpy() if torch.is_tensor(results["overall"]["outputs"]) else np.array(results["overall"]["outputs"])
        )

        overall_df = pd.DataFrame(
            {
                "sample_id": range(len(overall_predictions)),
                "prediction_prob": overall_predictions,
                "target": overall_targets,
                "raw_output": overall_outputs,
                "binary_prediction": (overall_predictions > 0.5).astype(int),
            }
        )

        overall_csv_file = os.path.join(predictions_dir, "overall_predictions.csv")
        overall_df.to_csv(overall_csv_file, index=False)

    # Save summary metrics as text
    summary_file = os.path.join(epoch_dir, f"{dataset_name}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Evaluation Results for {dataset_name} - Epoch {epoch}\n")
        f.write("=" * 60 + "\n\n")

        for task_name, metrics in results.items():
            if task_name == "overall":
                f.write("OVERALL METRICS:\n")
            else:
                f.write(f"Task: {task_name}\n")

            f.write(f"  Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            if "auc" in metrics:
                f.write(f"  AUC: {metrics['auc']:.4f}\n")
            if "count" in metrics:
                f.write(f"  Sample Count: {metrics['count']}\n")
            f.write("\n")

    logger.info(f"Saved evaluation results for {dataset_name} epoch {epoch} to {epoch_dir}")
    logger.info(f"Saved prediction CSVs to {predictions_dir}")


__all__ = ["save_evaluation_results"]
