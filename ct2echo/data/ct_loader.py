"""
CT data loader utilities.

Contains:
- CTLoader: High-level loader class for CT datasets
- dinov3_collate_fn: Collate function for variable-length DINOv3 image stacks
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from volumentations import Compose, CenterCrop, RandomCrop, Flip

from ct2echo.data.mtl_dataset import MTLDataset
from ct2echo.data.preprocessing import preprocess_dataframe_columns


def dinov3_collate_fn(batch):
    """Pad variable-length DINOv3 image stacks and build a mask for valid frames."""
    if not batch:
        return {}

    max_frames = max(sample["ct"].shape[1] for sample in batch)

    for sample in batch:
        ct_tensor = sample["ct"]
        existing_mask = sample.get("ct_mask")

        if existing_mask is None:
            existing_mask = torch.ones(ct_tensor.shape[1], dtype=torch.bool)
        else:
            existing_mask = existing_mask.to(dtype=torch.bool)

        frame_count = ct_tensor.shape[1]
        if frame_count < max_frames:
            pad_frames = max_frames - frame_count
            pad_tensor = ct_tensor.new_zeros(ct_tensor.shape[0], pad_frames, ct_tensor.shape[2], ct_tensor.shape[3])
            sample["ct"] = torch.cat([ct_tensor, pad_tensor], dim=1)

            padded_mask = torch.zeros(max_frames, dtype=torch.bool)
            padded_mask[:frame_count] = existing_mask
            sample["ct_mask"] = padded_mask
        else:
            if existing_mask.shape[0] != max_frames:
                padded_mask = torch.zeros(max_frames, dtype=torch.bool)
                padded_mask[:existing_mask.shape[0]] = existing_mask
                sample["ct_mask"] = padded_mask
            else:
                sample["ct_mask"] = existing_mask

    return default_collate(batch)


class CTLoader:
    def __init__(
        self,
        base_dir,
        task_embeddings,
        blob_service_client,
        batch_size,
        num_workers,
        use_dinov3=False,
        use_radio_labels=False,
        allowed_tasks=None,
        container_name="echo-data-lake",
    ) -> None:
        self.train_transform = Compose(
            [RandomCrop((144, 144, 164), always_apply=True, p=1.0), Flip(0, p=0.5)],
            p=1.0,
        )
        self.test_transform = Compose([CenterCrop((144, 144, 164), always_apply=True, p=1)], p=1.0)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_dir = base_dir
        self.task_embeddings = task_embeddings
        self.blob_service_client = blob_service_client
        self.use_dinov3 = use_dinov3  # Flag for DINOv3 preprocessing
        self.use_radio_labels = use_radio_labels
        self.allowed_tasks = allowed_tasks
        self.container_name = container_name
        if self.allowed_tasks is not None:
            print(f"🎯 CTLoader restricted to tasks: {self.allowed_tasks}")

        if self.use_dinov3:
            print("🔄 CTLoader configured for DINOv3 preprocessing")
        else:
            print("🔄 CTLoader configured for 3D volumetric preprocessing")
        if self.use_radio_labels:
            print("📻 CTLoader using radiology label definitions")

        self.file_mapping = {
            "train": "cu_train_merged_with_notes_labeled.csv",
            "val": "cu_val_merged_with_notes_labeled.csv",
            "ctest": "cu_test_merged_with_notes_labeled.csv",
            "wtest": "wcm_test_merged_with_notes_labeled.csv",
            "wprospect": "wcm_prospective_merged_with_notes.csv",
            "cprospect": "cu_prospective_merged_with_notes.csv",
            # "wproblem": "cu_wproblem_merged_with_notes.csv",
            # "cproblem": "columbia_problematic.csv",
            # "opportun": "opportunistic_screening_final_new.csv",
        }

    def run(self, mode):
        files = pd.read_csv(os.path.join(self.base_dir, self.file_mapping[mode]))

        # Apply column merging preprocessing
        files = preprocess_dataframe_columns(files)

        # files = files.dropna(subset=["ECHO_pasp_value"])
        contrast = pd.read_csv(os.path.join(self.base_dir, "contrast_issue.csv"))

        # amyloids = pd.read_csv(os.path.join(self.base_dir, "Selected_CT_ECHO_PYP_16sept.csv"))
        # pyp = pd.read_csv(os.path.join(self.base_dir, "Selected_CT_ECG_ECHO_PYP_all.csv"))
        #! Remove contrast
        files = files[~files["Path"].isin(contrast["Path"].to_list())]

        #! Remove PYP?
        # files = files[~files["Study Instance UID"].isin(pyp["Study Instance UID_x"].unique())]
        #! Male or Female in train
        # if mode == 'train':
        # files = files.query('Patients_Sex == "M" ')
        # files = files.query('Patients_Sex == "F" ')
        #! Remove amyloids
        # files = files[~files["Study Instance UID"].isin(amyloids["Study Instance UID"].to_list())]

        # dataset = CTDataset(files, self.train_transform, self.blob_service_client)

        if mode == "train":
            # files = pd.concat([files, pd.read_csv(os.path.join(self.base_dir, "columbia_prospective_additional.csv"))])
            dataset = MTLDataset(
                files,
                self.train_transform,
                self.task_embeddings,
                self.blob_service_client,
                mode="train",
                use_dinov3=self.use_dinov3,
                use_radio_labels=self.use_radio_labels,
                return_all_tasks=True,
                allowed_tasks=self.allowed_tasks,
                container_name=self.container_name,
            )
            self.metadata_dim = getattr(dataset, "metadata_dim", 0)
            loader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=dinov3_collate_fn if self.use_dinov3 else None,
            )
        elif mode == "val" or mode == "ctest" or mode == "wtest":
            dataset = MTLDataset(
                files,
                self.test_transform,
                self.task_embeddings,
                self.blob_service_client,
                mode="eval",
                use_dinov3=self.use_dinov3,
                use_radio_labels=self.use_radio_labels,
                allowed_tasks=self.allowed_tasks,
                container_name=self.container_name,
            )
            self.metadata_dim = getattr(dataset, "metadata_dim", 0)
            loader = DataLoader(
                dataset,
                self.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=self.num_workers,
                collate_fn=dinov3_collate_fn if self.use_dinov3 else None,
            )
        return loader, files

    def run_eval(self, mode):
        files = pd.read_csv(os.path.join(self.base_dir, self.file_mapping[mode]))

        # Apply column merging preprocessing
        files = preprocess_dataframe_columns(files)

        # Apply same filtering as in run() method
        contrast = pd.read_csv(os.path.join(self.base_dir, "contrast_issue.csv"))
        files = files[~files["Path"].isin(contrast["Path"].to_list())]

        # Choose appropriate augmentation based on mode
        if mode == "train":
            # For training mode, use training augmentation even in eval format
            # This gives us: ALL task labels per sample + training augmentation
            transform = self.train_transform
            shuffle = True  # Keep shuffling for training
        else:
            # For validation/test modes, use test augmentation
            transform = self.test_transform
            shuffle = False

        dataset = MTLDataset(
            files,
            transform,
            self.task_embeddings,
            self.blob_service_client,
            mode="eval",
            use_dinov3=self.use_dinov3,
            use_radio_labels=self.use_radio_labels,
            allowed_tasks=self.allowed_tasks,
            container_name=self.container_name,
        )
        self.metadata_dim = getattr(dataset, "metadata_dim", 0)
        loader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=dinov3_collate_fn if self.use_dinov3 else None,
        )
        return loader, files


__all__ = ["CTLoader", "dinov3_collate_fn"]
