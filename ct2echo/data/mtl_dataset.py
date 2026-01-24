"""
Multi-task learning dataset for CT volumes.

Contains:
- MTLDataset: PyTorch Dataset for multi-task CT classification
"""

import logging
import time
import torch
import pandas as pd
import numpy as np
import random
from typing import Any, Dict, cast
from torch.utils.data import Dataset
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

from ct2echo.preprocess.task_prompts import (
    label_query_func,
    label_checking,
    radio_label_query_func,
    radio_label_checking,
)
from ct2echo.data.metadata_utils import MetadataProcessor
from ct2echo.data.preprocessing import preprocess_for_dinov3

logger = logging.getLogger(__name__)


class MTLDataset(Dataset):
    def __init__(
        self,
        data_file,
        transform,
        task_embeddings,
        blob_service_client: BlobServiceClient,
        mode="train",
        use_dinov3=False,
        use_radio_labels=False,
        return_all_tasks=False,
        allowed_tasks=None,
        container_name="echo-data-lake",
    ):
        self.dataset = data_file
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self._standardize_metadata_columns()
        self.metadata_processor = MetadataProcessor(self.dataset)
        self.metadata_dim = self.metadata_processor.metadata_dim
        # Ensure task embeddings are on CPU for DataLoader compatibility
        if task_embeddings:
            self.task_embeddings = {k: v.detach().cpu() for k, v in task_embeddings.items()}
        else:
            self.task_embeddings = task_embeddings
        self.transform = transform
        self.mode = mode  # "train" or "eval"
        self.return_all_tasks = return_all_tasks or (mode == "eval")
        self.allowed_tasks = allowed_tasks
        self.use_dinov3 = use_dinov3  # Flag for DINOv3 preprocessing mode
        self.use_radio_labels = use_radio_labels
        if self.use_radio_labels:
            self.label_queries = {**label_query_func, **radio_label_query_func}
            self.label_checking = {**label_checking, **radio_label_checking}
        else:
            self.label_queries = label_query_func
            self.label_checking = label_checking

        # Cache available tasks for this dataset based on required columns
        self.available_tasks = self._get_available_tasks()
        if self.allowed_tasks is not None:
            original_available = list(self.available_tasks)
            allowed_set = set(self.allowed_tasks)
            filtered_tasks = [task for task in self.available_tasks if task in allowed_set]
            missing_requested = sorted(allowed_set - set(filtered_tasks))
            if not filtered_tasks:
                raise ValueError(
                    "No overlap between requested tasks and available tasks in dataset. "
                    f"Requested: {sorted(self.allowed_tasks)}, Available: {sorted(original_available)}"
                )
            if missing_requested:
                print(
                    "⚠️ Requested tasks missing from dataset columns: "
                    + ", ".join(missing_requested)
                )
            self.available_tasks = filtered_tasks
        print(f"Available tasks for this dataset: {self.available_tasks}")

        if self.use_dinov3:
            print("🔄 Dataset configured for DINOv3 preprocessing (164→165→RGB)")
        else:
            print("🔄 Dataset configured for 3D volumetric preprocessing")

        # Note: For evaluation mode, we'll use original indices and loop through tasks on-demand

    def _standardize_metadata_columns(self):
        """Ensure metadata columns exist with consistent naming and numeric types."""
        column_aliases = {
            "Patients_Sex": ["Patients_Sex", "Patients Sex", "sex", "Sex"],
            "Patients_Age": ["Patients_Age", "Patients Age", "age_at_study", "age_at_study_ct", "Age", "age"],
            "race_1": ["race_1", "Race_1", "race", "Race"],
            "systolic_bp": ["systolic_bp", "Systolic_BP", "systolic blood pressure"],
            "diastolic_bp": ["diastolic_bp", "Diastolic_BP", "diastolic blood pressure"],
            "heart_rate": ["heart_rate", "Heart_Rate", "heart rate"],
            "ethnicity": ["ethnicity", "Ethnicity", "ETHNIC_GROUP", "ETHNIC_GROUP_C"],
        }

        for target, aliases in column_aliases.items():
            if target in self.dataset.columns:
                continue
            for alias in aliases:
                if alias in self.dataset.columns:
                    self.dataset[target] = self.dataset[alias]
                    break
            else:
                self.dataset[target] = pd.Series([pd.NA] * len(self.dataset))

        numeric_columns = ["Patients_Age", "systolic_bp", "diastolic_bp", "heart_rate"]
        for col in numeric_columns:
            if col in self.dataset.columns:
                self.dataset[col] = pd.to_numeric(self.dataset[col], errors="coerce")

    def _get_available_tasks(self):
        """Check which tasks are available based on required columns and their logic types"""
        available_tasks = []

        for task_name, task_config in self.label_checking.items():
            task_type = task_config["type"]
            requirements = task_config["requirements"]

            task_available = self._check_task_requirements(requirements, task_type, self.dataset.columns, self.dataset)

            if task_available:
                available_tasks.append(task_name)

        return available_tasks

    def _check_task_requirements(self, requirements, requirement_type, available_columns, dataset=None):
        """
        Check if task requirements are met based on the requirement type.

        Args:
            requirements: List of columns
            requirement_type: Type of requirement logic
            available_columns: Available columns in the dataset
            dataset: Dataset to check for non-null values (optional)

        Returns:
            bool: True if requirements are satisfied
        """
        if requirement_type == "all_required":
            # All columns must be present and have some non-null data
            for col in requirements:
                if col not in available_columns:
                    return False
                if dataset is not None and dataset[col].isna().all():
                    return False
            return True

        elif requirement_type == "any_in_group":
            # At least one column from the group must be present and have some non-null data
            for col in requirements:
                if col in available_columns:
                    if dataset is None or not dataset[col].isna().all():
                        return True
            return False

        else:
            raise ValueError(f"Unknown requirement type: {requirement_type}")

    def _check_row_requirements(self, requirements, requirement_type, row_data):
        """
        Check if a specific row satisfies the task requirements.

        Args:
            requirements: List of columns
            requirement_type: Type of requirement logic
            row_data: Single row DataFrame

        Returns:
            bool: True if row satisfies requirements, False if should abstain
        """
        if requirement_type == "all_required":
            # All columns must have non-null values in this row
            for col in requirements:
                if col not in row_data.columns or row_data[col].isna().any():
                    return False
            return True

        elif requirement_type == "any_in_group":
            # At least one column must have non-null value in this row
            for col in requirements:
                if col in row_data.columns and not row_data[col].isna().any():
                    return True
            return False

        else:
            raise ValueError(f"Unknown requirement type: {requirement_type}")

    def _generate_label_for_task(self, idx, task_name):
        """Generate label for a specific task given the sample index and task name."""
        if task_name in self.available_tasks:
            # Get the query function for this task
            query_str = self.label_queries[task_name]

            # Get the task configuration
            task_config = self.label_checking[task_name]
            requirements = task_config["requirements"]
            requirement_type = task_config["type"]

            # Apply query to the specific row to generate label
            row_data = self.dataset.iloc[idx : idx + 1]  # Get single row as DataFrame

            # Check if this specific row satisfies the task requirements
            if not self._check_row_requirements(requirements, requirement_type, row_data):
                return -1  # Requirements not met for this row - abstain from evaluation

            try:
                # Evaluate the query on this specific row
                result = row_data.query(query_str)
                # If row satisfies query, label is 1, otherwise 0
                label = 1 if len(result) > 0 else 0
                return label
            except Exception as e:
                print(f"Error evaluating query for task {task_name} at index {idx}: {e}")
                return -1  # Invalid data - should be abstained from evaluation
        else:
            print(f"Warning: Task {task_name} not found in available tasks - abstaining from evaluation")
            return -1  # Task not available - should be abstained from evaluation

    def _process_label_value(self, label_value):
        """Process label value to ensure it's a valid float."""
        if pd.isna(label_value):
            return 0.0
        try:
            return float(label_value)
        except (ValueError, TypeError):
            return 0.0

    def _sample_task_and_generate_label(self, idx):
        """Sample a random available task for this specific row and generate label"""
        if not self.available_tasks:
            return None, -1

        # Get row data for checking requirements
        row_data = self.dataset.iloc[idx : idx + 1]  # Get single row as DataFrame

        # Find which tasks are available for this specific row
        available_tasks_for_row = []
        for task_name in self.available_tasks:
            task_config = self.label_checking[task_name]
            requirements = task_config["requirements"]
            requirement_type = task_config["type"]

            # Check if this row meets the task requirements
            if self._check_row_requirements(requirements, requirement_type, row_data):
                available_tasks_for_row.append(task_name)

        # If no tasks are available for this row, return abstain
        if not available_tasks_for_row:
            return None, -1

        # Randomly sample from available tasks for this row
        sampled_task = random.choice(available_tasks_for_row)

        # Generate label using existing function to avoid code duplication
        task_label = self._generate_label_for_task(idx, sampled_task)

        return sampled_task, task_label

    def __len__(self):
        # For both training and evaluation modes, use the original dataset length
        return len(self.dataset)

    def _download_blob_with_retry(self, fname: str, max_retries: int = 3) -> np.ndarray:
        """Download blob with exponential backoff retry logic.

        Args:
            fname: Path to the blob file
            max_retries: Maximum number of retry attempts

        Returns:
            numpy array of the downloaded data

        Raises:
            AzureError: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, blob=fname
                )
                download_stream = blob_client.download_blob()
                return np.frombuffer(download_stream.readall(), dtype=np.float64)
            except AzureError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download blob after {max_retries} attempts: {fname}")
                    raise
                logger.warning(f"Blob download attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def __getitem__(self, idx):
        # Use regular indexing for both training and evaluation modes
        sample_idx = idx
        fname = self.dataset.iloc[idx]["Path_New"]

        # Load and process the CT data with retry logic
        download_arr = self._download_blob_with_retry(fname)
        data = np.array(download_arr).reshape(164, 164, 164)
        data[data < -1000] = -1000
        data[data > 1000] = 1000

        if self.use_dinov3:
            # DINOv3 preprocessing with empty-slice filtering
            if self.transform is not None:
                data = self.transform(**{"image": data})["image"]

            processed, prep_details = preprocess_for_dinov3(data, return_details=True)
            prep_details = cast(Dict[str, Any], prep_details)
            data = processed.permute(1, 0, 2, 3)
            slice_mask = prep_details["mask"]

        else:
            # 3D volumetric preprocessing (legacy pipeline)
            data = self.transform(**{"image": data})["image"]
            data = torch.FloatTensor(data)
            data = data.permute((2, 0, 1))
            data = data.unsqueeze(0)
            slice_mask = None

        metadata = self.metadata_processor.encode_row(self.dataset.iloc[sample_idx])

        multi_task_mode = self.mode == "eval" or self.return_all_tasks

        if multi_task_mode:
            # For evaluation or multi-task training mode, generate labels for ALL available tasks
            task_results = {}
            task_embeddings = {}
            randomize_embeddings = self.mode != "eval"

            for task_name in self.available_tasks:
                task_label = self._generate_label_for_task(sample_idx, task_name)
                task_results[task_name] = torch.tensor(task_label).float()

                # Get task embedding for this task
                if task_name in self.task_embeddings:
                    task_embeddings_for_task = self.task_embeddings[task_name]
                    if task_embeddings_for_task.dim() > 1:
                        if randomize_embeddings:
                            num_embeddings = task_embeddings_for_task.shape[0]
                            random_idx = torch.randint(0, num_embeddings, (1,)).item()
                            task_embedding = task_embeddings_for_task[random_idx]
                        else:
                            task_embedding = task_embeddings_for_task[0]
                    else:
                        task_embedding = task_embeddings_for_task
                    task_embeddings[task_name] = task_embedding.cpu()
                else:
                    # Fallback to dummy embedding
                    task_embeddings[task_name] = torch.randn(1024)

            sample = {
                "ct": data,
                "metadata": metadata,
                "target": task_results,  # Dict mapping task_name -> label
                "task_embedding": task_embeddings,  # Dict mapping task_name -> embedding
                "idx": sample_idx,
                "task_name": self.available_tasks,
            }
            if slice_mask is not None:
                sample["ct_mask"] = slice_mask
            return sample
        else:
            # For training mode, randomly sample task and generate label (original behavior)
            sampled_task, task_label = self._sample_task_and_generate_label(sample_idx)
            label = torch.tensor(task_label).float()

            # Get the corresponding task embedding for the sampled task
            if sampled_task and sampled_task in self.task_embeddings:
                task_embeddings_for_task = self.task_embeddings[sampled_task]
                # Randomly sample one embedding from the available embeddings for this task
                num_embeddings = task_embeddings_for_task.shape[0]
                random_idx = torch.randint(0, num_embeddings, (1,)).item()
                task_embedding = task_embeddings_for_task[random_idx]
                task_embedding = task_embedding.cpu()
            else:
                # Fallback - use first available task embedding if task not found
                if self.task_embeddings:
                    first_task_embeddings = list(self.task_embeddings.values())[0]
                    if first_task_embeddings.dim() > 1:
                        num_embeddings = first_task_embeddings.shape[0]
                        random_idx = torch.randint(0, num_embeddings, (1,)).item()
                        task_embedding = first_task_embeddings[random_idx]
                    else:
                        task_embedding = first_task_embeddings
                    task_embedding = task_embedding.cpu()
                else:
                    task_embedding = torch.randn(1024)

            sample = {
                "ct": data,
                "metadata": metadata,
                "target": label,
                "task_embedding": task_embedding,
                "idx": sample_idx,
                "task_name": sampled_task,
            }
            if slice_mask is not None:
                sample["ct_mask"] = slice_mask
            return sample


__all__ = ["MTLDataset"]
