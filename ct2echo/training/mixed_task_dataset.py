"""
Mixed-task batch dataset and collation utilities for multi-task training.

Contains:
- MixedTaskBatchDataset: Dataset that creates mixed-task batches on-the-fly
- mixed_task_collate_fn: Collate function for mixed-task batches
- create_mixed_task_dataloader: Factory function for mixed-task DataLoaders
"""

import torch
import random
from typing import Dict, List, Optional
from torch.utils.data import Dataset, DataLoader


class MixedTaskBatchDataset(Dataset):
    """
    Dataset that creates mixed-task batches on-the-fly
    Each batch contains samples from different tasks
    """

    def __init__(
        self,
        base_dataset,
        task_embeddings: Dict[str, torch.Tensor],
        medical_tasks: List[str],
        samples_per_task: Optional[int] = None,
        task_distribution: str = "uniform",
    ):
        self.base_dataset = base_dataset
        self.task_embeddings = task_embeddings
        self.medical_tasks = medical_tasks
        self.samples_per_task = samples_per_task or (len(base_dataset) // len(medical_tasks))
        self.task_distribution = task_distribution

        # Create task assignments for each sample
        self._create_task_assignments()

    def _create_task_assignments(self):
        """Assign tasks to samples based on distribution strategy"""
        self.sample_task_assignments = []

        if self.task_distribution == "uniform":
            # Uniform distribution across tasks
            for i in range(len(self.base_dataset)):
                task_idx = i % len(self.medical_tasks)
                task_name = self.medical_tasks[task_idx]
                self.sample_task_assignments.append(task_name)
        elif self.task_distribution == "random":
            # Random task assignment
            for i in range(len(self.base_dataset)):
                task_name = random.choice(self.medical_tasks)
                self.sample_task_assignments.append(task_name)
        else:
            raise ValueError(f"Unknown task distribution: {self.task_distribution}")

        # Shuffle to ensure mixed batches
        combined = list(zip(range(len(self.base_dataset)), self.sample_task_assignments))
        random.shuffle(combined)
        indices, tasks = zip(*combined)
        self.shuffled_indices = list(indices)
        self.sample_task_assignments = list(tasks)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the actual sample index after shuffling
        actual_idx = self.shuffled_indices[idx]

        # Get base sample
        base_sample = self.base_dataset[actual_idx]

        # Handle different base dataset formats
        if isinstance(base_sample, dict):
            inputs = base_sample["ct"]
            targets = base_sample["target"]
            metadata = base_sample.get("metadata")
        elif isinstance(base_sample, tuple):
            inputs, targets = base_sample[:2]
            metadata = base_sample[2] if len(base_sample) > 2 else None
        else:
            raise ValueError(f"Unsupported base dataset format: {type(base_sample)}")

        # Get assigned task
        task_name = self.sample_task_assignments[idx]
        task_embedding = self.task_embeddings[task_name]

        return {
            "inputs": inputs,
            "targets": targets,
            "metadata": metadata,
            "task_name": task_name,
            "task_embedding": task_embedding,
            "sample_idx": actual_idx,
        }


def mixed_task_collate_fn(batch):
    """
    Collate function for mixed-task batches
    Creates batch-level task embeddings without task grouping
    """
    inputs = torch.stack([sample["inputs"] for sample in batch])
    targets = torch.stack([sample["targets"] for sample in batch])
    task_names = [sample["task_name"] for sample in batch]
    task_embeddings = torch.stack([sample["task_embedding"] for sample in batch])
    sample_indices = torch.tensor([sample["sample_idx"] for sample in batch])
    metadata = None
    if batch[0].get("metadata") is not None:
        metadata = torch.stack([sample["metadata"] for sample in batch])

    return {
        "inputs": inputs,
        "targets": targets,
        "metadata": metadata,
        "task_names": task_names,
        "task_embeddings": task_embeddings,
        "sample_indices": sample_indices,
    }


def create_mixed_task_dataloader(
    base_dataset,
    task_embeddings: Dict[str, torch.Tensor],
    medical_tasks: List[str],
    batch_size: int = 8,
    task_distribution: str = "uniform",
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for mixed-task training

    Args:
        base_dataset: Original dataset
        task_embeddings: Dict mapping task names to embeddings
        medical_tasks: List of medical task names
        batch_size: Batch size
        task_distribution: Task distribution strategy
        num_workers: Number of worker processes
        shuffle: Whether to shuffle

    Returns:
        DataLoader that yields mixed-task batches
    """

    mixed_dataset = MixedTaskBatchDataset(
        base_dataset=base_dataset,
        task_embeddings=task_embeddings,
        medical_tasks=medical_tasks,
        task_distribution=task_distribution,
    )

    dataloader = DataLoader(
        mixed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mixed_task_collate_fn,
        pin_memory=True,
    )

    return dataloader


__all__ = [
    "MixedTaskBatchDataset",
    "mixed_task_collate_fn",
    "create_mixed_task_dataloader",
]
