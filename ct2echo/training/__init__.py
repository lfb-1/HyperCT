"""
Training subpackage - Training logic and utilities.

Contains:
- training: train_epoch_* functions for hypernet training
- dynamic_lora: HookBasedLoRAManager, mixed task training (backward compatibility shim)
- lora_hooks: HookBasedLoRAManager class
- mixed_task_dataset: MixedTaskBatchDataset, collate functions
- dynamic_lora_manager: DynamicLoRAForwardManager, context manager, training functions
"""

from ct2echo.training.training import (
    train_epoch_with_hypernet,
    train_epoch_standard,
)

# Backward compatible imports from dynamic_lora.py shim
from ct2echo.training.dynamic_lora import (
    HookBasedLoRAManager,
    MixedTaskBatchDataset,
    DynamicLoRAForwardManager,
    dynamic_lora_context,
    create_mixed_task_dataloader,
    train_epoch_mixed_tasks_dynamic,
    mixed_task_collate_fn,
    analyze_mixed_batch_distribution,
)

__all__ = [
    # training
    "train_epoch_with_hypernet",
    "train_epoch_standard",
    # dynamic_lora (lora_hooks, mixed_task_dataset, dynamic_lora_manager)
    "HookBasedLoRAManager",
    "MixedTaskBatchDataset",
    "DynamicLoRAForwardManager",
    "dynamic_lora_context",
    "create_mixed_task_dataloader",
    "train_epoch_mixed_tasks_dynamic",
    "mixed_task_collate_fn",
    "analyze_mixed_batch_distribution",
]
