"""
Backward compatibility shim - imports moved to submodules.

This file re-exports all symbols from the split modules for backward compatibility.
New code should import directly from:
- ct2echo.training.lora_hooks
- ct2echo.training.mixed_task_dataset
- ct2echo.training.dynamic_lora_manager
"""

# LoRA hooks
from ct2echo.training.lora_hooks import HookBasedLoRAManager

# Mixed task dataset
from ct2echo.training.mixed_task_dataset import (
    MixedTaskBatchDataset,
    mixed_task_collate_fn,
    create_mixed_task_dataloader,
)

# Dynamic LoRA manager
from ct2echo.training.dynamic_lora_manager import (
    DynamicLoRAForwardManager,
    dynamic_lora_context,
    train_epoch_mixed_tasks_dynamic,
    analyze_mixed_batch_distribution,
)


__all__ = [
    # lora_hooks
    "HookBasedLoRAManager",
    # mixed_task_dataset
    "MixedTaskBatchDataset",
    "mixed_task_collate_fn",
    "create_mixed_task_dataloader",
    # dynamic_lora_manager
    "DynamicLoRAForwardManager",
    "dynamic_lora_context",
    "train_epoch_mixed_tasks_dynamic",
    "analyze_mixed_batch_distribution",
]
