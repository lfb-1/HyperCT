"""
Dynamic LoRA forward manager and training utilities.

Contains:
- DynamicLoRAForwardManager: Manages dynamic LoRA weight generation and application
- dynamic_lora_context: Context manager for dynamic LoRA training
- train_epoch_mixed_tasks_dynamic: Training function for mixed-task dynamic LoRA
- analyze_mixed_batch_distribution: Utility for analyzing task distribution
"""

import torch
from typing import Optional
from contextlib import contextmanager

from ct2echo.training.lora_hooks import HookBasedLoRAManager


class DynamicLoRAForwardManager:
    """
    Manages dynamic LoRA weight generation and application during training
    Based on hyper_modulator.py principles
    """

    def __init__(self, model, hypernet, scaling_factor=1.0):
        self.model = model
        self.hypernet = hypernet
        self.scaling_factor = scaling_factor
        self.hook_manager = HookBasedLoRAManager(model, hypernet, scaling_factor)

    def setup(self):
        """Setup the dynamic LoRA system"""
        self.hook_manager.register_hooks()

    def cleanup(self):
        """Cleanup hooks and resources"""
        self.hook_manager.remove_hooks()

    def _generate_batch_lora_weights(self, task_embeddings, metadata=None):
        metadata_enabled = metadata is not None and getattr(self.hypernet, "metadata_encoder", None) is not None
        if metadata_enabled:
            try:
                return self.hypernet.generate_full_model_lora(task_embeddings, metadata)
            except TypeError:
                pass
        return self.hypernet.generate_full_model_lora(task_embeddings)

    def forward_with_dynamic_lora(self, inputs, task_embeddings, metadata=None, masks=None):
        """
        Forward pass with dynamic LoRA weights for per-sample embeddings

        Args:
            inputs: Batch input tensor [batch_size, ...]
            task_embeddings: Task embeddings [batch_size, embed_dim]
            metadata: Optional metadata features [batch_size, metadata_dim]

        Returns:
            Model outputs with task-specific LoRA weights applied
        """
        # Ensure inputs require gradients for proper gradient flow
        # if not inputs.requires_grad:
        #     inputs = inputs.requires_grad_(True)

        batch_lora_weights = self._generate_batch_lora_weights(task_embeddings, metadata)

        # Set LoRA weights and perform forward pass
        self.hook_manager.set_lora_weights(batch_lora_weights)
        self.hook_manager.activate()

        try:
            # Forward pass through base model with LoRA applied
            # The base model now includes mlp_head, so we get final predictions directly
            if masks is not None and getattr(self.model, "supports_volume_mask", False):
                outputs = self.model(inputs, mask=masks)
            else:
                outputs = self.model(inputs)
        finally:
            self.hook_manager.deactivate()

        return outputs

    def precompute_lora_weights(self, task_embeddings, metadata=None):
        """Generate LoRA weights once for a given task embedding (for caching)."""
        with torch.no_grad():
            return self._generate_batch_lora_weights(task_embeddings, metadata)

    def forward_with_cached_lora(self, inputs, cached_lora_weights, masks=None):
        """Forward pass that reuses precomputed LoRA weights."""
        self.hook_manager.set_lora_weights(cached_lora_weights)
        self.hook_manager.activate()

        try:
            if masks is not None and getattr(self.model, "supports_volume_mask", False):
                outputs = self.model(inputs, mask=masks)
            else:
                outputs = self.model(inputs)
        finally:
            self.hook_manager.deactivate()

        return outputs

    def set_grad_logging(self, interval: Optional[int]) -> None:
        self.hook_manager.set_grad_logging(interval)


@contextmanager
def dynamic_lora_context(model, hypernet, scaling_factor=1.0):
    """Context manager for dynamic LoRA training (cross-attention removed)"""
    manager = DynamicLoRAForwardManager(model, hypernet, scaling_factor)
    manager.setup()
    try:
        yield manager
    finally:
        manager.cleanup()


def train_epoch_mixed_tasks_dynamic(model, hypernet, train_loader, optimizer, criterion, scaler, device, args):
    """
    Train one epoch with dynamic mixed-task LoRA weights

    Args:
        model: Base encoder model (e.g., DINOv3_Encoder)
        hypernet: LoRA_Hypernet instance
        train_loader: Mixed-task DataLoader
        optimizer: Optimizer
        criterion: Loss function
        scaler: GradScaler for mixed precision
        device: Device to run on
        args: Arguments containing lora_scaling
    """
    # Keep base model frozen, only train hypernet
    model.eval()  # Base model stays frozen
    hypernet.train()  # Only hypernet is trainable
    total_loss = 0.0
    num_batches = 0

    with dynamic_lora_context(model, hypernet, args.lora_scaling) as lora_manager:
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            task_embeddings = batch["task_embeddings"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)

            optimizer.zero_grad()

            # Use autocast only on CUDA
            autocast_enabled = torch.cuda.is_available()
            with torch.autocast(device_type="cuda" if autocast_enabled else "cpu", enabled=autocast_enabled):
                outputs = lora_manager.forward_with_dynamic_lora(inputs, task_embeddings, metadata)
                loss = criterion(outputs, targets)

            # Use scaler only when autocast is enabled
            if autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# Example usage and testing functions
def analyze_mixed_batch_distribution(dataloader, num_batches=5):
    """Analyze task distribution in mixed batches"""

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break


__all__ = [
    "DynamicLoRAForwardManager",
    "dynamic_lora_context",
    "train_epoch_mixed_tasks_dynamic",
    "analyze_mixed_batch_distribution",
]
