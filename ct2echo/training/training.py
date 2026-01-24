"""
Training-related functions for multi-task learning
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from ct2echo.training.dynamic_lora import dynamic_lora_context


def _log_grad_norm(module, name: str) -> None:
    """Safely log the L2 norm of gradients for a module."""
    squared_norm = 0.0
    has_grad = False
    for param in module.parameters():
        if param.grad is not None:
            has_grad = True
            squared_norm += param.grad.data.norm(2).item() ** 2
    if has_grad:
        total_norm = squared_norm ** 0.5
        logger.info(f"{name} grad norm: {total_norm:.6f}")


def _sample_tasks_from_multilabel_batch(
    target_dict: Dict[str, torch.Tensor],
    task_embedding_dict: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[str]]]:
    """Select a single valid task per sample from a dict of task labels.

    Returns tensors for selected labels, optional embeddings, and the chosen task names.
    Samples with no valid labels remain at -1 and should be filtered by the caller.
    """

    if not isinstance(target_dict, dict) or not target_dict:
        raise ValueError("target_dict must be a non-empty dictionary of task tensors")

    # Move task label tensors to the working device once
    device_targets: Dict[str, torch.Tensor] = {
        task_name: task_tensor.to(device)
        for task_name, task_tensor in target_dict.items()
    }

    first_task_tensor = next(iter(device_targets.values()))
    batch_size = first_task_tensor.shape[0]
    selected_labels = torch.full((batch_size,), -1.0, dtype=torch.float32, device=device)
    selected_task_names: List[Optional[str]] = [None] * batch_size

    embeddings_tensor: Optional[torch.Tensor] = None
    device_embeddings: Optional[Dict[str, torch.Tensor]] = None
    if task_embedding_dict:
        device_embeddings = {
            task_name: embedding.to(device)
            for task_name, embedding in task_embedding_dict.items()
        }
        example_embedding = next(iter(device_embeddings.values()))
        embed_dim = example_embedding.shape[-1]
        embeddings_tensor = torch.zeros((batch_size, embed_dim), dtype=example_embedding.dtype, device=device)

    for sample_idx in range(batch_size):
        valid_candidates = []
        for task_name, task_tensor in device_targets.items():
            label_value = task_tensor[sample_idx].item()
            if label_value != -1:
                valid_candidates.append((task_name, task_tensor[sample_idx]))

        if not valid_candidates:
            continue

        chosen_task, chosen_label = random.choice(valid_candidates)
        selected_task_names[sample_idx] = chosen_task
        selected_labels[sample_idx] = chosen_label

        if embeddings_tensor is not None and device_embeddings is not None:
            embedding_value = device_embeddings.get(chosen_task)
            if embedding_value is not None:
                embeddings_tensor[sample_idx] = embedding_value

    return selected_labels, embeddings_tensor, selected_task_names


def train_epoch_with_hypernet(
    model,
    hypernet,
    learnable_task_embedding,
    train_loader,
    optimizer,
    criterion,
    device,
    grad_log_interval: Optional[int] = None,
):
    """
    Train one epoch with hypernet-generated LoRA weights and learnable task embeddings
    """
    # Keep base model frozen, only train hypernet and learnable embeddings
    model.eval()  # Base model stays frozen
    hypernet.train()  # Only hypernet is trainable
    learnable_task_embedding.train()  # Task embeddings are trainable
    
    total_loss = 0.0
    num_batches = 0
    num_valid_samples = 0
    num_total_samples = 0

    grad_log_interval = grad_log_interval or 0
    grad_log_counter = 0

    for batch_idx, batch in enumerate(train_loader):
        grad_log_counter += 1
        # Move batch to device
        data = batch["ct"].to(device)
        metadata = batch.get("metadata")
        if metadata is not None:
            metadata = metadata.to(device)
        slice_mask = batch.get("ct_mask")
        if slice_mask is not None:
            slice_mask = slice_mask.to(device)
        
        targets_dict = batch["target"]
        selected_labels, _, selected_task_names = _sample_tasks_from_multilabel_batch(
            targets_dict,
            None,
            device,
        )

        # Filter out samples with -1 labels (abstain condition)
        valid_mask = selected_labels != -1
        num_total_samples += len(selected_labels)
        
        # Skip batch if no valid samples
        if not valid_mask.any():
            continue
            
        # Filter inputs, targets, and task embeddings to only include valid samples
        valid_data = data[valid_mask]
        valid_labels = selected_labels[valid_mask]
        valid_task_names = [task for idx, task in enumerate(selected_task_names) if valid_mask[idx] and task is not None]
        if not valid_task_names:
            continue

        batch_task_embeddings = learnable_task_embedding(valid_task_names)
        valid_task_embeddings = batch_task_embeddings
        valid_metadata = metadata[valid_mask] if metadata is not None else None
        valid_slice_mask = slice_mask[valid_mask] if slice_mask is not None else None
        num_valid_samples += len(valid_labels)

        # Ensure labels are the right shape for BCEWithLogitsLoss
        if valid_labels.dim() > 1:
            valid_labels = valid_labels.squeeze()

        optimizer.zero_grad()

        # Use dynamic LoRA context manager
        with dynamic_lora_context(model, hypernet, 1.0) as lora_manager:
            if grad_log_interval and grad_log_counter % grad_log_interval == 0:
                lora_manager.set_grad_logging(1)
            else:
                lora_manager.set_grad_logging(None)
            # Forward pass with proper gradient tracking
            outputs = lora_manager.forward_with_dynamic_lora(
                valid_data,
                valid_task_embeddings,
                valid_metadata,
                masks=valid_slice_mask,
            )
            
            # Handle output shape carefully to avoid dimension mismatches
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(-1)  # Only squeeze the last dimension
            
            # Ensure outputs and targets are both 1D tensors with same length
            if outputs.dim() == 0:  # Scalar case
                outputs = outputs.unsqueeze(0)
            if valid_labels.dim() == 0:  # Scalar target case
                valid_labels = valid_labels.unsqueeze(0)
            
            # Ensure outputs has the same shape as targets
            if outputs.shape != valid_labels.shape:
                logger.warning(f"Output shape {outputs.shape} doesn't match label shape {valid_labels.shape}")
                continue
            
            loss = criterion(outputs, valid_labels)

            # Only proceed with backward if loss has gradients
            if loss.requires_grad:
                # Backward pass - gradients will flow through hypernet AND learnable embeddings
                loss.backward()

                if grad_log_interval and grad_log_counter % grad_log_interval == 0:
                    _log_grad_norm(hypernet, "Hypernet")
                    if learnable_task_embedding is not None:
                        _log_grad_norm(learnable_task_embedding, "Learnable task embedding")

                # Perform optimizer step within the context manager
                optimizer.step()
            else:
                logger.error("Loss does not require gradients! Cannot perform backward pass.")
                return 0.0

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(
                f"Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Valid samples: {len(valid_labels)}/{len(selected_labels)}"
            )
            if batch_idx == 0:
                logger.info(f"First batch - Loss: {loss.item():.6f}, Learning rate: {optimizer.param_groups[0]['lr']}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Training completed - Valid samples: {num_valid_samples}/{num_total_samples} ({100*num_valid_samples/num_total_samples:.1f}%)")
    return avg_loss


def train_epoch_standard(model, train_loader, optimizer, criterion, device):
    """
    Standard training without hypernet
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_valid_samples = 0
    num_total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        inputs = batch["ct"].to(device)
        targets_dict = batch["target"]
        selected_labels, _, _ = _sample_tasks_from_multilabel_batch(targets_dict, None, device)
        slice_mask = batch.get("ct_mask")
        if slice_mask is not None:
            slice_mask = slice_mask.to(device)

        # Filter out samples with -1 labels (abstain condition)
        valid_mask = selected_labels != -1
        num_total_samples += len(selected_labels)

        # Skip batch if no valid samples
        if not valid_mask.any():
            continue

        # Filter inputs and targets to only include valid samples
        valid_inputs = inputs[valid_mask]
        valid_targets = selected_labels[valid_mask]
        valid_slice_mask = slice_mask[valid_mask] if slice_mask is not None else None
        num_valid_samples += len(valid_targets)

        # Ensure targets are the right shape for BCEWithLogitsLoss
        if valid_targets.dim() > 1:
            valid_targets = valid_targets.squeeze()

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=torch.cuda.is_available()):
            if valid_slice_mask is not None and getattr(model, "supports_volume_mask", False):
                outputs = model(valid_inputs, mask=valid_slice_mask)
            else:
                outputs = model(valid_inputs)
            loss = criterion(outputs.squeeze(), valid_targets.squeeze())
            loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(
                f"Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Valid samples: {len(valid_targets)}/{len(selected_labels)}"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Training completed - Valid samples: {num_valid_samples}/{num_total_samples} ({100*num_valid_samples/num_total_samples:.1f}%)")
    return avg_loss
