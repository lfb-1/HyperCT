"""
Checkpoint I/O utilities for saving and loading model checkpoints.

Contains:
- save_hypernet_checkpoint: Save merged hypernet weights and prediction head
- load_merged_hypernet_checkpoint: Load merged checkpoint
- load_trained_model_with_hypernet: Load trained model with hypernet
- save_model_with_hypernet_final: Save final trained model
- load_final_merged_checkpoint: Load final merged checkpoint
- load_epoch_merged_checkpoint: Load epoch merged checkpoint
- save_final_models: Save final model and hypernet states
"""

import os
import pickle

import torch
from loguru import logger

from ct2echo.utils.state_dict_utils import (
    _remap_hypernet_state_dict_for_current_modules,
    _extract_head_state_dict,
)


def save_hypernet_checkpoint(hypernet, base_model, epoch, output_dir):
    """Save merged hypernet weights and any prediction head state for the epoch."""
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    merged_checkpoint = {
        'hypernet_state_dict': hypernet.state_dict(),
        'epoch': epoch
    }

    head_attr, head_state = _extract_head_state_dict(base_model)
    if head_state is not None:
        merged_checkpoint['head_attr'] = head_attr
        merged_checkpoint['head_state_dict'] = head_state
    else:
        logger.warning("No prediction head (mlp_head/classifier) found on base model; saving hypernet only.")

    merged_file = os.path.join(epoch_dir, "merged_hypernet_checkpoint.pth")
    torch.save(merged_checkpoint, merged_file)
    if head_state is not None:
        logger.info(f"Saved merged hypernet+{head_attr} checkpoint to {merged_file}")
    else:
        logger.info(f"Saved hypernet checkpoint (no prediction head) to {merged_file}")


def load_merged_hypernet_checkpoint(base_model, hypernet, checkpoint_path, device="cuda"):
    """
    Load merged hypernet and base model mlp_head from a single checkpoint

    Args:
        base_model: The base encoder model (e.g., DINOv3_Encoder)
        hypernet: The LoRA_Hypernet instance
        checkpoint_path: Path to the merged checkpoint file
        device: Device to load tensors to

    Returns:
        Tuple of (loaded_base_model, loaded_hypernet, success_flags, epoch)
    """
    success_flags = {"hypernet": False, "head": False}
    epoch = None

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load hypernet weights
            if 'hypernet_state_dict' in checkpoint:
                raw_state_dict = checkpoint['hypernet_state_dict']
                remapped_state, remap_meta = _remap_hypernet_state_dict_for_current_modules(raw_state_dict, hypernet)
                load_result = hypernet.load_state_dict(remapped_state, strict=False)

                if remap_meta["remapped"]:
                    remap_samples = ", ".join(f"{src}→{dst}" for src, dst in remap_meta["remapped"][:3])
                    logger.info(
                        "♻️ Remapped legacy hypernet keys ({} total, showing first 3): {}",
                        len(remap_meta["remapped"]),
                        remap_samples,
                    )
                    if len(remap_meta["remapped"]) > 3:
                        logger.info("…and {} more", len(remap_meta["remapped"]) - 3)

                if remap_meta["dropped"]:
                    logger.warning(
                        "⚠️ Dropped incompatible hypernet keys: {}",
                        ", ".join(remap_meta["dropped"][:5]),
                    )

                if load_result.missing_keys:
                    logger.warning("⚠️ Missing hypernet parameters: {}", ", ".join(load_result.missing_keys))
                if load_result.unexpected_keys:
                    logger.warning("⚠️ Unexpected hypernet parameters: {}", ", ".join(load_result.unexpected_keys))

                success_flags["hypernet"] = True
                logger.info(f"✅ Loaded hypernet weights from: {checkpoint_path}")

            # Load prediction head weights (mlp_head/classifier)
            head_state = None
            head_attr = None

            if 'head_state_dict' in checkpoint:
                head_state = checkpoint['head_state_dict']
                head_attr = checkpoint.get('head_attr', 'mlp_head')
            elif 'mlp_head_state_dict' in checkpoint:
                head_state = checkpoint['mlp_head_state_dict']
                head_attr = 'mlp_head'

            if head_state is not None and head_attr is not None:
                if hasattr(base_model, head_attr):
                    getattr(base_model, head_attr).load_state_dict(head_state)
                    success_flags["head"] = True
                    logger.info(f"✅ Loaded {head_attr} weights from: {checkpoint_path}")
                else:
                    logger.warning(f"⚠️ Checkpoint contains {head_attr} weights but base model lacks this attribute")

            # Get epoch info if available
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                logger.info(f"✅ Loaded checkpoint from epoch: {epoch}")

        except Exception as e:
            logger.error(f"❌ Failed to load merged checkpoint from {checkpoint_path}: {e}")
    else:
        logger.warning(f"⚠️ Merged checkpoint not found at {checkpoint_path}")

    return base_model, hypernet, success_flags, epoch


def load_trained_model_with_hypernet(base_model, hypernet, checkpoint_dir, epoch_or_final="final", device="cuda"):
    """
    Load a trained model with hypernet and base model mlp_head from checkpoints

    Args:
        base_model: The base encoder model (e.g., DINOv3_Encoder)
        hypernet: The LoRA_Hypernet instance
        checkpoint_dir: Directory containing the checkpoints
        epoch_or_final: Either "final" or epoch number (e.g., 5)
        device: Device to load tensors to

    Returns:
        Tuple of (loaded_base_model, loaded_hypernet, success_flags)
    """
    success_flags = {"hypernet": False, "head": False}

    if epoch_or_final == "final":
        # Load from final directory
        hypernet_path = os.path.join(checkpoint_dir, "final", "final_hypernet.pth")
        head_candidates = [
            ("mlp_head", os.path.join(checkpoint_dir, "final", "final_base_model_mlp_head.pth")),
            ("classifier", os.path.join(checkpoint_dir, "final", "final_base_model_classifier.pth")),
        ]
    else:
        # Load from epoch directory
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch_or_final}")
        hypernet_path = os.path.join(epoch_dir, "hypernet.pth")
        head_candidates = [
            ("mlp_head", os.path.join(epoch_dir, "base_model_mlp_head.pth")),
            ("classifier", os.path.join(epoch_dir, "base_model_classifier.pth")),
        ]

    head_attr = None
    head_path = None
    for attr, path in head_candidates:
        if os.path.exists(path):
            head_attr, head_path = attr, path
            break

    # Load hypernet
    if os.path.exists(hypernet_path):
        try:
            raw_state_dict = torch.load(hypernet_path, map_location=device)
            remapped_state, remap_meta = _remap_hypernet_state_dict_for_current_modules(raw_state_dict, hypernet)
            load_result = hypernet.load_state_dict(remapped_state, strict=False)

            if remap_meta["remapped"]:
                remap_samples = ", ".join(f"{src}→{dst}" for src, dst in remap_meta["remapped"][:3])
                logger.info(
                    "♻️ Remapped legacy hypernet keys ({} total, showing first 3): {}",
                    len(remap_meta["remapped"]),
                    remap_samples,
                )
                if len(remap_meta["remapped"]) > 3:
                    logger.info("…and {} more", len(remap_meta["remapped"]) - 3)

            if remap_meta["dropped"]:
                logger.warning(
                    "⚠️ Dropped incompatible hypernet keys: {}",
                    ", ".join(remap_meta["dropped"][:5]),
                )

            if load_result.missing_keys:
                logger.warning("⚠️ Missing hypernet parameters: {}", ", ".join(load_result.missing_keys))
            if load_result.unexpected_keys:
                logger.warning("⚠️ Unexpected hypernet parameters: {}", ", ".join(load_result.unexpected_keys))

            logger.info(f"✅ Loaded hypernet from: {hypernet_path}")
            success_flags["hypernet"] = True
        except Exception as e:
            logger.error(f"❌ Failed to load hypernet from {hypernet_path}: {e}")
    else:
        logger.warning(f"⚠️ Hypernet checkpoint not found at {hypernet_path}")

    # Load base model prediction head
    if head_path is not None:
        try:
            head_state = torch.load(head_path, map_location=device)
            if hasattr(base_model, head_attr):
                getattr(base_model, head_attr).load_state_dict(head_state)
                logger.info(f"✅ Loaded base model {head_attr} from: {head_path}")
                success_flags["head"] = True
            else:
                logger.warning(
                    f"⚠️ Found checkpoint for {head_attr} at {head_path} but base model lacks this attribute"
                )
        except Exception as e:
            logger.error(f"❌ Failed to load base model {head_attr or 'prediction head'} from {head_path}: {e}")
    else:
        logger.warning("⚠️ Base model prediction head checkpoint not found")

    return base_model, hypernet, success_flags


def save_model_with_hypernet_final(base_model, hypernet, output_dir, args=None):
    """
    Save final trained model and hypernet with merged checkpoint approach

    Args:
        base_model: The base encoder model (e.g., DINOv3_Encoder)
        hypernet: The LoRA_Hypernet instance
        output_dir: Output directory
        args: Training arguments to save
    """
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    # Save merged final checkpoint (hypernet + prediction head)
    merged_final_checkpoint = {
        'hypernet_state_dict': hypernet.state_dict(),
        'epoch': 'final'
    }

    head_attr, head_state = _extract_head_state_dict(base_model)
    if head_state is not None:
        merged_final_checkpoint['head_attr'] = head_attr
        merged_final_checkpoint['head_state_dict'] = head_state

    final_merged_file = os.path.join(final_dir, "final_merged_checkpoint.pth")
    torch.save(merged_final_checkpoint, final_merged_file)
    if head_state is not None:
        logger.info(f"Saved final merged hypernet+{head_attr} checkpoint to {final_merged_file}")
    else:
        logger.info(f"Saved final merged hypernet checkpoint (no prediction head) to {final_merged_file}")

    # Save base model without prediction head for reference
    if head_attr is not None:
        head_prefix = f"{head_attr}."
        model_state_without_head = {k: v for k, v in base_model.state_dict().items() if not k.startswith(head_prefix)}
        final_model_file = os.path.join(final_dir, f"final_model_without_{head_attr}.pth")
        torch.save(model_state_without_head, final_model_file)
        logger.info(f"Saved final model (without {head_attr}) to {final_model_file}")

        head_file = os.path.join(final_dir, f"final_base_model_{head_attr}.pth")
        torch.save(head_state, head_file)
        logger.info(f"Saved final base model {head_attr} to {head_file}")
    else:
        final_model_file = os.path.join(final_dir, "final_model_without_head.pth")
        torch.save(base_model.state_dict(), final_model_file)
        logger.info(f"Saved final model (no head removed) to {final_model_file}")

    # Save complete model for convenience (includes everything)
    final_complete_model_file = os.path.join(final_dir, "final_complete_model.pth")
    torch.save(base_model.state_dict(), final_complete_model_file)
    logger.info(f"Saved final complete model to {final_complete_model_file}")

    # Save training arguments if provided
    if args is not None:
        args_file = os.path.join(output_dir, "training_args.pkl")
        with open(args_file, "wb") as f:
            pickle.dump(vars(args), f)
        logger.info(f"Saved training arguments to {args_file}")


def load_final_merged_checkpoint(base_model, hypernet, output_dir, device="cuda"):
    """
    Load final merged checkpoint (hypernet + mlp_head)

    Args:
        base_model: The base encoder model (e.g., DINOv3_Encoder)
        hypernet: The LoRA_Hypernet instance
        output_dir: Directory containing the final checkpoint
        device: Device to load tensors to

    Returns:
        Tuple of (loaded_base_model, loaded_hypernet, success_flags)
    """
    final_merged_path = os.path.join(output_dir, "final", "final_merged_checkpoint.pth")
    return load_merged_hypernet_checkpoint(base_model, hypernet, final_merged_path, device)[:3]  # Exclude epoch


def load_epoch_merged_checkpoint(base_model, hypernet, output_dir, epoch, device="cuda"):
    """
    Load epoch merged checkpoint (hypernet + mlp_head)

    Args:
        base_model: The base encoder model (e.g., DINOv3_Encoder)
        hypernet: The LoRA_Hypernet instance
        output_dir: Directory containing the epoch checkpoints
        epoch: Epoch number to load
        device: Device to load tensors to

    Returns:
        Tuple of (loaded_base_model, loaded_hypernet, success_flags, epoch)
    """
    epoch_merged_path = os.path.join(output_dir, f"epoch_{epoch}", "merged_hypernet_checkpoint.pth")
    return load_merged_hypernet_checkpoint(base_model, hypernet, epoch_merged_path, device)


def save_final_models(model, hypernet, args, output_dir):
    """Save final model and hypernet states"""
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    if hypernet is not None:
        # Save final hypernet
        final_hypernet_file = os.path.join(final_dir, "final_hypernet.pth")
        torch.save(hypernet.state_dict(), final_hypernet_file)
        logger.info(f"Saved final hypernet to {final_hypernet_file}")

    # Save final base model
    final_model_file = os.path.join(final_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_file)
    logger.info(f"Saved final model to {final_model_file}")

    # Save training arguments
    args_file = os.path.join(output_dir, "training_args.pkl")
    with open(args_file, "wb") as f:
        pickle.dump(vars(args), f)
    logger.info(f"Saved training arguments to {args_file}")

    logger.info(f"All results saved to: {output_dir}")


__all__ = [
    "save_hypernet_checkpoint",
    "load_merged_hypernet_checkpoint",
    "load_trained_model_with_hypernet",
    "save_model_with_hypernet_final",
    "load_final_merged_checkpoint",
    "load_epoch_merged_checkpoint",
    "save_final_models",
]
