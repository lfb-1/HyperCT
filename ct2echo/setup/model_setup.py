"""
Model setup and hypernet integration functions
"""

from typing import Optional

import torch
from ct2echo.models.hypernet import (
    LoRA_Hypernet,
    get_dinov3_target_modules_with_features,
)
from ct2echo.models.archs import load_dinov3_encoder
from loguru import logger


def setup_dinov3_with_hypernet(
    dinov3_path,
    config,
    device,
    metadata_dim: int = 0,
    train_base_mlp_head: Optional[bool] = None,
    task_embedding_dim: Optional[int] = None,
):
    """
    Setup DINOv3_Encoder with LoRA_Hypernet integration

    Args:
        dinov3_path: Path to DINOv3 model directory
        config: Configuration object
        device: Target device

    Returns:
        Tuple of (model, hypernet, target_modules)
    """

    if train_base_mlp_head is None:
        train_base_mlp_head = getattr(config.model, "train_base_mlp_head", False)

    # Load DINOv3 model
    logger.info(f"Loading DINOv3 model from {dinov3_path}")
    model = load_dinov3_encoder(
        dinov3_path,
        feature_dim=768,  # DINOv3 hidden size
        num_images=55,  # 165 slices / 3 = 55 RGB images
        aggregation="mean",  # Can be 'mean', 'max', 'cls_token'
        freeze_base=True,
        train_mlp_head=train_base_mlp_head,
    )
    model = model.to(device)

    logger.info("DINOv3 model loaded successfully")
    if train_base_mlp_head:
        logger.info("🟢 DINOv3 mlp_head parameters are trainable")

    requested_target_modules = getattr(config.model, "target_modules", None)
    if requested_target_modules:
        logger.info(
            "Using target modules specified in configuration: %s",
            requested_target_modules,
        )

    # Get target modules and features from DINOv3 model
    target_modules, in_features, out_features = get_dinov3_target_modules_with_features(
        model,
        allowed_modules=requested_target_modules,
    )

    logger.info(f"Found {len(target_modules)} DINOv3 target modules: {sorted(target_modules)}")
    logger.info("Target modules: attention (q/k/v/o) + MLP (up/down) projections")

    # Get head input size (768 for DINOv3)
    head_in_size = 768

    # For now, use the basic LoRA_Hypernet with DINOv3 parameters
    # TODO: Add from_dinov3_encoder factory methods to other hypernet classes
    logger.info("Using LoRA_Hypernet (MLP-based) for DINOv3")
    logger.info("⚠️ Advanced hypernet architectures (transformer/attention-only) not yet implemented for DINOv3")

    # Create MLP-based hypernetwork using the basic constructor
    # We'll need to manually set up the parameters since from_dinov3_encoder doesn't exist yet
    hypernet = LoRA_Hypernet(
        target_modules=list(target_modules),
        in_features=in_features,
        out_features=out_features,
        lora_rank=config.model.lora_rank,
        task_embedding_dim=task_embedding_dim or getattr(config.model, "learnable_embedding_dim", 128),
        latent_size=config.model.latent_size,
        head_in_size=head_in_size,
        device=device,
        metadata_dim=metadata_dim,
    )

    # Move hypernet to device
    hypernet = hypernet.to(device)

    # Verify consistency between extracted target modules and hypernet target modules
    assert set(target_modules) == set(
        hypernet.target_modules
    ), f"Mismatch between extracted target modules {target_modules} and hypernet target modules {hypernet.target_modules}"

    logger.info("DINOv3_Encoder setup complete:")
    logger.info(f"  - Architecture: {type(hypernet).__name__}")
    logger.info(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"  - Hypernet parameters: {sum(p.numel() for p in hypernet.parameters()):,}")
    logger.info(
        f"  - Trainable hypernet parameters: {sum(p.numel() for p in hypernet.parameters() if p.requires_grad):,}"
    )
    logger.info(f"  - Target modules ({len(target_modules)}): {target_modules}")
    logger.info(f"  - LoRA rank: {hypernet.lora_rank}")
    logger.info(f"  - Feature dimensions: 768 (DINOv3 standard)")
    logger.info(
        "  - Task embedding projection: %d → %d",
        getattr(hypernet, "task_embedding_input_dim", task_embedding_dim or 0),
        head_in_size,
    )

    return model, hypernet, target_modules
