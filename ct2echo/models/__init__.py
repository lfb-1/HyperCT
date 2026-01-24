"""
Models subpackage - Core model definitions.

Contains:
- archs: DINOv3_Encoder for CT scan processing
- hypernet: LoRA_Hypernet classes and utilities
"""

from ct2echo.models.archs import DINOv3_Encoder, load_dinov3_encoder
from ct2echo.models.hypernet import (
    LoRA_Hypernet,
    TaskEncoder,
    get_dinov3_target_modules_with_features,
)

__all__ = [
    # archs
    "DINOv3_Encoder",
    "load_dinov3_encoder",
    # hypernet
    "LoRA_Hypernet",
    "TaskEncoder",
    "get_dinov3_target_modules_with_features",
]
