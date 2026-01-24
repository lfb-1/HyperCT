"""
CT2ECHO: Low-Rank Hypernet for Unified Chest CT Analysis

Main package exports for convenient access to core components.
"""

from ct2echo.models.archs import DINOv3_Encoder
from ct2echo.models.hypernet import LoRA_Hypernet
from ct2echo.data.dataset import CTLoader, MTLDataset
from ct2echo.setup.model_setup import setup_dinov3_with_hypernet

__all__ = [
    # Models
    "DINOv3_Encoder",
    "LoRA_Hypernet",
    # Data
    "CTLoader",
    "MTLDataset",
    # Setup
    "setup_dinov3_with_hypernet",
]

__version__ = "0.1.0"
