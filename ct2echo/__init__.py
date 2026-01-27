"""
CT2ECHO: Low-Rank Hypernet for Unified Chest CT Analysis

Main package exports for convenient access to core components.
Uses lazy imports to avoid loading heavy ML dependencies when only utilities are needed.
"""

__version__ = "0.1.0"

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


def __getattr__(name):
    """Lazy import heavy dependencies only when accessed."""
    if name == "DINOv3_Encoder":
        from ct2echo.models.archs import DINOv3_Encoder
        return DINOv3_Encoder
    elif name == "LoRA_Hypernet":
        from ct2echo.models.hypernet import LoRA_Hypernet
        return LoRA_Hypernet
    elif name == "CTLoader":
        from ct2echo.data.dataset import CTLoader
        return CTLoader
    elif name == "MTLDataset":
        from ct2echo.data.dataset import MTLDataset
        return MTLDataset
    elif name == "setup_dinov3_with_hypernet":
        from ct2echo.setup.model_setup import setup_dinov3_with_hypernet
        return setup_dinov3_with_hypernet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
