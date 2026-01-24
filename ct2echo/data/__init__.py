"""
Data subpackage - Data loading and processing.

Contains:
- dataset: CTLoader, MTLDataset (backward compatibility shim)
- slice_utils: Slice manipulation utilities
- preprocessing: CT preprocessing utilities
- mtl_dataset: MTLDataset class
- ct_loader: CTLoader and collate functions
- metadata_utils: MetadataProcessor, MetadataStats
"""

# Backward compatible imports from dataset.py shim
from ct2echo.data.dataset import CTLoader, MTLDataset, dinov3_collate_fn
from ct2echo.data.metadata_utils import MetadataProcessor, MetadataStats

# Also export preprocessing utilities for direct access
from ct2echo.data.preprocessing import (
    preprocess_for_dinov3,
    preprocess_dataframe_columns,
)

__all__ = [
    # Core classes
    "CTLoader",
    "MTLDataset",
    "dinov3_collate_fn",
    # Metadata utilities
    "MetadataProcessor",
    "MetadataStats",
    # Preprocessing utilities
    "preprocess_for_dinov3",
    "preprocess_dataframe_columns",
]
