"""
Backward compatibility shim - imports moved to submodules.

This file re-exports all symbols from the split modules for backward compatibility.
New code should import directly from:
- ct2echo.data.slice_utils
- ct2echo.data.preprocessing
- ct2echo.data.mtl_dataset
- ct2echo.data.ct_loader
"""

# Slice utilities
from ct2echo.data.slice_utils import (
    _infer_slice_axis,
    _move_slice_axis_to_front,
    _move_slice_axis_back,
    _remove_empty_slices_axis0,
    _ensure_length_divisible_by_three_axis0,
    convert_164_to_165_slices,
)

# Preprocessing utilities
from ct2echo.data.preprocessing import (
    convert_165_to_rgb_images,
    apply_dinov3_preprocessing,
    preprocess_for_dinov3,
    preprocess_dataframe_columns,
)

# MTL Dataset
from ct2echo.data.mtl_dataset import MTLDataset

# CT Loader
from ct2echo.data.ct_loader import CTLoader, dinov3_collate_fn


__all__ = [
    # slice_utils
    "_infer_slice_axis",
    "_move_slice_axis_to_front",
    "_move_slice_axis_back",
    "_remove_empty_slices_axis0",
    "_ensure_length_divisible_by_three_axis0",
    "convert_164_to_165_slices",
    # preprocessing
    "convert_165_to_rgb_images",
    "apply_dinov3_preprocessing",
    "preprocess_for_dinov3",
    "preprocess_dataframe_columns",
    # mtl_dataset
    "MTLDataset",
    # ct_loader
    "CTLoader",
    "dinov3_collate_fn",
]
