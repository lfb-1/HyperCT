"""
Slice manipulation utilities for CT volumes.

Contains functions for:
- Inferring slice axis from volume shape
- Moving slice axis for processing
- Removing empty slices
- Ensuring slice count divisibility by 3 for RGB conversion
"""

import numpy as np


def _infer_slice_axis(shape):
    """Choose an axis to treat as the slice dimension, preferring the last max-sized axis."""
    if len(shape) < 3:
        raise ValueError(f"Expected a 3D volume, got shape {shape}")
    max_dim = max(shape)
    candidates = [idx for idx, dim in enumerate(shape) if dim == max_dim]
    return candidates[-1]


def _move_slice_axis_to_front(data, slice_axis):
    """Move the chosen slice axis to the front and ensure contiguous layout."""
    if slice_axis == 0:
        return np.ascontiguousarray(data), slice_axis
    return np.ascontiguousarray(np.moveaxis(data, slice_axis, 0)), slice_axis


def _move_slice_axis_back(data, slice_axis):
    """Restore slice axis to its original position."""
    if slice_axis == 0:
        return data
    return np.moveaxis(data, 0, slice_axis)


def _remove_empty_slices_axis0(slices, empty_value=-1000.0, tolerance=1.0):
    """Remove slices that are uniformly equal to the empty value within tolerance."""
    if slices.ndim != 3:
        raise ValueError(f"Expected slices with shape (num_slices, H, W), got {slices.shape}")

    diff = np.abs(slices - empty_value)
    max_diff = diff.reshape(diff.shape[0], -1).max(axis=1)
    non_empty_mask = max_diff > tolerance

    if not non_empty_mask.any():
        return slices, {"removed": 0, "all_empty": True, "non_empty_mask": np.ones_like(non_empty_mask, dtype=bool)}

    filtered = np.ascontiguousarray(slices[non_empty_mask])
    removed = int((~non_empty_mask).sum())
    return filtered, {"removed": removed, "all_empty": False, "non_empty_mask": non_empty_mask}


def _ensure_length_divisible_by_three_axis0(slices):
    """Adjust slices so the leading dimension is divisible by 3 with minimal change."""
    if slices.ndim != 3:
        raise ValueError(f"Expected slices with shape (num_slices, H, W), got {slices.shape}")

    working = np.ascontiguousarray(slices)
    info = {"added": 0, "removed": 0}

    if working.shape[0] == 0:
        return working, info

    if working.shape[0] < 3:
        repeat_src = working[-1:]
        while working.shape[0] < 3:
            working = np.concatenate([working, repeat_src], axis=0)
            info["added"] += 1

    remainder = working.shape[0] % 3

    if remainder == 1:
        working = working[:-1]
        info["removed"] += 1
    elif remainder == 2:
        repeat_slice = working[-1:]
        working = np.concatenate([working, repeat_slice], axis=0)
        info["added"] += 1

    return working, info


def convert_164_to_165_slices(data):
    """Ensure the slice dimension is divisible by 3 with minimal adjustments."""
    slice_axis = _infer_slice_axis(data.shape)
    moved, _ = _move_slice_axis_to_front(data, slice_axis)
    adjusted, _ = _ensure_length_divisible_by_three_axis0(moved)
    restored = _move_slice_axis_back(adjusted, slice_axis)
    return restored


__all__ = [
    "_infer_slice_axis",
    "_move_slice_axis_to_front",
    "_move_slice_axis_back",
    "_remove_empty_slices_axis0",
    "_ensure_length_divisible_by_three_axis0",
    "convert_164_to_165_slices",
]
