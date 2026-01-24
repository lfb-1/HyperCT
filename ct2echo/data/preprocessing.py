"""
Preprocessing utilities for CT volumes.

Contains functions for:
- Converting slices to RGB images
- Applying DINOv3-specific preprocessing
- Preprocessing dataframe columns (merging duplicates)
"""

import torch
import numpy as np
import pandas as pd

from ct2echo.data.slice_utils import (
    _infer_slice_axis,
    _move_slice_axis_to_front,
    _remove_empty_slices_axis0,
    _ensure_length_divisible_by_three_axis0,
)
from ct2echo.preprocess.task_prompts import merge_columns


def convert_165_to_rgb_images(data):
    """
    Convert slices to RGB images (non-overlapping triplets).

    Args:
        data: numpy array containing slices along one axis (length divisible by 3)

    Returns:
        numpy array of shape (N, 3, height, width) for RGB images
    """
    slice_axis = _infer_slice_axis(data.shape)

    if data.shape[slice_axis] % 3 != 0:
        raise ValueError(f"Slice dimension {data.shape[slice_axis]} is not divisible by 3")

    if slice_axis != 0:
        data = np.moveaxis(data, slice_axis, 0)

    num_rgb_images = data.shape[0] // 3
    rgb_images = data.reshape(num_rgb_images, 3, *data.shape[1:])

    return rgb_images


def apply_dinov3_preprocessing(rgb_images):
    """Apply DINOv3-specific preprocessing to RGB images before tensor conversion."""
    # Convert to tensor and normalize per image
    processed_images = []

    for i in range(rgb_images.shape[0]):
        # Convert single RGB image to proper format
        # rgb_images[i] shape: (3, height, width)
        rgb_image = rgb_images[i]  # (3, H, W)

        # Normalize to [0, 1] range (assuming input is in CT hounsfield units)
        rgb_image = (rgb_image + 1000) / 2000  # Map [-1000, 1000] to [0, 1]
        rgb_image = np.clip(rgb_image, 0, 1)

        processed_image = torch.from_numpy(rgb_image).float().contiguous()
        processed_images.append(processed_image)

    # Stack all processed images
    processed_tensor = torch.stack(processed_images, dim=0)  # (num_images, 3, target_height, target_width)

    return processed_tensor


def preprocess_for_dinov3(data, return_details=False, empty_value=-1000.0, empty_tolerance=1.0):
    """Preprocess a CT volume for DINOv3 following slice filtering and normalization."""
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume for DINOv3 preprocessing, got shape {data.shape}")

    slice_axis = _infer_slice_axis(data.shape)
    moved, _ = _move_slice_axis_to_front(data, slice_axis)

    filtered, removal_info = _remove_empty_slices_axis0(moved, empty_value=empty_value, tolerance=empty_tolerance)
    if removal_info.get("all_empty", False):
        filtered = moved
        removal_info["removed"] = 0

    adjusted, adjustment_info = _ensure_length_divisible_by_three_axis0(filtered)

    rgb_images = adjusted.reshape(-1, 3, *adjusted.shape[1:])
    processed_tensor = apply_dinov3_preprocessing(rgb_images)

    if not return_details:
        return processed_tensor

    num_images = processed_tensor.shape[0]
    details = {
        "mask": torch.ones(num_images, dtype=torch.bool),
        "num_rgb_images": num_images,
        "slice_axis": slice_axis,
        "removed_empty_slices": removal_info.get("removed", 0),
        "added_slices": adjustment_info.get("added", 0),
        "dropped_slices": adjustment_info.get("removed", 0),
        "all_slices_empty": removal_info.get("all_empty", False),
    }

    return processed_tensor, details


def preprocess_dataframe_columns(df):
    """
    Merge duplicate columns based on the merge_columns dictionary.

    Args:
        df (pd.DataFrame): Input dataframe with potentially duplicate columns

    Returns:
        pd.DataFrame: Processed dataframe with merged columns
    """
    df_processed = df.copy()

    for output_column, source_columns in merge_columns.items():
        # Check which source columns exist in the dataframe
        existing_columns = [col for col in source_columns if col in df_processed.columns]

        if not existing_columns:
            continue  # Skip if none of the source columns exist

        # If output column name is different from the first existing column, we need to merge
        if len(existing_columns) > 1 or existing_columns[0] != output_column:
            # Create the merged column by taking the first non-null value across source columns
            merged_series = None
            for col in existing_columns:
                if merged_series is None:
                    merged_series = df_processed[col].copy()
                else:
                    # Fill null values in merged_series with values from current column
                    merged_series = merged_series.fillna(df_processed[col])

            # Add the merged column with the output name
            df_processed[output_column] = merged_series

            # Drop the original source columns if they're different from the output column
            columns_to_drop = [col for col in existing_columns if col != output_column]
            if columns_to_drop:
                df_processed = df_processed.drop(columns=columns_to_drop)
                print(f"Merged columns {existing_columns} -> {output_column}")

    return df_processed


__all__ = [
    "convert_165_to_rgb_images",
    "apply_dinov3_preprocessing",
    "preprocess_for_dinov3",
    "preprocess_dataframe_columns",
]
