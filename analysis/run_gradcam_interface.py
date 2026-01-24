"""Command-line entry point for generating Grad-CAM visualisations with the DINOv3 hypernet.

This script wraps ``generate_dinov3_hypernet_gradcam`` and handles checkpoint loading,
input preprocessing, and result persistence under ``save_weights/gradcam``.
"""

from __future__ import annotations

import sys
import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

# Add package root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from ct2echo.data.dataset import preprocess_for_dinov3, preprocess_dataframe_columns
from analysis.gradcam_hypernet import (
    generate_dinov3_hypernet_gradcam,
    generate_dinov3_precomputed_lora_gradcam,
)
from ct2echo.utils.utils import init_azure
from ct2echo.data.metadata_utils import MetadataProcessor
from ct2echo.preprocess.task_prompts import (
    label_checking,
    label_query_func,
    radio_label_checking,
    radio_label_query_func,
)
from volumentations import Compose, CenterCrop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_ROOT = Path(
    "/mnt/azureml/cr/j/a83faa40078a487a93c6cb36d9894c3d/cap/data-capability/wd/checkpoint_dir/"
)
FINAL_DIR = CHECKPOINT_ROOT / "final"
TASK_EMBED_PATH = FINAL_DIR / "final_learnable_task_embeddings.pth"
PRECOMPUTED_LORA_PATH = Path("./precomputed_lora_weights.pt")


DEFAULT_DINOV3_PATH = Path(
    "/mnt/azureml/cr/j/a83faa40078a487a93c6cb36d9894c3d/cap/data-capability/wd/dinov3_dir/DINOV3_ViTb16"
)
DEFAULT_CONFIG_PATH = Path("conf/model/dinov3.yaml")
DEFAULT_OUTPUT_ROOT = Path("save_weights") / "gradcam"


# Evaluation transform mirrors dataset.py test pipeline
DINOV3_EVAL_TRANSFORM = Compose(
    [CenterCrop((144, 144, 164), always_apply=True, p=1.0)],
    p=1.0,
)

TASK_METRIC_COLUMNS: Dict[str, List[str]] = {
    "reduced_rv_systolic_function": ["ECHO_rv_systolic_function_value"],
    "reduced_lv_systolic_function": ["ECHO_lvef_value"],
    "pulmonary_hypertension": ["ECHO_pasp_value"],
    "atrial_chamber_enlargement": ["ECHO_la_volume_index_value"],
    "ventricular_enlargement": ["ECHO_lv_d_measurement"],
    "left_atrial_filling_pressure": [
        "ECHO_e_a_ratio",
        "ECHO_la_volume_index_value",
        "ECHO_tr_max_velocity_value",
    ],
    "right_atrial_filling_pressure": ["ECHO_tr_max_velocity_value"],
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _create_overlay_grid(images: List[np.ndarray], cols: int, output_path: Path) -> Optional[Path]:
    if not images:
        return None

    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if images[0].ndim == 3 else 1
    grid = np.zeros((rows * h, cols * w, channels), dtype=images[0].dtype)

    for idx, img in enumerate(images):
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        r = idx // cols
        c = idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    cv2.imwrite(str(output_path), grid)
    return output_path


def _row_satisfies_requirements(row: pd.Series, requirements: List[str], requirement_type: str) -> bool:
    if requirement_type == "all_required":
        return all((col in row.index and not pd.isna(row[col])) for col in requirements)
    if requirement_type == "any_in_group":
        return any(col in row.index and not pd.isna(row[col]) for col in requirements)
    raise ValueError(f"Unknown requirement type: {requirement_type}")


def _extract_positive_tasks_from_row(row: pd.Series, candidate_tasks: Iterable[str]) -> List[str]:
    positives: List[str] = []
    row_df = row.to_frame().T
    for task in candidate_tasks:
        # Support both echo-derived tasks and radiology tasks
        if task in label_checking:
            cfg = label_checking[task]
            query = label_query_func.get(task)
        elif task in radio_label_checking:
            cfg = radio_label_checking[task]
            query = radio_label_query_func.get(task)
        else:
            continue

        if not _row_satisfies_requirements(row, cfg["requirements"], cfg["type"]):
            continue
        if not query:
            continue
        try:
            result = row_df.query(query)
            label = 1 if len(result) > 0 else 0
        except Exception:
            label = 0
        if label == 1:
            positives.append(task)
    return positives


def _extract_positive_radio_labels_from_row(row: pd.Series) -> List[str]:
    """Return all radiology labels that are positive for this row.

    Uses the radio-specific requirement and query definitions from
    ``task_prompts.radio_label_checking`` and ``radio_label_query_func``.
    """

    positives: List[str] = []
    row_df = row.to_frame().T
    for task, cfg in radio_label_checking.items():
        if not _row_satisfies_requirements(row, cfg["requirements"], cfg["type"]):
            continue
        query = radio_label_query_func.get(task)
        if not query:
            continue
        try:
            result = row_df.query(query)
            label = 1 if len(result) > 0 else 0
        except Exception:
            label = 0
        if label == 1:
            positives.append(task)
    return positives


def _load_blob_volume(
    blob_path: str,
    blob_service_client,
    container: str,
) -> np.ndarray:
    """Download a CT volume from Azure Blob Storage using dataset.py conventions."""

    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_path)
    download_stream = blob_client.download_blob()
    download_arr = np.frombuffer(download_stream.readall(), dtype=np.float64)
    expected_voxels = 164 * 164 * 164
    if download_arr.size == expected_voxels:
        data = download_arr.reshape(164, 164, 164)
    else:
        raise ValueError(
            f"Unexpected blob size for {blob_path}: expected {expected_voxels} voxels, got {download_arr.size}"
        )
    data = np.clip(data, -1000, 1000)
    return data


def _load_volume(
    volume_path: str,
    blob_service_client,
    azure_container: str = "echo-data-lake",
) -> np.ndarray:
    """Load volume from Azure blob storage; local files are not supported."""

    if blob_service_client is None:
        raise ValueError("Blob service client is required for loading CT volumes")

    blob_path = str(volume_path)
    if "://" in blob_path:
        _, remainder = blob_path.split("://", 1)
        if "/" in remainder:
            remainder = remainder.split("/", 1)[1]
        blob_path = remainder
    blob_path = blob_path.lstrip("/")

    return _load_blob_volume(blob_path, blob_service_client, azure_container)


def _filter_blank_triplets(triplets: torch.Tensor, min_range: float = 5e-3, min_variance: float = 1e-6) -> torch.Tensor:
    """Remove triplets that lack contrast so overlays always include anatomy."""

    flattened = triplets.view(triplets.shape[0], -1)
    intensity_range = flattened.max(dim=1).values - flattened.min(dim=1).values
    variance = flattened.var(dim=1, unbiased=False)
    non_zero = flattened.abs().sum(dim=1) > 0

    valid_mask = non_zero & ((intensity_range > min_range) | (variance > min_variance))

    if not valid_mask.any():
        raise ValueError("All triplets appear empty or flat after preprocessing; cannot proceed")

    if valid_mask.all():
        return triplets

    logger.debug(
        "Filtered out %d flat triplets (kept %d)",
        (~valid_mask).sum().item(),
        valid_mask.sum().item(),
    )

    return triplets[valid_mask]


def _sanitize_for_filename(value: Any) -> str:
    if value is None:
        return "NA"
    text = str(value).strip()
    if not text:
        return "NA"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _format_metric_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (np.floating, np.integer)):
        return f"{float(value):.2f}"
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    try:
        numeric = float(value)
        return f"{numeric:.2f}"
    except (TypeError, ValueError):
        cleaned = _sanitize_for_filename(value)
        return cleaned if cleaned else "NA"


def _build_task_filename(
    task: str,
    volume_path: str,
    row: pd.Series,
    row_index: int,
) -> str:
    path_component = _sanitize_for_filename(volume_path)
    empi_component = _sanitize_for_filename(row.get("EMPI", "NA"))
    accession_component = _sanitize_for_filename(row.get("CT_AccessionNumber", "NA"))

    metric_columns = TASK_METRIC_COLUMNS.get(task, [])
    metric_parts: List[str] = []
    for column in metric_columns:
        value = row.get(column, None)
        if pd.isna(value):
            continue
        metric_parts.append(f"{column}-{_format_metric_value(value)}")

    if not metric_parts:
        metric_parts.append("metrics-NA")

    metrics_component = _sanitize_for_filename("__".join(metric_parts))

    components = [
        _sanitize_for_filename(task),
        f"idx-{row_index}",
        f"empi-{empi_component}",
        f"acc-{accession_component}",
        f"metrics-{metrics_component}",
    ]

    return "__".join(component for component in components if component)


def _collect_metric_values(row: pd.Series, task: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for column in TASK_METRIC_COLUMNS.get(task, []):
        value = row.get(column, None)
        if pd.isna(value):
            continue
        if isinstance(value, (np.integer, int)):
            metrics[column] = int(value)
        elif isinstance(value, (np.floating, float)):
            metrics[column] = float(value)
        else:
            metrics[column] = str(value)
    return metrics


def _to_jsonable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def _prepare_model_input(array: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw array into [1, 3, N, H, W] tensor, dropping blank slices."""
    tensor = torch.from_numpy(array).float()

    if tensor.dim() == 5:  # e.g. [B, 3, N, H, W] or [B, N, 3, H, W]
        if tensor.shape[0] != 1:
            raise ValueError("Grad-CAM expects a single volume (batch dimension == 1)")
        if tensor.shape[1] == 3:
            triplets = tensor.squeeze(0).permute(1, 0, 2, 3).contiguous()
        elif tensor.shape[2] == 3:
            triplets = tensor.squeeze(0).contiguous()
        else:
            raise ValueError(f"Cannot infer channel dimension from tensor shape {tuple(tensor.shape)}")
        triplets = _filter_blank_triplets(triplets)
        return triplets.permute(1, 0, 2, 3).contiguous().unsqueeze(0), triplets

    if tensor.dim() == 4:
        # Possible shapes: [3, N, H, W], [N, 3, H, W]
        if tensor.shape[0] == 3:
            triplets = tensor.permute(1, 0, 2, 3).contiguous()
        elif tensor.shape[1] == 3:
            triplets = tensor.contiguous()
        else:
            raise ValueError(f"Cannot identify channel axis in tensor shape {tuple(tensor.shape)}")
        triplets = _filter_blank_triplets(triplets)
        return triplets.permute(1, 0, 2, 3).contiguous().unsqueeze(0), triplets

    if tensor.dim() == 3:
        volume = tensor.numpy().astype(np.float32)

        transformed = DINOV3_EVAL_TRANSFORM(image=volume)["image"]
        processed, _details = preprocess_for_dinov3(transformed, return_details=True)

        triplets = _filter_blank_triplets(processed.contiguous())
        return triplets.permute(1, 0, 2, 3).contiguous().unsqueeze(0), triplets

    raise ValueError(
        "Input volume must resolve to shape [1, 3, N, H, W] after preprocessing; "
        f"got {tuple(tensor.shape)}"
    )


def _load_task_embeddings(tasks: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Load learnable task embeddings and return dict limited to selected tasks."""
    checkpoint = torch.load(TASK_EMBED_PATH, map_location="cpu")
    task_names = checkpoint["task_names"]
    embeddings = checkpoint["embeddings"]  # [num_tasks, dim]

    available = {name: embeddings[idx] for idx, name in enumerate(task_names)}
    missing = [task for task in tasks if task not in available]
    if missing:
        raise ValueError(f"Requested tasks not found in embeddings checkpoint: {missing}")

    return {task: available[task].clone() for task in tasks}


def _save_task_result(
    task_dir: Path,
    analysis,
    task: str,
    file_stem: str,
    grid_cols: int,
    sample_metadata: Optional[torch.Tensor] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist Grad-CAM artefacts for a single task-sample pair."""

    result = analysis.per_task.get(task)
    if result is None:
        logger.warning(f"Grad-CAM analysis did not return results for task {task}")
        return

    overlay_root = task_dir / "overlay_grids"
    heatmap_root = task_dir / "heatmap_grids"
    summary_root = task_dir / "summaries"
    metadata_root = task_dir / "metadata"

    overlay_root.mkdir(parents=True, exist_ok=True)
    heatmap_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    if sample_metadata is not None:
        metadata_root.mkdir(parents=True, exist_ok=True)

    prob = float(1.0 / (1.0 + math.exp(-result.logit)))

    heatmap_stack = result.heatmap.detach().cpu().numpy()
    heatmap_images: List[np.ndarray] = []
    for heat_slice in heatmap_stack:
        heat_uint8 = (heat_slice * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heatmap_images.append(heat_color)

    heat_grid_path = _create_overlay_grid(
        heatmap_images,
        grid_cols,
        heatmap_root / f"{file_stem}.png",
    )

    overlay_grid_path = _create_overlay_grid(
        result.overlays,
        grid_cols,
        overlay_root / f"{file_stem}.png",
    )

    summary = {
        "task": task,
        "logit": result.logit,
        "prob": prob,
        "heatmap_grid": str(heat_grid_path) if heat_grid_path else None,
        "overlay_grid": str(overlay_grid_path) if overlay_grid_path else None,
        "lora_layer_norms": {
            module: tensor.detach().cpu().tolist() for module, tensor in result.lora_layer_norms.items()
        },
        "extra_info": extra_info or {},
    }

    summary_path = summary_root / f"{file_stem}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if sample_metadata is not None:
        metadata_path = metadata_root / f"{file_stem}.npy"
        np.save(metadata_path, sample_metadata.cpu().numpy())


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM maps for the DINOv3 hypernet")
    parser.add_argument("--ct-volume", type=Path, help="Path to CT volume (.npy/.npz/.pt)")
    parser.add_argument(
        "--csv",
        type=Path,
        help="CSV file containing per-study metadata with a column pointing to CT volume paths",
    )
    parser.add_argument(
        "--csv-col",
        type=str,
        default="Path_New",
        help="Column inside the CSV that stores the CT path (default: Path_New)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional CSV column to use as sample identifier when saving outputs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of rows processed from the CSV",
    )
    parser.add_argument(
        "--contrast-csv",
        type=Path,
        default=None,
        help="Optional contrast exclusion CSV; defaults to contrast_issue.csv next to the metadata file",
    )
    parser.add_argument(
        "--positive-tasks",
        nargs="*",
        default=None,
        help="Task names with positive labels to visualise (defaults to all tasks)",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=8,
        help="Number of columns when building overlay grids",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of task names to analyse; defaults to all available tasks",
    )
    parser.add_argument(
        "--dinov3-path",
        type=Path,
        default=DEFAULT_DINOV3_PATH,
        help="Directory containing pretrained DINOv3 weights",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model config used to instantiate the hypernet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (defaults to save_weights/gradcam/<timestamp>)",
    )
    parser.add_argument(
        "--azure-container",
        type=str,
        default="echo-data-lake",
        help="Azure Blob Storage container name (default: echo-data-lake)",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=30,
        help="Number of positive visualisations to generate per task",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g. cuda, cuda:1, cpu)",
    )
    parser.add_argument(
        "--use-precomputed-lora",
        action="store_true",
        help=(
            "Use precomputed LoRA weights (from precomputed_lora_weights.pt) "
            "and run true Grad-CAM on task-specific models instead of input-gradient maps."
        ),
    )
    parser.add_argument(
        "--cam-layer-index",
        type=int,
        default=-1,
        help=(
            "Hidden-state index inside DINOv3 to use for Grad-CAM (supports negative indices; "
            "-1 corresponds to the final transformer layer)."
        ),
    )
    parser.add_argument(
        "--cam-threshold",
        type=float,
        default=None,
        help=(
            "Optional per-volume threshold applied after Grad-CAM normalisation. "
            "Values below this float (expected 0-1) are zeroed to suppress low-confidence slices."
        ),
    )
    parser.add_argument(
        "--cam-smoothing-kernel",
        type=int,
        default=3,
        help=(
            "Odd kernel size for avg-pooling-based smoothing applied to hypernet saliency maps. "
            "Use 1 to disable smoothing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ct_volume and not args.csv:
        raise ValueError("Provide either --ct-volume or --csv for batch processing")

    try:
        blob_service_client = init_azure()
    except Exception as exc:  # pragma: no cover - credential configuration dependent
        raise RuntimeError("Failed to initialise Azure BlobServiceClient; check credentials") from exc

    checkpoint = torch.load(TASK_EMBED_PATH, map_location="cpu")
    all_tasks = checkpoint["task_names"]
    selected_tasks = args.tasks or all_tasks

    if args.positive_tasks:
        selected_tasks = [task for task in selected_tasks if task in set(args.positive_tasks)]
        if not selected_tasks:
            raise ValueError("None of the specified --positive-tasks overlap with available tasks")

    task_embeddings_all = _load_task_embeddings(selected_tasks)

    # Optionally load precomputed LoRA weights for true Grad-CAM
    precomputed_lora = None
    if args.use_precomputed_lora:
        if not PRECOMPUTED_LORA_PATH.exists():
            raise FileNotFoundError(
                f"--use-precomputed-lora set but {PRECOMPUTED_LORA_PATH} not found; "
                "run precompute_task_specific_models.py first."
            )
        lora_payload = torch.load(PRECOMPUTED_LORA_PATH, map_location="cpu")
        precomputed_lora = lora_payload.get("precomputed_lora", {})

    output_root = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_per_task = max(1, args.samples_per_task)

    collected_counts = {task: 0 for task in selected_tasks}
    task_output_dirs = {task: output_root / task for task in selected_tasks}
    generated_any = False

    if args.csv:
        df = pd.read_csv(args.csv)
        df = preprocess_dataframe_columns(df)
        if args.csv_col not in df.columns:
            raise ValueError(f"Column '{args.csv_col}' not found in CSV {args.csv}")
        if args.id_column and args.id_column not in df.columns:
            raise ValueError(f"Identifier column '{args.id_column}' not present in CSV")

        contrast_csv = args.contrast_csv
        if contrast_csv is None:
            candidate = args.csv.parent / "contrast_issue.csv"
            if candidate.exists():
                contrast_csv = candidate
        if contrast_csv is not None and Path(contrast_csv).exists():
            contrast_df = pd.read_csv(contrast_csv)
            if "Path" in df.columns and "Path" in contrast_df.columns:
                df = df[~df["Path"].isin(contrast_df["Path"].tolist())]

        rows = df
        if args.max_samples is not None:
            rows = rows.head(args.max_samples)

        if rows.empty:
            print("No samples remain after filtering; nothing to process.")
            return

        metadata_processor = MetadataProcessor(rows)

        for row_counter, (idx, row) in enumerate(rows.iterrows()):
            raw_path = row.get(args.csv_col, None)
            if pd.isna(raw_path):
                continue

            volume_path = str(raw_path)

            metadata_tensor = metadata_processor.encode_row(row)
            positive_tasks = _extract_positive_tasks_from_row(row, selected_tasks)
            radio_positive = _extract_positive_radio_labels_from_row(row)
            tasks_needed = [
                task
                for task in positive_tasks
                if task in collected_counts and collected_counts[task] < samples_per_task
            ]

            if not tasks_needed:
                continue

            ct_array = _load_volume(
                volume_path,
                blob_service_client=blob_service_client,
                azure_container=args.azure_container,
            )
            model_inputs, triplet_images = _prepare_model_input(ct_array)

            metadata_batch = metadata_tensor.unsqueeze(0) if metadata_tensor is not None else None

            embeddings_subset = {
                task: task_embeddings_all[task].unsqueeze(0)
                for task in tasks_needed
                if task in task_embeddings_all
            }
            if not embeddings_subset:
                continue

            if args.use_precomputed_lora and precomputed_lora is not None:
                task_to_lora = {
                    task: precomputed_lora[task]
                    for task in tasks_needed
                    if task in precomputed_lora
                }
                analysis = generate_dinov3_precomputed_lora_gradcam(
                    dinov3_path=args.dinov3_path,
                    model_config=args.config,
                    task_to_lora=task_to_lora,
                    tasks=list(task_to_lora.keys()),
                    inputs=model_inputs,
                    hypernet_checkpoint=str(CHECKPOINT_ROOT),
                    device=args.device,
                    metadata=metadata_batch,
                    triplet_images=triplet_images,
                    cam_layer_index=args.cam_layer_index,
                    cam_threshold=args.cam_threshold,
                )
            else:
                analysis = generate_dinov3_hypernet_gradcam(
                    dinov3_path=args.dinov3_path,
                    model_config=args.config,
                    task_embeddings=embeddings_subset,
                    inputs=model_inputs,
                    hypernet_checkpoint=str(CHECKPOINT_ROOT),
                    device=args.device,
                    metadata=metadata_batch,
                    triplet_images=triplet_images,
                    cam_threshold=args.cam_threshold,
                    cam_smoothing_kernel=args.cam_smoothing_kernel,
                )

            for task in tasks_needed:
                file_stem = _build_task_filename(task, volume_path, row, row_counter)
                metrics = _collect_metric_values(row, task)
                metrics = {key: _to_jsonable(val) for key, val in metrics.items()}
                extra_info = {
                    "ct_path": volume_path,
                    "EMPI": _to_jsonable(row.get("EMPI", None)),
                    "CT_AccessionNumber": _to_jsonable(row.get("CT_AccessionNumber", None)),
                    "metrics": metrics,
                    "echo_positive_tasks": positive_tasks,
                    "radio_positive_tasks": radio_positive,
                }
                _save_task_result(
                    task_output_dirs[task],
                    analysis,
                    task,
                    file_stem,
                    grid_cols=args.grid_cols,
                    sample_metadata=metadata_batch,
                    extra_info=extra_info,
                )
                collected_counts[task] += 1
                generated_any = True

            if all(count >= samples_per_task for count in collected_counts.values()):
                break
    else:
        if not args.positive_tasks:
            raise ValueError("--positive-tasks must be provided when no CSV is specified")
        direct_tasks = [task for task in selected_tasks if task in set(args.positive_tasks)]
        if not direct_tasks:
            raise ValueError("None of the specified --positive-tasks overlap with available tasks")

        ct_array = _load_volume(
            str(args.ct_volume),
            blob_service_client=blob_service_client,
            azure_container=args.azure_container,
        )
        model_inputs, triplet_images = _prepare_model_input(ct_array)
        embeddings_subset = {
            task: task_embeddings_all[task].unsqueeze(0)
            for task in direct_tasks
            if task in task_embeddings_all
        }
        if embeddings_subset:
            if args.use_precomputed_lora and precomputed_lora is not None:
                task_to_lora = {
                    task: precomputed_lora[task]
                    for task in direct_tasks
                    if task in precomputed_lora
                }
                analysis = generate_dinov3_precomputed_lora_gradcam(
                    dinov3_path=args.dinov3_path,
                    model_config=args.config,
                    task_to_lora=task_to_lora,
                    tasks=list(task_to_lora.keys()),
                    inputs=model_inputs,
                    hypernet_checkpoint=str(CHECKPOINT_ROOT),
                    device=args.device,
                    triplet_images=triplet_images,
                    cam_layer_index=args.cam_layer_index,
                    cam_threshold=args.cam_threshold,
                )
            else:
                analysis = generate_dinov3_hypernet_gradcam(
                    dinov3_path=args.dinov3_path,
                    model_config=args.config,
                    task_embeddings=embeddings_subset,
                    inputs=model_inputs,
                    hypernet_checkpoint=str(CHECKPOINT_ROOT),
                    device=args.device,
                    triplet_images=triplet_images,
                    cam_threshold=args.cam_threshold,
                    cam_smoothing_kernel=args.cam_smoothing_kernel,
                )

            empty_row = pd.Series({}, dtype=object)
            for idx, task in enumerate(direct_tasks):
                file_stem = _build_task_filename(task, str(args.ct_volume), empty_row, idx)
                extra_info = {
                    "ct_path": str(args.ct_volume),
                    "EMPI": None,
                    "CT_AccessionNumber": None,
                    "metrics": {},
                }
                _save_task_result(
                    task_output_dirs[task],
                    analysis,
                    task,
                    file_stem,
                    grid_cols=args.grid_cols,
                    sample_metadata=None,
                    extra_info=extra_info,
                )
                collected_counts[task] = collected_counts.get(task, 0) + 1
                generated_any = True

    for task, count in collected_counts.items():
        if count < samples_per_task:
            logger.warning(
                "Task %s produced %d/%d positive samples", task, count, samples_per_task
            )

    print("Grad-CAM artefacts saved to:")
    for task in selected_tasks:
        directory = task_output_dirs[task]
        if directory.exists():
            print(f"  - {task}: {directory} (generated {collected_counts.get(task, 0)} samples)")
    if not generated_any:
        print("  (No artefacts generated)")


if __name__ == "__main__":
    main()
