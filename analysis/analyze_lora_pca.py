"""PCA analysis of precomputed task-specific LoRA weights.

This script loads `precomputed_lora_weights.pt` (or a provided path),
extracts flattened LoRA vectors per task, runs PCA across tasks, and
saves 2D scatter plots (or CSV fallbacks) of task embeddings. It
supports:

    • Aggregated PCA (all modules concatenated)
    • Per-module PCA with ``--per-module``
    • Optional "clinical" styling (``--clinical-style``) that colours tasks by
        user-provided clinical group metadata and adds better label handling
        when matplotlib + seaborn + pandas + adjustText are available.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on task-specific LoRA weights and plot them")
    parser.add_argument(
        "--lora-file",
        type=Path,
        default=Path("precomputed_lora_weights.pt"),
        help="Path to precomputed LoRA weights (output of precompute_task_specific_models.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lora_pca_plots"),
        help="Directory to save PCA plots",
    )
    parser.add_argument(
        "--per-module",
        action="store_true",
        help="Additionally run PCA separately for each LoRA module",
    )
    parser.add_argument(
        "--component-level",
        action="store_true",
        help=(
            "Treat each individual LoRA matrix (A/B for each task and module) "
            "as a separate component in the PCA space."
        ),
    )
    parser.add_argument(
        "--module-filter",
        type=str,
        default=None,
        help=(
            "Optional substring filter applied to module names when using "
            "--component-level (e.g. 'query' to only include attention_query)."
        ),
    )
    parser.add_argument(
        "--matrix-kind",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="Which LoRA matrices to include at component level: A, B, or both.",
    )
    parser.add_argument(
        "--clinical-style",
        action="store_true",
        help="Use seaborn/adjustText aesthetics when plotting (requires pandas, seaborn, adjustText)",
    )
    parser.add_argument(
        "--task-group-file",
        type=Path,
        default=None,
        help="Optional CSV with columns task_name,group[,color] for clinical styling",
    )
    return parser.parse_args()


def _load_precomputed_lora(path: Path) -> tuple[List[str], Dict[str, Dict[str, Dict[str, torch.Tensor]]]]:
    payload = torch.load(path, map_location="cpu")
    if "task_names" not in payload or "precomputed_lora" not in payload:
        raise ValueError(
            f"Expected keys 'task_names' and 'precomputed_lora' in {path}, "
            f"got {list(payload.keys())}"
        )
    task_names: List[str] = list(payload["task_names"])
    precomputed = payload["precomputed_lora"]

    new_task_names = []
    for t, p in zip(task_names, precomputed):
        if 'enlarge' in t or 'atrial' in t or 'systolic' in t:
            new_task_names.append(t)

    # return new_task_names, precomputed
    return task_names, precomputed


def _load_task_groups(
    path: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load task -> group mapping (and optional colors) from CSV.

    Expected columns: ``task_name``, ``group``; optional ``color`` hex or name.
    Returns (task_to_group, group_to_color).
    """

    if path is None:
        return {}, {}

    if not path.exists():
        raise FileNotFoundError(f"Task group file not found: {path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pandas is required to parse --task-group-file for clinical styling"
        ) from exc

    df = pd.read_csv(path)
    required_cols = {"task_name", "group"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Task group file must contain columns {sorted(required_cols)}, got {sorted(df.columns)}"
        )

    task_to_group: Dict[str, str] = {}
    group_to_color: Dict[str, str] = {}

    for _, row in df.iterrows():
        task = str(row["task_name"]).strip()
        group = str(row["group"]).strip()
        if not task or not group:
            continue
        task_to_group[task] = group
        if "color" in df.columns and isinstance(row.get("color"), str):
            color_val = row.get("color")
            if isinstance(color_val, str) and color_val.strip():
                group_to_color[group] = color_val.strip()

    return task_to_group, group_to_color


def _build_task_vectors(
    task_names: List[str],
    precomputed: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> torch.Tensor:
    """Convert nested LoRA dict into a matrix [num_tasks, dim]."""

    vectors: List[torch.Tensor] = []
    reference_dim: int | None = None

    for task in task_names:
        task_dict = precomputed[task]
        segments: List[torch.Tensor] = []

        # Sort module names for consistency
        for module_name in sorted(task_dict.keys()):
            tensors = task_dict[module_name]
            lora_A = tensors["lora_A"].reshape(-1).float()
            lora_B = tensors["lora_B"].reshape(-1).float()
            segments.append(torch.cat([lora_A, lora_B], dim=0))

        vec = torch.cat(segments, dim=0)

        if reference_dim is None:
            reference_dim = vec.numel()
        elif vec.numel() != reference_dim:
            raise ValueError(
                f"Inconsistent vector size for task {task}: expected {reference_dim}, got {vec.numel()}"
            )

        vectors.append(vec)

    return torch.stack(vectors, dim=0)  # [num_tasks, dim]


def _build_module_vectors(
    task_names: List[str],
    precomputed: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """Create per-module matrices of shape [num_tasks, dim_module]."""

    if not task_names:
        return {}

    reference_modules = sorted(precomputed[task_names[0]].keys())
    module_vectors: Dict[str, List[torch.Tensor]] = {
        module: [] for module in reference_modules}
    module_dims: Dict[str, int] = {}

    for task in task_names:
        task_dict = precomputed[task]
        missing = set(reference_modules) - set(task_dict.keys())
        if missing:
            raise ValueError(f"Task {task} missing modules: {sorted(missing)}")

        for module in reference_modules:
            tensors = task_dict[module]
            lora_A = tensors["lora_A"].reshape(-1).float()
            lora_B = tensors["lora_B"].reshape(-1).float()
            vec = torch.cat([lora_A, lora_B], dim=0)

            if module not in module_dims:
                module_dims[module] = vec.numel()
            elif module_dims[module] != vec.numel():
                raise ValueError(
                    f"Module {module} has inconsistent dim: expected {module_dims[module]}, got {vec.numel()}"
                )

            module_vectors[module].append(vec)

    return {module: torch.stack(vectors, dim=0) for module, vectors in module_vectors.items()}


def _build_component_vectors(
    task_names: List[str],
    precomputed: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    module_filter: Optional[str] = None,
    matrix_kind: str = "both",
) -> tuple[torch.Tensor, List[str]]:
    """Flatten individual LoRA matrices as separate samples.

    Each sample corresponds to a single (task, module, matrix_kind) triple,
    e.g. "task0_blocks.0.attention_query_A". Only matrices with a consistent
    flattened dimension are kept so they can share one PCA space.
    """

    vectors: List[torch.Tensor] = []
    labels: List[str] = []
    reference_dim: int | None = None

    kinds_to_use: List[str]
    if matrix_kind == "A":
        kinds_to_use = ["A"]
    elif matrix_kind == "B":
        kinds_to_use = ["B"]
    else:
        kinds_to_use = ["A", "B"]

    for task_idx, task in enumerate(task_names):
        task_dict = precomputed[task]
        for module_name in sorted(task_dict.keys()):
            if module_filter is not None and module_filter not in module_name:
                continue

            tensors = task_dict[module_name]

            for kind in kinds_to_use:
                key = "lora_A" if kind == "A" else "lora_B"
                if key not in tensors:
                    continue

                vec = tensors[key].reshape(-1).float()

                if reference_dim is None:
                    reference_dim = vec.numel()
                elif vec.numel() != reference_dim:
                    # Skip matrices with mismatched size so PCA has consistent dim
                    logger.debug(
                        "Skipping {} for task {} (dim {} != reference {})",
                        f"{module_name}_{kind}",
                        task,
                        vec.numel(),
                        reference_dim,
                    )
                    continue

                vectors.append(vec)
                # Use the actual task name in the label so legends and CSVs
                # clearly expose which clinical task each component belongs to.
                labels.append(f"{task}_{module_name}_{kind}")

    if not vectors:
        raise ValueError(
            "No components collected for component-level PCA; "
            "check --module-filter and --matrix-kind."
        )

    return torch.stack(vectors, dim=0), labels


def _run_pca(x: torch.Tensor, n_components: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Run PCA via SVD on centered data.

    Returns (projected, explained_variance_ratio) where:
        projected: [num_samples, n_components]
        explained_variance_ratio: [num_components_total]

    Note: ``explained_variance_ratio`` always contains *all* principal components
    (upto the rank of ``x``), regardless of ``n_components`` used for the
    low-dimensional projection. This is important for producing a full scree
    plot to test low-rank structure.
    """

    if x.ndim != 2:
        raise ValueError(
            f"Input to PCA must be 2D (num_samples, dim), got {x.shape}")

    x = x.float()
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    # SVD on centered data
    u, s, vh = torch.linalg.svd(x_centered, full_matrices=False)

    # Use the requested number of components only for the projection.
    n_components = min(n_components, vh.shape[0])
    components = vh[:n_components]  # [n_components, dim]

    projected = x_centered @ components.T  # [num_samples, n_components]

    # Explained variance ratio across *all* components (for a full scree plot)
    var = s**2
    total_var = var.sum()
    explained_var_ratio = (var / total_var).cpu()

    return projected.cpu(), explained_var_ratio


def _plot_pca(
    projected: torch.Tensor,
    task_names: List[str],
    explained_var_ratio: torch.Tensor,
    output_path: Path,
    *,
    plot_title: str | None = None,
    clinical_style: bool = False,
    task_to_group: Optional[Dict[str, str]] = None,
    preset_group_colors: Optional[Dict[str, str]] = None,
    use_index_labels: bool = True,
) -> None:
    """Plot PCA scatter if matplotlib is available; otherwise dump CSV.

    This avoids hard dependency on a working matplotlib/NumPy binary pair.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = projected[:, 0].cpu().numpy()
    y = projected[:, 1].cpu().numpy() if projected.shape[1] > 1 else None

    # Compute a symmetric axis range so that x and y share the same scale and
    # the origin stays centered. This makes distances and directions
    # comparable across different PCA runs and components.
    if y is not None:
        max_abs = float(max(abs(x).max(), abs(y).max()))
    else:
        max_abs = float(abs(x).max())
    if max_abs == 0.0:
        max_abs = 1.0
    axis_min, axis_max = -max_abs, max_abs

    # First try to import matplotlib lazily, while silencing stderr to avoid
    # noisy stack traces when binary wheels are incompatible with NumPy.
    try:  # pragma: no cover - import-time binary issues are environment-specific
        stderr_buf = io.StringIO()
        stdout_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf), contextlib.redirect_stdout(stdout_buf):
            import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        # Fall back to saving raw PCA coordinates for external plotting.
        logger.warning(
            "matplotlib could not be imported ({}); "
            "saving PCA coordinates to CSV instead of PNG.",
            exc,
        )

        csv_path = output_path.with_suffix(".csv")
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Always identify points by their full label (task/component name).
            header = ["label", "pc1", "pc2"] if y is not None else [
                "label", "pc1"]
            writer.writerow(header)
            for i, name in enumerate(task_names):
                row = [name, float(x[i])]
                if y is not None:
                    row.append(float(y[i]))
                writer.writerow(row)

        logger.info("Saved PCA coordinates to {}", csv_path)
        return

    use_clinical = clinical_style
    clinical_support_ok = False
    if use_clinical:
        try:
            import pandas as pd  # type: ignore
            import seaborn as sns  # type: ignore
        except Exception as exc:
            logger.warning(
                "clinical-style plotting requested but pandas/seaborn missing: {}; falling back to basic plot",
                exc,
            )
            use_clinical = False
        else:
            clinical_support_ok = True
        try:
            from adjustText import adjust_text  # type: ignore
        except Exception:
            adjust_text = None  # type: ignore
    if not use_clinical:
        adjust_text = None  # type: ignore

    pc1_var = explained_var_ratio[0].item() * 100.0
    pc2_var = explained_var_ratio[1].item(
    ) * 100.0 if len(explained_var_ratio) > 1 else 0.0

    title_text = plot_title or "PCA of task-specific LoRA weights"

    if use_clinical and clinical_support_ok:
        plt.style.use("seaborn-v0_8-whitegrid")
        import pandas as pd  # type: ignore  # re-import for type checkers
        import seaborn as sns  # type: ignore

        df = pd.DataFrame({
            "task_name": task_names,
            "PC1": x,
            "PC2": y if y is not None else [0.0] * len(x),
        })
        mapping = task_to_group or {}
        df["group"] = df["task_name"].map(mapping).fillna("Unspecified")
        unique_groups = df["group"].unique().tolist()

        palette = preset_group_colors.copy() if preset_group_colors else {}
        remaining_groups = [g for g in unique_groups if g not in palette]
        if remaining_groups:
            palette_gen = sns.color_palette("tab10", len(remaining_groups))
            from matplotlib.colors import to_hex  # type: ignore

            for group_name, color in zip(remaining_groups, palette_gen):
                palette[group_name] = to_hex(color)

        # Use a more pronounced rectangular canvas for better readability
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.scatterplot(
            data=df,
            x="PC1",
            y="PC2",
            hue="group",
            palette=palette,
            s=220,
            ax=ax,
        )

        # Add smooth confidence regions for each group
        try:
            sns.kdeplot(
                data=df,
                x="PC1",
                y="PC2",
                hue="group",
                levels=5,  # multiple contour levels to satisfy filled contour requirement
                fill=True,
                alpha=0.12,
                palette=palette,
                legend=False,
                ax=ax,
            )
        except Exception as exc:  # pragma: no cover - cosmetic
            logger.warning("kdeplot for confidence regions failed: {}", exc)

        # Add group centroids to show group "centers" explicitly
        try:
            centroids = df.groupby("group")["PC1", "PC2"].mean().reset_index()
        except Exception:
            centroids = None

        if centroids is not None:
            centroid_colors = [palette.get(g, "black")
                               for g in centroids["group"]]
            ax.scatter(
                centroids["PC1"],
                centroids["PC2"],
                s=260,
                marker="X",
                c=centroid_colors,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

        # Draw 1-based indices inside each point (bold for visibility)
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            ax.text(
                row.PC1,
                row.PC2,
                str(idx),
                fontsize=9,
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

        if adjust_text is not None:
            # We no longer use adjust_text for indices, but keep hook for future use.
            pass

        # Apply consistent symmetric limits for both axes
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        ax.set_title(title_text, fontsize=18, fontweight="bold")
        ax.set_xlabel(
            f"PC1 ({pc1_var:.1f}% var)", fontsize=14, fontweight="bold"
        )
        ax.set_ylabel(
            f"PC2 ({pc2_var:.1f}% var)", fontsize=14, fontweight="bold"
        )

        # Make tick labels slightly larger and bold
        ax.tick_params(axis="both", labelsize=11)
        for tick_lbl in ax.get_xticklabels() + ax.get_yticklabels():
            tick_lbl.set_fontweight("bold")

        # Legends: clinical groups (color) and index → task mapping
        from matplotlib.lines import Line2D  # type: ignore

        handles, labels_ = ax.get_legend_handles_labels()
        group_legend = ax.legend(
            handles,
            labels_,
            title="Screening Group",
            loc="upper right",
            fontsize=9,
            title_fontsize=10,
            prop={"weight": "bold"},
        )

        index_entries: list[Line2D] = []
        index_labels: list[str] = []
        for idx, name in enumerate(task_names, start=1):
            raw_label = str(name)
            shown = raw_label if len(
                raw_label) <= 32 else raw_label[:29] + "..."
            index_entries.append(
                Line2D([0], [0], marker="o", linestyle="",
                       color="gray", markersize=5)
            )
            index_labels.append(f"{idx}: {shown}")

        ax.legend(
            index_entries,
            index_labels,
            title="Index  task",
            loc="lower left",
            fontsize=6,
            title_fontsize=8,
            prop={"weight": "bold"},
        )
        ax.add_artist(group_legend)
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    # Basic matplotlib scatter fallback
    # Use a more rectangular canvas for better readability
    fig, ax = plt.subplots(figsize=(14, 8))
    marker_size = 220.0
    if y is not None:
        ax.scatter(x, y, s=marker_size)
    else:
        ax.scatter(x, [0.0] * len(x), s=marker_size)

    # Put 1-based index inside each dot (bold for visibility)
    for i, _name in enumerate(task_names):
        idx_label = str(i + 1)
        ax.text(
            x[i],
            y[i] if y is not None else 0.0,
            idx_label,
            fontsize=9,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    # Optional legend listing index -> task label inside the figure
    try:
        from matplotlib.lines import Line2D  # type: ignore

        legend_entries = []
        legend_labels = []
        for i, name in enumerate(task_names):
            # Build "index: label" strings; long labels are truncated
            raw_label = str(name)
            shown = raw_label if len(
                raw_label) <= 32 else raw_label[:29] + "..."
            legend_entries.append(
                Line2D([0], [0], marker="o", linestyle="",
                       color="gray", markersize=5)
            )
            legend_labels.append(f"{i + 1}: {shown}")

        # Place legend inside figure
        ax.legend(
            legend_entries,
            legend_labels,
            title="Index  task",
            loc="upper right",
            fontsize=6,
            title_fontsize=8,
            prop={"weight": "bold"},
        )
    except Exception:
        pass

    # Apply symmetric axis limits for consistent scaling
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel(
        f"PC1 ({pc1_var:.1f}% var)", fontsize=14, fontweight="bold"
    )
    if y is not None:
        ax.set_ylabel(
            f"PC2 ({pc2_var:.1f}% var)", fontsize=14, fontweight="bold"
        )
    else:
        ax.set_ylabel("PC2", fontsize=14, fontweight="bold")

    # Make tick labels slightly larger and bold
    ax.tick_params(axis="both", labelsize=11)
    for tick_lbl in ax.get_xticklabels() + ax.get_yticklabels():
        tick_lbl.set_fontweight("bold")

    ax.set_title(title_text, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "module"


def _save_scree_plot(
    explained_var_ratio: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a scree plot (or CSV fallback) of PCA explained variance.

    This quantifies how many PCs are needed to capture most variance,
    which is the key evidence for a low-dimensional subspace.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = explained_var_ratio.cpu().numpy()
    pcs = list(range(1, len(values) + 1))
    cumulative = values.cumsum()

    try:  # pragma: no cover - environment dependent
        stderr_buf = io.StringIO()
        stdout_buf = io.StringIO()
        with contextlib.redirect_stderr(stderr_buf), contextlib.redirect_stdout(stdout_buf):
            import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        import csv

        logger.warning(
            "matplotlib could not be imported ({}); saving scree data to CSV instead of PNG.",
            exc,
        )
        csv_path = output_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["pc_index", "explained_variance_ratio", "cumulative_variance_ratio"])
            for idx, var, cum in zip(pcs, values, cumulative):
                writer.writerow([idx, float(var), float(cum)])
        logger.info("Saved scree data to {}", csv_path)
        return

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, values, marker="o", label="Individual variance")
    plt.plot(pcs, cumulative, marker="s", label="Cumulative variance")
    plt.xlabel("Principal component index")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree plot of LoRA weight subspace")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    logger.info(f"Loading precomputed LoRA weights from {args.lora_file} ...")
    task_names, precomputed = _load_precomputed_lora(args.lora_file)

    if args.component_level:
        logger.info(
            "Found {} tasks; building component-level LoRA matrices (module_filter={}, matrix_kind={}) ...",
            len(task_names),
            args.module_filter,
            args.matrix_kind,
        )
        x, labels = _build_component_vectors(
            task_names,
            precomputed,
            module_filter=args.module_filter,
            matrix_kind=args.matrix_kind,
        )
    else:
        logger.info(
            f"Found {len(task_names)} tasks; building flattened LoRA vectors ...")
        x = _build_task_vectors(task_names, precomputed)
        labels = task_names

    task_to_group, group_colors = _load_task_groups(args.task_group_file)

    # If no explicit task group file is provided and we're in the standard
    # 25-task setting, treat the first 7 tasks as "Opportunistic" and the
    # remaining 18 as "Conventional" so we can color them differently.
    if (not args.component_level) and not task_to_group and len(labels) == 25:
        logger.info(
            "No task-group file provided; assigning first 7 tasks as 'Opportunistic' "
            "and the remaining {} tasks as 'Conventional'.",
            len(labels) - 7,
        )
        for idx, name in enumerate(labels):
            task_to_group[name] = "Opportunistic" if idx < 7 else "Conventional"

        # Provide default colors if none are specified
        if not group_colors:
            group_colors = {
                "Opportunistic": "tab:blue",
                "Conventional": "tab:orange",
            }

    logger.info("Running PCA (2 components) across tasks ...")
    projected, explained_var_ratio = _run_pca(x, n_components=2)

    # Scree plot (explained variance across all PCs)
    scree_path = args.output_dir / "tasks_lora_pca_scree.png"
    logger.info(f"Saving scree plot / data to {scree_path} ...")
    _save_scree_plot(explained_var_ratio, scree_path)

    # When doing component-level PCA, also save a legend mapping from
    # integer index (used in plots/CSVs) to the full component label.
    if args.component_level:
        import csv

        legend_path = args.output_dir / "tasks_lora_pca_labels.csv"
        with open(legend_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label"])
            for i, name in enumerate(labels):
                writer.writerow([i, name])
        logger.info("Saved component label legend to {}", legend_path)

    output_path = args.output_dir / "tasks_lora_pca.pdf"
    logger.info(f"Plotting PCA scatter to {output_path} ...")
    _plot_pca(
        projected,
        labels,
        explained_var_ratio,
        output_path,
        clinical_style=args.clinical_style,
        task_to_group=task_to_group,
        preset_group_colors=group_colors,
        use_index_labels=True,
    )

    if args.per_module:
        logger.info("Running per-module PCA ...")
        module_vectors = _build_module_vectors(task_names, precomputed)
        for module_name, matrix in module_vectors.items():
            logger.info("  Module {}", module_name)
            module_projected, module_var = _run_pca(matrix, n_components=2)
            safe_name = _sanitize_filename(module_name)
            module_output = args.output_dir / f"module_{safe_name}_pca.png"
            _plot_pca(
                module_projected,
                task_names,
                module_var,
                module_output,
                plot_title=f"PCA of LoRA module: {module_name}",
                clinical_style=args.clinical_style,
                task_to_group=task_to_group,
                preset_group_colors=group_colors,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
