"""Grad-CAM analysis utilities for the DINOv3 hypernetwork setup.

The functions in this module do not alter training or evaluation code paths.
They provide a standalone workflow for:
  • Loading the frozen DINOv3 encoder paired with a trained LoRA hypernet
  • Generating task-conditioned LoRA weights from the hypernet
  • Running a forward/backward pass to obtain Grad-CAM heatmaps
  • Summarising LoRA deltas per task and comparing them across tasks

Example usage is documented in ``generate_dinov3_hypernet_gradcam``.
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Iterable

# Add package root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from ct2echo.training.dynamic_lora import HookBasedLoRAManager
from ct2echo.utils.io_utils import load_merged_hypernet_checkpoint, load_trained_model_with_hypernet
from ct2echo.setup.model_setup import setup_dinov3_with_hypernet

try:  # Optional dependency – only required for config loading
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - handled gracefully at runtime
    OmegaConf = None  # type: ignore


@dataclass
class TaskGradCAMResult:
    """Container for Grad-CAM artefacts produced for a single task."""

    task: str
    logit: float
    heatmap: torch.Tensor  # [num_images, H, W] normalized gradient-based saliency
    overlays: List[np.ndarray]
    lora_layer_norms: Dict[str, torch.Tensor]
    lora_delta_flat: Dict[str, torch.Tensor]


@dataclass
class GradCAMAnalysis:
    """Returned by ``generate_dinov3_hypernet_gradcam``."""

    per_task: Dict[str, TaskGradCAMResult]
    pairwise_lora_stats: Dict[Tuple[str, str], Dict[str, Dict[str, float]]]
    model_device: torch.device


# ---------------------------------------------------------------------------
# Model / hypernet loading
# ---------------------------------------------------------------------------


def _load_model_config(config: Optional[str | Path | object]) -> object:
    """Load a Hydra-style model config or pass through existing objects."""

    if config is None:
        raise ValueError("A model config (DictConfig/path) is required for Grad-CAM analysis")

    if OmegaConf is not None and isinstance(config, (str, Path)):
        cfg = OmegaConf.load(str(config))
        return SimpleNamespace(model=cfg)

    if hasattr(config, "model"):
        return config

    if isinstance(config, dict):
        return SimpleNamespace(model=SimpleNamespace(**config))

    raise TypeError(f"Unsupported config type: {type(config)!r}")


def load_dinov3_hypernet_for_analysis(
    dinov3_path: str | Path,
    model_config: str | Path | object,
    hypernet_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
) -> Tuple[torch.nn.Module, torch.nn.Module, object]:
    """Instantiate the frozen DINOv3 encoder and its hypernet for analysis."""

    cfg = _load_model_config(model_config)
    device_obj = torch.device(device)

    model, hypernet, _ = setup_dinov3_with_hypernet(
        dinov3_path=str(dinov3_path),
        config=cfg,
        device=device_obj,
    )
    model.eval().to(device_obj)
    hypernet.eval().to(device_obj)

    if hypernet_checkpoint:
        checkpoint_path = Path(hypernet_checkpoint)
        if checkpoint_path.is_dir():
            # Support either a directory containing final/*.pth or the final folder itself
            final_dir = checkpoint_path
            if not (final_dir / "final_hypernet.pth").exists():
                if (final_dir / "final" / "final_hypernet.pth").exists():
                    final_dir = final_dir / "final"

            load_trained_model_with_hypernet(
                base_model=model,
                hypernet=hypernet,
                checkpoint_dir=str(final_dir.parent) if (final_dir.parent / "final" / "final_hypernet.pth").exists() else str(final_dir),
                epoch_or_final="final",
                device=device_obj,
            )
            logger.info(f"Loaded hypernet weights from directory {hypernet_checkpoint}")
        else:
            load_merged_hypernet_checkpoint(
                base_model=model,
                hypernet=hypernet,
                checkpoint_path=str(checkpoint_path),
                device=device_obj,
            )
            logger.info(f"Loaded hypernet checkpoint from {hypernet_checkpoint}")

    return model, hypernet, cfg


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------


def _forward_with_hidden_states(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_layer_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass returning logits plus token activations from a chosen layer."""

    if inputs.dim() != 5:
        raise ValueError(
            f"Expected input tensor with shape [batch, 3, num_images, H, W], got {inputs.shape}"
        )

    batch_size, channels, num_images, height, width = inputs.shape
    if channels != 3:
        raise ValueError(f"DINOv3 expects channel dimension == 3, got {channels}")

    orig_num_images = getattr(model, "num_images", num_images)
    if num_images != orig_num_images:
        logger.warning(
            "Input provides %d images but model.num_images=%d; temporarily overriding for analysis",
            num_images,
            orig_num_images,
        )
        model.num_images = num_images

    x_reshaped = inputs.reshape(batch_size * num_images, 3, height, width)
    x_preprocessed = model.preprocess(x_reshaped)

    # Enable gradients w.r.t. the backbone inputs so that the
    # transformer's token activations (including patch tokens) remain
    # part of the computation graph. This allows Grad-CAM to propagate
    # from the scalar score back to all tokens instead of treating
    # them as an isolated leaf tensor.
    if not x_preprocessed.requires_grad:
        x_preprocessed.requires_grad_(True)

    outputs = model.dinov3(pixel_values=x_preprocessed, output_hidden_states=True)
    final_hidden_state = outputs.last_hidden_state  # [batch*num_images, seq, hidden]
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("DINOv3 did not return hidden_states; set output_hidden_states=True")

    layer_count = len(hidden_states)
    target_idx = target_layer_index if target_layer_index >= 0 else layer_count + target_layer_index
    if target_idx < 0 or target_idx >= layer_count:
        raise ValueError(
            f"target_layer_index {target_layer_index} out of range for {layer_count} hidden states"
        )

    target_hidden_state = hidden_states[target_idx]
    grad_container: dict[str, torch.Tensor] = {}

    def _capture_grad(grad: torch.Tensor) -> None:
        grad_container["grad"] = grad.detach()

    target_hidden_state.retain_grad()
    target_hidden_state.register_hook(_capture_grad)

    seq_len = final_hidden_state.shape[1]
    hidden_size = final_hidden_state.shape[2]

    # Reshape to split batch and image axes for downstream analysis
    cam_hidden_state = target_hidden_state.view(batch_size, num_images, seq_len, hidden_size)
    final_hidden_state = final_hidden_state.view(batch_size, num_images, seq_len, hidden_size)

    cls_features = final_hidden_state[:, :, 0, :]

    if model.aggregation == "mean":
        aggregated = cls_features.mean(dim=1)
    elif model.aggregation == "max":
        aggregated = cls_features.max(dim=1).values
    elif model.aggregation == "cls_token":
        aggregated = cls_features[:, 0, :]
    else:
        raise ValueError(f"Unsupported aggregation mode: {model.aggregation}")

    logits = model.mlp_head(aggregated)

    if num_images != orig_num_images:
        model.num_images = orig_num_images

    return logits, cam_hidden_state, cls_features, aggregated, grad_container


def _compute_patch_dims(seq_len: int, num_register_tokens: int) -> Tuple[int, int, int]:
    """Return patch count and grid dimensions for ViT tokens.

    Originally this helper assumed that the number of patch tokens formed a
    perfect square, which is true for ideal ViT grids but can fail when extra
    special tokens are present or when the model configuration changes.

    To be more robust, we now:
      1. Compute the number of patch tokens after removing CLS + register.
      2. Find a near-square factorisation (h, w) such that h * w == patch_tokens.
         If no non-trivial square-like factor exists, fall back to a 1 x N strip.
    This avoids hard failures like the "85 is not a perfect square" error and
    still produces a sensible 2D grid for Grad-CAM visualisation.
    """

    # Token layout for DINOv3ViTEmbeddings (HuggingFace implementation):
    #   [CLS] + [register_tokens...] + [patch_tokens...]
    #
    # The sequence length observed from the model therefore satisfies
    #   seq_len = 1 (CLS) + num_register_tokens + num_patches
    # and we want to recover ``num_patches`` here.

    patch_tokens = seq_len - 1 - num_register_tokens
    if patch_tokens <= 0:
        raise ValueError(f"Invalid patch token count derived from sequence length {seq_len}")

    # Start from the integer sqrt and search downward for a divisor of
    # patch_tokens so that grid_h * grid_w == patch_tokens.
    grid_h = int(math.sqrt(patch_tokens))
    while grid_h > 1 and patch_tokens % grid_h != 0:
        grid_h -= 1

    if grid_h <= 1:
        # Fall back to a 1 x N grid (still valid for visualisation).
        grid_h = 1
        grid_w = patch_tokens
    else:
        grid_w = patch_tokens // grid_h

    if grid_h * grid_w != patch_tokens:
        # This should not happen given the construction above, but guard anyway.
        raise ValueError(
            f"Failed to factor patch token count {patch_tokens} into a grid; got {grid_h}x{grid_w}"
        )

    return patch_tokens, grid_h, grid_w


def _compute_gradcam_from_tokens(
    token_activations: torch.Tensor,
    token_grads: torch.Tensor,
    num_register_tokens: int,
) -> torch.Tensor:
    """Compute a Grad-CAM map from ViT token activations and gradients."""

    patch_tokens, grid_h, grid_w = _compute_patch_dims(token_activations.shape[1], num_register_tokens)

    if token_activations.dim() == 2:
        token_activations = token_activations.unsqueeze(0)
    if token_grads.dim() == 2:
        token_grads = token_grads.unsqueeze(0)

    # DINOv3 token ordering is [CLS] + [register_tokens] + [patch_tokens].
    # Skip the CLS and any register tokens so that Grad-CAM is computed
    # strictly on spatial patch tokens.
    start_idx = 1 + max(int(num_register_tokens), 0)
    end_idx = start_idx + patch_tokens

    patch_activations = token_activations[:, start_idx:end_idx, :]
    patch_grads = token_grads[:, start_idx:end_idx, :]

    cam_tokens = (patch_grads * patch_activations).sum(dim=-1)
    cam_grid = cam_tokens.view(-1, grid_h, grid_w)

    # Defer normalisation so volumes can be scaled consistently across
    # their slice dimension. Returning raw CAM values preserves relative
    # magnitudes for the later per-volume normalisation step.
    return cam_grid


def _upsample_cam(cam: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Upsample patch-level CAM to input resolution."""

    cam = cam.unsqueeze(1)  # [N, 1, H, W]
    cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
    return cam.squeeze(1)


def _generate_overlay_image(
    base_slice: torch.Tensor,
    heat_slice: torch.Tensor,
    *,
    assume_heat_normalised: bool = False,
) -> np.ndarray:
    base_np = base_slice.detach().cpu().numpy()
    base_norm = (base_np - base_np.min()) / (base_np.max() - base_np.min() + 1e-6)
    base_uint8 = (base_norm * 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_uint8, cv2.COLOR_GRAY2BGR)

    heat_np = heat_slice.detach().cpu().numpy()
    if assume_heat_normalised:
        heat_norm = np.clip(heat_np, 0.0, 1.0)
    else:
        heat_norm = (heat_np - heat_np.min()) / (heat_np.max() - heat_np.min() + 1e-6)
    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET).astype(np.float32)

    overlay = base_bgr.astype(np.float32)
    heat_alpha = heat_norm[..., None]

    # Suppress low-activation regions to keep background grayscale
    alpha = np.clip((heat_alpha - 0.15) / 0.85, 0.0, 1.0)
    overlay = overlay * (1.0 - alpha) + heat_color * alpha
    return overlay.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# LoRA weight summaries
# ---------------------------------------------------------------------------


def _flatten_lora_delta(module_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return flattened LoRA delta matrices per layer."""

    lora_A = module_weights["lora_A"].squeeze(0)  # [layers, rank, in]
    lora_B = module_weights["lora_B"].squeeze(0)  # [layers, out, rank]
    delta = torch.einsum("lor,lri->loi", lora_B, lora_A)
    return delta.view(delta.shape[0], -1)


def _summarise_lora_weights(
    lora_weights: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute per-layer Frobenius norms and flattened deltas for each module."""

    layer_norms: Dict[str, torch.Tensor] = {}
    delta_flat: Dict[str, torch.Tensor] = {}

    for module, weights in lora_weights.items():
        delta = _flatten_lora_delta(weights)
        norms = torch.linalg.norm(delta, dim=1)
        layer_norms[module] = norms.detach().cpu()
        delta_flat[module] = delta.detach().cpu()

    return layer_norms, delta_flat


def _compare_lora_pairs(
    per_task_deltas: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[Tuple[str, str], Dict[str, Dict[str, float]]]:
    """Pairwise comparison of LoRA deltas between tasks."""

    tasks = list(per_task_deltas.keys())
    comparison: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}

    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1 :]:
            module_stats: Dict[str, Dict[str, float]] = {}
            delta_a = per_task_deltas[task_a]
            delta_b = per_task_deltas[task_b]

            for module in delta_a.keys():
                flat_a = delta_a[module]
                flat_b = delta_b[module]
                if flat_a.shape != flat_b.shape:
                    raise ValueError(
                        f"Incompatible LoRA shapes for module {module}: {flat_a.shape} vs {flat_b.shape}"
                    )

                diff = flat_a - flat_b
                fro_layer = torch.linalg.norm(diff, dim=1)
                cosine = F.cosine_similarity(flat_a, flat_b, dim=1)

                module_stats[module] = {
                    "fro_norm_mean": float(fro_layer.mean().item()),
                    "fro_norm_max": float(fro_layer.max().item()),
                    "cosine_mean": float(cosine.mean().item()),
                    "cosine_min": float(cosine.min().item()),
                }

            comparison[(task_a, task_b)] = module_stats

    return comparison


def _smooth_heatmap_stack(heatmaps: torch.Tensor, kernel_size: Optional[int]) -> torch.Tensor:
    """Lightly smooth CAM slices with avg pooling to reduce sparsity."""

    if not kernel_size or kernel_size <= 1:
        return heatmaps
    if kernel_size % 2 == 0:
        raise ValueError("cam_smoothing_kernel must be an odd integer >= 1")

    padding = kernel_size // 2
    smoothed = F.avg_pool2d(
        heatmaps.unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return smoothed.squeeze(1)



# ---------------------------------------------------------------------------
# Grad-CAM with precomputed LoRA weights
# ---------------------------------------------------------------------------


def generate_dinov3_precomputed_lora_gradcam(
    dinov3_path: str | Path,
    model_config: str | Path | object,
    task_to_lora: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    tasks: Iterable[str],
    inputs: torch.Tensor,
    hypernet_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
    metadata: Optional[torch.Tensor] = None,
    target_logit_index: int = 0,
    lora_scaling: Optional[float] = None,
    triplet_images: Optional[torch.Tensor] = None,
    cam_layer_index: int = -1,
    cam_threshold: Optional[float] = None,
) -> GradCAMAnalysis:
    """Generate proper Grad-CAM maps using *precomputed* LoRA weights.

    This variant assumes LoRA weights have already been generated and saved
    (e.g. by ``precompute_task_specific_models.py``). For each requested task:

      1. The corresponding LoRA weights are applied via ``HookBasedLoRAManager``.
      2. A forward pass is run while capturing ViT token activations.
      3. We backpropagate the target logit and compute a Grad-CAM map from
         token activations and their gradients (not from input gradients).

    Args:
        cam_layer_index: Index into DINOv3 hidden states to use for CAM. Accepts
            negative indices (e.g., -1 for final layer, -2 for penultimate).
        cam_threshold: Optional float in [0, 1] applied after per-volume
            normalisation to zero-out low-confidence activations.
    """

    if inputs.dim() != 5:
        raise ValueError("Input tensor must be [batch, 3, num_images, H, W]")

    # Validate tasks against available LoRA weights
    task_list = list(tasks)
    if not task_list:
        raise ValueError("No tasks provided for Grad-CAM analysis")

    missing = [t for t in task_list if t not in task_to_lora]
    if missing:
        raise ValueError(f"Precomputed LoRA weights missing for tasks: {missing}")

    device_obj = torch.device(device)
    inputs = inputs.to(device_obj)
    metadata = metadata.to(device_obj) if metadata is not None else None

    model, hypernet, cfg = load_dinov3_hypernet_for_analysis(
        dinov3_path=dinov3_path,
        model_config=model_config,
        hypernet_checkpoint=hypernet_checkpoint,
        device=device_obj,
    )

    scaling = float(lora_scaling) if lora_scaling is not None else float(
        getattr(cfg.model, "lora_scaling", 1.0)
    )

    hook_manager = HookBasedLoRAManager(model, hypernet, scaling_factor=scaling)
    hook_manager.register_hooks()

    per_task_results: Dict[str, TaskGradCAMResult] = {}
    per_task_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    per_task_norms: Dict[str, Dict[str, torch.Tensor]] = {}
    per_task_logits: Dict[str, float] = {}
    per_task_heatmaps: Dict[str, torch.Tensor] = {}

    height, width = inputs.shape[-2], inputs.shape[-1]

    try:
        for task_name in task_list:
            raw_lora = task_to_lora[task_name]

            # Move LoRA weights to the model device
            lora_on_device: Dict[str, Dict[str, torch.Tensor]] = {}
            for module_name, tensors in raw_lora.items():
                lora_on_device[module_name] = {
                    key: tensor.to(device_obj) for key, tensor in tensors.items()
                }

            layer_norms, delta_flat = _summarise_lora_weights(raw_lora)
            per_task_norms[task_name] = layer_norms
            per_task_deltas[task_name] = delta_flat

            hook_manager.set_lora_weights(lora_on_device)

            try:
                hook_manager.activate()

                model.zero_grad(set_to_none=True)
                hypernet.zero_grad(set_to_none=True)

                # Forward pass with token activations and gradient capture
                logits, cam_hidden_state, cls_features, aggregated, grad_container = _forward_with_hidden_states(
                    model,
                    inputs,
                    target_layer_index=cam_layer_index,
                )

                batch_logits = logits[:, target_logit_index]
                score = batch_logits.mean()
                score.backward()

                per_task_logits[task_name] = float(batch_logits.detach().mean().cpu().item())

                if "grad" not in grad_container:
                    raise RuntimeError(
                        "Failed to obtain gradients w.r.t. token activations for Grad-CAM"
                    )

                token_grads = grad_container["grad"].detach()  # [B*num_images, seq, hidden]

                batch_size, num_images, seq_len, hidden_size = cam_hidden_state.shape
                token_activations = cam_hidden_state.view(batch_size * num_images, seq_len, hidden_size)

                # DINOv3 exposes the number of register tokens on the
                # configuration object rather than the model itself.
                # Fall back gracefully if the attribute is missing.
                dinov3_cfg = getattr(model.dinov3, "config", model.dinov3)
                num_register_tokens = int(getattr(dinov3_cfg, "num_register_tokens", 0))
                cam_tokens = _compute_gradcam_from_tokens(
                    token_activations,
                    token_grads,
                    num_register_tokens=num_register_tokens,
                )  # [B*num_images, H_p, W_p]

                cam_upsampled = _upsample_cam(cam_tokens, height=height, width=width)
                cam_upsampled = cam_upsampled.view(batch_size, num_images, height, width)

                per_task_heatmaps[task_name] = cam_upsampled[0].detach().cpu()
            finally:
                hook_manager.deactivate()
    finally:
        hook_manager.remove_hooks()

    pairwise = _compare_lora_pairs(per_task_deltas) if len(per_task_deltas) > 1 else {}

    if triplet_images is not None:
        # triplet_images expected shape [num_images, 3, H, W]
        base_triplets = triplet_images.detach().cpu()
    else:
        base_triplets = inputs[0].detach().cpu().permute(1, 0, 2, 3)

    for task_name, cam_stack in per_task_heatmaps.items():
        grad_stack = cam_stack.clone()
        global_min = grad_stack.min()
        global_max = grad_stack.max()
        if (global_max - global_min) > 1e-6:
            grad_stack = (grad_stack - global_min) / (global_max - global_min)
        else:
            grad_stack = torch.zeros_like(grad_stack)

        if cam_threshold is not None:
            grad_stack = grad_stack.masked_fill(grad_stack < cam_threshold, 0.0)

        overlay_images: List[np.ndarray] = []
        for idx in range(grad_stack.shape[0]):
            triplet = base_triplets[idx]
            middle_slice = triplet[1]
            overlay_images.append(
                _generate_overlay_image(
                    middle_slice,
                    grad_stack[idx],
                    assume_heat_normalised=True,
                )
            )

        per_task_results[task_name] = TaskGradCAMResult(
            task=task_name,
            logit=per_task_logits.get(task_name, 0.0),
            heatmap=grad_stack.cpu(),
            overlays=overlay_images,
            lora_layer_norms=per_task_norms[task_name],
            lora_delta_flat=per_task_deltas[task_name],
        )

    return GradCAMAnalysis(
        per_task=per_task_results,
        pairwise_lora_stats=pairwise,
        model_device=device_obj,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dinov3_hypernet_gradcam(
    dinov3_path: str | Path,
    model_config: str | Path | object,
    task_embeddings: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    hypernet_checkpoint: Optional[str | Path] = None,
    device: str | torch.device = "cuda",
    metadata: Optional[torch.Tensor] = None,
    target_logit_index: int = 0,
    lora_scaling: Optional[float] = None,
    heatmap_output_dir: Optional[str | Path] = None,
    triplet_images: Optional[torch.Tensor] = None,
    cam_threshold: Optional[float] = None,
    cam_smoothing_kernel: Optional[int] = 3,
) -> GradCAMAnalysis:
    """Generate Grad-CAM maps for multiple tasks using the DINOv3 hypernet.

    Args:
        dinov3_path: Directory containing the pretrained DINOv3 weights.
        model_config: Path to ``conf/model/dinov3.yaml`` or an equivalent config object.
        task_embeddings: Mapping from task names to embedding tensors shaped [embed_dim] or [1, embed_dim].
        inputs: Tensor shaped [batch, 3, num_images, H, W]; typically batch==1 for analysis.
        hypernet_checkpoint: Optional checkpoint containing trained hypernet weights.
        device: Device for computation.
        metadata: Optional tensor matching the metadata_dim used during training.
        target_logit_index: Index of the logit to backpropagate (default 0 for binary heads).
        lora_scaling: Optional override of the LoRA scaling factor. Falls back to config/model value.
        heatmap_output_dir: Directory to save overlays. If ``None`` or matplotlib unavailable, nothing is saved.
        cam_threshold: Optional float threshold applied after per-volume
            normalisation.
        cam_smoothing_kernel: Optional odd integer kernel (default 3) used for
            average-pooling-based smoothing of gradient maps prior to
            normalisation. Set to 1 or ``None`` to disable.

    Returns:
        GradCAMAnalysis with per-task heatmaps and LoRA comparison statistics.
    """

    if inputs.dim() != 5:
        raise ValueError("Input tensor must be [batch, 3, num_images, H, W]")

    device_obj = torch.device(device)
    inputs = inputs.to(device_obj)
    if not inputs.requires_grad:
        inputs.requires_grad_(True)
    metadata = metadata.to(device_obj) if metadata is not None else None

    model, hypernet, cfg = load_dinov3_hypernet_for_analysis(
        dinov3_path=dinov3_path,
        model_config=model_config,
        hypernet_checkpoint=hypernet_checkpoint,
        device=device_obj,
    )

    scaling = lora_scaling
    if scaling is None:
        scaling = float(getattr(cfg.model, "lora_scaling", 1.0))

    hook_manager = HookBasedLoRAManager(model, hypernet, scaling_factor=scaling)
    hook_manager.register_hooks()

    per_task_results: Dict[str, TaskGradCAMResult] = {}
    per_task_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    per_task_norms: Dict[str, Dict[str, torch.Tensor]] = {}
    per_task_logits: Dict[str, float] = {}
    per_task_heatmaps: Dict[str, torch.Tensor] = {}

    try:
        for task_name, embedding in task_embeddings.items():
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            elif embedding.dim() != 2 or embedding.shape[0] != 1:
                raise ValueError(
                    f"Task embedding for {task_name} must be [embed_dim] or [1, embed_dim], got {embedding.shape}"
                )

            embedding = embedding.to(device_obj)

            with torch.no_grad():
                lora_weights = hypernet.generate_full_model_lora(
                    embedding,
                    metadata if metadata is not None else None,
                )

            layer_norms, delta_flat = _summarise_lora_weights(lora_weights)
            per_task_norms[task_name] = layer_norms
            per_task_deltas[task_name] = delta_flat

            hook_manager.set_lora_weights(lora_weights)

            try:
                hook_manager.activate()

                model.zero_grad(set_to_none=True)
                hypernet.zero_grad(set_to_none=True)

                if inputs.grad is not None:
                    inputs.grad.zero_()

                logits = model(inputs)
                batch_logits = logits[:, target_logit_index]
                targets = torch.ones_like(batch_logits)
                loss = F.binary_cross_entropy_with_logits(batch_logits, targets)
                loss.backward()
                per_task_logits[task_name] = float(batch_logits.detach().mean().cpu().item())

                if inputs.grad is None:
                    raise RuntimeError("Failed to obtain gradients w.r.t. input volume")

                grad_stack = inputs.grad[0].detach().abs().mean(dim=0)  # [num_images, H, W]
                per_task_heatmaps[task_name] = grad_stack.cpu()
                inputs.grad.zero_()
            finally:
                hook_manager.deactivate()
    finally:
        hook_manager.remove_hooks()

    pairwise = _compare_lora_pairs(per_task_deltas) if len(per_task_deltas) > 1 else {}

    if triplet_images is not None:
        # triplet_images expected shape [num_images, 3, H, W]
        base_triplets = triplet_images.detach().cpu()
    else:
        base_triplets = inputs[0].detach().cpu().permute(1, 0, 2, 3)

    for task_name, grad_stack in per_task_heatmaps.items():
        grad_stack = grad_stack.clone()
        grad_stack = _smooth_heatmap_stack(grad_stack, cam_smoothing_kernel)
        global_min = grad_stack.min()
        global_max = grad_stack.max()
        if (global_max - global_min) > 1e-6:
            grad_stack = (grad_stack - global_min) / (global_max - global_min)
        else:
            grad_stack = torch.zeros_like(grad_stack)

        if cam_threshold is not None:
            grad_stack = grad_stack.masked_fill(grad_stack < cam_threshold, 0.0)

        overlay_images: List[np.ndarray] = []
        for idx in range(grad_stack.shape[0]):
            triplet = base_triplets[idx]
            middle_slice = triplet[1]
            overlay_images.append(
                _generate_overlay_image(
                    middle_slice,
                    grad_stack[idx],
                    assume_heat_normalised=True,
                )
            )

        per_task_results[task_name] = TaskGradCAMResult(
            task=task_name,
            logit=per_task_logits.get(task_name, 0.0),
            heatmap=grad_stack.cpu(),
            overlays=overlay_images,
            lora_layer_norms=per_task_norms[task_name],
            lora_delta_flat=per_task_deltas[task_name],
        )

    return GradCAMAnalysis(
        per_task=per_task_results,
        pairwise_lora_stats=pairwise,
        model_device=device_obj,
    )


__all__ = [
    "TaskGradCAMResult",
    "GradCAMAnalysis",
    "load_dinov3_hypernet_for_analysis",
    "generate_dinov3_hypernet_gradcam",
    "generate_dinov3_precomputed_lora_gradcam",
]
