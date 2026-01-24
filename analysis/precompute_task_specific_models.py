"""Precompute task-specific LoRA weights and construct task-specific models.

This script:
  1. Loads learnable task embeddings from a checkpoint.
  2. Instantiates the DINOv3 encoder + LoRA hypernetwork with trained weights.
  3. Loops over tasks and precomputes full-model LoRA weights for each task.
  4. Saves the per-task LoRA weights, and exposes a simple wrapper class
     to combine the base model with cached LoRA for task-specific inference.

The design mirrors the dynamic LoRA infrastructure used during training
(`DynamicLoRAForwardManager` and `HookBasedLoRAManager`) but computes the
hypernetwork outputs just once per task instead of per-batch.

Example CLI usage:

    python precompute_task_specific_models.py \
        --dinov3-path /path/to/DINOV3_ViTb16 \
        --config conf/model/dinov3.yaml \
        --hypernet-checkpoint checkpoint_dir \
        --task-embeddings checkpoint_dir/final/final_learnable_task_embeddings.pth \
        --output precomputed_lora_weights.pt

After running, you can load the saved file and use
`TaskSpecificLoRAModel` for inference.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add package root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from ct2echo.training.dynamic_lora import DynamicLoRAForwardManager
from analysis.gradcam_hypernet import load_dinov3_hypernet_for_analysis


def _load_task_embedding_checkpoint(path: Path) -> Tuple[List[str], torch.Tensor]:
    """Load learnable task embeddings from a saved checkpoint.

    The expected format matches `LearnableTaskEmbedding.save_embeddings`:
      {"task_names": [...], "embeddings": tensor[num_tasks, dim], ...}.
    """

    checkpoint = torch.load(path, map_location="cpu")
    if "task_names" not in checkpoint or "embeddings" not in checkpoint:
        raise ValueError(
            f"Task embedding checkpoint at {path} is missing 'task_names' or 'embeddings' keys."
        )

    task_names: List[str] = list(checkpoint["task_names"])
    embeddings: torch.Tensor = checkpoint["embeddings"]

    if embeddings.ndim != 2 or embeddings.shape[0] != len(task_names):
        raise ValueError(
            "Embeddings tensor must have shape [num_tasks, dim] matching number of task_names; "
            f"got shape {tuple(embeddings.shape)} for {len(task_names)} tasks."
        )

    return task_names, embeddings


def _precompute_lora_for_tasks(
    model: torch.nn.Module,
    hypernet: torch.nn.Module,
    task_names: List[str],
    embeddings: torch.Tensor,
    device: torch.device,
    metadata_dim: int = 0,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """Precompute full-model LoRA weights for each task.

    Returns a nested dictionary compatible with `HookBasedLoRAManager`:

        {
          task_name: {
            base_module_name: {
              "lora_A": [1, num_layers, rank, in_features],
              "lora_B": [1, num_layers, out_features, rank],
            },
            ...
          },
          ...
        }
    """

    manager = DynamicLoRAForwardManager(model, hypernet, scaling_factor=1.0)
    # We only need the hypernet; hooks are not required for precomputation,
    # but reusing the helper keeps things consistent.

    precomputed: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    embeddings = embeddings.to(device)

    use_metadata = metadata_dim > 0 and getattr(hypernet, "metadata_encoder", None) is not None
    dummy_metadata = None
    if use_metadata:
        dummy_metadata = torch.zeros(1, metadata_dim, device=device, dtype=embeddings.dtype)

    for idx, task in enumerate(task_names):
        task_embedding = embeddings[idx : idx + 1]  # [1, dim]
        logger.info(f"Precomputing LoRA weights for task {task} (index {idx})")

        with torch.no_grad():
            lora_weights = manager.precompute_lora_weights(task_embedding, dummy_metadata)

        # Move weights to CPU for storage
        cpu_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        for module_name, tensors in lora_weights.items():
            cpu_weights[module_name] = {
                "lora_A": tensors["lora_A"].cpu(),
                "lora_B": tensors["lora_B"].cpu(),
            }

        precomputed[task] = cpu_weights

    return precomputed


class TaskSpecificLoRAModel(torch.nn.Module):
    """Wrap a base model with cached, task-specific LoRA weights.

    This uses the same hook-based mechanism as during training, but the
    LoRA tensors are provided upfront instead of being generated on the fly.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        hypernet: torch.nn.Module,
        cached_lora_weights: Dict[str, Dict[str, torch.Tensor]],
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hypernet = hypernet
        self.cached_lora_weights = cached_lora_weights
        self._manager = DynamicLoRAForwardManager(self.base_model, self.hypernet, scaling_factor=scaling_factor)
        self._manager.setup()

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        return self._manager.forward_with_cached_lora(inputs, self.cached_lora_weights, masks=masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute task-specific LoRA weights for DINOv3 hypernet")
    parser.add_argument(
        "--dinov3-path",
        type=Path,
        required=True,
        help="Path to pretrained DINOv3 directory (e.g., DINOV3_ViTb16)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model config used to instantiate the hypernet (e.g., conf/model/dinov3.yaml)",
    )
    parser.add_argument(
        "--hypernet-checkpoint",
        type=Path,
        required=True,
        help="Directory or file containing trained hypernet weights (same as training/eval)",
    )
    parser.add_argument(
        "--task-embeddings",
        type=Path,
        required=True,
        help="Checkpoint with learnable task embeddings (final_learnable_task_embeddings.pth)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to save the precomputed LoRA weights (e.g., precomputed_lora_weights.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (e.g. cuda, cuda:0, cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    logger.info("Loading DINOv3 base model + hypernet ...")
    model, hypernet, cfg = load_dinov3_hypernet_for_analysis(
        dinov3_path=args.dinov3_path,
        model_config=args.config,
        hypernet_checkpoint=args.hypernet_checkpoint,
        device=device,
    )

    model.eval()
    hypernet.eval()

    logger.info(f"Base model on {next(model.parameters()).device}, hypernet on {next(hypernet.parameters()).device}")

    logger.info(f"Loading task embeddings from {args.task_embeddings} ...")
    task_names, embeddings = _load_task_embedding_checkpoint(args.task_embeddings)

    metadata_dim = getattr(hypernet, "metadata_dim", 0)

    logger.info("Precomputing LoRA weights for all tasks ...")
    precomputed = _precompute_lora_for_tasks(
        model=model,
        hypernet=hypernet,
        task_names=task_names,
        embeddings=embeddings,
        device=device,
        metadata_dim=metadata_dim,
    )

    payload = {
        "task_names": task_names,
        "precomputed_lora": precomputed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    logger.info(f"Saved precomputed LoRA weights for {len(task_names)} tasks to {args.output}")


if __name__ == "__main__":
    main()
