import os
import sys
from pathlib import Path

# Add package root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from ct2echo.data.dataset import CTLoader
from ct2echo.training.dynamic_lora import dynamic_lora_context
from ct2echo.evaluation.evaluation import evaluate_model_with_dynamic_lora
from ct2echo.utils.io_utils import load_merged_hypernet_checkpoint, save_evaluation_results, save_hypernet_checkpoint
from ct2echo.task_embeddings.learnable_task_embeddings import create_learnable_task_embeddings, create_learnable_task_embedding_dict

# Import organized modules
from ct2echo.setup.model_setup import setup_dinov3_with_hypernet
from ct2echo.preprocess.task_prompts import v2_medical_task_descriptions as medical_task_descriptions
from ct2echo.training.training import train_epoch_with_hypernet
from ct2echo.utils.utils import init_azure, seed_all
from ct2echo.utils.azure_logging import suppress_azure_logs

# Suppress Azure Storage logs globally BEFORE Hydra configures logging
# This must be done at module level to take effect before @hydra.main
suppress_azure_logs()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration"""

    logger.info("Starting CT2ECHO training with Hydra configuration")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Configuration: {cfg}")

    # Seed everything for reproducibility
    seed_all(cfg)
    blob_service_client = init_azure()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define medical tasks
    medical_tasks = list(medical_task_descriptions.keys())

    # Initialize learnable task embeddings
    logger.info("🎯 Using learnable task embeddings")

    learnable_task_embedding = create_learnable_task_embeddings(
        embedding_dim=getattr(cfg.model, "learnable_embedding_dim", 128),
        init_std=getattr(cfg.model, "learnable_embedding_init_std", 0.02),
        device=str(device),
        use_radio_labels=getattr(cfg.model, "use_radio_labels", False),
    )

    # Create compatible task embeddings dictionary for data loading
    task_embeddings = create_learnable_task_embedding_dict(learnable_task_embedding, batch_size=1)

    logger.info(f"✨ Initialized learnable embeddings for {len(medical_tasks)} medical tasks")
    logger.info(f"   📏 Embedding dimension: {learnable_task_embedding.embedding_dim}")

    task_embedding_dim_override = learnable_task_embedding.embedding_dim
    logger.info(f"Task embedding input dimension set to {task_embedding_dim_override}")

    # Create output directory for saving results
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {cfg.paths.output_dir}")

    # Create data loaders (always use DINOv3 preprocessing)
    use_radio_labels = getattr(cfg.model, "use_radio_labels", False)
    loader = CTLoader(
        cfg.paths.train_val_dir,
        task_embeddings,
        blob_service_client,
        cfg.training.batch_size,
        cfg.training.num_workers,
        use_dinov3=True,
        use_radio_labels=use_radio_labels,
    )

    # Create all the data loaders
    cornell_prospective_loader, _ = loader.run_eval("wprospect")
    columbia_prospective_loader, _ = loader.run_eval("cprospect")
    cornell_test_loader, _ = loader.run("wtest")
    columbia_train_loader, _ = loader.run("train")
    columbia_test_loader, _ = loader.run("ctest")
    columbia_val_loader, _ = loader.run("val")

    logger.info("Data loaders created successfully")

    metadata_dim = getattr(columbia_train_loader.dataset, "metadata_dim", 0)
    logger.info(f"Metadata feature vector dimension: {metadata_dim}")

    # Setup DINOv3 model with LoRA Hypernet
    logger.info("🔄 Setting up DINOv3 with LoRA Hypernet")
    from types import SimpleNamespace

    temp_config = SimpleNamespace()
    temp_config.model = SimpleNamespace()
    for key, value in cfg.model.items():
        setattr(temp_config.model, key, value)
    temp_config.paths = SimpleNamespace()
    for key, value in cfg.paths.items():
        setattr(temp_config.paths, key, value)

    # Setup DINOv3 model and hypernet
    dinov3_path = cfg.paths.dinov3_dir + "/DINOV3_ViTb16"
    train_base_head_flag = getattr(cfg.model, "train_base_mlp_head", False)
    model, hypernet, _ = setup_dinov3_with_hypernet(
        dinov3_path,
        temp_config,
        device,
        metadata_dim=metadata_dim,
        train_base_mlp_head=train_base_head_flag,
        task_embedding_dim=task_embedding_dim_override,
    )

    logger.info("🎯 DINOv3 Integration Complete:")
    logger.info("   • Base Model: DINOv3-B (Vision Transformer)")
    logger.info("   • Input Processing: 164→165→RGB (55 images)")
    logger.info("   • Feature Dimension: 768")
    logger.info("   • Target Modules: 6 (attention + MLP)")
    logger.info(f"   • LoRA Rank: {cfg.model.lora_rank}")

    # Setup optimizer and criterion
    hypernet_lr = cfg.training.lr
    logger.info(f"🔧 Using lr={hypernet_lr:.2e} for MLP hypernetwork")

    # Collect parameters to optimize
    optimizer_params = list(hypernet.parameters())

    # Add learnable task embeddings if not frozen
    if getattr(cfg.model, "use_frozen_task_embeddings", False) == False:
        optimizer_params.extend(learnable_task_embedding.parameters())
        logger.info("🎯 Added learnable task embedding parameters to optimizer")

    if getattr(cfg.model, "train_base_mlp_head", False) and isinstance(
        getattr(model, "mlp_head", None), torch.nn.Module
    ):
        head_params = [p for p in getattr(model, "mlp_head").parameters() if p.requires_grad]
        if head_params:
            optimizer_params.extend(head_params)
            logger.info("🧠 Added base mlp_head parameters to optimizer for joint training")

    # Use conservative Adam settings for stability
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=hypernet_lr,
        weight_decay=cfg.training.wd,
        betas=(0.9, 0.95),  # More conservative beta2 for stability
        eps=1e-8,
    )
    logger.info("Optimizer: AdamW with conservative settings for hypernet stability")

    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    logger.info(f"Using CosineAnnealingLR scheduler with T_max={cfg.training.num_epochs}")

    logger.info("Starting training...")

    # Track best validation performance
    best_val_auc = -1.0
    best_epoch = 0
    best_model_path = None

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        # Train with learnable task embeddings
        avg_loss = train_epoch_with_hypernet(
            model,
            hypernet,
            learnable_task_embedding,
            columbia_train_loader,
            optimizer,
            criterion,
            device,
            getattr(cfg.training, "lora_grad_log_interval", 0),
        )
        logger.info(f"Epoch {epoch + 1} mixed-task loss: {avg_loss:.4f}")

        # Log embedding parameter norm
        embedding_param_norm = sum(torch.norm(p).item() for p in learnable_task_embedding.parameters())
        logger.info(f"🎯 Learnable embedding parameter norm: {embedding_param_norm:.6f}")

        total_param_norm = sum(torch.norm(p).item() for p in hypernet.parameters())
        param_norm_change = total_param_norm - getattr(main, "_last_param_norm", total_param_norm)
        main._last_param_norm = total_param_norm
        logger.info(f"🔧 Hypernet parameter norm: {total_param_norm:.6f} (change: {param_norm_change:+.6f})")

        save_hypernet_checkpoint(hypernet, model, epoch + 1, cfg.paths.output_dir)

        # Save learnable task embeddings
        embeddings_save_path = os.path.join(
            cfg.paths.output_dir, f"epoch_{epoch + 1}", "learnable_task_embeddings.pth"
        )
        os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
        learnable_task_embedding.save_embeddings(embeddings_save_path)

        with dynamic_lora_context(
            model,
            hypernet,
            cfg.model.lora_scaling,
        ) as lora_manager:
            hypernet.eval()
            learnable_task_embedding.eval()

            val_metrics = evaluate_model_with_dynamic_lora(
                model, lora_manager, learnable_task_embedding, columbia_val_loader, criterion, device, "Val"
            )

            save_evaluation_results(val_metrics, epoch + 1, cfg.paths.output_dir, "validation")

            current_val_auc = val_metrics.get("overall", {}).get("average_auc", 0.0)
            if current_val_auc is None:
                current_val_auc = 0.0
            if not torch.isnan(torch.tensor(current_val_auc)) and current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch = epoch + 1
                best_model_path = os.path.join(
                    cfg.paths.output_dir, f"epoch_{epoch + 1}", "merged_hypernet_checkpoint.pth"
                )
                logger.info(f"🏆 New best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

        scheduler.step()

    logger.info("Training completed!")

    # Load best model for final evaluation
    logger.info(f"Loading best model from epoch {best_epoch} with validation AUC {best_val_auc:.4f}")

    if best_model_path and os.path.exists(best_model_path):
        model, hypernet, success_flags, _ = load_merged_hypernet_checkpoint(
            model, hypernet, best_model_path, str(device)
        )
        if success_flags["hypernet"] and success_flags["head"]:
            logger.info("✅ Loaded best hypernet and mlp_head weights for final evaluation")
        else:
            logger.warning(
                f"⚠️ Partial loading from best checkpoint: hypernet={success_flags['hypernet']}, mlp_head={success_flags['mlp_head']}"
            )

    # Final evaluation on test sets
    logger.info("Final evaluation on test sets...")
    logger.info("Evaluating with dynamic LoRA system...")

    with dynamic_lora_context(
        model,
        hypernet,
        cfg.model.lora_scaling,
    ) as lora_manager:
        hypernet.eval()
        learnable_task_embedding.eval()

        columbia_test_metrics = evaluate_model_with_dynamic_lora(
            model,
            lora_manager,
            learnable_task_embedding,
            columbia_test_loader,
            criterion,
            device,
            "Columbia Test",
        )
        save_evaluation_results(columbia_test_metrics, "final", cfg.paths.output_dir, "columbia_test")

        cornell_test_metrics = evaluate_model_with_dynamic_lora(
            model,
            lora_manager,
            learnable_task_embedding,
            cornell_test_loader,
            criterion,
            device,
            "Cornell Test",
        )
        save_evaluation_results(cornell_test_metrics, "final", cfg.paths.output_dir, "cornell_test")

        columbia_prospective_metrics = evaluate_model_with_dynamic_lora(
            model,
            lora_manager,
            learnable_task_embedding,
            columbia_prospective_loader,
            criterion,
            device,
            "Columbia Prospective",
        )
        save_evaluation_results(
            columbia_prospective_metrics, "final", cfg.paths.output_dir, "columbia_prospective"
        )

        cornell_prospective_metrics = evaluate_model_with_dynamic_lora(
            model,
            lora_manager,
            learnable_task_embedding,
            cornell_prospective_loader,
            criterion,
            device,
            "Cornell Prospective",
        )
        save_evaluation_results(
            cornell_prospective_metrics, "final", cfg.paths.output_dir, "cornell_prospective"
        )

    logger.info("Evaluation completed!")

    # Save final model and hypernet
    final_dir = os.path.join(cfg.paths.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    head_attr = None

    final_hypernet_file = os.path.join(final_dir, "final_hypernet.pth")
    torch.save(hypernet.state_dict(), final_hypernet_file)
    logger.info(f"Saved final hypernet to {final_hypernet_file}")

    # Save final learnable task embeddings
    final_embeddings_file = os.path.join(final_dir, "final_learnable_task_embeddings.pth")
    learnable_task_embedding.save_embeddings(final_embeddings_file)
    logger.info(f"💾 Saved final learnable task embeddings to {final_embeddings_file}")

    head_module = None
    for attr in ("mlp_head", "classifier"):
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            head_attr, head_module = attr, module
            break

    if head_module is not None:
        try:
            head_state = head_module.state_dict()
            head_file = os.path.join(final_dir, f"final_base_model_{head_attr}.pth")
            torch.save(head_state, head_file)
            logger.info(f"Saved final base model {head_attr} to {head_file}")
        except (AttributeError, TypeError) as e:
            logger.warning(f"Could not save {head_attr}: {e}")
    else:
        logger.warning("No prediction head (mlp_head/classifier) found to save.")

    # Save final models
    head_prefix = f"{head_attr}." if head_attr is not None else None
    if head_prefix is not None:
        model_state_without_head = {k: v for k, v in model.state_dict().items() if not k.startswith(head_prefix)}
        final_model_file = os.path.join(final_dir, f"final_model_without_{head_attr}.pth")
    else:
        model_state_without_head = model.state_dict()
        final_model_file = os.path.join(final_dir, "final_model_without_head.pth")

    torch.save(model_state_without_head, final_model_file)
    if head_attr is not None:
        logger.info(f"Saved final model (without {head_attr}) to {final_model_file}")
    else:
        logger.info(f"Saved final model (no head removed) to {final_model_file}")

    final_complete_model_file = os.path.join(final_dir, "final_complete_model.pth")
    torch.save(model.state_dict(), final_complete_model_file)
    logger.info(f"Saved final complete model to {final_complete_model_file}")

    # Save Hydra configuration
    from omegaconf import OmegaConf

    config_file = os.path.join(cfg.paths.output_dir, "hydra_config.yaml")
    OmegaConf.save(cfg, config_file)
    logger.info(f"Saved Hydra configuration to {config_file}")

    logger.info(f"All results saved to: {cfg.paths.output_dir}")

    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING AND EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best validation AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    logger.info(f"Best model path: {best_model_path}")
    logger.info(f"Final evaluation performed using best model from epoch {best_epoch}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
