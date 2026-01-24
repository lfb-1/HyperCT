"""
Evaluation-related functions for multi-task learning
"""

import torch
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score


def evaluate_model_with_dynamic_lora(model, lora_manager, learnable_task_embedding, test_loader, criterion, device, loader_name=""):
    """
    Evaluate model on test set with dynamic LoRA weights and learnable task embeddings
    """
    model.eval()
    learnable_task_embedding.eval()

    # Dictionary to store final results by task (flat structure for saving)
    final_results = {}
    
    # Dictionary to store intermediate results by task
    task_results = {}
    total_batches = len(test_loader)

    logger.info(f"Starting dynamic LoRA evaluation with learnable embeddings for {loader_name} - {total_batches} batches")

    cached_task_lora_weights = {}
    hypernet_device = device
    try:
        hypernet_device = next(lora_manager.hypernet.parameters()).device
    except Exception:
        logger.warning("Could not infer hypernet device; defaulting to model device for cached LoRA weights")

    task_names = getattr(learnable_task_embedding, "task_names", None)
    if task_names:
        logger.info(f"Precomputing dynamic LoRA weights for {len(task_names)} learnable tasks")
        for task_name in task_names:
            task_embedding = (
                learnable_task_embedding.get_task_embedding(task_name)
                .detach()
                .to(hypernet_device)
                .unsqueeze(0)
            )
            cached_task_lora_weights[task_name] = lora_manager.precompute_lora_weights(task_embedding)
    else:
        logger.warning("No task names found on learnable_task_embedding; dynamic generation will be used lazily")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:  # Print every 10 batches
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

            inputs = batch["ct"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)
            slice_mask = batch.get("ct_mask")
            if slice_mask is not None:
                slice_mask = slice_mask.to(device)

            # Evaluation mode: batch["target"] is a dict {task_name: tensor}
            targets_dict = batch["target"]

            # Use the dictionary keys as the definitive source of available tasks
            available_task_names = list(targets_dict.keys())

            # Process each task separately
            for task_name in available_task_names:
                if task_name not in targets_dict:
                    continue

                # Get targets for this specific task
                task_targets = targets_dict[task_name].to(device)

                if task_name not in cached_task_lora_weights:
                    logger.warning(f"Task {task_name} missing cached LoRA weights; generating on-the-fly")
                    fallback_embedding = (
                        learnable_task_embedding.get_task_embedding(task_name)
                        .detach()
                        .to(hypernet_device)
                        .unsqueeze(0)
                    )
                    cached_task_lora_weights[task_name] = lora_manager.precompute_lora_weights(fallback_embedding)

                # Check for valid samples in this task
                valid_mask = task_targets != -1

                if not valid_mask.any():
                    # No valid samples for this task in this batch
                    continue

                # Filter to only valid samples
                valid_inputs = inputs[valid_mask]
                valid_targets = task_targets[valid_mask]
                valid_slice_mask = slice_mask[valid_mask] if slice_mask is not None else None

                # Forward pass with dynamic LoRA weights using learnable embeddings
                outputs = lora_manager.forward_with_cached_lora(
                    valid_inputs,
                    cached_task_lora_weights[task_name],
                    masks=valid_slice_mask,
                )

                # Ensure outputs are the right shape for binary classification
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(-1)  # Only squeeze the last dimension
                
                # Ensure outputs and targets are both 1D tensors with same length
                if outputs.dim() == 0:  # Scalar case
                    outputs = outputs.unsqueeze(0)
                if valid_targets.dim() == 0:  # Scalar target case
                    valid_targets = valid_targets.unsqueeze(0)
                
                # Ensure both outputs and targets have the same shape for loss calculation
                if outputs.shape != valid_targets.shape:
                    logger.warning(f"Task {task_name}: Output shape {outputs.shape} doesn't match target shape {valid_targets.shape}")
                    continue

                # Calculate loss for this task
                loss = criterion(outputs, valid_targets)

                # Initialize task results if not exists
                if task_name not in task_results:
                    task_results[task_name] = {"predictions": [], "targets": [], "losses": []}

                # Store results
                task_results[task_name]["predictions"].extend(outputs.sigmoid().cpu().numpy())
                task_results[task_name]["targets"].extend(valid_targets.cpu().numpy())
                task_results[task_name]["losses"].append(loss.item())

    # Calculate metrics for each task
    # Process task results and prepare flat output structure (same as regular evaluation)
    all_aucs = []

    for task_name, task_data in task_results.items():
        if len(task_data["predictions"]) == 0:
            logger.warning(f"No predictions for task {task_name}")
            continue

        predictions = task_data["predictions"]
        targets = task_data["targets"]
        avg_loss = sum(task_data["losses"]) / len(task_data["losses"]) if task_data["losses"] else 0.0

        # Calculate metrics
        binary_preds = [1 if p > 0.5 else 0 for p in predictions]
        binary_targets = [int(t) for t in targets]
        accuracy = sum(1 for p, t in zip(binary_preds, binary_targets) if p == t) / len(binary_targets)

        # Calculate AUC
        try:
            auc_score = roc_auc_score(targets, predictions)
            all_aucs.append(auc_score)
        except ValueError as e:
            logger.warning(f"Could not calculate AUC for task {task_name}: {e}")
            auc_score = 0.0

        # Calculate F1 score
        try:
            f1 = f1_score(binary_targets, binary_preds, zero_division=0)
        except ValueError as e:
            logger.warning(f"Could not calculate F1 for task {task_name}: {e}")
            f1 = 0.0

        # Calculate balanced accuracy
        try:
            balanced_acc = balanced_accuracy_score(binary_targets, binary_preds)
        except ValueError as e:
            logger.warning(f"Could not calculate balanced accuracy for task {task_name}: {e}")
            balanced_acc = 0.0

        # Store in flat structure (same as regular evaluation)
        final_results[task_name] = {
            "auc": auc_score, 
            "loss": avg_loss, 
            "accuracy": accuracy,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "count": len(predictions)
        }

        logger.info(f"{loader_name} - Task: {task_name}, AUC: {auc_score:.4f}, F1: {f1:.4f}, Bal_Acc: {balanced_acc:.4f}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Samples: {len(predictions)}")

    # Calculate overall metrics
    if all_aucs:
        average_auc = sum(all_aucs) / len(all_aucs)
        logger.info(f"{loader_name} - Overall Average AUC: {average_auc:.4f} across {len(all_aucs)} tasks")
    else:
        average_auc = 0.0
        logger.warning(f"{loader_name} - No valid AUC scores calculated")
    
    # Calculate overall loss (weighted average across all tasks)
    all_losses = []
    all_accuracies = []
    all_f1_scores = []
    all_balanced_accs = []
    all_sample_counts = []
    for task_name, task_data in task_results.items():
        if task_data["losses"]:
            task_avg_loss = sum(task_data["losses"]) / len(task_data["losses"])
            all_losses.append(task_avg_loss)
            
            # Calculate task metrics
            predictions = task_data["predictions"]
            targets = task_data["targets"]
            binary_preds = [1 if p > 0.5 else 0 for p in predictions]
            binary_targets = [int(t) for t in targets]
            task_accuracy = sum(1 for p, t in zip(binary_preds, binary_targets) if p == t) / len(binary_targets)
            all_accuracies.append(task_accuracy)
            
            try:
                task_f1 = f1_score(binary_targets, binary_preds, zero_division=0)
                all_f1_scores.append(task_f1)
            except:
                all_f1_scores.append(0.0)
            
            try:
                task_bal_acc = balanced_accuracy_score(binary_targets, binary_preds)
                all_balanced_accs.append(task_bal_acc)
            except:
                all_balanced_accs.append(0.0)
            
            all_sample_counts.append(len(task_data["losses"]))
    
    if all_losses:
        # Weighted average metrics by number of samples per task
        total_samples = sum(all_sample_counts)
        overall_loss = sum(loss * count for loss, count in zip(all_losses, all_sample_counts)) / total_samples
        overall_accuracy = sum(acc * count for acc, count in zip(all_accuracies, all_sample_counts)) / total_samples
        overall_f1 = sum(f1 * count for f1, count in zip(all_f1_scores, all_sample_counts)) / total_samples
        overall_balanced_acc = sum(ba * count for ba, count in zip(all_balanced_accs, all_sample_counts)) / total_samples
        
        final_results["overall"] = {
            "loss": overall_loss,
            "accuracy": overall_accuracy,
            "f1_score": overall_f1,
            "balanced_accuracy": overall_balanced_acc,
            "average_auc": average_auc,
            "count": total_samples
        }
        logger.info(f"{loader_name} - Overall Loss: {overall_loss:.4f}, Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}, Bal_Acc: {overall_balanced_acc:.4f}")
    else:
        final_results["overall"] = {
            "loss": 0.0,
            "accuracy": 0.0,
            "f1_score": 0.0,
            "balanced_accuracy": 0.0,
            "average_auc": 0.0,
            "count": 0
        }
        logger.warning(f"{loader_name} - No valid losses found for overall calculation")

    return final_results


def evaluate_model(model, test_loader, criterion, device, loader_name=""):
    """
    Evaluate model on test set with multi-task batches
    """
    model.eval()

    # Dictionary to store results by task
    task_results = {}
    total_batches = len(test_loader)

    logger.info(f"Starting evaluation for {loader_name} - {total_batches} batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:  # Print every 10 batches
                logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

            inputs = batch["ct"].to(device)
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)
            slice_mask = batch.get("ct_mask")
            if slice_mask is not None:
                slice_mask = slice_mask.to(device)
            targets_dict = batch["target"]

            if slice_mask is not None and getattr(model, "supports_volume_mask", False):
                outputs = model(inputs, mask=slice_mask)
            else:
                outputs = model(inputs)

            # Ensure outputs are the right shape for binary classification
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze()

            # Process each task separately using the shared model outputs
            for task_name, task_targets in targets_dict.items():
                task_targets = task_targets.to(device)
                valid_mask = task_targets != -1

                if not valid_mask.any():
                    continue

                task_outputs = outputs[valid_mask]
                valid_targets = task_targets[valid_mask]

                if task_outputs.shape[0] != valid_targets.shape[0]:
                    logger.warning(
                        f"Output shape {task_outputs.shape} doesn't match target shape {valid_targets.shape} for task {task_name}"
                    )
                    continue

                loss = criterion(task_outputs.squeeze(), valid_targets.squeeze())

                if task_name not in task_results:
                    task_results[task_name] = {"outputs": [], "targets": [], "losses": []}

                task_results[task_name]["outputs"].append(task_outputs.detach().cpu())
                task_results[task_name]["targets"].append(valid_targets.detach().cpu())
                task_results[task_name]["losses"].append(loss.item())

    logger.info(f"Completed evaluation data collection. Computing metrics for {len(task_results)} tasks...")

    # Calculate metrics for each task
    final_results = {}
    overall_outputs = []
    overall_targets = []

    for task_idx, (task_name, task_data) in enumerate(task_results.items()):
        logger.info(f"Computing metrics for task {task_idx + 1}/{len(task_results)}: {task_name}")
        task_outputs = torch.cat(task_data["outputs"])
        task_targets = torch.cat(task_data["targets"])

        # Apply sigmoid to convert logits to probabilities
        task_probs = torch.sigmoid(task_outputs)
        task_preds = (task_probs > 0.5).float()

        # Calculate metrics
        task_loss = sum(task_data["losses"]) / len(task_data["losses"])
        task_accuracy = torch.mean((task_preds == task_targets).float()).item()

        # Calculate F1 score
        try:
            task_f1 = f1_score(task_targets.numpy(), task_preds.numpy(), zero_division=0)
        except Exception as e:
            logger.warning(f"Could not compute F1 for {task_name}: {e}")
            task_f1 = float("nan")

        # Calculate balanced accuracy
        try:
            task_balanced_acc = balanced_accuracy_score(task_targets.numpy(), task_preds.numpy())
        except Exception as e:
            logger.warning(f"Could not compute balanced accuracy for {task_name}: {e}")
            task_balanced_acc = float("nan")

        # Improved AUC calculation for standard evaluation
        try:
            unique_targets = torch.unique(task_targets)
            logger.info(
                f"Task {task_name} - Target range: {task_targets.min():.3f} to {task_targets.max():.3f}, Unique values: {len(unique_targets)}"
            )

            if len(unique_targets) > 1:  # Check if we have variation in targets
                # For binary classification (0/1 targets), use probabilities directly
                if torch.all((task_targets == 0) | (task_targets == 1)):
                    task_auc = roc_auc_score(task_targets.numpy(), task_probs.numpy())
                    logger.info(f"Binary classification AUC for {task_name}: {task_auc:.4f}")
                else:
                    # For continuous targets, convert to binary based on median split or threshold
                    # This is common for regression tasks like LVEF prediction
                    target_median = torch.median(task_targets)
                    binary_targets = (task_targets > target_median).float()

                    # Use probabilities for AUC calculation
                    task_auc = roc_auc_score(binary_targets.numpy(), task_probs.numpy())
                    logger.info(f"Continuous targets (median={target_median:.2f}) converted to binary AUC for {task_name}: {task_auc:.4f}")
            else:
                task_auc = float("nan")  # AUC undefined with only one value
                logger.warning(f"Only one unique target value for {task_name}: {unique_targets[0]:.3f}")
        except Exception as e:
            logger.warning(f"Could not compute AUC for {task_name}: {e}")
            task_auc = float("nan")

        final_results[task_name] = {
            "loss": task_loss,
            "accuracy": task_accuracy,
            "f1_score": task_f1,
            "balanced_accuracy": task_balanced_acc,
            "auc": task_auc,
            "count": task_outputs.shape[0],
            "predictions": task_probs,  # Save prediction probabilities
            "targets": task_targets,  # Save actual targets
            "outputs": task_outputs,  # Save raw logits
        }

        logger.info(
            f"{loader_name} {task_name} - Loss: {task_loss:.4f}, Accuracy: {task_accuracy:.4f}, F1: {task_f1:.4f}, Bal_Acc: {task_balanced_acc:.4f}, AUC: {task_auc:.4f}, Samples: {task_outputs.shape[0]}"
        )

        # Add to overall results
        overall_outputs.extend(task_data["outputs"])
        overall_targets.extend(task_data["targets"])

    # Calculate overall metrics
    logger.info("Computing overall metrics...")
    if overall_outputs:
        overall_outputs = torch.cat(overall_outputs)
        overall_targets = torch.cat(overall_targets)
        overall_probs = torch.sigmoid(overall_outputs)
        overall_preds = (overall_probs > 0.5).float()

        overall_accuracy = torch.mean((overall_preds == overall_targets).float()).item()
        overall_loss = sum(sum(task_data["losses"]) for task_data in task_results.values()) / sum(
            len(task_data["losses"]) for task_data in task_results.values()
        )

        # Calculate overall F1 score
        try:
            overall_f1 = f1_score(overall_targets.numpy(), overall_preds.numpy(), zero_division=0)
        except Exception as e:
            logger.warning(f"Could not compute overall F1: {e}")
            overall_f1 = float("nan")

        # Calculate overall balanced accuracy
        try:
            overall_balanced_acc = balanced_accuracy_score(overall_targets.numpy(), overall_preds.numpy())
        except Exception as e:
            logger.warning(f"Could not compute overall balanced accuracy: {e}")
            overall_balanced_acc = float("nan")

        # Calculate overall AUC with improved logic
        try:
            unique_overall_targets = torch.unique(overall_targets)
            logger.info(
                f"Overall targets - Range: {overall_targets.min():.3f} to {overall_targets.max():.3f}, Unique values: {len(unique_overall_targets)}"
            )

            if len(unique_overall_targets) > 1:
                # For binary targets, use probabilities directly
                if torch.all((overall_targets == 0) | (overall_targets == 1)):
                    overall_auc = roc_auc_score(overall_targets.numpy(), overall_probs.numpy())
                    logger.info(f"Overall binary classification AUC: {overall_auc:.4f}")
                else:
                    # For continuous targets, convert to binary based on median split
                    overall_median = torch.median(overall_targets)
                    binary_overall_targets = (overall_targets > overall_median).float()
                    overall_auc = roc_auc_score(binary_overall_targets.numpy(), overall_probs.numpy())
                    logger.info(f"Overall continuous targets (median={overall_median:.2f}) converted to binary AUC: {overall_auc:.4f}")
            else:
                overall_auc = float("nan")
                logger.warning(f"Overall: Only one unique target value: {unique_overall_targets[0]:.3f}")
        except Exception as e:
            logger.warning(f"Could not compute overall AUC: {e}")
            overall_auc = float("nan")

        logger.info(f"{loader_name} Overall - Loss: {overall_loss:.4f}, Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}, Bal_Acc: {overall_balanced_acc:.4f}, AUC: {overall_auc:.4f}")

        # Calculate average AUC across tasks (excluding NaN values)
        task_aucs = [task_data["auc"] for task_data in final_results.values() if not torch.isnan(torch.tensor(task_data["auc"]))]
        if task_aucs:
            average_auc = sum(task_aucs) / len(task_aucs)
            logger.info(f"{loader_name} Average AUC across {len(task_aucs)} tasks: {average_auc:.4f}")
        else:
            average_auc = 0.0  # Use 0.0 instead of NaN for consistency
        logger.info(f"{loader_name} Average AUC across {len(task_aucs)} tasks: {average_auc:.4f}")

        final_results["overall"] = {
            "loss": overall_loss,
            "accuracy": overall_accuracy,
            "f1_score": overall_f1,
            "balanced_accuracy": overall_balanced_acc,
            "auc": overall_auc,  # AUC computed from concatenated predictions/targets
            "average_auc": average_auc,  # Average of individual task AUCs
            "predictions": overall_probs,  # Save overall prediction probabilities
            "targets": overall_targets,  # Save overall targets
            "outputs": overall_outputs,  # Save overall raw logits
        }

    logger.info(f"Evaluation completed for {loader_name}")
    return final_results
