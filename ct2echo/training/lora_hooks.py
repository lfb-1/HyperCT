"""
Hook-based LoRA weight management for dynamic adaptation.

Contains:
- HookBasedLoRAManager: Manages dynamic LoRA weight application using forward hooks
"""

import torch
import torch.nn as nn
from typing import Optional
from loguru import logger


class HookBasedLoRAManager:
    """
    Manages dynamic LoRA weight application using forward hooks
    Similar to hyper_modulator.py approach
    """

    def __init__(self, model, hypernet, scaling_factor=1.0):
        self.model = model
        self.hypernet = hypernet
        self.scaling_factor = scaling_factor
        self.hooks = []
        self.current_lora_weights = {}
        self.active = False
        self._debug_logged = False
        self.log_grad_interval: Optional[int] = None
        self._grad_step = 0
        self._logged_grad_modules: set = set()

        # Create mapping from module names to layer indices
        self._create_module_to_layer_mapping()

    def _create_module_to_layer_mapping(self):
        """Create a mapping from module names to their layer indices"""
        self.module_to_layer = {}

        # Determine spatial depth to offset temporal layers
        spatial_depth = 0
        if hasattr(self.model, "enc_spatial_transformer") and hasattr(self.model.enc_spatial_transformer, "layers"):
            try:
                spatial_depth = len(self.model.enc_spatial_transformer.layers)
            except Exception:
                spatial_depth = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Extract layer index from module name
                parts = name.split(".")
                layer_idx = 0  # default

                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                    elif part == "layer" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                    elif "layer" in part.lower():
                        try:
                            layer_idx = int("".join(filter(str.isdigit, part)))
                            break
                        except ValueError:
                            continue

                # Offset temporal layers to follow spatial layers in a single 0..N-1 index space
                if "enc_temporal_transformer" in name:
                    layer_idx += spatial_depth

                self.module_to_layer[name] = layer_idx

    def _extract_base_module_name(self, full_name: str) -> str:
        """
        Extract base module name from full module path with spatial/temporal distinction
        """
        parts = full_name.split(".")

        # Determine if this is spatial or temporal transformer
        transformer_prefix = ""
        if "enc_spatial_transformer" in full_name:
            transformer_prefix = "spatial_"
        elif "enc_temporal_transformer" in full_name:
            transformer_prefix = "temporal_"
        # For modules not in transformers (like patch_emb), keep original naming

        # Handle DINOv3 module patterns
        if "dinov3" in full_name or ".encoder.layer." in full_name:
            if "q_proj" in full_name:
                return "attention_query"
            if "k_proj" in full_name:
                return "attention_key"
            if "v_proj" in full_name:
                return "attention_value"
            if "o_proj" in full_name:
                return "attention_output"
            if "mlp.up_proj" in full_name:
                return "mlp_up"
            if "mlp.down_proj" in full_name:
                return "mlp_down"

        # Handle mlp_head
        if "mlp_head" in full_name:
            return "mlp_head"

        # Fallback: use the last meaningful part
        if parts[-1].isdigit() and len(parts) > 1:
            return f"{transformer_prefix}{parts[-2]}"
        return f"{transformer_prefix}{parts[-1]}"

    def _create_hook_fn(self, module_name, original_weight):
        """Create a forward hook function for a specific module"""

        def hook_fn(module, input, output):
            input_tensor = input[0] if isinstance(input, tuple) else input

            base_module_name = self._extract_base_module_name(module_name)

            if not self.active or base_module_name not in self.current_lora_weights:
                zero_lora_output = torch.zeros_like(output, requires_grad=True)
                return output + zero_lora_output

            lora_weights = self.current_lora_weights[base_module_name]
            lora_A = lora_weights.get("lora_A")
            lora_B = lora_weights.get("lora_B")

            if lora_A is None or lora_B is None:
                zero_lora_output = torch.zeros_like(output, requires_grad=True)
                return output + zero_lora_output

            input_features = input_tensor.shape[-1]
            if input_features != lora_A.shape[-1]:
                zero_lora_output = torch.zeros_like(output, requires_grad=True)
                return output + zero_lora_output

            input_batch_size = input_tensor.shape[0]
            layer_idx = self.module_to_layer.get(module_name, 0)
            if layer_idx >= lora_A.shape[1]:
                layer_idx = 0

            current_lora_A = lora_A[:, layer_idx, :, :]
            current_lora_B = lora_B[:, layer_idx, :, :]

            lora_batch = current_lora_A.shape[0]
            repeat_factor = 1
            if lora_batch > 0 and input_batch_size % lora_batch == 0:
                repeat_factor = input_batch_size // lora_batch
            else:
                if input_batch_size < lora_batch:
                    current_lora_A = current_lora_A[:input_batch_size]
                    current_lora_B = current_lora_B[:input_batch_size]
                    lora_batch = current_lora_A.shape[0]
                elif input_batch_size > lora_batch:
                    repeats = input_batch_size - lora_batch
                    if repeats > 0:
                        last_A = current_lora_A[-1:].expand(repeats, -1, -1)
                        last_B = current_lora_B[-1:].expand(repeats, -1, -1)
                        current_lora_A = torch.cat([current_lora_A, last_A], dim=0)
                        current_lora_B = torch.cat([current_lora_B, last_B], dim=0)
                        lora_batch = current_lora_A.shape[0]

            if len(input_tensor.shape) == 3:
                seq_len = input_tensor.shape[1]
                if repeat_factor > 1:
                    if input_tensor.shape[0] != lora_batch * repeat_factor:
                        zero_lora_output = torch.zeros_like(output, requires_grad=True)
                        return output + zero_lora_output
                    reshaped_input = input_tensor.view(lora_batch, repeat_factor, seq_len, input_features)
                    reshaped_input = reshaped_input.view(lora_batch, repeat_factor * seq_len, input_features)
                    proj = torch.matmul(reshaped_input, current_lora_A.transpose(-1, -2))
                    lora_flat = torch.matmul(proj, current_lora_B.transpose(-1, -2))
                    lora_output = lora_flat.view(lora_batch, repeat_factor, seq_len, -1)
                    lora_output = lora_output.view(*input_tensor.shape[:-1], lora_output.shape[-1])
                else:
                    proj = torch.matmul(
                        input_tensor.view(lora_batch, seq_len, input_features), current_lora_A.transpose(-1, -2)
                    )
                    lora_output = torch.matmul(proj, current_lora_B.transpose(-1, -2))
                    lora_output = lora_output.view(*input_tensor.shape[:-1], lora_output.shape[-1])
            elif len(input_tensor.shape) == 2:
                if repeat_factor > 1:
                    if input_tensor.shape[0] != lora_batch * repeat_factor:
                        zero_lora_output = torch.zeros_like(output, requires_grad=True)
                        return output + zero_lora_output
                    reshaped_input = input_tensor.view(lora_batch, repeat_factor, input_features)
                    proj = torch.matmul(reshaped_input, current_lora_A.transpose(-1, -2))
                    lora_output = torch.matmul(proj, current_lora_B.transpose(-1, -2))
                    lora_output = lora_output.view(input_tensor.shape[0], -1)
                else:
                    proj = torch.bmm(input_tensor.unsqueeze(1), current_lora_A.transpose(-1, -2)).squeeze(1)
                    lora_output = torch.bmm(proj.unsqueeze(1), current_lora_B.transpose(-1, -2)).squeeze(1)
            else:
                return output

            result = output + self.scaling_factor * lora_output

            log_key = (base_module_name, layer_idx)

            if (
                self.log_grad_interval
                and self.log_grad_interval > 0
                and self._grad_step % self.log_grad_interval == 0
                and log_key not in self._logged_grad_modules
            ):

                def _grad_hook(grad, base_module_name=base_module_name, layer_idx=layer_idx):
                    try:
                        grad_norm = grad.norm().item()
                    except Exception:
                        grad_norm = float("nan")
                    logger.info(f"LoRA grad norm [{base_module_name}] layer={layer_idx}: {grad_norm:.6f}")
                    return grad

                result.register_hook(_grad_hook)
                self._logged_grad_modules.add(log_key)

            if not lora_output.requires_grad and (lora_A.requires_grad or lora_B.requires_grad):
                lora_output = lora_output.requires_grad_(True)

            if lora_output.requires_grad or torch.norm(lora_output) > 1e-8:
                if not result.requires_grad and lora_output.requires_grad:
                    result = result.detach().requires_grad_(True) + self.scaling_factor * lora_output
                return result
            else:
                zero_lora_output = torch.zeros_like(output, requires_grad=True)
                return output + zero_lora_output

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all Linear layers"""
        self.remove_hooks()  # Remove existing hooks first

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._create_hook_fn(name, module.weight.data.clone()))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def set_lora_weights(self, lora_weights_dict):
        """Set LoRA weights for the current forward pass"""
        self.current_lora_weights = lora_weights_dict
        self._grad_step += 1
        self._logged_grad_modules.clear()

    def activate(self):
        """Activate LoRA weight application"""
        self.active = True

    def deactivate(self):
        """Deactivate LoRA weight application"""
        self.active = False
        self.current_lora_weights = {}

    def set_grad_logging(self, interval: Optional[int]) -> None:
        if interval is None or interval <= 0:
            self.log_grad_interval = None
        else:
            self.log_grad_interval = int(interval)


__all__ = ["HookBasedLoRAManager"]
