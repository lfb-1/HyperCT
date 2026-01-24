"""
State dict remapping utilities for hypernet checkpoints.

Contains:
- LEGACY_HYPERNET_MODULE_MAP: Mapping for legacy module names
- _remap_hypernet_state_dict_for_current_modules: Adapt older checkpoints to current naming
- _extract_head_state_dict: Extract prediction head state dict from model
"""

from collections import OrderedDict
import torch


LEGACY_HYPERNET_MODULE_MAP = {
    "to_q": ["spatial_to_q", "temporal_to_q"],
    "to_kv": ["spatial_to_kv", "temporal_to_kv"],
    "to_out": ["spatial_to_out", "temporal_to_out"],
    "ff_net_1": ["spatial_ff_net_1", "temporal_ff_net_1"],
    "ff_net_4": ["spatial_ff_net_4", "temporal_ff_net_4"],
}

MODULE_PREFIXES_WITH_NAMES = (
    "heads",
    "adaptive_weight_generators",
    "local_refinement_networks",
    "cross_attention_layers",
    "cross_attention_queries",
)


def _remap_hypernet_state_dict_for_current_modules(state_dict, hypernet):
    """Adapt older hypernet checkpoints to the current module naming scheme."""
    if not hasattr(hypernet, "target_modules"):
        return state_dict, {"remapped": [], "dropped": []}

    expected_modules = set(hypernet.target_modules)
    if not expected_modules:
        return state_dict, {"remapped": [], "dropped": []}

    remapped_state = OrderedDict()
    remapped_pairs = []
    dropped_keys = []

    for key, value in state_dict.items():
        handled = False
        for prefix in MODULE_PREFIXES_WITH_NAMES:
            prefix_with_sep = f"{prefix}."
            if key.startswith(prefix_with_sep):
                handled = True
                remainder = key[len(prefix_with_sep) :]
                if "." in remainder:
                    module_name, suffix = remainder.split(".", 1)
                    suffix = f".{suffix}"
                else:
                    module_name, suffix = remainder, ""

                if module_name in expected_modules:
                    remapped_state[key] = value
                else:
                    legacy_targets = LEGACY_HYPERNET_MODULE_MAP.get(module_name, [])
                    valid_targets = [m for m in legacy_targets if m in expected_modules]

                    if valid_targets:
                        for new_module in valid_targets:
                            new_key = f"{prefix_with_sep}{new_module}{suffix}"
                            remapped_state[new_key] = value.clone() if torch.is_tensor(value) else value
                            remapped_pairs.append((key, new_key))
                    else:
                        # Drop keys that no longer correspond to any module
                        dropped_keys.append(key)
                break
        if handled:
            continue

        remapped_state[key] = value

    # Remove legacy embedding rows if their shape no longer matches (module count changed)
    embedding_key = "layer_type_encoder.0.weight"
    if (hasattr(hypernet, "layer_type_encoder")
            and embedding_key in remapped_state
            and torch.is_tensor(remapped_state[embedding_key])):
        expected_rows = hypernet.layer_type_encoder[0].weight.shape[0]
        if remapped_state[embedding_key].shape[0] != expected_rows:
            dropped_keys.append(embedding_key)
            remapped_state.pop(embedding_key, None)

    return remapped_state, {"remapped": remapped_pairs, "dropped": dropped_keys}


def _extract_head_state_dict(base_model):
    """Return (attribute_name, state_dict) for the prediction head if available."""
    for attr in ("mlp_head", "classifier"):
        module = getattr(base_model, attr, None)
        if isinstance(module, torch.nn.Module):
            return attr, module.state_dict()
    return None, None


__all__ = [
    "LEGACY_HYPERNET_MODULE_MAP",
    "MODULE_PREFIXES_WITH_NAMES",
    "_remap_hypernet_state_dict_for_current_modules",
    "_extract_head_state_dict",
]
