"""
Backward compatibility shim - imports moved to submodules.

This file re-exports all symbols from the split modules for backward compatibility.
New code should import directly from:
- ct2echo.utils.state_dict_utils
- ct2echo.utils.checkpoint_io
- ct2echo.utils.evaluation_io
"""

# State dict utilities
from ct2echo.utils.state_dict_utils import (
    LEGACY_HYPERNET_MODULE_MAP,
    MODULE_PREFIXES_WITH_NAMES,
    _remap_hypernet_state_dict_for_current_modules,
    _extract_head_state_dict,
)

# Checkpoint I/O
from ct2echo.utils.checkpoint_io import (
    save_hypernet_checkpoint,
    load_merged_hypernet_checkpoint,
    load_trained_model_with_hypernet,
    save_model_with_hypernet_final,
    load_final_merged_checkpoint,
    load_epoch_merged_checkpoint,
    save_final_models,
)

# Evaluation I/O
from ct2echo.utils.evaluation_io import save_evaluation_results


__all__ = [
    # state_dict_utils
    "LEGACY_HYPERNET_MODULE_MAP",
    "MODULE_PREFIXES_WITH_NAMES",
    "_remap_hypernet_state_dict_for_current_modules",
    "_extract_head_state_dict",
    # checkpoint_io
    "save_hypernet_checkpoint",
    "load_merged_hypernet_checkpoint",
    "load_trained_model_with_hypernet",
    "save_model_with_hypernet_final",
    "load_final_merged_checkpoint",
    "load_epoch_merged_checkpoint",
    "save_final_models",
    # evaluation_io
    "save_evaluation_results",
]
