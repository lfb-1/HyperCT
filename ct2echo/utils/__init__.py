"""
Utils subpackage - General utilities.

Contains:
- utils: General utility functions (seed_all, metrics, etc.)
- io_utils: I/O utilities for saving/loading checkpoints (backward compatibility shim)
- state_dict_utils: State dict remapping utilities
- checkpoint_io: Checkpoint save/load utilities
- evaluation_io: Evaluation results I/O
- azure_logging: Azure SDK logging utilities
"""

from ct2echo.utils.utils import (
    seed_all,
    init_azure,
    filter_empty_input,
    save_pred,
    plot_roc,
    log_metrics,
)

# Backward compatible imports from io_utils.py shim
from ct2echo.utils.io_utils import (
    save_evaluation_results,
    save_hypernet_checkpoint,
    load_merged_hypernet_checkpoint,
    load_trained_model_with_hypernet,
    save_model_with_hypernet_final,
    load_final_merged_checkpoint,
    load_epoch_merged_checkpoint,
    save_final_models,
)

from ct2echo.utils.azure_logging import (
    suppress_azure_logs,
    quiet_azure_logs,
    AZURE_LOGGERS,
)

__all__ = [
    # utils
    "seed_all",
    "init_azure",
    "filter_empty_input",
    "save_pred",
    "plot_roc",
    "log_metrics",
    # io_utils (state_dict_utils, checkpoint_io, evaluation_io)
    "save_evaluation_results",
    "save_hypernet_checkpoint",
    "load_merged_hypernet_checkpoint",
    "load_trained_model_with_hypernet",
    "save_model_with_hypernet_final",
    "load_final_merged_checkpoint",
    "load_epoch_merged_checkpoint",
    "save_final_models",
    # azure_logging
    "suppress_azure_logs",
    "quiet_azure_logs",
    "AZURE_LOGGERS",
]
