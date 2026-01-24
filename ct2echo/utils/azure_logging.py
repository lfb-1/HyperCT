"""Azure SDK logging utilities.

Centralized module for suppressing verbose Azure SDK logging across the codebase.
"""
import logging
from contextlib import contextmanager

AZURE_LOGGERS = [
    "azure",
    "azure.ai.ml",
    "azure.core",
    "azure.core.pipeline",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.storage",
    "azure.storage.blob",
    "azure.identity",
    "urllib3",
    "urllib3.connectionpool",
    "msrest",
    "msrest.http_logger",
    "adal-python",
]


def suppress_azure_logs(level: int = logging.WARNING) -> None:
    """Suppress verbose Azure SDK logging.

    Args:
        level: Logging level to set for Azure loggers. Defaults to WARNING.
    """
    for name in AZURE_LOGGERS:
        logging.getLogger(name).setLevel(level)
    # Extra noisy loggers get CRITICAL level
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)


@contextmanager
def quiet_azure_logs():
    """Context manager for temporary Azure log suppression.

    Saves original logging levels, suppresses Azure logs, then restores
    original levels on exit.

    Example:
        with quiet_azure_logs():
            ml_client = MLClient(credential, ...)
    """
    original = {name: logging.getLogger(name).level for name in AZURE_LOGGERS}
    suppress_azure_logs()
    try:
        yield
    finally:
        for name, level in original.items():
            logging.getLogger(name).setLevel(level)
