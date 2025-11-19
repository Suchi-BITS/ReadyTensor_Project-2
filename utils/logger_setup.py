import logging
import os
from logging.handlers import RotatingFileHandler


def setup_execution_logger(log_path: str = "execution_trace.log") -> logging.Logger:
    """
    Configure a secure, production-ready logger for FinOps agent execution.

    Enhancements:
    - Validates path to prevent directory traversal
    - Ensures absolute path resolution
    - Uses RotatingFileHandler to avoid infinite log growth
    - Adds timestamps, levels, module info
    - Handles permission and file errors gracefully
    """

    # ---------------------------
    # Validate and secure log path
    # ---------------------------
    if not isinstance(log_path, str) or not log_path.strip():
        raise ValueError("log_path must be a non-empty string.")

    # Guardrail: prevent directory traversal
    forbidden = ["..", "//", "\\", "$(", ";", "|", "`"]
    if any(f in log_path for f in forbidden):
        raise ValueError(f"Unsafe log_path detected: {log_path}")

    # Convert to absolute path for safety
    log_path = os.path.abspath(log_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # ---------------------------
    # Create logger
    # ---------------------------
    logger = logging.getLogger("finops_trace")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:

        try:
            # Rotating logs (5 MB max, keep 3 files)
            handler = RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing log file handler: {e}")

        # -------------------------------------------
        # Better log format: timestamp + level + module
        # -------------------------------------------
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
