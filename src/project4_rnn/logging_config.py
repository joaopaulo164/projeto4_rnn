"""Logging configuration for project4_rnn."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "project4_rnn",
    log_level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Name of the logger.
        log_level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to write log records to file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Check if file handler for this exact path already exists
        file_handler_exists = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
            for h in logger.handlers
        )
        if not file_handler_exists:
            file_formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger
