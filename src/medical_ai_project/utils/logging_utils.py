"""Logging configuration utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name.
        log_file: Optional file path to persist logs.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
