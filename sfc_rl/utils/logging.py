"""Logging utilities with colorlog (colored console + file)."""

import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup a logger with colored console and optional colored file output.

    Args:
        name: Logger name
        log_file: Optional path to log file (colors included)
        level: Logging level (e.g., logging.INFO)
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 🎨 Colored formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s %(levelname)-8s %(asctime)s  [%(name)s]   %(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
    )

    # 🖥️ Console handler (standard logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 📁 File handler (standard logging)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


