"""Logging utilities with colorlog (colored console + file)."""

import logging
import sys
from pathlib import Path
from typing import Optional
import colorlog

# Track which loggers we've already set up to prevent duplicates
_setup_loggers = set()

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    force_setup: bool = False,
) -> logging.Logger:
    """Setup a logger with colored console and optional colored file output.

    Args:
        name: Logger name
        log_file: Optional path to log file (colors included)
        level: Logging level (e.g., logging.INFO)
        force_setup: If True, reconfigure logger even if already set up
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if name in _setup_loggers and not force_setup and logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Only clear handlers if we're forcing setup or this is first time
    if force_setup or name not in _setup_loggers:
        logger.handlers.clear()

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # 🎨 Colored formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        reset=True,
    )

    # 🖥️ Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 📁 File handler
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _setup_loggers.add(name)
    return logger
