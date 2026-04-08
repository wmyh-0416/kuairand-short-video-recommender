from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "kuairand_rec",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create a console logger with an optional file handler.

    Repeated calls with the same logger name are idempotent: old handlers are
    cleared so scripts do not duplicate log lines when imported in notebooks.
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()

    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = level

    logger.setLevel(level_value)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level_value)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_value)
        logger.addHandler(file_handler)

    return logger
