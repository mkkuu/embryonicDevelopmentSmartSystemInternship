"""
Structured logging for experiment runs. Every run's full output is
persisted to a file, not only printed to a terminal that may already be
closed by the time anyone wants to check the numbers again — the same gap
already identified while designing the reproduction protocol for the
original classifier training run, filled here for anything built on this
evaluation framework.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_path: Path, logger_name: str = "evaluation") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if called more than once in one process

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
