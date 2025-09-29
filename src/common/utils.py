# common/utils.py

import random
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def setup_logger(
    name: str = "log",
    log_dir: str = "logs",
    log_level: str = "DEBUG",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> logger:
    """Sets up a Loguru logger with customizable parameters.

    This function removes any existing logger handlers and configures new ones
    for file-based logging and stderr output for errors.

    Args:
        name (str): Name of the log file (without extension).
        log_dir (str): Directory to store log files.
        log_level (str): Minimum log level to capture (e.g., "INFO", "DEBUG").
        rotation (str): Condition for rotating the log file (e.g., "10 MB").
        retention (str): How long to keep log files (e.g., "1 week").

    Returns:
        logger: The configured Loguru logger instance.
    """
    logger.remove()
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.add(
        log_path / f"{name}.log",
        rotation=rotation,
        retention=retention,
        level=log_level.upper(),
        format=log_format,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(sys.stderr, level="ERROR", format=log_format)
    return logger


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The integer value to use as the seed.
    """
    print(f"Using {seed} as seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The following lines can be enabled for perfect reproducibility, but may
    # impact performance.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
