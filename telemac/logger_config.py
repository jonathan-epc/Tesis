import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    name="log", log_dir="logs", log_level="DEBUG", rotation="10 MB", retention="1 week"
):
    """
    Set up a Loguru logger with customizable parameters.

    Args:
        name (str): Name of the log file (without extension)
        log_dir (str): Directory to store log files
        log_level (str): Minimum log level to capture
        rotation (str): When to rotate the log file (e.g., "10 MB", "1 day")
        retention (str): How long to keep log files (e.g., "1 week", "1 month")

    Returns:
        logger: Configured Loguru logger object
    """
    # Remove any existing handlers
    logger.remove()

    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Define log format
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Add a handler to write all messages to the log file
    logger.add(
        f"{log_dir}/{name}.log",
        rotation=rotation,
        retention=retention,
        level=log_level.upper(),
        format=log_format,
        enqueue=True,  # Makes logging thread-safe
        backtrace=True,  # Adds exception tracebacks to error logs
        diagnose=True,  # Adds extra diagnostic info to error logs
    )

    # Add a handler to write only error messages to sys.stderr
    logger.add(sys.stderr, level="ERROR", format=log_format)

    # Optionally, you can add a handler for critical errors to be sent via email
    # This requires additional setup and dependencies
    # logger.add(send_email, level="CRITICAL")

    return logger