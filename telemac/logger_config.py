from loguru import logger
import sys

def setup_logger(name="log"):
    logger.remove()

    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"

    # Add a handler to write all messages to the log file
    logger.add(f"logs/{name}.log", rotation="10 MB", level="DEBUG", format=log_format)

    # Add a handler to write only error messages to sys.stderr
    logger.add(sys.stderr, level="ERROR", format=log_format)
