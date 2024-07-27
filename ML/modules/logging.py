# logging_utils.py
from loguru import logger
from tqdm.autonotebook import tqdm


def setup_logger():
    logger.remove()
    logger.add("logs/file_{time}.log", rotation="500 MB")
    #logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
    return logger