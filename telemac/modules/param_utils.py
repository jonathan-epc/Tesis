# Parameter management module (param_utils.py)
from loguru import logger
import pandas as pd
def load_parameters(file_path):
    try:
        return pd.read_csv(file_path, index_col="id")
    except Exception as e:
        logger.error(f"Error loading parameters from {file_path}: {e}")
        raise