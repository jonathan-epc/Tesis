# Parameter management module (param_utils.py)
from loguru import logger
import pandas as pd


def load_parameters(file_path):
    """
    Loads parameters from a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded parameters.

    Raises
    ------
    Exception
        If an error occurs while loading the parameters.

    Examples
    --------
    >>> load_parameters('parameters.csv')
       param1  param2
    id
    1      10      20
    2      15      25
    """
    try:
        return pd.read_csv(file_path, index_col="id")
    except Exception as e:
        logger.error(f"Error loading parameters from {file_path}: {e}")
        raise
