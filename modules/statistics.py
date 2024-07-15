import pandas as pd
import numpy as np
def calculate_statistics(variable_data: pd.Series) -> pd.Series:
    """
    Calculate statistics for a variable.

    Args:
        variable_data (pd.Series): The data for the variable.

    Returns:
        pd.Series: The statistics for the variable.
    """
    variable_stats = variable_data.describe()
    variable_stats.rename({"std": "variance"}, inplace=True)
    variable_stats["variance"] = variable_stats["variance"] ** 2
    return variable_stats

def combine_statistics(stats1: pd.Series, stats2: pd.Series) -> pd.Series:
    """
    Combine statistics from two variables.

    Args:
        stats1 (pd.Series): The first set of statistics.
        stats2 (pd.Series): The second set of statistics.

    Returns:
        pd.Series: The combined statistics.
    """
    count = stats1["count"] + stats2["count"]
    mean = (stats1["count"] * stats1["mean"] + stats2["count"] * stats2["mean"]) / count
    var = (
        (stats1["count"] - 1) * stats1["variance"]
        + (stats2["count"] - 1) * stats2["variance"]
    ) / (count - 1) + (
        stats1["count"] * stats2["count"] * (stats1["mean"] - stats2["mean"]) ** 2
    ) / (
        count * (count - 1)
    )
    min_ = np.min([stats1["min"], stats2["min"]])
    max_ = np.max([stats1["max"], stats2["max"]])
    return pd.Series(
        {"count": count, "mean": mean, "variance": var, "min": min_, "max": max_}
    )

def normalize_statistics(stat_value, stat_name, table):
    """
    Normalize a statistic value based on the statistics table.

    Args:
        stat_value (float): The value to be normalized.
        stat_name (str): The name of the statistic.
        table (pd.DataFrame): The table containing mean and variance for normalization.

    Returns:
        float: The normalized value.
    """
    mean = table.loc[table["names"] == stat_name]["mean"].item()
    variance = table.loc[table["names"] == stat_name]["variance"].item()
    return (stat_value - mean) / np.sqrt(variance)
