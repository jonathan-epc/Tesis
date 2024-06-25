import os
import re

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm.autonotebook import tqdm


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


def process_simulation_results(
    base_dir: str, output_file: str, normalized_output_file: str
):
    """
    Process simulation results and save them to HDF5 files.

    Args:
        base_dir (str): The base directory containing simulation results and parameters.
        output_file (str): The path to save the HDF5 file with raw data.
        normalized_output_file (str): The path to save the HDF5 file with normalized data.
    """
    logger.info("Processing start")

    # Load parameters
    parameters = pd.read_csv(os.path.join(base_dir, "parameters.csv"))
    parameter_names = ["H0", "Q0", "SLOPE", "n"]
    parameter_stats = {param: parameters[param].describe() for param in parameter_names}
    parameter_table = (
        pd.DataFrame(parameter_stats)
        .T.reset_index()
        .rename(columns={"index": "names", "std": "variance"})
    )
    parameter_table["variance"] = parameter_table["variance"] ** 2

    # Define variable names
    variable_names = ["F", "B", "H", "Q", "S", "U", "V"]

    # Get list of result files
    result_files = [
        f
        for f in os.listdir(os.path.join(base_dir, "results"))
        if f.startswith("results_") and f.endswith(".slf")
    ]

    with h5py.File(output_file, "w") as hdf5_file:
        overall_stats = {}
        for idx, result_file in enumerate(tqdm(result_files, desc="Processing files")):
            try:
                i = int(re.split(r"_|\.", result_file)[1])
                simulation_name = os.path.splitext(result_file)[0]

                result = xr.open_dataset(
                    os.path.join(base_dir, "results", result_file), engine="selafin"
                )
                simulation_parameters = parameters.iloc[i]
                simulation_group = hdf5_file.create_group(simulation_name)

                variable_stats = {}
                for variable_name, variable_data in result.items():
                    simulation_group.create_dataset(
                        variable_name, data=variable_data[-1]
                    )
                    variable_stats[variable_name] = calculate_statistics(
                        pd.Series(variable_data[-1].values)
                    )

                for parameter_name, parameter_value in simulation_parameters.items():
                    simulation_group.attrs[parameter_name] = parameter_value

                if idx == 0:
                    overall_stats = variable_stats
                else:
                    overall_stats = {
                        var_name: combine_statistics(
                            overall_stats[var_name], variable_stats[var_name]
                        )
                        for var_name in variable_names
                    }
                logger.info(f"Processed file: {result_file}")

            except Exception as e:
                logger.error(f"An error occurred when processing {result_file}: {e}")

        variable_table = (
            pd.DataFrame(overall_stats)
            .T.reset_index()
            .rename(columns={"index": "names"})
        )
        table = pd.concat([parameter_table, variable_table])
        statistics_group = hdf5_file.create_group("statistics")
        for name in table["names"]:
            for stat in ["count", "min", "max", "variance", "mean"]:
                statistics_group.attrs[f"{name}_{stat}"] = table.loc[
                    table["names"] == name
                ][stat].item()
    logger.success("Data saved successfully")

    with h5py.File(normalized_output_file, "w") as hdf5_file:
        for idx, result_file in enumerate(tqdm(result_files, desc="Normalizing files")):
            try:
                i = int(re.split(r"_|\.", result_file)[1])
                simulation_name = os.path.splitext(result_file)[0]

                result = xr.open_dataset(
                    os.path.join(base_dir, "results", result_file), engine="selafin"
                )
                simulation_parameters = parameters.iloc[i]
                
                simulation_group = hdf5_file.create_group(simulation_name)

                
                for variable_name, variable_data in result.items():
                    normalized_data = normalize_statistics(
                        variable_data[-1], variable_name, table
                    )
                    
                    simulation_group.create_dataset(variable_name, data=normalized_data)

                for parameter_name, parameter_value in simulation_parameters.items():
                    if parameter_name in parameter_names:
                        normalized_value = normalize_statistics(
                            parameter_value, parameter_name, table
                        )
                        simulation_group.attrs[parameter_name] = normalized_value
                    else:
                        simulation_group.attrs[parameter_name] = parameter_value
                logger.info(f"Processed file: {result_file}")

            except Exception as e:
                logger.error(f"An error occurred when processing {result_file}: {e}")

        statistics_group = hdf5_file.create_group("statistics")
        for name in table["names"]:
            for stat in ["count", "min", "max", "variance", "mean"]:
                statistics_group.attrs[f"{name}_{stat}"] = table.loc[
                    table["names"] == name
                ][stat].item()
    logger.success("Normalized data saved successfully")

    logger.success("Script completed successfully")

if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log", format="{time} {level} {message}", level="INFO"
    )

    BASE_DIR = "telemac"
    OUTPUT_FILE = "ML/simulation_data.hdf5"
    NORMALIZED_OUTPUT_FILE = "ML/simulation_data_normalized.hdf5"

    process_simulation_results(BASE_DIR, OUTPUT_FILE, NORMALIZED_OUTPUT_FILE)
