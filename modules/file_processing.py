import os
import re

import h5py
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm.autonotebook import tqdm

from modules.statistics import (
    calculate_statistics,
    combine_statistics,
    normalize_statistics,
)


def process_and_save(
    hdf5_file,
    condition,
    base_dir,
    parameters,
    parameter_table,
    parameter_names,
    variable_names,
    result_files,
):
    overall_stats = {}
    for idx, result_file in enumerate(
        tqdm(result_files, desc="Processing files"),
    ):
        try:
            i = int(re.split(r"_|\.", result_file)[0])

            if parameters.iloc[i]["BOTTOM"] != condition:
                logger.info(f"Skipping file: {result_file} for not being {condition}")
                continue

            result = xr.open_dataset(
                os.path.join(base_dir, "results", result_file), engine="selafin"
            )
            simulation_parameters = parameters.iloc[i]
            simulation_group = hdf5_file.create_group(f"simulation_{i}")

            variable_stats = {}
            for variable_name, variable_data in result.items():
                simulation_group.create_dataset(variable_name, data=variable_data[-1])
                if variable_name not in overall_stats:
                    overall_stats[variable_name] = []
                overall_stats[variable_name].extend(variable_data[-1].values.flatten())

            for parameter_name, parameter_value in simulation_parameters.items():
                simulation_group.create_dataset(parameter_name, data=parameter_value)
                if parameter_name not in overall_stats:
                    overall_stats[parameter_name] = []
                overall_stats[parameter_name].append(parameter_value)

            if idx == 0 or not overall_stats:
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


    # Calculate and store overall statistics
    statistics_group = hdf5_file.create_group("statistics")
    stats_data = []
    
    for name, values in overall_stats.items():
        series = pd.Series(values)
        stats = calculate_statistics(series)
        for stat, value in stats.items():
            statistics_group.attrs[f"{name}_{stat}"] = value
        stats['name'] = name
        stats_data.append(stats)
    
    # Create a DataFrame with all statistics
    stats_table = pd.DataFrame(stats_data)
    
    return stats_table