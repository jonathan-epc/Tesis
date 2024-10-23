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
                variable_stats[variable_name] = calculate_statistics(
                    pd.Series(variable_data[-1].values)
                )

            for parameter_name, parameter_value in simulation_parameters.items():
                simulation_group.attrs[parameter_name] = parameter_value

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

    variable_table = (
        pd.DataFrame(overall_stats).T.reset_index().rename(columns={"index": "names"})
    )
    table = pd.concat([parameter_table, variable_table])
    statistics_group = hdf5_file.create_group("statistics")
    for name in table["names"]:
        for stat in ["count", "min", "max", "variance", "mean"]:
            statistics_group.attrs[f"{name}_{stat}"] = table.loc[
                table["names"] == name
            ][stat].item()
    return table


def process_and_save_normalized(
    hdf5_file,
    condition,
    table,
    base_dir,
    parameters,
    parameter_names,
    variable_names,
    result_files,
):
    for idx, result_file in enumerate(tqdm(result_files, desc="Normalizing files")):
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