import os
import re

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
    channel_length,
    channel_width,
    parameters,
    parameter_table,
    parameter_names,
    variable_names,
    result_files,
):
    def _compute_adimensional_numbers(case):
        H0, Q0, SLOPE = (
            case["H0"],
            case["Q0"],
            case["SLOPE"],
        )
        xc, yc = channel_length, channel_width
        bc, hc = SLOPE * xc, H0
        uc = Q0 / (H0 * yc)

        return {
            "H*": case["H"] / hc,
            "U*": case["U"] / uc,
            "V*": case["V"] / uc,
            "B*": case["B"] / bc,
        }

    overall_stats = {}
    for idx, result_file in enumerate(
        tqdm(result_files, desc="Processing files"),
    ):
        try:
            i = int(re.split(r"_|\.", result_file)[0])

            if parameters.iloc[i]["BOTTOM"] != condition:
                logger.info(f"Skipping file: {result_file} for not being {condition}")
                continue

            # Open the primary result file
            result = xr.open_dataset(
                os.path.join(base_dir, "results", result_file), engine="selafin"
            )
            logger.debug(f"Opened {result_file} successfully")

            # Dynamically construct the path for the geometry file to fetch variable B
            bottom_type = parameters.iloc[i]["BOTTOM"]
            geometry_path = os.path.join(
                base_dir, "geometry", f"3x3_{bottom_type}_{i}.slf"
            )
            geometry_data = xr.open_dataset(geometry_path, engine="selafin")
            variable_B = geometry_data["B"][0]

            # Create simulation group in the HDF5 file
            simulation_parameters = parameters.iloc[i]
            simulation_group = hdf5_file.create_group(f"simulation_{i}")

            # Extract variables and calculate their statistics
            variable_stats = {}
            for variable_name, variable_data in result.items():
                if variable_name in variable_names:
                    simulation_group.create_dataset(
                        variable_name, data=variable_data[-1]
                    )
                    variable_stats[variable_name] = calculate_statistics(
                        pd.Series(variable_data[-1].values)
                    )

            # Add variable B and its stats
            simulation_group.create_dataset("B", data=variable_B.values)
            variable_stats["B"] = calculate_statistics(pd.Series(variable_B.values))

            # Prepare case dictionary for adimensional numbers
            case = {
                "H0": simulation_parameters["H0"],
                "Q0": simulation_parameters["Q0"],
                "n": simulation_parameters["n"],
                "nut": simulation_parameters["nut"],
                "SLOPE": simulation_parameters["SLOPE"],
                "H": result["H"][-1].values,
                "U": result["U"][-1].values,
                "V": result["V"][-1].values,
                "B": variable_B.values,
            }

            # Compute and save adimensional numbers
            adimensional_numbers = _compute_adimensional_numbers(case)

            # Define adimensional numbers to be added as groups
            grouped_adim_numbers = {"B*", "H*", "U*", "V*"}
            for adim_name, adim_value in adimensional_numbers.items():
                if adim_name in grouped_adim_numbers:
                    # Add specific adimensional numbers as datasets
                    simulation_group.create_dataset(adim_name, data=adim_value)
                    variable_stats[adim_name] = calculate_statistics(
                        pd.Series(adim_value)
                    )
                else:
                    # Add other adimensional numbers as simulation attributes
                    simulation_group.attrs[adim_name] = adim_value

            # Add simulation parameters as attributes
            for parameter_name, parameter_value in simulation_parameters.items():
                simulation_group.attrs[parameter_name] = parameter_value

            # Update overall statistics
            if idx == 0 or not overall_stats:
                overall_stats = variable_stats
            else:
                overall_stats = {
                    var_name: combine_statistics(
                        overall_stats[var_name], variable_stats[var_name]
                    )
                    for var_name in {**variable_stats, **overall_stats}
                }

            logger.info(f"Processed file: {result_file}")

        except Exception as e:
            logger.error(f"An error occurred when processing {result_file}: {e}")

    # Create the variable table and save statistics
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
    logger.info("Saved statistics to HDF5 file")
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
    for _idx, result_file in enumerate(tqdm(result_files, desc="Normalizing files")):
        try:
            i = int(re.split(r"_|\.", result_file)[0])

            if parameters.iloc[i]["BOTTOM"] != condition:
                logger.info(f"Skipping file: {result_file} for not being {condition}")
                continue

            # Open the primary result file
            result = xr.open_dataset(
                os.path.join(base_dir, "results", result_file), engine="selafin"
            )
            logger.debug(f"Opened {result_file} succesfully")

            # Dynamically construct the path for the geometry file to fetch variable B
            bottom_type = parameters.iloc[i]["BOTTOM"]
            geometry_path = os.path.join(
                base_dir, "geometry", f"3x3_{bottom_type}_{i}.slf"
            )
            logger.debug(f"Constructed path for geometry of {result_file} succesfully")

            # Open the geometry file and extract variable B
            geometry_data = xr.open_dataset(geometry_path, engine="selafin")
            variable_B = geometry_data["B"]
            logger.debug(f"Extracted geometry for {result_file} succesfully")

            # Retrieve simulation parameters
            simulation_parameters = parameters.iloc[i]

            # Create a simulation group in the HDF5 file
            simulation_group = hdf5_file.create_group(f"simulation_{i}")

            for variable_name, variable_data in result.items():
                normalized_data = normalize_statistics(
                    variable_data[-1], variable_name, table
                )
                simulation_group.create_dataset(variable_name, data=normalized_data)

            # Normalize and save variable B
            normalized_B = normalize_statistics(variable_B, "B", table)
            simulation_group.create_dataset("B", data=normalized_B)

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

    # Create a statistics group and store statistics in HDF5 attributes
    statistics_group = hdf5_file.create_group("statistics")
    for name in table["names"]:
        for stat in ["count", "min", "max", "variance", "mean"]:
            statistics_group.attrs[f"{name}_{stat}"] = table.loc[
                table["names"] == name
            ][stat].item()
