import os
import re

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.autonotebook import tqdm
from loguru import logger


def calculate_statistics(variable_data):
    """Calculate statistics for a variable."""
    variable_stats = variable_data.describe()
    variable_stats.rename({"std": "var"}, inplace=True)
    variable_stats["var"] = variable_stats["var"] ** 2
    return variable_stats


def combine_statistics(stats1, stats2):
    """Combine statistics from two variables."""
    count = stats1["count"] + stats2["count"]
    mean = (stats1["count"] * stats1["mean"] + stats2["count"] * stats2["mean"]) / count
    var = (
        (stats1["count"] - 1) * stats1["var"] + (stats2["count"] - 1) * stats2["var"]
    ) / (count - 1) + (
        stats1["count"] * stats2["count"] * (stats1["mean"] - stats2["mean"]) ** 2
    ) / (
        count * (count - 1)
    )
    min_ = np.min([stats1["min"], stats2["min"]])
    max_ = np.max([stats1["max"], stats2["max"]])
    return pd.Series(
        {"count": count, "mean": mean, "var": var, "min": min_, "max": max_}
    )


# Configure the logger
logger.add("simulation.log", format="{time} {level} {message}", level="INFO")

# Define the base directory
base_dir = "telemac"

# Load the parameters from the CSV file
parameters = pd.read_csv(os.path.join(base_dir, "parameters.csv"))
parameter_names = ["H0", "Q0", "SLOPE", "n"]
parameter_stats = {
    parameter: parameters[parameter].describe() for parameter in parameter_names
}
parameter_table = pd.DataFrame(parameter_stats).T
parameter_table.reset_index(inplace=True)
parameter_table.rename(columns={"index": "names", "std": "var"}, inplace=True)
parameter_table["var"] = parameter_table["var"] ** 2

# Define variable names
variable_names = ["F", "B", "H", "Q", "S", "U", "V"]

# Create a new HDF5 file
hdf5_file = h5py.File("ML/simulation_data.hdf5", "w")

# Get a list of all the files in the directory that match the pattern "result_i.slf"
result_files = [
    f
    for f in os.listdir(os.path.join(base_dir, "results"))
    if f.startswith("results_") and f.endswith(".slf")
]

# Loop over the results of each simulation
for result_file in tqdm(result_files):
    # Get the index of the simulation
    i = int(re.split(r"_|\.", result_file)[1])

    # Get the name of the simulation (without the extension)
    simulation_name = os.path.splitext(result_file)[0]

    try:
        # Load the results of the simulation
        result = xr.open_dataset(
            os.path.join(base_dir, "results", result_file), engine="selafin"
        )
        # Get the parameters for this simulation
        simulation_parameters = parameters.iloc[i]
        # Create a new group for the results of this simulation
        simulation_group = hdf5_file.create_group(simulation_name)

        # Store the results in the group
        variable_stats = {}
        for variable_name, variable_data in result.items():
            simulation_group.create_dataset(variable_name, data=variable_data[-1])
            variable_stats[variable_name] = calculate_statistics(
                pd.Series(variable_data[-1].values)
            )

        # Store the parameters as attributes of the group
        for parameter_name, parameter_value in simulation_parameters.items():
            simulation_group.attrs[parameter_name] = parameter_value

        # Combine statistics for this simulation with the overall statistics
        if i == 0:
            overall_stats = variable_stats
        else:
            overall_stats = {
                variable_name: combine_statistics(
                    overall_stats[variable_name], variable_stats[variable_name]
                )
                for variable_name in variable_names
            }
        logger.info("Processed file: {}", result_file)

    except Exception as e:
        logger.error("An error occurred when processing {}: {}", result_file, e)

# Create a table of overall statistics
variable_table = pd.DataFrame(overall_stats).T
variable_table.reset_index(inplace=True)
variable_table.rename(columns={"index": "names"}, inplace=True)

# Concatenate parameter and variable tables
table = pd.concat([parameter_table, variable_table])

statistics_group = hdf5_file.create_group("statistics")
for name in list(table["names"]):
    statistics_group.attrs[name + "_count"] = table.loc[table["names"] == name]["count"]
    statistics_group.attrs[name + "_min"] = table.loc[table["names"] == name]["min"]
    statistics_group.attrs[name + "_max"] = table.loc[table["names"] == name]["max"]
    statistics_group.attrs[name + "_var"] = table.loc[table["names"] == name]["var"]
    statistics_group.attrs[name + "_mean"] = table.loc[table["names"] == name]["mean"]

# Close the HDF5 file
hdf5_file.close()

logger.success("Script completed successfully")
