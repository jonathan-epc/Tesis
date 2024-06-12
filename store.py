import os
import re

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.autonotebook import tqdm

result_files = [
    f
    for f in os.listdir("./telemac/results/")
    if f.startswith("results_") and f.endswith(".slf")
]

# Define the base directory
base_dir = "telemac"
# Load the parameters from the CSV file
parameters = pd.read_csv(os.path.join(base_dir, "parameters.csv"))
parameter_names = ["H0", "Q0", "SLOPE", "n"]
stats = {parameter: parameters[parameter].describe() for parameter in parameter_names}
table1 = pd.DataFrame(
    {
        "names": parameter_names,
        "count": [stats[parameter]["count"] for parameter in parameter_names],
        "min": [stats[parameter]["min"] for parameter in parameter_names],
        "max": [stats[parameter]["max"] for parameter in parameter_names],
        "var": [stats[parameter]["std"] ** 2 for parameter in parameter_names],
        "mean": [stats[parameter]["mean"] for parameter in parameter_names],
    }
)

variable_names = ["F", "B", "H", "Q", "S", "U", "V"]

stats = {
    variable: pd.Series(
        {"count": 0, "var": 0.0, "min": 9999, "max": -9999, "mean": 9999}
    )
    for variable in variable_names
}

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
        for variable_name, variable_data in result.items():
            simulation_group.create_dataset(variable_name, data=variable_data[-1])
            variable_stats = pd.Series(variable_data[-1].values).describe()
            stats[variable_name]["min"] = np.min(
                [stats[variable_name]["min"], variable_stats["min"]]
            )
            stats[variable_name]["max"] = np.max(
                [stats[variable_name]["max"], variable_stats["max"]]
            )
            stats[variable_name]["var"] = (
                (stats[variable_name]["count"] - 1) * stats[variable_name]["var"]
                + (variable_stats["count"] - 1) * variable_stats["std"] ** 2
            ) / (stats[variable_name]["count"] + variable_stats["count"] - 1) + (
                (stats[variable_name]["count"] * variable_stats["count"])
                * (stats[variable_name]["mean"] - variable_stats["mean"]) ** 2
            ) / (
                (stats[variable_name]["count"] + variable_stats["count"])
                * (stats[variable_name]["count"] + variable_stats["count"] - 1)
            )
            stats[variable_name]["mean"] = (
                stats[variable_name]["mean"] * stats[variable_name]["count"]
                + variable_stats["mean"] * variable_stats["count"]
            ) / (stats[variable_name]["count"] + variable_stats["count"])
            stats[variable_name]["count"] = (
                stats[variable_name]["count"] + variable_stats["count"]
            )

        # Store the parameters as attributes of the group
        for parameter_name, parameter_value in simulation_parameters.items():
            simulation_group.attrs[parameter_name] = parameter_value

    except Exception as e:
        print(f"An error occurred when processing {result_file}: {e}")

table2 = pd.DataFrame(
    {
        "names": variable_names,
        "count": [stats[variable]["count"] for variable in variable_names],
        "min": [stats[variable]["min"] for variable in variable_names],
        "max": [stats[variable]["max"] for variable in variable_names],
        "var": [stats[variable]["var"] for variable in variable_names],
        "mean": [stats[variable]["mean"] for variable in variable_names],
    }
)

table = pd.concat([table1, table2])

# Create a new group for the statistics
statistics_group = hdf5_file.create_group("statistics")
for name in list(table["names"]):
    statistics_group.attrs[name + "_count"] = table.loc[table["names"] == name]["count"]
    statistics_group.attrs[name + "_min"] = table.loc[table["names"] == name]["min"]
    statistics_group.attrs[name + "_max"] = table.loc[table["names"] == name]["max"]
    statistics_group.attrs[name + "_var"] = table.loc[table["names"] == name]["var"]
    statistics_group.attrs[name + "_mean"] = table.loc[table["names"] == name]["mean"]


# Close the HDF5 file
hdf5_file.close()
