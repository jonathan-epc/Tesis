import h5py
import pandas as pd
import numpy as np
import xarray as xr
import os
import re
from tqdm.autonotebook import tqdm

result_files = [f for f in os.listdir('./telemac/results/') if f.startswith('results_') and f.endswith('.slf')]

# Define the base directory
base_dir = 'telemac'

# Load the parameters from the CSV file
parameters = pd.read_csv(os.path.join(base_dir, 'parameters.csv'))

# Create a new HDF5 file
hdf5_file = h5py.File('simulation_data.hdf5', 'w')

# Get a list of all the files in the directory that match the pattern "result_i.slf"
result_files = [f for f in os.listdir(os.path.join(base_dir, 'results')) if f.startswith('results_') and f.endswith('.slf')]

# Loop over the results of each simulation
for result_file in tqdm(result_files):
    # Get the index of the simulation
    i = int(re.split("_|\.", result_file)[1])

    # Get the name of the simulation (without the extension)
    simulation_name = os.path.splitext(result_file)[0]

    try:
        # Load the results of the simulation
        result = xr.open_dataset(os.path.join(base_dir, 'results', result_file), engine="selafin")

        # Get the parameters for this simulation
        simulation_parameters = parameters.iloc[i]

        # Create a new group for the results of this simulation
        simulation_group = hdf5_file.create_group(simulation_name)

        # Store the results in the group
        for variable_name, variable_data in result.items():
            simulation_group.create_dataset(variable_name, data=variable_data)

        # Store the parameters as attributes of the group
        for parameter_name, parameter_value in simulation_parameters.items():
            simulation_group.attrs[parameter_name] = parameter_value

    except Exception as e:
        print(f'An error occurred when processing {result_file}: {e}')

# Close the HDF5 file
hdf5_file.close()
