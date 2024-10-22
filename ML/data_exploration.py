import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from config import get_config
from modules.data import HDF5Dataset

def load_dataset(config, file_path):
    return HDF5Dataset(
        file_path=file_path,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize=(True, True),
        device='cpu',
    )

def extract_data(dataset, config):
    """
    Extract data from the dataset into separate dictionaries for inputs and outputs.
    The dataset now returns (inputs, outputs) where each is a list of tensors.
    """
    input_data = {var: [] for var in config.data.inputs}
    output_data = {var: [] for var in config.data.outputs}

    for idx in range(len(dataset)):
        inputs, outputs = dataset[idx]
        
        # Process inputs
        for var_name, tensor in zip(config.data.inputs, inputs):
            input_data[var_name].append(tensor)
            
        # Process outputs
        for var_name, tensor in zip(config.data.outputs, outputs):
            output_data[var_name].append(tensor)

    # Stack tensors for each variable
    return {var: torch.stack(tensors) for var, tensors in input_data.items()}, \
           {var: torch.stack(tensors) for var, tensors in output_data.items()}

def compute_statistics(data):
    """Compute statistical measures for the input tensor."""
    numpy_data = data.cpu().numpy().flatten()
    return {
        "mean": np.mean(numpy_data),
        "std": np.std(numpy_data),
        "min": np.min(numpy_data),
        "max": np.max(numpy_data),
        "skewness": stats.skew(numpy_data),
        "kurtosis": stats.kurtosis(numpy_data),
        "percentile_25": np.percentile(numpy_data, 25),
        "median": np.median(numpy_data),
        "percentile_75": np.percentile(numpy_data, 75),
    }

def plot_distribution(data, title, xlabel):
    """Plot the distribution of the input tensor."""
    plt.figure(figsize=(8, 5))
    plt.hist(data.cpu().numpy().flatten(), bins=50, density=True, alpha=0.6, label="Histogram")
    plt.title(f"Distribution of {title}")
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    config = get_config()
    dataset = load_dataset(config, "data/bars.hdf5")
    
    # Extract and analyze the data
    input_data, output_data = extract_data(dataset, config)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Compute statistics and create plots
    stats_dict = {}
    
    # Process input variables
    for param, data in input_data.items():
        stats_dict[f"input_{param}"] = compute_statistics(data)
        plot_distribution(data, f"Input Parameter: {param}", param)

    # Process output variables
    for var, data in output_data.items():
        stats_dict[f"output_{var}"] = compute_statistics(data)
        plot_distribution(data, f"Output Variable: {var}", var)

    # Convert stats_dict to a pandas DataFrame
    stats_df = pd.DataFrame(stats_dict).T

    # Print the statistics as a table
    print("\nStatistics Table:")
    print(stats_df)

if __name__ == "__main__":
    main()
