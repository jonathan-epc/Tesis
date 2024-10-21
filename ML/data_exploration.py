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
        variables=config.data.variables,
        parameters=config.data.parameters,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize=(True, True),
        device=config.device,
    )

def extract_data(dataset, config):
    input_data = {param: [] for param in config.data.parameters}
    output_data = {var: [] for var in config.data.variables}

    for case in dataset:
        inputs, outputs = case
        params, param_B = inputs

        for i, param in enumerate(config.data.parameters):
            input_data[param].append(param_B if param == "B" else params[i])

        for i, var in enumerate(config.data.variables):
            output_data[var].append(outputs[i, :, :])

    return {param: torch.stack(values) for param, values in input_data.items()}, \
           {var: torch.stack(values) for var, values in output_data.items()}

def compute_statistics(data):
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
    input_data, output_data = extract_data(dataset, config)
    plt.style.use('seaborn-v0_8-whitegrid')

    stats_dict = {}
    for param, data in input_data.items():
        stats_dict[f"input_{param}"] = compute_statistics(data)
        plot_distribution(data, f"Input Parameter: {param}", param)

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
