import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from config import CONFIG

from modules.data import HDF5Dataset


def load_datasets(normalized_file, original_file):
    return (
        HDF5Dataset(
            file_path=normalized_file,
            variables=CONFIG["data"]["variables"],
            parameters=CONFIG["data"]["parameters"],
            numpoints_x=CONFIG["data"]["numpoints_x"],
            numpoints_y=CONFIG["data"]["numpoints_y"],
            normalize=CONFIG["data"]["normalize"],
            device=CONFIG["device"],
        ),
        HDF5Dataset(
            file_path=original_file,
            variables=CONFIG["data"]["variables"],
            parameters=CONFIG["data"]["parameters"],
            numpoints_x=CONFIG["data"]["numpoints_x"],
            numpoints_y=CONFIG["data"]["numpoints_y"],
            normalize=CONFIG["data"]["normalize"],
            device=CONFIG["device"],
        ),
    )


def extract_parameter_data(dataset, param_index):
    return np.array([item[0][0][param_index].item() for item in dataset])


def extract_variable_data(dataset, var_index):
    return torch.cat([item[1][var_index].flatten() for item in dataset]).cpu().numpy()


def plot_distributions(data, titles, fig_size=(20, 15)):
    fig, axes = plt.subplots(len(data), 2, figsize=fig_size)
    for (original, normalized), (ax_orig, ax_norm), title in zip(data, axes, titles):
        for ax, d, label in [
            (ax_orig, original, "Original"),
            (ax_norm, normalized, "Normalized"),
        ]:
            mean, std = np.mean(d), np.std(d)
            ax.hist(
                d,
                bins=200,
                alpha=0.7,
                label=f"{title} (mean={mean:.3f}, std={std:.3f})",
            )
            ax.legend()
            ax.set_title(f"Distribution of {title} ({label})")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def calculate_statistics(data):
    mean = np.mean(data)
    std = np.std(data)
    return {
        "Mean": mean,
        "Std Dev": std,
        "Min": np.min(data),
        "P1": np.percentile(data, 1),
        "P5": np.percentile(data, 5),
        "P10": np.percentile(data, 10),
        "P25": np.percentile(data, 25),
        "Median": np.percentile(data, 50),
        "P75": np.percentile(data, 75),
        "P90": np.percentile(data, 90),
        "P95": np.percentile(data, 95),
        "P99": np.percentile(data, 99),
        "Max": np.max(data),
        "Skewness": np.mean((data - mean) ** 3) / std**3,
        "Kurtosis": np.mean((data - mean) ** 4) / std**4 - 3,
    }


def main():
    normalized_dataset, dataset = load_datasets(
        "simulation_data_normalized_noise.hdf5", "simulation_data_noise.hdf5"
    )

    parameters = dataset.parameters
    variables = dataset.variables

    # Extract parameter data
    parameter_data = [
        (
            extract_parameter_data(dataset, i),
            extract_parameter_data(normalized_dataset, i),
        )
        for i in range(len(parameters))
    ]
    # Add bottom parameter
    parameter_data.append(
        (
            extract_variable_data(dataset, 0),
            extract_variable_data(normalized_dataset, 0),
        )
    )
    parameters.append("B")
    plot_distributions(parameter_data, parameters)

    stats = [calculate_statistics(data) for data, _ in parameter_data]
    df = pd.DataFrame(stats, index=parameters)
    df["Range"] = df["Max"] - df["Min"]
    print(df)

    # Extract variable data
    variable_data = [
        (
            extract_variable_data(dataset, i),
            extract_variable_data(normalized_dataset, i),
        )
        for i in range(len(variables))
    ]

    plot_distributions(variable_data, variables)

    stats = [calculate_statistics(data) for data, _ in variable_data]
    df = pd.DataFrame(stats, index=variables)
    df["Range"] = df["Max"] - df["Min"]
    print(df)


if __name__ == "__main__":
    main()
