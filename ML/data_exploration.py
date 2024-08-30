17%5

17//5



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import get_config
from modules.data import HDF5Dataset
from scipy import stats

def load_dataset(config, file_path):
    return HDF5Dataset(
        file_path=file_path,
        variables=config.data.variables,
        parameters=config.data.parameters,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize=(False, False),
        device=config.device,
    )

def extract_parameter_data(dataset, param_index):
    return np.array([item[0][0][param_index].item() for item in dataset])

def extract_variable_data(dataset, var_index):
    return torch.cat([item[1][var_index].flatten() for item in dataset]).cpu().numpy()

def plot_distributions(data, titles, fig_size=(20, 15)):
    fig, axes = plt.subplots(len(data), 1, figsize=fig_size)
    for d, ax, title in zip(data, axes, titles):
        mean, std = np.mean(d), np.std(d)
        ax.hist(
            d,
            bins=200,
            alpha=0.7,
            label=f"{title} (mean={mean:.3f}, std={std:.3f})",
        )
        ax.legend()
        ax.set_title(f"Distribution of {title}")
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
        "Skewness": stats.skew(data),
        "Kurtosis": stats.kurtosis(data),
        "IQR": np.percentile(data, 75) - np.percentile(data, 25),
        "CV": std / mean if mean != 0 else np.nan,
        "Shapiro-Wilk p-value": stats.shapiro(data[:5000])[1]
        if len(data) > 5000
        else stats.shapiro(data)[1],
        "Range": np.ptp(data),
        "n": len(data)
    }

def correlation_analysis(data, labels, max_points=10000):
    if len(data[0]) > max_points:
        indices = np.random.choice(len(data[0]), max_points, replace=False)
        sampled_data = [d[indices] for d in data]
    else:
        sampled_data = data

    corr_matrix = np.corrcoef(sampled_data)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    return corr_matrix

def main():
    config = get_config()
    dataset = load_dataset(config, "data/noise.hdf5")
    parameters = dataset.parameters
    variables = dataset.variables

    # Extract parameter data
    parameter_data = [
        extract_parameter_data(dataset, i) for i in range(len(parameters))
    ]
    parameter_data.append(extract_variable_data(dataset, 0))
    parameters.append("B")

    plot_distributions(parameter_data, parameters)
    stats = [calculate_statistics(data) for data in parameter_data]
    df = pd.DataFrame(stats, index=parameters)
    print("Parameter Statistics:")
    print(df)

    # Correlation analysis for parameters
    print("\nParameter Correlation Matrix:")
    param_corr = correlation_analysis(parameter_data, parameters)

    # Extract variable data
    variable_data = [extract_variable_data(dataset, i) for i in range(len(variables))]
    plot_distributions(variable_data, variables)
    stats = [calculate_statistics(data) for data in variable_data]
    df = pd.DataFrame(stats, index=variables)
    print("\nVariable Statistics:")
    print(df)

    # Correlation analysis for variables
    print("\nVariable Correlation Matrix:")
    var_corr = correlation_analysis(variable_data, variables)

if __name__ == "__main__":
    main()
