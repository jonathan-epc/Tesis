import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from scipy import stats
from config import get_config
from modules.data import HDF5Dataset

def load_dataset(config, file_path):
    """Load the dataset based on the given configuration."""
    return HDF5Dataset(
        file_path=file_path,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        channel_length=config.channel.length,
        channel_width=config.channel.width,
        # normalize_input=config.data.normalize_input,
        # normalize_output=config.data.normalize_output,
        normalize_input=False,
        normalize_output=False,
        device="cpu",
        preload=True,
    )

def extract_data(dataset, config):
    """Extract input and output data from the dataset."""
    input_data = {key: [] for key in config.data.inputs}
    output_data = {key: [] for key in config.data.outputs}

    for data in dataset:
        inputs, outputs = data
        field_inputs, scalar_inputs = inputs
        field_outputs, scalar_outputs = outputs

        for key, value in zip(config.data.inputs, field_inputs + scalar_inputs):
            input_data[key].append(value)
        for key, value in zip(config.data.outputs, field_outputs + scalar_outputs):
            output_data[key].append(value)

    # Convert lists to tensors
    input_data = {key: torch.stack(values) for key, values in input_data.items()}
    output_data = {key: torch.stack(values) for key, values in output_data.items()}
    return input_data, output_data

def compute_statistics(data):
    """Compute statistical measures for the given data."""
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

def check_missing_values(input_data, output_data):
    """Check for missing or NaN values in input and output data."""
    return {
        key: (data.isnan().sum().item(), data.numel())
        for key, data in {**input_data, **output_data}.items()
    }

def create_summary_plot(data_dict, title_prefix, output_file, plot_type="histogram"):
    """Create and save a summary plot for all variables."""
    num_vars = len(data_dict)
    cols = 3
    rows = (num_vars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, (var, data) in enumerate(data_dict.items()):
        numpy_data = data.cpu().numpy().flatten()
        stats = compute_statistics(data)

        ax = axes[i]
        if plot_type == "histogram":
            sns.histplot(numpy_data, bins=50, kde=True, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=numpy_data, orient="h", ax=ax)

        ax.set_title(f"{title_prefix} {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Density")

        # Add statistics to the plot
        stat_text = (f"Mean: {stats['mean']:.2f}\n"
                     f"Std: {stats['std']:.2f}\n"
                     f"Min: {stats['min']:.2f}\n"
                     f"Max: {stats['max']:.2f}\n"
                     f"Skew: {stats['skewness']:.2f}\n"
                     f"Kurt: {stats['kurtosis']:.2f}")
        ax.text(0.95, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))

    # Hide unused subplots
    for ax in axes[num_vars:]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    config = get_config()
    dataset = load_dataset(config, "data/barsa.hdf5")
    print(dataset)

    # Extract data
    input_data, output_data = extract_data(dataset, config)

    # Set plot style
    plt.style.use("default")

    # Compute statistics
    stats_dict = {
        **{f"input_{key}": compute_statistics(data) for key, data in input_data.items()},
        **{f"output_{key}": compute_statistics(data) for key, data in output_data.items()},
    }

    # Check for missing values
    missing_values = check_missing_values(input_data, output_data)
    print("\nMissing Values:")
    for var, (missing, total) in missing_values.items():
        print(f"{var}: {missing}/{total} missing ({100 * missing / total:.2f}%)")

    # Save summary plots
    create_summary_plot(input_data, "Input Parameter:", "plots/exploration/adim_input_summary.png", plot_type="histogram")
    create_summary_plot(output_data, "Output Variable:", "plots/exploration/adim_output_summary.png", plot_type="histogram")

    # Convert statistics to a DataFrame and print
    stats_df = pd.DataFrame(stats_dict).T
    print("\nStatistics Table:")
    print(stats_df)

if __name__ == "__main__":
    main()
