import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cmap import Colormap
from config import get_config
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split
from torcheval.metrics.functional import r2_score

from modules.data import HDF5Dataset
from modules.models import *
from modules.utils import get_hparams

def load_model(model_path: str, model_class, config, hparams: dict) -> nn.Module:
    model = model_class(
        len(config.data.parameters),
        len(config.data.variables),
        config.data.numpoints_x,
        config.data.numpoints_y,
        **hparams,
    ).to(config.device)
    model.load_state_dict(
        torch.load(model_path, map_location=config.device, weights_only=True)
    )
    return model

def evaluate_model(
    model: nn.Module, dataloader: DataLoader, config
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [input.to(config.device) for input in inputs]
            targets = targets.to(config.device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)
    return torch.cat(all_outputs), torch.cat(all_targets)

def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> dict:
    mse = mean_squared_error(
        targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten()
    )
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(
        targets.cpu().numpy().flatten(), outputs.cpu().numpy().flatten()
    )
    r2 = r2_score(outputs.flatten(), targets.flatten()).item()
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def plot_results(outputs, targets, config, save_path=None, case_idx=None):
    if case_idx:
        outputs_np = outputs[case_idx].cpu().numpy()
        targets_np = targets[case_idx].cpu().numpy()
    else:
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
    num_channels = len(config.data.variables)
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle("Output vs Target")
    cm = Colormap("google:turbo").to_mpl()
    ranges = [
        [
            0.000000e00,
            0.011718,
            0.061830,
            0.094836,
            0.167014,
            0.265499,
            0.400344,
            0.630483,
            0.917442,
            2.132721,
            60.392731,
        ],  # F
        [
            1.368279e-08,
            0.002357,
            0.014414,
            0.021451,
            0.033815,
            0.048432,
            0.064224,
            0.080132,
            0.090736,
            0.111490,
            0.191505,
        ],  # H
        [
            0.000000e00,
            0.000018,
            0.000762,
            0.001668,
            0.004088,
            0.008512,
            0.016031,
            0.028384,
            0.041222,
            0.065866,
            0.246938,
        ],  # Q
        [
            2.051122e-02,
            0.100002,
            0.126540,
            0.147548,
            0.208702,
            0.350817,
            0.584520,
            0.830296,
            0.966522,
            1.159716,
            1.373890,
        ],  # S
        [
            -8.723943e-01,
            0.001870,
            0.033025,
            0.055383,
            0.105482,
            0.176719,
            0.281774,
            0.447547,
            0.625206,
            1.263860,
            3.263138,
        ],  # U
        [
            -4.792528e-01,
            -0.083140,
            -0.037877,
            -0.023468,
            -0.006728,
            0.000000,
            0.011588,
            0.031165,
            0.047374,
            0.097615,
            0.510785,
        ],  # V
    ]
    num_points = len(outputs_np.flatten())
    n_bins = int(num_points ** (1 / 3))
    for i in range(num_channels):

        targets = targets_np[:, i].flatten()
        outputs = outputs_np[:, i].flatten()
        
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        ax.hist2d(
            targets,
            outputs,
            bins=n_bins,
            cmap=cm,
            range=[[ranges[i][1], ranges[i][9]], [ranges[i][1], ranges[i][9]]],
        )
        ax.set_xlabel("Target")
        ax.set_ylabel("Output")
        ax.set_title(config.data.variables[i])

        # Add identity line
        min_val = ranges[i][1]
        max_val = ranges[i][9]
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=0.5)

        # Calculate and display R2 score
        try:
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            output_tensor = torch.tensor(outputs, dtype=torch.float32)
            r2 = r2_score(output_tensor, target_tensor)
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                    verticalalignment='top', fontsize=10, color='red')
        except Exception as e:
            print(f"Error calculating R² score: {e}")

        # Set the aspect ratio to be equal
        ax.set_aspect("equal", adjustable="box")

        # Set the x and y limits to be the same
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + ".png")  # Save the figure to the specified path
    plt.show()
    plt.close()

def plot_residuals(outputs, targets, config, save_path=None, case_idx=None):
    if case_idx:
        outputs_np = outputs[case_idx].cpu().numpy()
        targets_np = targets[case_idx].cpu().numpy()
    else:
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
    residuals_np = outputs_np - targets_np
    num_channels = len(config.data.variables)
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    # Make sure axes is always iterable (for both 1D and 2D cases)
    if num_channels == 1:
        axes = np.array([axes])

    fig.suptitle("Residuals Distribution", fontsize=16)
    cm = Colormap("google:turbo").to_mpl()

    num_points = len(outputs_np.flatten())
    n_bins = int(num_points ** (1 / 3))

    for i in range(num_channels):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]

        # Compute bin edges for better visualization
        target_min, target_max = np.min(targets_np[:, i]), np.max(targets_np[:, i])
        residual_min, residual_max = (
            np.min(residuals_np[:, i]),
            np.max(residuals_np[:, i]),
        )

        # Plot 2D histogram
        c = ax.hist2d(
            targets_np[:, i].flatten(),
            residuals_np[:, i].flatten(),
            bins=n_bins,
            cmap=cm,
            range=[[target_min, target_max], [residual_min, residual_max]],
        )
        ax.set_xlabel("Target")
        ax.set_ylabel("Residual")
        ax.set_title(f"Channel: {config.data.variables[i]}", fontsize=12)

    # Remove empty subplots
    if num_channels < rows * cols:
        for j in range(num_channels, rows * cols):
            fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])  # Adjust layout to fit suptitle
    if save_path:
        plt.savefig(save_path + ".png")  # Save the figure to the specified path
    plt.show()
    plt.close()

def plot_2d_field(data, title, cmap, vmin=None, vmax=None, norm=None):
    """Helper function to plot 2D fields with proper colormap scaling."""
    if norm is not None:
        im = plt.imshow(data, cmap=cmap, norm=norm)
    else:
        im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, orientation="horizontal")
    plt.title(title)
    plt.axis("off")
    return im

def plot_field_comparison(outputs, targets, config, save_path=None, case_idx=None):
    """Plot expected, calculated, and difference 2D fields for each variable."""
    output_case = outputs[case_idx].cpu().numpy()
    target_case = targets[case_idx].cpu().numpy()
    num_channels = output_case.shape[0]
    cmap = Colormap(
        "google:turbo"
    ).to_mpl()  # Keep your original colormap for the first two plots
    diff_cmap = plt.get_cmap(
        "seismic"
    )  # Use a diverging colormap for the difference plot

    for i in range(num_channels):
        variable_name = config.data.variables[i]
        variable_unit = config.data.variable_units[i]

        # Determine overall min and max for consistent colorbar
        vmin = min(np.min(target_case[i]), np.min(output_case[i]))
        vmax = max(np.max(target_case[i]), np.max(output_case[i]))

        plt.figure(figsize=(15, 5))

        # Plot expected field
        plt.subplot(3, 1, 1)
        im1 = plot_2d_field(
            target_case[i],
            f"Expected {variable_name} / {variable_unit}",
            cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Plot calculated field
        plt.subplot(3, 1, 2)
        im2 = plot_2d_field(
            output_case[i],
            f"Calculated {variable_name} / {variable_unit}",
            cmap,
            vmin=vmin,
            vmax=vmax,
        )

        # Plot difference
        plt.subplot(3, 1, 3)
        diff = target_case[i] - output_case[i]
        max_diff = np.abs(
            diff
        ).max()  # Get the max absolute difference for symmetric limits
        norm = TwoSlopeNorm(
            vmin=-max_diff, vcenter=0, vmax=max_diff
        )  # Normalize colormap symmetrically around 0
        im3 = plot_2d_field(
            diff, f"Difference {variable_name} / {variable_unit}", diff_cmap, norm=norm
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + variable_name + ".png")
        plt.show()

def select_random_case(test_dataset) -> int:
    num_cases = len(test_dataset)
    print(f"Number of test cases: {num_cases}")
    case_idx = random.randint(0, num_cases - 1)
    print(f"Randomly selected case index: {case_idx}")
    return case_idx

def create_output_directories(directories: List[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main(config, model_name: str, model_class, hparams: dict, case_idx: int = None):
    # Create output directories
    create_output_directories(["savepoints", "plots"])

    # Load the model
    model_path = f"savepoints/{model_name}_best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, model_class, config, hparams)

    full_dataset = HDF5Dataset(
        file_path=config.data.file_name,
        variables=config.data.variables,
        parameters=config.data.parameters,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize=config.data.normalize,
        device=config.device,
    )

    test_size = int(config.training.test_frac * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    # Evaluate the model
    outputs, targets = evaluate_model(model, test_dataloader, config)

    # Calculate metrics
    metrics = calculate_metrics(outputs, targets)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # If no specific case_idx is provided, select one randomly
    if case_idx is None:
        case_idx = select_random_case(test_dataset)

    # Get the original case index from the full dataset
    original_case_idx = test_dataset.indices[case_idx]

    # Print the key of the selected case
    print(
        f"Selected case index: {case_idx}, corresponding key: {full_dataset.keys[original_case_idx]}"
    )

    plot_results(outputs, targets, config, "plots/model_results")
    plot_residuals(outputs, targets, config, "plots/model_residuals")

    # Plot fields comparison (expected, calculated, difference)
    plot_field_comparison(outputs, targets, config, "plots/model_difference", case_idx)

    # Plot correlation with R² and identity line
    # Plot and save results
    plot_results(outputs, targets, config, "plots/model_results_single", case_idx)
    plot_residuals(outputs, targets, config, "plots/model_residuals_single", case_idx)

if __name__ == "__main__":
    config = get_config()
    hparams = {
        "n_layers": config.model.n_layers,
        "n_modes_x": config.model.n_modes_x,
        "n_modes_y": config.model.n_modes_y,
        "hidden_channels": config.model.hidden_channels,
        "lifting_channels": config.model.lifting_channels,
        "projection_channels": config.model.projection_channels,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "accumulation_steps": config.training.accumulation_steps,
    }
    model_class = globals()[config.model.class_name]
    main(config, "study2_FNOwRnet_trial_12", model_class, hparams)
