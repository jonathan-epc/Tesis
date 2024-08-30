import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cmap import Colormap
from config import get_config
from torch.utils.data import DataLoader, random_split
from torcheval.metrics.functional import r2_score

from modules.data import HDF5Dataset
from modules.models import *
from modules.utils import get_hparams


def load_model(model_path, model_class, config, hparams):
    model = model_class(
        len(config.data.parameters),
        len(config.data.variables),
        config.data.numpoints_x,
        config.data.numpoints_y,
        **hparams,
    ).to(config.device)
    model.load_state_dict(torch.load(model_path))
    return model


def evaluate_model(model, dataloader, config):
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


def calculate_metrics(outputs, targets):
    mse = nn.MSELoss()(outputs, targets).item()
    r2 = r2_score(outputs.flatten(), targets.flatten()).item()
    return mse, r2


def plot_results(outputs, targets, config, save_path):
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    num_channels = outputs_np.shape[1]
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

    for i in range(num_channels):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        ax.hist2d(
            targets_np[:, i].flatten(),
            outputs_np[:, i].flatten(),
            bins=512,
            cmap=cm,
            range=[[ranges[i][1], ranges[i][9]], [ranges[i][1], ranges[i][9]]],
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Target")
        ax.set_ylabel("Output")
        ax.set_title(config.data.variables[i])

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_residuals(outputs, targets, config, save_path):
    outputs_np = outputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    residuals_np = outputs_np - targets_np
    num_channels = outputs_np.shape[1]
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 12*rows))

    # Make sure axes is always iterable (for both 1D and 2D cases)
    if num_channels == 1:
        axes = np.array([axes])

    fig.suptitle("Residuals Distribution", fontsize=16)
    cm = Colormap("google:turbo").to_mpl()

    for i in range(num_channels):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]

        # Compute bin edges for better visualization
        target_min, target_max = np.min(targets_np[:, i]), np.max(targets_np[:, i])
        residual_min, residual_max = (
            np.min(residuals_np[:, i]),
            np.max(residuals_np[:, i]),
        )
        bins = [512, 512]

        # Plot 2D histogram
        c = ax.hist2d(
            targets_np[:, i].flatten(),
            residuals_np[:, i].flatten(),
            bins=bins,
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
    plt.savefig(save_path)  # Save the figure to the specified path
    plt.show()
    plt.close()


def main(config,model_name, model_class, hparams):
    # Load the model
    model = load_model(
        f"savepoints/{model_name}_best_model.pth",
        model_class,
        config,
        hparams,
    )

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
    mse, r2 = calculate_metrics(outputs, targets)
    print(f"MSE: {mse}, R2: {r2}")

    # Plot and save results
    plot_results(outputs, targets, config, "plots/model_results.png")
    plot_residuals(outputs, targets, config, "plots/model_results.png")


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
    main(config, 'FNO_20240829_132709', model_class, hparams)
