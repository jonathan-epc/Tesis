import os
import random
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmap import Colormap
from config import get_config
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import autocast
from torch.utils.data import DataLoader, random_split
from torcheval.metrics.functional import r2_score

from modules.data import HDF5Dataset
from modules.models import *
from modules.utils import get_hparams, set_seed


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate regression metrics between prediction and target tensors."""
    # Flatten tensors to handle both 2D and 4D inputs
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    # Basic metrics
    mse = F.mse_loss(pred_flat, target_flat).item()
    mae = F.l1_loss(pred_flat, target_flat).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # R-squared calculation
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    # Additional metrics
    diff = torch.abs(pred_flat - target_flat)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = torch.mean(torch.abs(diff / target_flat)) * 100 if torch.all(target_flat != 0) else float('nan')

    # Normalized Root Mean Square Error (NRMSE)
    target_range = target_flat.max() - target_flat.min()
    nrmse = rmse / target_range.item() if target_range != 0 else float('nan')

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = torch.mean(2 * torch.abs(diff) / (torch.abs(target_flat) + torch.abs(pred_flat))) * 100

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "nrmse": nrmse,
        "smape": smape
    }


def evaluate_predictions(
    field_predictions: torch.Tensor,
    field_targets: torch.Tensor,
    scalar_predictions: torch.Tensor,
    scalar_targets: torch.Tensor,
    config,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate combined metrics for both field and scalar predictions.

    Args:
        field_predictions: Field predictions [batch_size, num_fields, height, width]
        field_targets: Field targets [batch_size, num_fields, height, width]
        scalar_predictions: Scalar predictions [batch_size, num_scalars]
        scalar_targets: Scalar targets [batch_size, num_scalars]
        config: Config object containing data.outputs, data.non_scalars, data.scalars

    Returns:
        Dictionary with overall and per-variable metrics
    """
    # Get variable names
    field_names = [var for var in config.data.outputs if var in config.data.non_scalars]
    scalar_names = [var for var in config.data.outputs if var in config.data.scalars]

    # Calculate overall metrics by concatenating flattened predictions
    field_pred_flat = field_predictions.reshape(field_predictions.shape[0], -1)
    field_targ_flat = field_targets.reshape(field_targets.shape[0], -1)
    scalar_pred_flat = scalar_predictions.reshape(scalar_predictions.shape[0], -1)
    scalar_targ_flat = scalar_targets.reshape(scalar_targets.shape[0], -1)

    all_pred = torch.cat([field_pred_flat, scalar_pred_flat], dim=1)
    all_targ = torch.cat([field_targ_flat, scalar_targ_flat], dim=1)

    metrics = {"overall": calculate_metrics(all_pred, all_targ)}

    # Calculate per-variable metrics
    for idx, name in enumerate(field_names):
        metrics[name] = calculate_metrics(
            field_predictions[:, idx], field_targets[:, idx]
        )

    for idx, name in enumerate(scalar_names):
        metrics[name] = calculate_metrics(
            scalar_predictions[:, idx], scalar_targets[:, idx]
        )

    return metrics


def evaluate_model(model, test_dataloader, config):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        all_field_outputs, all_scalar_outputs = [], []
        all_field_targets, all_scalar_targets = [], []
        all_field_inputs, all_scalar_inputs = [], []

        for batch in test_dataloader:
            inputs, targets = batch
            input_fields, input_scalars = inputs
            target_fields, target_scalars = targets

            # Move inputs and targets to device
            input_fields = [field.to(config.device) for field in input_fields]
            input_scalars = (
                [scalar.to(config.device) for scalar in input_scalars]
                if input_scalars
                else []
            )
            target_fields = [field.to(config.device) for field in target_fields]
            target_scalars = (
                [scalar.to(config.device) for scalar in target_scalars]
                if target_scalars
                else []
            )

            # Reconstruct the tuples
            inputs = (input_fields, input_scalars)
            targets = (target_fields, target_scalars)

            # No autocast context here, to prevent type conversion
            # Forward pass
            outputs = model(inputs)

            # Collect field and scalar outputs and targets
            field_outputs, scalar_outputs = outputs

            if field_outputs is not None:
                all_field_outputs.append(field_outputs)
                if target_fields:
                    all_field_targets.append(torch.stack(target_fields, dim=1))

            # Only collect scalar outputs and targets if they exist
            if scalar_outputs is not None:
                all_scalar_outputs.append(scalar_outputs)
                if target_scalars:
                    all_scalar_targets.append(
                        torch.stack(target_scalars, dim=1)
                    )  # Stack to preserve batch structure

        # After processing all batches, concatenate outputs and targets along batch dimension
        all_field_outputs = (
            torch.cat(all_field_outputs, dim=0) if all_field_outputs else None
        )
        all_field_targets = (
            torch.cat(all_field_targets, dim=0) if all_field_targets else None
        )

        # Only concatenate scalar outputs if they exist
        all_scalar_outputs = (
            torch.cat(all_scalar_outputs, dim=0) if all_scalar_outputs else None
        )
        all_scalar_targets = (
            torch.cat(all_scalar_targets, dim=0) if all_scalar_targets else None
        )

        return (
            all_field_outputs,
            all_field_targets,
            all_scalar_outputs,
            all_scalar_targets,
        )


def create_prediction_plots(
    field_outputs,
    field_targets,
    field_names,
    scalar_outputs=None,
    scalar_targets=None,
    scalar_names=None,
    metrics=None,
    cols=3,
):
    """
    Creates prediction vs target plots for both field and scalar variables.
    """
    # Combine field and scalar data if scalar data exists
    all_outputs = [field_outputs]
    all_targets = [field_targets]
    all_names = [field_names]

    if scalar_outputs is not None and scalar_targets is not None:
        all_outputs.append(scalar_outputs)
        all_targets.append(scalar_targets)
        all_names.append(scalar_names)

    total_plots = sum(outputs.size(1) for outputs in all_outputs)
    rows = int(np.ceil(total_plots / cols))

    # Create subplots for predictions and residuals
    for plot_type in ["predictions", "residuals"]:
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        ax = ax.flatten()

        plot_idx = 0
        for outputs, targets, names in zip(all_outputs, all_targets, all_names):
            for i in range(outputs.size(1)):
                # Get data
                target_data = targets[:, i].cpu().numpy().flatten()
                output_data = outputs[:, i].cpu().numpy().flatten()

                # Calculate plot ranges
                min_val = min(target_data.min(), output_data.min())
                max_val = max(target_data.max(), output_data.max())

                # Prepare y-data based on plot type
                y_data = (
                    output_data
                    if plot_type == "predictions"
                    else (target_data - output_data)
                )

                # Create 2D histogram
                hist = ax[plot_idx].hist2d(
                    x=target_data,
                    y=y_data,
                    bins=(100, 100),
                    range=np.array([(min_val, max_val), (min_val, max_val)]),
                    cmap="turbo",
                )

                # Add identity line
                ax[plot_idx].plot(
                    [min_val, max_val], [min_val, max_val], "k--", linewidth=1
                )

                # Set labels and title
                ax[plot_idx].set_xlabel("Targets")
                ax[plot_idx].set_ylabel(
                    "Predictions" if plot_type == "predictions" else "Residuals"
                )
                plot_title = f"{'Predictions' if plot_type == 'predictions' else 'Residuals'} vs Targets for {names[i]}"
                ax[plot_idx].set_title(plot_title)
                ax[plot_idx].axis("square")

                # Add metrics if available
                if metrics and names[i] in metrics:
                    metric_text = "\n".join(
                        [f"{k.upper()}: {v:.4f}" for k, v in metrics[names[i]].items()]
                    )
                    ax[plot_idx].text(
                        0.05,
                        0.95,
                        metric_text,
                        transform=ax[plot_idx].transAxes,
                        verticalalignment="top",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.5),
                    )

                plot_idx += 1

        # Hide unused subplots
        for j in range(total_plots, len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()
        plt.show()


def plot_field_comparisons(field_outputs, field_targets, field_names, case_idx=43):
    """
    Creates comparison plots for field variables showing expected, calculated, and difference,
    with added error metrics for the difference plot.
    """
    for i in range(field_outputs.size(1)):
        target = field_targets[case_idx, i].cpu().numpy()
        output = field_outputs[case_idx, i].cpu().numpy()

        # Calculate ranges for consistent coloring
        vmin = min(target.min(), output.min())
        vmax = max(target.max(), output.max())
        diff = target - output
        diff_max = max(abs(diff.min()), abs(diff.max()))

        # Calculate error metrics
        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # Normalized error metrics
        mape = np.mean(np.abs(diff / target)) * 100 if np.all(target != 0) else np.nan
        nrmse = rmse / (target.max() - target.min()) if target.max() != target.min() else np.nan
        smape = np.mean(2 * np.abs(diff) / (np.abs(target) + np.abs(output))) * 100

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))

        # Plot expected, calculated, and difference
        plots = [
            (ax1, target, "Expected", "turbo", (None, vmin, vmax)),
            (ax2, output, "Calculated", "turbo", (None, vmin, vmax)),
            (
                ax3,
                diff,
                "Difference",
                "seismic",
                (TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max), None, None),
            ),
        ]

        for ax, data, title, cmap, (norm, vmin, vmax) in plots:
            im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} {field_names[i]}")
            plt.colorbar(im, ax=ax, fraction=0.2, pad=0.4, orientation="horizontal")

        # Add error metrics to the Difference plot
        ax3.text(
            0.05, 0.95,
            f"MAE: {mae:.4f}\nMax Error: {max_error:.4f}\nRMSE: {rmse:.4f}\n"
            f"MAPE: {mape:.2f}%\nNRMSE: {nrmse:.4f}\nSMAPE: {smape:.2f}%",
            ha="left",
            va="top",
            transform=ax3.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        fig.suptitle(f"Case {case_idx}")
        plt.tight_layout()
        plt.show()


def plot_field_fourier(field_outputs, field_targets, field_names, case_idx=43):
    """
    Creates comparison plots for Fourier-transformed field variables showing expected, calculated,
    phase differences, and magnitude differences (absolute and relative).
    """
    for i in range(field_outputs.size(1)):
        # Retrieve fields and compute their Fourier transforms
        target = field_targets[case_idx, i].cpu().numpy()
        output = field_outputs[case_idx, i].cpu().numpy()

        # Compute Fourier transforms
        ft_target = np.fft.fftshift(np.fft.fft2(target))
        ft_output = np.fft.fftshift(np.fft.fft2(output))

        # Compute magnitude and phase
        mag_target = np.abs(ft_target)
        mag_output = np.abs(ft_output)
        phase_target = np.angle(ft_target)
        phase_output = np.angle(ft_output)

        # Normalize magnitudes for better contrast in visualization
        mag_target /= mag_target.max()
        mag_output /= mag_output.max()

        # Compute magnitude differences (absolute and relative)
        mag_diff = mag_target - mag_output
        rel_mag_diff = np.abs(mag_diff) / (
            mag_target + 1e-10
        )  # Prevent division by zero
        phase_diff = phase_target - phase_output

        # Set up plots
        fig, axes = plt.subplots(5, 1, figsize=(25, 5))
        fig.suptitle(
            f"Fourier Comparison for {field_names[i]} for case {case_idx}", fontsize=16
        )

        # Plot Expected and Calculated Fourier Magnitudes
        plots = [
            (axes[0], np.log1p(mag_target), "Expected Fourier Magnitude", "turbo"),
            (axes[1], np.log1p(mag_output), "Calculated Fourier Magnitude", "turbo"),
            (
                axes[2],
                np.log1p(np.abs(mag_diff)),
                "Absolute Magnitude Difference",
                "seismic",
            ),
            (axes[3], rel_mag_diff, "Relative Magnitude Difference", "seismic"),
            (axes[4], phase_diff, "Phase Difference", "twilight_shifted"),
        ]

        for ax, data, title, cmap in plots:
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

        plt.tight_layout()
        plt.show()


def get_input_output_counts(config: Any) -> Dict[str, int]:
    """Calculate input and output dimensions based on configuration."""
    return {
        "field_inputs": len(
            [p for p in config.data.inputs if p in config.data.non_scalars]
        ),
        "scalar_inputs": len(
            [p for p in config.data.inputs if p in config.data.scalars]
        ),
        "field_outputs": len(
            [p for p in config.data.outputs if p in config.data.non_scalars]
        ),
        "scalar_outputs": len(
            [p for p in config.data.outputs if p in config.data.scalars]
        ),
    }


def get_output_names(config: Any) -> Dict[str, List[str]]:
    """Get lists of field and scalar output variable names."""
    return {
        "field": [var for var in config.data.outputs if var in config.data.non_scalars],
        "scalar": [var for var in config.data.outputs if var in config.data.scalars],
    }


def setup_datasets(
    config: Any, hparams: Dict
) -> Tuple[DataLoader, List[str], List[str]]:
    """Initialize and split datasets, create data loader."""
    full_dataset = HDF5Dataset(
        file_path=config.data.file_name,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        channel_length=config.channel.length,
        channel_width=config.channel.width,
        normalize_input=config.data.normalize_input,
        normalize_output=config.data.normalize_output,
        device=config.device,
        preload=False,
    )

    # Split dataset
    test_size = int(config.training.test_frac * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    _, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    return test_dataloader


def load_model(
    config: Any, hparams: Dict, io_counts: Dict[str, int], model_name: str
) -> torch.nn.Module:
    """Initialize and load the model."""
    model_class = getattr(sys.modules[__name__], config.model.class_name)
    model = model_class(
        field_inputs_n=io_counts["field_inputs"],
        scalar_inputs_n=io_counts["scalar_inputs"],
        field_outputs_n=io_counts["field_outputs"],
        scalar_outputs_n=io_counts["scalar_outputs"],
        **hparams,
    ).to(config.device)

    model.load_state_dict(
        torch.load(
            f"savepoints/{model_name}_best_model.pth",
            map_location=config.device,
            weights_only=True,
        )
    )
    return model


def main() -> None:
    """Main function to run the evaluation pipeline."""
    # Initial setup
    config = get_config()
    set_seed(config.seed)

    # Get model dimensions and variable names
    io_counts = get_input_output_counts(config)
    output_names = get_output_names(config)

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        load_if_exists=True,
        storage=config.optuna.storage,
    )

    # Get model name from config or command line args
    model_name = "study16i_FNOnet_trial_17"  # Assuming it's in config, adjust as needed

    hparams = study.trials[17].params

    # Setup model and data
    model = load_model(config, hparams, io_counts, model_name)
    test_dataloader = setup_datasets(config, hparams)

    # Evaluate model
    all_field_outputs, all_field_targets, all_scalar_outputs, all_scalar_targets = (
        evaluate_model(model, test_dataloader, config)
    )

    # Calculate metrics and create visualizations
    metrics = evaluate_predictions(
        all_field_outputs,
        all_field_targets,
        all_scalar_outputs,
        all_scalar_targets,
        config,
    )

    random_idx = random.randint(0, len(all_field_outputs) - 1)
    # Generate visualization plots
    create_prediction_plots(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        all_scalar_outputs,
        all_scalar_targets,
        output_names["scalar"],
        metrics,
    )

    plot_field_comparisons(
        all_field_outputs, all_field_targets, output_names["field"], case_idx=random_idx
    )
    plot_field_fourier(
        all_field_outputs, all_field_targets, output_names["field"], case_idx=random_idx
    )


if __name__ == "__main__":
    main()
