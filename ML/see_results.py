import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmap import Colormap
from config import get_config
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import autocast
from torch.utils.data import DataLoader, random_split
from torcheval.metrics.functional import r2_score

from modules.data import HDF5Dataset
from modules.models import *
from modules.utils import denormalize_outputs_and_targets, get_hparams, set_seed


class PlotManager:
    def __init__(self, base_dir="plots"):
        self.base_dir = Path(base_dir)
        self.counters = {}
        self._ensure_directory()

    def _ensure_directory(self):
        """Create the base directory if it doesn't exist."""
        self.base_dir.mkdir(exist_ok=True)

    def save_plot(
        self,
        fig,
        plot_type,
        dpi=400,
        version=None,
        formats=None,
        author=None,
        additional_metadata=None,
    ):
        """
        Save a plot with multiple formats, versioning, and metadata.

        Args:
            fig: matplotlib figure object
            plot_type (str): Type/name of the plot
            dpi (int): Resolution for raster formats
            version (int, optional): Version number for the plot
            formats (list, optional): List of formats to save (default: ['png', 'pdf'])
            author (str, optional): Author name for metadata
            additional_metadata (dict, optional): Additional metadata to include
        """
        # Set defaults
        formats = formats or ["png", "pdf"]
        self.counters[plot_type] = self.counters.get(plot_type, 0) + 1

        # Generate version string
        if version is None:
            version = self.counters[plot_type]
        version_str = f"_v{version}"

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct base filename
        base_filename = f"{plot_type}{version_str}_{timestamp}"

        # Prepare metadata
        metadata = {
            "Creator": author or "Unknown",
            "Date": timestamp,
            "Version": str(version),
            "PlotType": plot_type,
        }
        if additional_metadata:
            metadata.update(additional_metadata)

        # Save in each format
        saved_files = []
        for fmt in formats:
            filepath = self.base_dir / f"{base_filename}.{fmt}"
            try:
                fig.savefig(
                    filepath,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.1,
                    facecolor="white",
                    format=fmt,
                )
                saved_files.append(filepath)
                print(f"✓ Successfully saved plot as {filepath}")
            except Exception as e:
                print(f"✗ Failed to save plot as {filepath}: {e}")
        plt.show()
        plt.close(fig)  # Clean up memory
        return saved_files


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
    mape = (
        torch.mean(torch.abs(diff / target_flat)) * 100
        if torch.all(target_flat != 0)
        else float("nan")
    )

    # Normalized Root Mean Square Error (NRMSE)
    target_range = target_flat.max() - target_flat.min()
    nrmse = rmse / target_range.item() if target_range != 0 else float("nan")

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = (
        torch.mean(
            2 * torch.abs(diff) / (torch.abs(target_flat) + torch.abs(pred_flat))
        )
        * 100
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "nrmse": nrmse,
        "smape": smape,
    }


def evaluate_predictions(
    field_predictions: Optional[torch.Tensor],
    field_targets: Optional[torch.Tensor],
    scalar_predictions: Optional[torch.Tensor],
    scalar_targets: Optional[torch.Tensor],
    config,
    csv_output_path: Optional[str] = None,
    original_indices: Optional[List[int]] = None,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Calculate combined metrics for both field and scalar predictions.
    Args:
        field_predictions: Field predictions [batch_size, num_fields, height, width], or None
        field_targets: Field targets [batch_size, num_fields, height, width], or None
        scalar_predictions: Scalar predictions [batch_size, num_scalars], or None
        scalar_targets: Scalar targets [batch_size, num_scalars], or None
        config: Config object containing data.outputs, data.non_scalars, data.scalars
        csv_output_path: Optional path to save the per-case metrics DataFrame as CSV
        original_indices: Optional list of original indices from the full dataset
    """
    metrics = {}
    per_case_metrics = []
    
    # Use original indices if provided, otherwise use range
    case_indices = original_indices if original_indices is not None else range(
        field_predictions.shape[0] if field_predictions is not None else scalar_predictions.shape[0]
    )
    
    # Get variable names
    field_names = (
        [var for var in config.data.outputs if var in config.data.non_scalars]
        if field_predictions is not None and field_targets is not None
        else []
    )
    scalar_names = (
        [var for var in config.data.outputs if var in config.data.scalars]
        if scalar_predictions is not None and scalar_targets is not None
        else []
    )
    
    # Initialize flattened predictions and targets
    all_pred = []
    all_targ = []
    
    # Process field predictions
    if field_predictions is not None and field_targets is not None:
        field_pred_flat = field_predictions.reshape(field_predictions.shape[0], -1)
        field_targ_flat = field_targets.reshape(field_targets.shape[0], -1)
        all_pred.append(field_pred_flat)
        all_targ.append(field_targ_flat)
        
        for idx, name in enumerate(field_names):
            pred = field_predictions[:, idx]
            targ = field_targets[:, idx]
            metrics[name] = calculate_metrics(pred, targ)
            
            for batch_idx, orig_idx in enumerate(case_indices):
                case_metrics = calculate_metrics(
                    pred[batch_idx].unsqueeze(0),
                    targ[batch_idx].unsqueeze(0)
                )
                per_case_metrics.append({
                    'case': orig_idx,
                    'variable': name,
                    'type': 'field',
                    **{key: value.item() if isinstance(value, torch.Tensor) else value for key, value in case_metrics.items()}
                })
    
    # Process scalar predictions
    if scalar_predictions is not None and scalar_targets is not None:
        scalar_pred_flat = scalar_predictions.reshape(scalar_predictions.shape[0], -1)
        scalar_targ_flat = scalar_targets.reshape(scalar_targets.shape[0], -1)
        all_pred.append(scalar_pred_flat)
        all_targ.append(scalar_targ_flat)
        
        for idx, name in enumerate(scalar_names):
            pred = scalar_predictions[:, idx]
            targ = scalar_targets[:, idx]
            metrics[name] = calculate_metrics(pred, targ)
            
            for batch_idx, orig_idx in enumerate(case_indices):
                case_metrics = calculate_metrics(
                    pred[batch_idx].unsqueeze(0),
                    targ[batch_idx].unsqueeze(0)
                )
                per_case_metrics.append({
                    'case': orig_idx,
                    'variable': name,
                    'type': 'scalar',
                    **{key: value.item() if isinstance(value, torch.Tensor) else value for key, value in case_metrics.items()}
                })
    
    # Calculate overall metrics if there are valid predictions and targets
    if all_pred and all_targ:
        all_pred = torch.cat(all_pred, dim=1)
        all_targ = torch.cat(all_targ, dim=1)
        metrics["overall"] = calculate_metrics(all_pred, all_targ)
        
        for batch_idx, orig_idx in enumerate(case_indices):
            case_metrics = calculate_metrics(
                all_pred[batch_idx].unsqueeze(0),
                all_targ[batch_idx].unsqueeze(0)
            )
            per_case_metrics.append({
                'case': orig_idx,
                'variable': 'overall',
                'type': 'combined',
                **{key: value.item() if isinstance(value, torch.Tensor) else value for key, value in case_metrics.items()}
            })
            
    else:
        metrics["overall"] = {}
    
    # Convert per-case metrics to DataFrame
    per_case_df = pd.DataFrame(per_case_metrics)
    
    # Pivot the DataFrame to have one row per case
    # First, get all metric columns (excluding 'case', 'variable', and 'type')
    metric_columns = [col for col in per_case_df.columns if col not in ['case', 'variable', 'type']]
    
    # Pivot the DataFrame
    pivoted_df = pd.DataFrame({'case': sorted(per_case_df['case'].unique())})
    
    for metric in metric_columns:
        # Create new column names in the format: variable_metric
        pivot_data = per_case_df.pivot(
            index='case',
            columns='variable',
            values=metric
        )
        
        # Rename columns to include the metric name
        pivot_data.columns = [f"{col}_{metric}" for col in pivot_data.columns]
        
        # Join with the main DataFrame
        pivoted_df = pivoted_df.merge(
            pivot_data,
            left_on='case',
            right_index=True,
            how='left'
        )
    
    # Save DataFrame to CSV if path is provided
    if csv_output_path is not None:
        pivoted_df.to_csv(csv_output_path, index=False)
    
    return metrics, pivoted_df


def evaluate_model(model, test_dataloader, config, dataset):
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

            # Forward pass
            outputs = model(inputs)
            outputs, targets = denormalize_outputs_and_targets(
                outputs,
                targets,
                dataset,
                config,
            )
            target_fields, target_scalars = targets
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


def plot_scatter(
    field_outputs,
    field_targets,
    field_names,
    scalar_outputs=None,
    scalar_targets=None,
    scalar_names=None,
    units=None,  # Dictionary mapping variable names to their units
    metrics=None,
    cols=3,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",  # 'en' for English, 'es' for Spanish
    detailed=True,  # Toggle between detailed and simple plots
    data_percentile=98,  # Percentile of data to include in plot range
):
    """
    Creates prediction vs target plots for both field and scalar variables.
    """
    # Language dictionary
    translations = {
        "en": {
            "targets": "Targets",
            "predictions": "Predictions",
            "residuals": "Residuals",
            "vs": "vs",
            "for": "for",
        },
        "es": {
            "targets": "Valores Reales",
            "predictions": "Predicciones",
            "residuals": "Residuos",
            "vs": "vs",
            "for": "para",
        },
    }
    lang = translations[language]

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

                # Calculate plot ranges based on percentile
                min_percentile = 100 - data_percentile
                min_val = max(
                    np.percentile(target_data, min_percentile),
                    np.percentile(output_data, min_percentile),
                )
                max_val = min(
                    np.percentile(target_data, data_percentile),
                    np.percentile(output_data, data_percentile),
                )

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
                    cmap=custom_cmap,
                )

                # Add identity line for predictions plot
                if plot_type == "predictions":
                    ax[plot_idx].plot(
                        [min_val, max_val], [min_val, max_val], "k--", linewidth=1
                    )

                # Get unit string if available
                unit_str = (
                    f" ({units[names[i]]})" if units and names[i] in units else ""
                )

                # Set labels and title
                ax[plot_idx].set_xlabel(f"{lang['targets']}{unit_str}")
                ax[plot_idx].set_ylabel(
                    f"{lang['predictions']}{unit_str}"
                    if plot_type == "predictions"
                    else f"{lang['residuals']}{unit_str}"
                )
                plot_title = (
                    f"{lang['predictions'] if plot_type == 'predictions' else lang['residuals']} "
                    f"{lang['vs']} {lang['targets']} {lang['for']} {names[i]}{unit_str}"
                )
                ax[plot_idx].set_title(plot_title)
                ax[plot_idx].axis("square")

                # Add metrics if available and detailed mode is on
                if detailed and metrics and names[i] in metrics:
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
        plot_manager.save_plot(
            fig, plot_type=plot_type, dpi=300, formats=["png", "pdf"]
        )
        plt.show()


def plot_field_comparisons(
    field_outputs,
    field_targets,
    field_names,
    units=None,  # Dictionary mapping variable names to their units
    case_idx=43,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",  # 'en' for English, 'es' for Spanish
    detailed=True,  # Toggle between detailed and simple plots
):
    """
    Creates comparison plots for field variables showing expected, calculated, and difference,
    with added error metrics for the difference plot.
    """
    # Language dictionary
    translations = {
        "en": {
            "expected": "Expected",
            "calculated": "Calculated",
            "difference": "Difference",
            "case": "Case",
        },
        "es": {
            "expected": "Esperado",
            "calculated": "Calculado",
            "difference": "Diferencia",
            "case": "Caso",
        },
    }
    lang = translations[language]

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
        rmse = np.sqrt(np.mean(diff**2))

        # Normalized error metrics
        mape = np.mean(np.abs(diff / target)) * 100 if np.all(target != 0) else np.nan
        nrmse = (
            rmse / (target.max() - target.min())
            if target.max() != target.min()
            else np.nan
        )
        smape = np.mean(2 * np.abs(diff) / (np.abs(target) + np.abs(output))) * 100

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))

        # Get unit string if available
        unit_str = (
            f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        )

        # Plot expected, calculated, and difference
        plots = [
            (ax1, target, lang["expected"], "turbo", (None, vmin, vmax)),
            (ax2, output, lang["calculated"], "turbo", (None, vmin, vmax)),
            (
                ax3,
                diff,
                lang["difference"],
                "seismic",
                (TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max), None, None),
            ),
        ]

        for ax, data, title, cmap, (norm, vmin, vmax) in plots:
            im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} {field_names[i]}{unit_str}")
            plt.colorbar(im, ax=ax, fraction=0.2, pad=0.4, orientation="horizontal")

        # Add error metrics to the Difference plot if detailed mode is on
        if detailed:
            metric_text = (
                f"MAE: {mae:.4f}{unit_str}\nMax Error: {max_error:.4f}{unit_str}\n"
                f"RMSE: {rmse:.4f}{unit_str}\nMAPE: {mape:.2f}%\n"
                f"NRMSE: {nrmse:.4f}\nSMAPE: {smape:.2f}%"
            )
            ax3.text(
                0.05,
                0.95,
                metric_text,
                ha="left",
                va="top",
                transform=ax3.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        fig.suptitle(f"{lang['case']} {case_idx}")
        plt.tight_layout()
        plot_manager.save_plot(
            fig,
            plot_type=f"{field_names[i]}_comparison_case_{case_idx}",
            dpi=300,
            formats=["png", "pdf"],
        )
        plt.show()


def plot_field_analysis(
    field_outputs,
    field_targets,
    field_names,
    units=None,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",
    detailed=True,
):
    """
    Creates analysis plots for field variables across the entire dataset,
    focusing on error distributions and spatial patterns.

    Parameters:
    -----------
    field_outputs : torch.Tensor
        Predicted field values, shape (n_cases, n_fields, height, width)
    field_targets : torch.Tensor
        Target field values, shape (n_cases, n_fields, height, width)
    field_names : list
        Names of the fields
    units : dict, optional
        Dictionary mapping variable names to their units
    custom_cmap : str, optional
        Custom colormap name
    plot_manager : PlotManager, optional
        Manager for saving plots
    language : str
        Language for plot labels ('en' or 'es')
    detailed : bool
        Whether to include detailed metrics
    """
    translations = {
        "en": {
            "spatial_error": "Spatial Error Distribution",
            "error_hist": "Error Histogram",
            "avg_error": "Average Error",
            "percentile": "Percentile",
            "frequency": "Frequency",
            "spatial_position": "Spatial Position",
            "error_magnitude": "Error Magnitude",
        },
        "es": {
            "spatial_error": "Distribución Espacial del Error",
            "error_hist": "Histograma de Error",
            "avg_error": "Error Promedio",
            "percentile": "Percentil",
            "frequency": "Frecuencia",
            "spatial_position": "Posición Espacial",
            "error_magnitude": "Magnitud del Error",
        },
    }
    lang = translations[language]

    for i in range(field_outputs.size(1)):
        # Convert to numpy and compute errors
        targets = field_targets[:, i].cpu().numpy()
        outputs = field_outputs[:, i].cpu().numpy()
        errors = targets - outputs

        # Get unit string if available
        unit_str = (
            f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        )

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Spatial error distribution (mean absolute error at each position)
        ax1 = fig.add_subplot(gs[0, 0])
        spatial_mae = np.mean(np.abs(errors), axis=0)
        im1 = ax1.imshow(spatial_mae, cmap="viridis")
        ax1.set_title(f"{lang['spatial_error']}\n{field_names[i]}{unit_str}")
        plt.colorbar(im1, ax=ax1)

        # 2. Error histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(errors.flatten(), bins=50, density=True)
        ax2.set_title(f"{lang['error_hist']}\n{field_names[i]}{unit_str}")
        ax2.set_xlabel(lang["error_magnitude"])
        ax2.set_ylabel(lang["frequency"])

        # 3. Error percentiles across spatial dimensions
        ax3 = fig.add_subplot(gs[1, 0])
        error_percentiles = np.percentile(
            np.abs(errors), q=[25, 50, 75, 90, 95, 99], axis=0
        )
        for q, p in zip([25, 50, 75, 90, 95, 99], error_percentiles):
            ax3.plot(p.mean(axis=1), label=f"{q}th percentile")
        ax3.set_title(f"{lang['error_magnitude']} {lang['percentile']}")
        ax3.set_xlabel(lang["spatial_position"])
        ax3.set_ylabel(f"{lang['error_magnitude']}{unit_str}")
        ax3.legend()

        # 4. Statistics text box
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")

        # Calculate comprehensive error metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))

        # Calculate relative errors
        mape = (
            np.mean(np.abs(errors / targets)) * 100 if np.all(targets != 0) else np.nan
        )
        nrmse = (
            rmse / (np.max(targets) - np.min(targets))
            if np.max(targets) != np.min(targets)
            else np.nan
        )
        smape = np.mean(2 * np.abs(errors) / (np.abs(targets) + np.abs(outputs))) * 100

        # Add spatial correlation metrics
        spatial_correlation = np.mean(
            [
                np.corrcoef(t.flatten(), o.flatten())[0, 1]
                for t, o in zip(targets, outputs)
            ]
        )

        stats_text = (
            f"Global Error Metrics:\n"
            f"MAE: {mae:.4f}{unit_str}\n"
            f"RMSE: {rmse:.4f}{unit_str}\n"
            f"Max Error: {max_error:.4f}{unit_str}\n"
            f"MAPE: {mape:.2f}%\n"
            f"NRMSE: {nrmse:.4f}\n"
            f"SMAPE: {smape:.2f}%\n\n"
            f"Spatial Correlation: {spatial_correlation:.4f}"
        )

        ax4.text(
            0.05,
            0.95,
            stats_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if plot_manager:
            plot_manager.save_plot(
                fig,
                plot_type=f"{field_names[i]}_global_analysis",
                dpi=300,
                formats=["png", "pdf"],
            )

        plt.show()


def plot_field_fourier(
    field_outputs,
    field_targets,
    field_names,
    units=None,  # Dictionary mapping variable names to their units
    case_idx=43,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",  # 'en' for English, 'es' for Spanish
    detailed=True,  # Toggle between detailed and simple plots
    data_percentile=98,  # Percentile of data to include in plot range
):
    """
    Creates comparison plots for Fourier-transformed field variables showing expected, calculated,
    phase differences, and magnitude differences (absolute and relative).
    """
    # Language dictionary
    translations = {
        "en": {
            "fourier_comparison": "Fourier Comparison for",
            "case": "case",
            "expected_magnitude": "Expected Fourier Magnitude",
            "calculated_magnitude": "Calculated Fourier Magnitude",
            "abs_mag_diff": "Absolute Magnitude Difference",
            "rel_mag_diff": "Relative Magnitude Difference",
            "phase_diff": "Phase Difference",
            "max_abs_diff": "Maximum Absolute Difference",
            "mean_abs_diff": "Mean Absolute Difference",
            "max_rel_diff": "Maximum Relative Difference",
            "mean_rel_diff": "Mean Relative Difference",
            "max_phase_diff": "Maximum Phase Difference",
            "mean_phase_diff": "Mean Phase Difference",
        },
        "es": {
            "fourier_comparison": "Comparación de Fourier para",
            "case": "caso",
            "expected_magnitude": "Magnitud de Fourier Esperada",
            "calculated_magnitude": "Magnitud de Fourier Calculada",
            "abs_mag_diff": "Diferencia Absoluta de Magnitud",
            "rel_mag_diff": "Diferencia Relativa de Magnitud",
            "phase_diff": "Diferencia de Fase",
            "max_abs_diff": "Diferencia Absoluta Máxima",
            "mean_abs_diff": "Diferencia Absoluta Media",
            "max_rel_diff": "Diferencia Relativa Máxima",
            "mean_rel_diff": "Diferencia Relativa Media",
            "max_phase_diff": "Diferencia de Fase Máxima",
            "mean_phase_diff": "Diferencia de Fase Media",
        },
    }
    lang = translations[language]

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

        # Calculate metrics if detailed mode is on
        if detailed:
            metrics = {
                "max_abs_diff": np.max(np.abs(mag_diff)),
                "mean_abs_diff": np.mean(np.abs(mag_diff)),
                "max_rel_diff": np.max(rel_mag_diff),
                "mean_rel_diff": np.mean(rel_mag_diff),
                "max_phase_diff": np.max(np.abs(phase_diff)),
                "mean_phase_diff": np.mean(np.abs(phase_diff)),
            }

        # Get unit string if available
        unit_str = (
            f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        )

        # Set up plots
        fig, axes = plt.subplots(5, 1, figsize=(25, 5))
        fig.suptitle(
            f"{lang['fourier_comparison']} {field_names[i]}{unit_str} {lang['case']} {case_idx}",
            fontsize=16,
        )

        # Plot Expected and Calculated Fourier Magnitudes
        plots = [
            (axes[0], np.log1p(mag_target), lang["expected_magnitude"], "turbo"),
            (axes[1], np.log1p(mag_output), lang["calculated_magnitude"], "turbo"),
            (axes[2], np.log1p(np.abs(mag_diff)), lang["abs_mag_diff"], "seismic"),
            (axes[3], rel_mag_diff, lang["rel_mag_diff"], "seismic"),
            (axes[4], phase_diff, lang["phase_diff"], "twilight_shifted"),
        ]

        for idx, (ax, data, title, cmap) in enumerate(plots):
            # Calculate plot ranges based on percentile if not phase difference
            if idx != 4:  # Not phase difference
                vmin = np.percentile(data, 100 - data_percentile)
                vmax = np.percentile(data, data_percentile)
            else:  # Phase difference ranges from -pi to pi
                vmin, vmax = -np.pi, np.pi

            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

            # Add metrics if detailed mode is on
            if detailed and idx >= 2:  # Only for difference plots
                metric_keys = {
                    2: ["max_abs_diff", "mean_abs_diff"],
                    3: ["max_rel_diff", "mean_rel_diff"],
                    4: ["max_phase_diff", "mean_phase_diff"],
                }
                if idx in metric_keys:
                    metric_text = "\n".join(
                        [f"{lang[k]}: {metrics[k]:.4f}" for k in metric_keys[idx]]
                    )
                    ax.text(
                        0.02,
                        0.98,
                        metric_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8),
                    )

        plt.tight_layout()
        plot_manager.save_plot(
            fig,
            plot_type=f"{field_names[i]}_fourier_comparison_case_{case_idx}",
            dpi=300,
            formats=["png", "pdf"],
        )
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
    # Get original indices from the test subset
    test_indices = test_dataset.indices
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    return test_dataloader, full_dataset, test_indices


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
    # Define the colors for our custom colormap
    colors = ["white", "#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    n_bins = 256  # Number of color gradations

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_turbo", colors, N=n_bins)
    # Get model dimensions and variable names
    io_counts = get_input_output_counts(config)
    output_names = get_output_names(config)

    study = optuna.create_study(
        study_name=config.optuna.study_name,
        load_if_exists=True,
        storage=config.optuna.storage,
    )

    # Get model name from config or command line args
    model_name = "study4i_FNOnet_trial_48"  # Assuming it's in config, adjust as needed

    hparams = study.trials[48].params

    # Setup model and data
    model = load_model(config, hparams, io_counts, model_name)
    test_dataloader, full_dataset, test_indices = setup_datasets(config, hparams)

    # Evaluate model
    all_field_outputs, all_field_targets, all_scalar_outputs, all_scalar_targets = (
        evaluate_model(model, test_dataloader, config, full_dataset)
    )

    # Calculate metrics and create visualizations
    metrics, per_case_df = evaluate_predictions(
        all_field_outputs,
        all_field_targets,
        all_scalar_outputs,
        all_scalar_targets,
        config,
        csv_output_path="case_metrics.csv",
        original_indices=test_indices  # Pass the indices from random_split
    )

    random_idx = random.randint(0, len(all_field_outputs) - 1)
    # Generate visualization plots
    units = {"U": "m/s", "V": "m/s", "H": "m", "B": "m", "V": "m/s", "H": "m"}

    plot_manager = PlotManager(base_dir=f"plots/{model_name}")
    plot_scatter(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        all_scalar_outputs,
        all_scalar_targets,
        output_names["scalar"],
        units=units,
        metrics=metrics,
        custom_cmap=custom_cmap,
        plot_manager=plot_manager,
        language="es",  # for Spanish
        detailed=True,  # for detailed plots
        data_percentile=99,  # to adjust plot ranges
    )

    plot_field_comparisons(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        case_idx=random_idx,
        custom_cmap=custom_cmap,
        plot_manager=plot_manager,
        units=units,
        language="es",  # for Spanish
        detailed=True,  # for detailed plots
    )
    # plot_field_analysis(
    #     all_field_outputs,
    #     all_field_targets,
    #     output_names["field"],
    #     custom_cmap=custom_cmap,
    #     plot_manager=plot_manager,
    #     units=units,
    #     language="es",  # for Spanish
    #     detailed=True,  # for detailed plots
    # )
    # plot_field_fourier(
    #     all_field_outputs,
    #     all_field_targets,
    #     output_names["field"],
    #     case_idx=random_idx,
    #     custom_cmap=custom_cmap,
    #     plot_manager=plot_manager,
    #     units=units,
    #     language="es",  # for Spanish
    #     detailed=True,  # for detailed plots
    #     data_percentile=95,  # to adjust plot ranges
    # )


if __name__ == "__main__":
    main()
