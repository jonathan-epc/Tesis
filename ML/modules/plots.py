from typing import Dict, List, Tuple, Type
import numpy as np
import torch
from pathlib import Path
import math
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import get_config
from cmap import Colormap
from matplotlib.gridspec import GridSpec
import re


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
        base_filename = re.sub(r'[<>:"/\\|?*\0]', 'a', base_filename)

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

def plot_scatter(
    field_outputs,
    field_targets,
    field_names,
    scalar_outputs=None,
    scalar_targets=None,
    scalar_names=None,
    units=None,         # Dictionary mapping variable names to their units
    long_names=None,    # Dictionary mapping variable symbols to full names (can be nested for languages)
    metrics=None,
    cols=3,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",      # 'en' for English, 'es' for Spanish
    detailed=True,      # Toggle between detailed and simple plots
    data_percentile=98, # Percentile of data to include in plot range
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

    # Create subplots for predictions and residuals with a global title.
    for plot_type in ["predictions", "residuals"]:
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        ax = ax.flatten()

        # Set a single, global title for the figure.
        global_title = f"{lang['predictions'] if plot_type=='predictions' else lang['residuals']} {lang['vs']} {lang['targets']}"
        fig.suptitle(global_title, fontsize=16)

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
                
                # Determine the display name, supporting multiple languages if available.
                if long_names:
                    if names[i] in long_names:
                        if isinstance(long_names[names[i]], dict):
                            display_name = f"{long_names[names[i]].get(language, names[i])} {names[i]}"
                        else:
                            display_name = f"{long_names[names[i]]} {names[i]}"
                    else:
                        display_name = names[i]
                else:
                    display_name = names[i]

                # Set only the variable name (with unit) as the subplot title.
                ax[plot_idx].set_title(f"{display_name}{unit_str}", fontsize=10)

                # Set labels
                ax[plot_idx].set_xlabel(f"{lang['targets']}{unit_str}")
                ax[plot_idx].set_ylabel(
                    f"{lang['predictions']}{unit_str}" if plot_type == "predictions"
                    else f"{lang['residuals']}{unit_str}"
                )
                ax[plot_idx].axis("square")

                # Add metrics if available and detailed mode is on
                if detailed and metrics and names[i] in metrics:
                    metric_text = "\n".join(
                        [f"{k.upper()}: {v:.4f}" for k, v in metrics[names[i]].items()]
                    )
                    ax[plot_idx].text(
                        0.05,
                        0.90,
                        metric_text,
                        transform=ax[plot_idx].transAxes,
                        verticalalignment="top",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.5),
                    )

                plot_idx += 1

        # Hide any unused subplots
        for j in range(total_plots, len(ax)):
            fig.delaxes(ax[j])
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
        plot_manager.save_plot(
            fig, plot_type=plot_type, dpi=300, formats=["png", "pdf"]
        )
        plt.show()


def plot_field_comparisons(
    field_outputs,
    field_targets,
    field_names,
    units=None,      # Dictionary mapping variable names to their units
    long_names=None, # Optional: Dictionary mapping variable names to full names (can be nested for languages)
    case_idx=43,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",   # 'en' for English, 'es' for Spanish
    detailed=True,   # Toggle between detailed and simple plots
):
    """
    Creates comparison plots for field variables showing expected, calculated, and difference,
    with added error metrics for the difference plot. The suptitle includes the case number, variable's long name,
    symbol, and unit, while each subplot shows only its individual label.
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
        nrmse = rmse / (target.max() - target.min()) if target.max() != target.min() else np.nan
        smape = np.mean(2 * np.abs(diff) / (np.abs(target) + np.abs(output))) * 100

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5))

        # Get unit string if available
        unit_str = f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        
        # Determine the display name, supporting multiple languages if available.
        if long_names:
            if field_names[i] in long_names:
                if isinstance(long_names[field_names[i]], dict):
                    display_name = long_names[field_names[i]].get(language, field_names[i])
                else:
                    display_name = long_names[field_names[i]]
            else:
                display_name = field_names[i]
        else:
            display_name = field_names[i]

        # Plot expected, calculated, and difference without variable info in each subplot title.
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

        for ax, data, title, cmap, (norm, vmin_local, vmax_local) in plots:
            im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin_local, vmax=vmax_local)
            ax.set_title(title, fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.2, pad=0.4, orientation="horizontal")

        # Add error metrics to the Difference plot if detailed mode is on
        if detailed:
            metric_text = (
                f"MAE   : {mae:.4f}{unit_str}\tMax Error: {max_error:.4f}{unit_str}\n"
                f"RMSE  : {rmse:.4f}{unit_str}\tMAPE     : {mape:.2f}%\n"
                f"NRMSE : {nrmse:.4f}\tSMAPE    : {smape:.2f}%"
            )
            metric_text = metric_text.expandtabs(8)
            ax3.text(
                0.00,
                -0.90,
                metric_text,
                ha="left",
                va="top",
                transform=ax3.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Set a suptitle including the case number, variable long name, symbol, and unit
        fig.suptitle(
            f"{lang['case']} {case_idx} - {display_name} ({field_names[i]}){unit_str}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
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
    custom_cmap=None, # custom_cmap is passed but not used in this specific function's plots
    plot_manager: PlotManager = None,
    language="en",
    detailed=True, # detailed is passed but not directly used to switch on/off parts of these specific plots
):
    """
    Creates individual analysis plots for field variables across the entire dataset,
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
        Custom colormap name (currently not used in this function's plots)
    plot_manager : PlotManager, optional
        Manager for saving plots
    language : str
        Language for plot labels ('en' or 'es')
    detailed : bool
        Whether to include detailed metrics (currently not used to gate plots)
    """
    translations = {
        "en": {
            "spatial_error_dist": "Spatial Error Distribution",
            "error_hist": "Error Histogram",
            "avg_error": "Average Error",
            "percentile_plot_title": "Error Magnitude Percentiles vs. Spatial Position",
            "error_magnitude": "Error Magnitude",
            "frequency": "Frequency",
            "spatial_position_y": "Spatial Position (Y-axis average)", # Clarified for percentile plot
            "statistics_title": "Global Error Metrics",
            "stat_mae": "MAE",
            "stat_rmse": "RMSE",
            "stat_max_error": "Max Error",
            "stat_mape": "MAPE",
            "stat_nrmse": "NRMSE",
            "stat_smape": "SMAPE",
            "stat_spatial_corr": "Spatial Correlation",
        },
        "es": {
            "spatial_error_dist": "Distribución Espacial del Error",
            "error_hist": "Histograma de Error",
            "avg_error": "Error Promedio",
            "percentile_plot_title": "Percentiles de Magnitud de Error vs. Posición Espacial",
            "error_magnitude": "Magnitud del Error",
            "frequency": "Frecuencia",
            "spatial_position_y": "Posición Espacial (promedio eje Y)",
            "statistics_title": "Métricas Globales de Error",
            "stat_mae": "MAE",
            "stat_rmse": "RMSE",
            "stat_max_error": "Error Máximo",
            "stat_mape": "MAPE",
            "stat_nrmse": "NRMSE",
            "stat_smape": "SMAPE",
            "stat_spatial_corr": "Correlación Espacial",
        },
    }
    lang = translations[language]

    for i in range(field_outputs.size(1)):
        # Convert to numpy and compute errors
        targets_np = field_targets[:, i].cpu().numpy()
        outputs_np = field_outputs[:, i].cpu().numpy()
        errors_np = targets_np - outputs_np
        current_field_name = field_names[i]

        # Get unit string if available
        unit_str = (
            f" ({units[current_field_name]})"
            if units and current_field_name in units
            else ""
        )
        title_suffix = f"\n{current_field_name}{unit_str}"

        # --- 1. Spatial error distribution (mean absolute error at each position) ---
        fig_spatial, ax_spatial = plt.subplots(figsize=(8, 7)) # Adjusted size for single plot
        spatial_mae = np.mean(np.abs(errors_np), axis=0)
        im_spatial = ax_spatial.imshow(spatial_mae, cmap="viridis", aspect='equal')
        ax_spatial.set_title(f"{lang['spatial_error_dist']}{title_suffix}")
        # Horizontal colorbar
        cbar_spatial = fig_spatial.colorbar(im_spatial, ax=ax_spatial, orientation='horizontal', pad=0.15, shrink=0.8)
        cbar_spatial.set_label(f"{lang['avg_error']}{unit_str}")
        ax_spatial.set_xlabel("X position")
        ax_spatial.set_ylabel("Y position")
        plt.tight_layout()
        if plot_manager:
            plot_manager.save_plot(
                fig_spatial,
                plot_type=f"{current_field_name}_spatial_error_distribution",
            )
        plt.show()

        # --- 2. Error histogram ---
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        ax_hist.hist(errors_np.flatten(), bins=50, density=True, color='skyblue', edgecolor='black')
        ax_hist.set_title(f"{lang['error_hist']}{title_suffix}")
        ax_hist.set_xlabel(f"{lang['error_magnitude']}{unit_str}")
        ax_hist.set_ylabel(lang['frequency'])
        ax_hist.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        if plot_manager:
            plot_manager.save_plot(
                fig_hist,
                plot_type=f"{current_field_name}_error_histogram",
            )
        plt.show()

        # --- 3. Error percentiles across spatial dimensions (averaging along one spatial axis) ---
        fig_perc, ax_perc = plt.subplots(figsize=(8, 6))
        # Assuming errors_np is (n_cases, height, width)
        # We plot percentiles of error magnitude averaged along the width (axis=2) vs height (y-position)
        # Or averaged along height (axis=1) vs width (x-position). Let's choose Y-axis average for now.
        # If your data has a clear "flow" direction, you might want to adjust this.
        abs_errors_np = np.abs(errors_np)
        percentiles_to_plot = [25, 50, 75, 90, 95, 99]
        
        # Calculate percentiles over cases for each spatial point: shape (num_percentiles, height, width)
        error_percentiles_spatial = np.percentile(abs_errors_np, q=percentiles_to_plot, axis=0)

        for q_idx, q_val in enumerate(percentiles_to_plot):
            # For each percentile, take its spatial map and average along the X-axis (width)
            # This gives a profile along the Y-axis (height)
            profile_along_y = error_percentiles_spatial[q_idx].mean(axis=1)
            ax_perc.plot(profile_along_y, label=f"{q_val}th")
        
        ax_perc.set_title(f"{lang['percentile_plot_title']}{title_suffix}")
        ax_perc.set_xlabel(lang["spatial_position_y"]) # This now refers to the Y-axis index
        ax_perc.set_ylabel(f"{lang['error_magnitude']}{unit_str}")
        ax_perc.legend()
        ax_perc.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if plot_manager:
            plot_manager.save_plot(
                fig_perc,
                plot_type=f"{current_field_name}_error_percentiles",
            )
        plt.show()

        # --- 4. Statistics text ---
        # Calculate comprehensive error metrics
        mae = np.mean(np.abs(errors_np))
        rmse = np.sqrt(np.mean(errors_np**2))
        max_error = np.max(np.abs(errors_np))
        
        # Calculate relative errors - ensure targets_np is not all zeros
        if np.any(targets_np != 0):
            # Mask to avoid division by zero, for elements close to zero in target
            safe_targets_mask = np.abs(targets_np) > 1e-9 # A small epsilon
            
            mape_values = np.abs(errors_np[safe_targets_mask] / targets_np[safe_targets_mask])
            mape = np.mean(mape_values) * 100 if mape_values.size > 0 else np.nan
            
            smape_values = 2 * np.abs(errors_np[safe_targets_mask]) / \
                           (np.abs(targets_np[safe_targets_mask]) + np.abs(outputs_np[safe_targets_mask]))
            smape = np.mean(smape_values) * 100 if smape_values.size > 0 else np.nan
        else:
            mape = np.nan
            smape = np.nan

        target_range = np.max(targets_np) - np.min(targets_np)
        nrmse = rmse / target_range if target_range > 1e-9 else np.nan # Avoid division by zero if range is tiny

        # Calculate spatial correlation per case and average
        correlations = []
        for case_idx in range(targets_np.shape[0]):
            target_flat = targets_np[case_idx].flatten()
            output_flat = outputs_np[case_idx].flatten()
            if np.std(target_flat) > 1e-9 and np.std(output_flat) > 1e-9: # Check for non-constant arrays
                correlations.append(np.corrcoef(target_flat, output_flat)[0, 1])
        spatial_correlation = np.nanmean(correlations) if correlations else np.nan


        stats_text_lines = [
            f"{lang['statistics_title']}:",
            f"{lang['stat_mae']}: {mae:.4f}{unit_str}",
            f"{lang['stat_rmse']}: {rmse:.4f}{unit_str}",
            f"{lang['stat_max_error']}: {max_error:.4f}{unit_str}",
            f"{lang['stat_mape']}: {mape:.2f}%" if not np.isnan(mape) else f"{lang['stat_mape']}: N/A",
            f"{lang['stat_nrmse']}: {nrmse:.4f}" if not np.isnan(nrmse) else f"{lang['stat_nrmse']}: N/A",
            f"{lang['stat_smape']}: {smape:.2f}%" if not np.isnan(smape) else f"{lang['stat_smape']}: N/A",
            f"{lang['stat_spatial_corr']}: {spatial_correlation:.4f}" if not np.isnan(spatial_correlation) else f"{lang['stat_spatial_corr']}: N/A",
        ]
        stats_text = "\n".join(stats_text_lines)

        # Create a figure just for the text
        fig_stats, ax_stats = plt.subplots(figsize=(6, 4)) # Adjust size as needed
        ax_stats.axis("off")
        ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_stats.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
        )
        fig_stats.suptitle(f"{lang['statistics_title']}{title_suffix}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
        if plot_manager:
            plot_manager.save_plot(
                fig_stats,
                plot_type=f"{current_field_name}_error_statistics",
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

def plot_difference():
    return
def plot_difference_im():
    return