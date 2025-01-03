from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import get_config
from cmap import Colormap

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

def plot_im(outputs: torch.Tensor, targets: torch.Tensor, step: int, name: str, fold_n: int) -> str:
    """
    Plot and save the outputs, targets, and their difference side by side for each channel.
    
    Args:
    outputs (torch.Tensor): Model outputs (batch_size, channels, numpoints_x, numpoints_y)
    targets (torch.Tensor): Ground truth targets (batch_size, channels, numpoints_x, numpoints_y)
    step (int): Current step (for filename)
    name (str): Name of the run (for filename)
    fold_n (int): Fold number (for filename)
    
    Returns:
    str: Path to the saved plot image
    """
    config = get_config()
    diff = (outputs - targets).abs()
    outputs_np = outputs.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    diff_np = diff.cpu().detach().numpy()
    
    num_channels = outputs_np.shape[1]
    fig, axes = plt.subplots(num_channels, 3, figsize=(15, num_channels))
    fig.suptitle(f'Output, Target, and Absolute Difference at Step {step}')
    
    for i in range(num_channels):
        ax_output = axes[i, 0] if num_channels > 1 else axes[0]
        ax_target = axes[i, 1] if num_channels > 1 else axes[1]
        ax_diff = axes[i, 2] if num_channels > 1 else axes[2]
        
        im_output = ax_output.imshow(outputs_np[0, i], cmap='viridis', vmin=-1, vmax=1, aspect='auto')
        im_target = ax_target.imshow(targets_np[0, i], cmap='viridis', vmin=-1, vmax=1, aspect='auto')
        im_diff = ax_diff.imshow(diff_np[0, i], cmap='viridis', vmin=0, vmax=2, aspect='auto')
        
        ax_output.set_title(f'Output - {config.data.variables[i]}')
        ax_target.set_title(f'Target - {config.data.variables[i]}')
        ax_diff.set_title(f'Absolute Difference - {config.data.variables[i]}')
        
        # Add colorbars with adjusted size
        divider = make_axes_locatable(ax_output)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_output, cax=cax)
        
        divider = make_axes_locatable(ax_target)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_target, cax=cax)
        
        divider = make_axes_locatable(ax_diff)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_diff, cax=cax)
    
    plt.tight_layout()
    filename = f"plots/{name}_im_step_{step}_fold_{fold_n}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filename

def plot_scatter(outputs: torch.Tensor, targets: torch.Tensor, step: int, name: str, fold_n: int) -> str:
    """
    Plot and save the scatter plot of outputs vs targets for each channel in a grid.

    Args:
    outputs (torch.Tensor): Model outputs (batch_size, channels, numpoints_x, numpoints_y)
    targets (torch.Tensor): Ground truth targets (batch_size, channels, numpoints_x, numpoints_y)
    step (int): Current step (for filename)
    name (str): Name of the run (for filename)
    fold_n (int): Fold number (for filename)

    Returns:
    str: Path to the saved plot image
    """
    config = get_config()
    outputs_np = outputs.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()

    num_channels = outputs_np.shape[1]
    rows = int(math.ceil(math.sqrt(num_channels)))
    cols = int(math.ceil(num_channels / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'Scatter Plot (Output vs Target) at Step {step}')
   
    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.scatter(targets_np[:, i].flatten(), outputs_np[:, i].flatten(), alpha=0.5)
        ax.set_xlabel('Target')
        ax.set_ylabel('Output')
        ax.set_title(f'{config.data.variables[i]}')
    # Remove any unused subplots
    for i in range(num_channels, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    plt.tight_layout()
    filename = f"plots/{name}_scatter_step_{step}_fold_{fold_n}.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

def plot_hist(outputs: torch.Tensor, targets: torch.Tensor, name: str, step: int, fold_n: int) -> str:
    """
    Plot and save the hexbin plot of outputs vs targets for each channel in a grid.

    Args:
    outputs (torch.Tensor): Model outputs (batch_size, channels, numpoints_x, numpoints_y)
    targets (torch.Tensor): Ground truth targets (batch_size, channels, numpoints_x, numpoints_y)
    step (int): Current step (for filename)
    name (str): Name of the run (for filename)
    fold_n (int): Fold number (for filename)

    Returns:
    str: Path to the saved plot image
    """
    config = get_config()
    outputs_np = outputs.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    
    num_channels = outputs_np.shape[1]
    cols = int(math.ceil(math.sqrt(num_channels)))
    rows = int(math.ceil(num_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'Hexbin Plot (Output vs Target) at Step {step}')

    cm = Colormap('google:turbo').to_mpl()  # case insensitive
    ranges=[
        [[0.0,1.0], [0.0,1.0]],
        [[0.0,0.1], [0.0,0.1]],
        [[-0.05,0.05], [-0.05,0.05]],
        [[0.0,0.8], [0.0,0.8]],
        [[0.0,0.4], [0.0,0.4]],
        [[-0.01,0.01], [-0.01,0.01]],
    ]

    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        # hb = ax.hexbin(targets_np[:, i].flatten(), outputs_np[:, i].flatten(), gridsize=50, cmap=cm, extent=(-2,2,-2,2))
        hb = ax.hist2d(targets_np[:, i].flatten(), outputs_np[:, i].flatten(), bins=256, cmap=cm, range=ranges[i])
        ax.set_aspect('equal')
        ax.set_xlabel('Target')
        ax.set_ylabel('Output')
        ax.set_title(f'{config.data.variables[i]}')

    # Remove any unused subplots
    for i in range(num_channels, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    plt.tight_layout()
    filename = f"plots/{name}_hexbin_step_{step}_fold_{fold_n}.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

