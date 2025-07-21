from typing import Dict, List, Tuple, Type, Optional
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
    """Enhanced plot manager with automatic directory creation and improved error handling."""
    
    def __init__(self, base_dir: str = "plots"):
        """
        Initialize the PlotManager.
        
        Args:
            base_dir: Base directory for saving plots (default: "plots")
        """
        self.base_dir = Path(base_dir)
        self.counters: Dict[str, int] = {}
        self._ensure_directory()
    
    def _ensure_directory(self, path: Optional[Path] = None) -> None:
        """
        Create directory if it doesn't exist.
        
        Args:
            path: Directory path to create (defaults to base_dir)
        """
        target_path = path or self.base_dir
        try:
            target_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create directory {target_path}: {e}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename by removing invalid characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove invalid characters for most filesystems
        sanitized = re.sub(r'[<>:"/\\|?*\0]', '_', filename)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        # Ensure filename isn't empty
        if not sanitized:
            sanitized = "plot"
        return sanitized
    
    def create_subdirectory(self, subdir: str) -> Path:
        """
        Create a subdirectory under base_dir.
        
        Args:
            subdir: Subdirectory name
            
        Returns:
            Path to the created subdirectory
        """
        subdir_path = self.base_dir / subdir
        self._ensure_directory(subdir_path)
        return subdir_path
    
    def save_plot(
        self,
        fig,
        plot_type: str,
        dpi: int = 400,
        version: Optional[int] = None,
        formats: Optional[List[str]] = None,
        author: Optional[str] = None,
        additional_metadata: Optional[Dict[str, str]] = None,
        subdirectory: Optional[str] = None,
        include_timestamp: bool = True,
        close_figure: bool = True,
        show_plot: bool = True,
        filename_prefix: str = "",
        filename_suffix: str = "",
    ) -> List[Path]:
        """
        Save a plot with multiple formats, versioning, and metadata.
        
        Args:
            fig: matplotlib figure object
            plot_type: Type/name of the plot
            dpi: Resolution for raster formats
            version: Version number for the plot (auto-incremented if None)
            formats: List of formats to save (default: ['png', 'pdf'])
            author: Author name for metadata
            additional_metadata: Additional metadata to include
            subdirectory: Subdirectory to save plots in
            include_timestamp: Whether to include timestamp in filename
            close_figure: Whether to close the figure after saving
            show_plot: Whether to display the plot
            filename_prefix: Prefix to add to filename
            filename_suffix: Suffix to add to filename (before extension)
            
        Returns:
            List of saved file paths
        """
        # Set defaults
        formats = formats or ["png", "pdf"]
        
        # Validate formats
        valid_formats = {'png', 'pdf', 'svg', 'eps', 'ps', 'jpg', 'jpeg', 'tiff', 'tif'}
        invalid_formats = set(formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid format(s): {invalid_formats}. Valid formats: {valid_formats}")
        
        # Determine save directory
        if subdirectory:
            save_dir = self.create_subdirectory(subdirectory)
        else:
            save_dir = self.base_dir
        
        # Handle versioning
        self.counters[plot_type] = self.counters.get(plot_type, 0) + 1
        if version is None:
            version = self.counters[plot_type]
        
        # Generate filename components
        components = []
        
        if filename_prefix:
            components.append(self._sanitize_filename(filename_prefix))
        
        components.append(self._sanitize_filename(plot_type))
        components.append(f"v{version}")
        
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            components.append(timestamp)
        
        if filename_suffix:
            components.append(self._sanitize_filename(filename_suffix))
        
        base_filename = "_".join(components)
        
        # Prepare format-specific metadata
        timestamp_datetime = datetime.now()
        timestamp_iso = timestamp_datetime.isoformat()
        
        # Save in each format
        saved_files = []
        errors = []
        
        for fmt in formats:
            filepath = save_dir / f"{base_filename}.{fmt}"
            try:
                # Format-specific settings
                save_kwargs = {
                    "dpi": dpi,
                    "bbox_inches": "tight",
                    "pad_inches": 0.1,
                    "facecolor": "white",
                    "format": fmt,
                }
                
                # Add metadata for formats that support it
                if fmt.lower() == 'pdf':
                    # PDF-specific metadata keys (PDF expects datetime objects)
                    pdf_metadata = {
                        "Creator": author or "Unknown",
                        "CreationDate": timestamp_datetime,
                        "Title": f"{plot_type} v{version}",
                        "Subject": plot_type,
                        "Keywords": f"plot,{plot_type},version_{version}"
                    }
                    # Add custom metadata to Keywords field
                    if additional_metadata:
                        custom_keys = ",".join(f"{k}:{v}" for k, v in additional_metadata.items())
                        pdf_metadata["Keywords"] += f",{custom_keys}"
                    save_kwargs["metadata"] = pdf_metadata
                    
                elif fmt.lower() == 'png':
                    # PNG metadata can be more flexible (accepts strings)
                    png_metadata = {
                        "Creator": author or "Unknown",
                        "Date": timestamp_iso,
                        "Version": str(version),
                        "PlotType": plot_type,
                        "DPI": str(dpi),
                    }
                    if additional_metadata:
                        png_metadata.update(additional_metadata)
                    save_kwargs["metadata"] = png_metadata
                
                fig.savefig(filepath, **save_kwargs)
                saved_files.append(filepath)
                print(f"✓ Successfully saved plot as {filepath}")
                
            except Exception as e:
                error_msg = f"Failed to save plot as {filepath}: {e}"
                errors.append(error_msg)
                print(f"✗ {error_msg}")
        
        # Display and cleanup
        if show_plot:
            plt.show()
        
        if close_figure:
            plt.close(fig)
        
        # Report any errors
        if errors:
            print(f"\n⚠️  {len(errors)} error(s) occurred during saving:")
            for error in errors:
                print(f"   - {error}")
        
        return saved_files
    
    def get_plot_count(self, plot_type: str) -> int:
        """
        Get the current count for a plot type.
        
        Args:
            plot_type: Type of plot
            
        Returns:
            Current count for the plot type
        """
        return self.counters.get(plot_type, 0)
    
    def reset_counter(self, plot_type: str) -> None:
        """
        Reset the counter for a specific plot type.
        
        Args:
            plot_type: Type of plot to reset
        """
        self.counters[plot_type] = 0
    
    def list_saved_plots(self, plot_type: Optional[str] = None) -> List[Path]:
        """
        List all saved plots, optionally filtered by type.
        
        Args:
            plot_type: Filter by plot type (optional)
            
        Returns:
            List of saved plot file paths
        """
        if not self.base_dir.exists():
            return []
        
        pattern = f"*{plot_type}*" if plot_type else "*"
        return list(self.base_dir.glob(f"**/{pattern}"))
    
    def __str__(self) -> str:
        """String representation of the PlotManager."""
        return f"PlotManager(base_dir='{self.base_dir}', counters={self.counters})"
    
    def __repr__(self) -> str:
        """Detailed representation of the PlotManager."""
        return self.__str__()


def plot_scatter(
    field_outputs,
    field_targets,
    field_names,
    scalar_outputs=None,
    scalar_targets=None,
    scalar_names=None,
    study_type: str = "N/A",  # ADDED
    bottom_type: str = "N/A", # ADDED
    units=None,
    long_names=None,
    metrics=None,
    cols=3,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",
    detailed=True,
    data_percentile=98,
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
    title_prefix = f"({study_type.upper()} on {bottom_type.capitalize()}) " # ADDED

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
        # UPDATED: Add prefix to global title
        global_title = f"{title_prefix}{lang['predictions'] if plot_type=='predictions' else lang['residuals']} {lang['vs']} {lang['targets']}"
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
    study_type: str = "N/A",
    bottom_type: str = "N/A",
    units=None,
    long_names=None,
    case_idx=43,
    custom_cmap=None,
    plot_manager=None,
    language="en",
    detailed=True,
):
    translations = {
        "en": {"expected": "Expected", "calculated": "Calculated", "difference": "Difference", "case": "Case"},
        "es": {"expected": "Esperado", "calculated": "Calculado", "difference": "Diferencia", "case": "Caso"},
    }
    lang = translations.get(language, translations["en"])
    title_prefix = f"({study_type.upper()} on {bottom_type.capitalize()}) "

    for i in range(field_outputs.size(1)):
        target = field_targets[case_idx, i].cpu().numpy()
        output = field_outputs[case_idx, i].cpu().numpy()

        vmin, vmax = min(target.min(), output.min()), max(target.max(), output.max())
        diff = target - output
        diff_max = max(abs(diff.min()), abs(diff.max()))

        mae = np.mean(np.abs(diff))
        max_error = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        mape = np.mean(np.abs(diff / target)) * 100 if np.all(target != 0) else np.nan
        nrmse = rmse / (target.max() - target.min()) if target.max() != target.min() else np.nan
        smape = np.mean(2 * np.abs(diff) / (np.abs(target) + np.abs(output))) * 100

        unit_str = f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        display_name = (
            long_names.get(field_names[i], {}).get(language, field_names[i])
            if long_names and isinstance(long_names.get(field_names[i]), dict)
            else long_names.get(field_names[i], field_names[i])
            if long_names else field_names[i]
        )

        # Layout using GridSpec
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.3])
        axes = [plt.subplot(gs[i]) for i in range(3)]
        metric_ax = plt.subplot(gs[3])
        metric_ax.axis("off")  # Hide metric subplot axes

        plots = [
            (axes[0], target, lang["expected"], "turbo", (None, vmin, vmax)),
            (axes[1], output, lang["calculated"], "turbo", (None, vmin, vmax)),
            (axes[2], diff, lang["difference"], "seismic", (TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max), None, None)),
        ]

        for ax, data, title, cmap, (norm, vmin_local, vmax_local) in plots:
            im = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin_local, vmax=vmax_local, aspect='auto')
            ax.set_title(title, fontsize=12)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="25%", pad=0.25)
            fig.colorbar(im, cax=cax, orientation="horizontal")

        if detailed:
            metric_text = (
                f"MAE: {mae:.4f}{unit_str}    Max Error: {max_error:.4f}{unit_str}    RMSE: {rmse:.4f}{unit_str}    MAPE: {mape:.2f}%    NRMSE: {nrmse:.4f}    SMAPE: {smape:.2f}%"
            )
            metric_ax.text(0, 0.9, metric_text, ha="left", va="top", fontsize=10,
                           bbox=dict(facecolor="white", alpha=0.7), transform=metric_ax.transAxes)

        fig.suptitle(f"{title_prefix}{lang['case']} {case_idx} - {display_name} ({field_names[i]}){unit_str}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if plot_manager:
            plot_manager.save_plot(
                fig,
                plot_type=f"{field_names[i]}_comparison_case_{case_idx}",
                dpi=300,
                formats=["png", "pdf"],
            )

        plt.show()
        
# NEW FUNCTION: For calculating global statistics
def calculate_field_analysis_metrics(
    field_outputs,
    field_targets,
    field_names,
    units=None,
    language="en"
):
    """
    Calculates comprehensive global error metrics for each field variable.
    Returns a dictionary of metrics with JSON-serializable types.
    """
    translations = { "en": {"statistics_title": "Global Error Metrics", "stat_mae": "MAE", "stat_rmse": "RMSE", "stat_max_error": "Max Error", "stat_mape": "MAPE", "stat_nrmse": "NRMSE", "stat_smape": "SMAPE", "stat_spatial_corr": "Spatial Correlation"}, "es": {"statistics_title": "Métricas Globales de Error", "stat_mae": "MAE", "stat_rmse": "RMSE", "stat_max_error": "Error Máximo", "stat_mape": "MAPE", "stat_nrmse": "NRMSE", "stat_smape": "SMAPE", "stat_spatial_corr": "Correlación Espacial"},}
    lang = translations[language]
    
    all_metrics = {}

    for i in range(field_outputs.size(1)):
        targets_np = field_targets[:, i].cpu().numpy()
        outputs_np = field_outputs[:, i].cpu().numpy()
        errors_np = targets_np - outputs_np
        current_field_name = field_names[i]
        
        # --- MODIFICATION START ---
        # Convert numpy types to native Python types using .item() or float()
        mae = float(np.mean(np.abs(errors_np)))
        rmse = float(np.sqrt(np.mean(errors_np**2)))
        max_error = float(np.max(np.abs(errors_np)))
        
        if np.any(targets_np != 0):
            safe_targets_mask = np.abs(targets_np) > 1e-9
            mape_values = np.abs(errors_np[safe_targets_mask] / targets_np[safe_targets_mask])
            mape = float(np.mean(mape_values) * 100) if mape_values.size > 0 else np.nan
            smape_values = 2 * np.abs(errors_np[safe_targets_mask]) / (np.abs(targets_np[safe_targets_mask]) + np.abs(outputs_np[safe_targets_mask]))
            smape = float(np.mean(smape_values) * 100) if smape_values.size > 0 else np.nan
        else:
            mape, smape = np.nan, np.nan

        target_range = float(np.max(targets_np) - np.min(targets_np))
        nrmse = rmse / target_range if target_range > 1e-9 else np.nan

        correlations = []
        for case_idx in range(targets_np.shape[0]):
            target_flat, output_flat = targets_np[case_idx].flatten(), outputs_np[case_idx].flatten()
            if np.std(target_flat) > 1e-9 and np.std(output_flat) > 1e-9:
                correlations.append(np.corrcoef(target_flat, output_flat)[0, 1])
        
        # Use .item() for single-value numpy arrays, or float() for scalars
        spatial_correlation = float(np.nanmean(correlations)) if correlations else np.nan
        # --- MODIFICATION END ---

        # Store metrics in a dictionary
        field_metrics = {
            lang['stat_mae']: mae,
            lang['stat_rmse']: rmse,
            lang['stat_max_error']: max_error,
            lang['stat_mape']: f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
            lang['stat_nrmse']: nrmse if not np.isnan(nrmse) else "N/A", # Keep nrmse as float or "N/A"
            lang['stat_smape']: f"{smape:.2f}%" if not np.isnan(smape) else "N/A",
            lang['stat_spatial_corr']: spatial_correlation if not np.isnan(spatial_correlation) else "N/A",
        }
        all_metrics[current_field_name] = field_metrics

    return all_metrics

def plot_field_analysis(
    field_outputs,
    field_targets,
    field_names,
    study_type: str = "N/A",
    bottom_type: str = "N/A",
    units=None,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",
    detailed=True,
    epsilon=1e-8 # ADDED: Epsilon for safe division
):
    """
    Creates individual analysis plots for field variables across the entire dataset,
    focusing on error distributions and spatial patterns.
    
    MODIFIED: The "Spatial Error Distribution" now shows the Mean Absolute Percentage Error (MAPE).
    """
    translations = {
        "en": {
            # MODIFIED: Changed title for spatial error
            "spatial_error_dist": "Spatial Mean Absolute Percentage Error (MAPE)", 
            "error_hist": "Error Histogram",
            # MODIFIED: Changed colorbar label
            "avg_error": "Average % Error", 
            "percentile_plot_title": "Error Magnitude Percentile Profiles",
            "y_profile_title": "Y-axis Profile",
            "x_profile_title": "X-axis Profile",
            "error_magnitude": "Error Magnitude",
            "frequency": "Frequency",
            "percentile": "Percentile",
            "y_position": "Y Position",
            "x_position": "X Position",
            "statistics_title": "Global Error Metrics",
            "stat_mae": "MAE", "stat_rmse": "RMSE", "stat_max_error": "Max Error",
            "stat_mape": "MAPE", "stat_nrmse": "NRMSE", "stat_smape": "SMAPE",
            "stat_spatial_corr": "Spatial Correlation",
        },
        "es": {
            # MODIFIED: Changed title for spatial error
            "spatial_error_dist": "Error Porcentual Absoluto Medio Espacial (MAPE)", 
            "error_hist": "Histograma de Error",
            # MODIFIED: Changed colorbar label
            "avg_error": "Error Promedio %", 
            "percentile_plot_title": "Perfiles de Percentiles de Magnitud de Error",
            "y_profile_title": "Perfil del Eje Y", "x_profile_title": "Perfil del Eje X",
            "error_magnitude": "Magnitud del Error", "frequency": "Frecuencia",
            "percentile": "Percentil", "y_position": "Posición Y", "x_position": "Posición X",
            "statistics_title": "Métricas Globales de Error",
            "stat_mae": "MAE", "stat_rmse": "RMSE", "stat_max_error": "Error Máximo",
            "stat_mape": "MAPE", "stat_nrmse": "NRMSE", "stat_smape": "SMAPE",
            "stat_spatial_corr": "Correlación Espacial",
        },
    }
    lang = translations[language]

    # ADDED: Create a title prefix for all plots in this function call
    title_prefix = f"({study_type.upper()} on {bottom_type.capitalize()}) "

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

        # # --- 1. Spatial error distribution (mean absolute error at each position) ---
        # fig_spatial, ax_spatial = plt.subplots(figsize=(8, 7))
        # spatial_mae = np.mean(np.abs(errors_np), axis=0)
        # im_spatial = ax_spatial.imshow(spatial_mae, cmap="viridis", aspect='equal')
        # # UPDATED: Add prefix to title
        # ax_spatial.set_title(f"{title_prefix}{lang['spatial_error_dist']}{title_suffix}")
        # cbar_spatial = fig_spatial.colorbar(im_spatial, ax=ax_spatial, orientation='horizontal', pad=0.15, shrink=0.8)
        # cbar_spatial.set_label(f"{lang['avg_error']}{unit_str}")
        # ax_spatial.set_xlabel("X position")
        # ax_spatial.set_ylabel("Y position")
        # plt.tight_layout()
        # if plot_manager:
        #     plot_manager.save_plot(
        #         fig_spatial,
        #         plot_type=f"{current_field_name}_spatial_error_distribution",
        #     )
        # plt.show()

        # --- 1. Spatial error distribution (mean absolute error at each position) ---
        fig_spatial, ax_spatial = plt.subplots(figsize=(10, 2)) # Adjusted figsize
        spatial_mae = np.mean(np.abs(errors_np), axis=0)
        im_spatial = ax_spatial.imshow(spatial_mae, cmap="viridis", aspect='auto')
        ax_spatial.set_title(f"{title_prefix}{lang['spatial_error_dist']}{title_suffix}")
        ax_spatial.set_xlabel("X position")
        ax_spatial.set_ylabel("Y position")
        
        # --- NEW: Colorbar placement logic ---
        divider = make_axes_locatable(ax_spatial)
        cax = divider.append_axes("bottom", size="25%", pad=0.5) # pad controls space
        cbar_spatial = fig_spatial.colorbar(im_spatial, cax=cax, orientation='horizontal')
        cbar_spatial.set_label(f"{lang['avg_error']}")
        # cbar_spatial.set_label(f"{lang['avg_error']}{unit_str}")
        # --- END NEW ---
        
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
        # UPDATED: Add prefix to title
        ax_hist.set_title(f"{title_prefix}{lang['error_hist']}{title_suffix}")
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

        # --- 3. Error percentile CONTOUR PLOTS (NEW IMPLEMENTATION) ---
        fig_perc, (ax_perc_y, ax_perc_x) = plt.subplots(1, 2, figsize=(16, 6)) # Create two subplots

        abs_errors_np = np.abs(errors_np)
        # Create a fine-grained set of percentiles for a continuous plot
        q_levels = np.arange(5, 100) # Percentiles from 5 to 99

        # Calculate percentiles over cases for each spatial point: shape (num_percentiles, height, width)
        error_q_spatial = np.percentile(abs_errors_np, q=q_levels, axis=0)

        # a) Y-axis Profile (averaging along width/X-axis)
        profile_data_y = error_q_spatial.mean(axis=2) # Shape: (num_percentiles, height)
        Y_y, Q_y = np.meshgrid(np.arange(profile_data_y.shape[1]), q_levels)
        contour_y = ax_perc_y.contourf(Y_y, Q_y, profile_data_y, cmap="turbo", levels=20)
        cbar_y = fig_perc.colorbar(contour_y, ax=ax_perc_y)
        cbar_y.set_label(f"{lang['error_magnitude']}{unit_str}")
        ax_perc_y.set_title(lang['y_profile_title'])
        ax_perc_y.set_xlabel(lang['y_position'])
        ax_perc_y.set_ylabel(lang['percentile'])

        # b) X-axis Profile (averaging along height/Y-axis)
        profile_data_x = error_q_spatial.mean(axis=1) # Shape: (num_percentiles, width)
        X_x, Q_x = np.meshgrid(np.arange(profile_data_x.shape[1]), q_levels)
        contour_x = ax_perc_x.contourf(X_x, Q_x, profile_data_x, cmap="turbo", levels=32)
        cbar_x = fig_perc.colorbar(contour_x, ax=ax_perc_x)
        cbar_x.set_label(f"{lang['error_magnitude']}{unit_str}")
        ax_perc_x.set_title(lang['x_profile_title'])
        ax_perc_x.set_xlabel(lang['x_position'])
        ax_perc_x.set_ylabel(lang['percentile'])
        
        # UPDATED: Add a global suptitle with all info
        fig_perc.suptitle(f"{title_prefix}{lang['percentile_plot_title']}{title_suffix}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust for suptitle
        if plot_manager:
            plot_manager.save_plot(
                fig_perc,
                plot_type=f"{current_field_name}_error_percentile_contours",
            )
        plt.show()


def plot_field_fourier(
    field_outputs,
    field_targets,
    field_names,
    study_type: str = "N/A",  # ADDED
    bottom_type: str = "N/A", # ADDED
    units=None,
    case_idx=43,
    custom_cmap=None,
    plot_manager: PlotManager = None,
    language="en",
    detailed=True,
    data_percentile=98,
):
    """
    Creates comparison plots for Fourier-transformed field variables.
    """
    translations = { "en": {"fourier_comparison": "Fourier Comparison for", "case": "case", "expected_magnitude": "Expected Fourier Magnitude", "calculated_magnitude": "Calculated Fourier Magnitude", "abs_mag_diff": "Absolute Magnitude Difference", "rel_mag_diff": "Relative Magnitude Difference", "phase_diff": "Phase Difference", "max_abs_diff": "Maximum Absolute Difference", "mean_abs_diff": "Mean Absolute Difference", "max_rel_diff": "Maximum Relative Difference", "mean_rel_diff": "Mean Relative Difference", "max_phase_diff": "Maximum Phase Difference", "mean_phase_diff": "Mean Phase Difference"}, "es": {"fourier_comparison": "Comparación de Fourier para", "case": "caso", "expected_magnitude": "Magnitud de Fourier Esperada", "calculated_magnitude": "Magnitud de Fourier Calculada", "abs_mag_diff": "Diferencia Absoluta de Magnitud", "rel_mag_diff": "Diferencia Relativa de Magnitud", "phase_diff": "Diferencia de Fase", "max_abs_diff": "Diferencia Absoluta Máxima", "mean_abs_diff": "Diferencia Absoluta Media", "max_rel_diff": "Diferencia Relativa Máxima", "mean_rel_diff": "Diferencia Relativa Media", "max_phase_diff": "Diferencia de Fase Máxima", "mean_phase_diff": "Diferencia de Fase Media"},}
    lang = translations[language]
    title_prefix = f"({study_type.upper()} on {bottom_type.capitalize()}) " # ADDED

    for i in range(field_outputs.size(1)):
        target, output = field_targets[case_idx, i].cpu().numpy(), field_outputs[case_idx, i].cpu().numpy()
        ft_target, ft_output = np.fft.fftshift(np.fft.fft2(target)), np.fft.fftshift(np.fft.fft2(output))
        mag_target, mag_output = np.abs(ft_target), np.abs(ft_output)
        phase_target, phase_output = np.angle(ft_target), np.angle(ft_output)
        mag_diff_raw = mag_target - mag_output
        mag_target_norm, mag_output_norm = mag_target / mag_target.max(), mag_output / mag_output.max()
        rel_mag_diff, phase_diff = np.abs(mag_diff_raw) / (mag_target + 1e-10), phase_target - phase_output

        metrics = {}
        if detailed:
            metrics = {"max_abs_diff": np.max(np.abs(mag_diff_raw)), "mean_abs_diff": np.mean(np.abs(mag_diff_raw)), "max_rel_diff": np.max(rel_mag_diff), "mean_rel_diff": np.mean(rel_mag_diff), "max_phase_diff": np.max(np.abs(phase_diff)), "mean_phase_diff": np.mean(np.abs(phase_diff))}
        
        unit_str = f" ({units[field_names[i]]})" if units and field_names[i] in units else ""
        
        fig, axes = plt.subplots(5, 1, figsize=(12, 12))
        # UPDATED: Add prefix to suptitle
        fig.suptitle(f"{title_prefix}{lang['fourier_comparison']} {field_names[i]}{unit_str} {lang['case']} {case_idx}", fontsize=16)

        plots = [
            (axes[0], np.log1p(mag_target_norm), lang["expected_magnitude"], "turbo"),
            (axes[1], np.log1p(mag_output_norm), lang["calculated_magnitude"], "turbo"),
            (axes[2], np.log1p(np.abs(mag_diff_raw)), lang["abs_mag_diff"], "seismic"),
            (axes[3], rel_mag_diff, lang["rel_mag_diff"], "seismic"),
            (axes[4], phase_diff, lang["phase_diff"], "twilight_shifted"),
        ]
        for idx, (ax, data, title, cmap) in enumerate(plots):
            vmin, vmax = (np.percentile(data, 100 - data_percentile), np.percentile(data, data_percentile)) if idx != 4 else (-np.pi, np.pi)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(title)
            divider, cax = make_axes_locatable(ax), divider.append_axes("bottom", size="10%", pad=0.5)
            fig.colorbar(im, cax=cax, orientation="horizontal")

            if detailed and idx >= 2:
                metric_keys = {2: ["max_abs_diff", "mean_abs_diff"], 3: ["max_rel_diff", "mean_rel_diff"], 4: ["max_phase_diff", "mean_phase_diff"]}
                if idx in metric_keys:
                    metric_text = "\n".join([f"{lang[k]}: {metrics[k]:.4f}" for k in metric_keys[idx]])
                    ax.text(-0.1, 0.5, metric_text, transform=ax.transAxes, verticalalignment='center', fontsize=8, bbox=dict(facecolor="white", alpha=0.8))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if plot_manager:
            plot_manager.save_plot(fig, plot_type=f"{field_names[i]}_fourier_comparison_case_{case_idx}")
        plt.show()

def plot_difference():
    return
def plot_difference_im():
    return