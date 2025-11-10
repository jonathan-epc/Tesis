# src/ML/modules/plots.py
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from common.utils import setup_logger

from .plot_config import LONG_NAMES, SYMBOLS, TRANSLATIONS, UNITS

logger = setup_logger()


class PlotManager:
    """Manages saving figures in multiple formats for publication and preview."""

    def __init__(
        self,
        output_path: str | Path,
        publication_style: bool = True,
        formats: list[str] = None,
        base_font_size: int = 16,
    ):
        """
        Initializes the PlotManager.

        Args:
            output_path (str | Path): The directory where plots will be saved.
            publication_style (bool): If True, applies publication-quality styling.
            formats (list[str]): A list of file formats to save each plot in.
            base_font_size (int): The base font size for scaling text elements.
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        if formats:
            self.formats = formats
        else:
            self.formats = ["pdf", "png"]

        self.font_sizes = {
            "base": base_font_size,
            "small": base_font_size - 2,
            "large": base_font_size + 2,
        }

        if publication_style:
            self.enable_publication_style()

    def enable_publication_style(self, base_style="seaborn-v0_8-whitegrid"):
        """Apply consistent, publication-quality Matplotlib style."""
        plt.style.use(base_style)

        base_size = self.font_sizes["base"]
        small_size = self.font_sizes["small"]
        large_size = self.font_sizes["large"]

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Palatino", "Computer Modern Roman"],
                "font.size": base_size,
                "axes.labelsize": base_size,
                "axes.titlesize": large_size,
                "axes.linewidth": 1.2,
                "axes.grid": True,
                "axes.grid.which": "major",
                "grid.alpha": 0.3,
                "xtick.labelsize": small_size,
                "ytick.labelsize": small_size,
                "legend.fontsize": small_size,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def save_plot(self, fig, filename_stem: str):
        """Saves the figure in all formats specified during initialization."""
        sanitized_stem = re.sub(r'[<>:"/\\|?*\0]', "_", filename_stem).strip(". ")

        for fmt in self.formats:
            filepath = self.output_path / f"{sanitized_stem}.{fmt}"
            try:
                # facecolor = "white" if fmt != "png" else "auto"
                fig.savefig(
                    filepath, dpi=300, bbox_inches="tight", transparent=(fmt == "png")
                )
                logger.info(f"Saved figure: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save {filepath}: {e}")

        plt.close(fig)


def _get_name(var, lang):
    return f"{LONG_NAMES.get(var, {}).get(lang, var)} ({SYMBOLS.get(var, var)})"


def _get_unit(var):
    unit = UNITS.get(var, "")
    return f" [{unit}]" if unit else ""


# ------------------------------------------------------------------------------
#                               SCATTER PLOTS
# ------------------------------------------------------------------------------


def _plot_single_scatter_ax(
    ax,
    pred,
    targ,
    name,
    metrics,
    lang_text,
    language,
    publication,
    panel_label=None,
    plot_manager=None,
):
    """Helper function to draw a single scatter plot on a given axis."""
    if len(pred) > 50000:
        indices = np.random.choice(len(pred), 50000, replace=False)
        pred, targ = pred[indices], targ[indices]

    hb = ax.hexbin(targ, pred, gridsize=50, cmap="viridis", mincnt=1, norm=LogNorm())

    fig = ax.get_figure()
    cbar = fig.colorbar(hb, ax=ax, format="%g")
    cbar.set_label(lang_text["count"])

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label=lang_text["identity"])
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    if not publication:
        ax.set_title(f"{_get_name(name, language)}{_get_unit(name)}")
    elif panel_label:
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=plot_manager.font_sizes["base"],
            fontweight="bold",
            va="top",
            ha="left",
        )

    ax.set_xlabel(lang_text["truth"])
    ax.set_ylabel(lang_text["pred"])

    var_metrics = metrics.get(name, {})
    r2 = var_metrics.get("r2", float("nan"))
    rmse = var_metrics.get("rmse", float("nan"))
    mae = var_metrics.get("mae", float("nan"))
    smape = var_metrics.get("smape", float("nan"))

    stats_text = (
        f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nSMAPE = {smape:.2f}%"
    )

    if publication and panel_label:
        text_x, ha_align = 0.95, "right"
    else:
        text_x, ha_align = 0.05, "left"

    ax.text(
        text_x,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=plot_manager.font_sizes["small"],
        va="top",
        ha=ha_align,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )


def plot_scatter_predictions(
    predictions,
    targets,
    field_names,
    scalar_names,
    plot_manager,
    metrics,
    title_prefix,
    language,
    publication=False,
    separate_plots=False,
):
    lang_text = TRANSLATIONS[language]
    all_preds, all_targs, all_names = [], [], []
    if predictions[0] is not None:
        all_preds.extend(torch.unbind(predictions[0], dim=1))
        all_targs.extend(torch.unbind(targets[0], dim=1))
        all_names.extend(field_names)
    if predictions[1] is not None:
        all_preds.extend(torch.unbind(predictions[1], dim=1))
        all_targs.extend(torch.unbind(targets[1], dim=1))
        all_names.extend(scalar_names)
    if not all_names:
        return

    num_plots = len(all_names)
    # --- Plotting Logic ---
    if separate_plots:
        # MODE 1: Generate one plot per file
        for i, name in enumerate(all_names):
            fig, ax = plt.subplots(figsize=(6, 5))  # Good size for a single panel
            pred, targ = all_preds[i].numpy().flatten(), all_targs[i].numpy().flatten()

            # For separate plots, panel labels don't make sense, so pass None.
            # The title will be shown if publication=False.
            _plot_single_scatter_ax(
                ax,
                pred,
                targ,
                name,
                metrics,
                lang_text,
                language,
                publication,
                panel_label=None,
                plot_manager=plot_manager,
            )
            plt.tight_layout()
            # Save each figure with a unique name
            plot_manager.save_plot(fig, f"scatter_{name}")

    else:
        # MODE 2: Generate one file with subplots (Original behavior)
        num_plots = len(all_names)
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 5.5, rows * 4.5), squeeze=False
        )

        if not publication:
            fig.suptitle(f"{title_prefix}\n{lang_text['scatter_title']}", fontsize=15)

        panel_labels = [f"({chr(65 + i)})" for i in range(num_plots)]

        for i, ax in enumerate(axes.flat):
            if i >= num_plots:
                ax.axis("off")
                continue

            name = all_names[i]
            pred = all_preds[i].numpy().flatten()
            targ = all_targs[i].numpy().flatten()

            _plot_single_scatter_ax(
                ax,
                pred,
                targ,
                name,
                metrics,
                lang_text,
                language,
                publication,
                panel_labels[i],
                plot_manager=plot_manager,
            )

        plt.tight_layout(rect=[0, 0, 1, 0.95] if not publication else [0, 0, 1, 1])
        plot_manager.save_plot(fig, "scatter_composite")


# ------------------------------------------------------------------------------
#                               FIELD COMPARISON
# ------------------------------------------------------------------------------
def plot_field_comparison(
    prediction,
    target,
    variable_name,
    plot_manager,
    case_id,
    title_prefix,
    language,
    publication=False,
    case_type: str = "example",
):
    lang_text = TRANSLATIONS[language]
    pred_np, targ_np = prediction.numpy(), target.numpy()
    epsilon = 1e-9
    diff_np = (targ_np - pred_np) / (targ_np + np.sign(targ_np) * epsilon) * 100

    vmin, vmax = np.percentile(targ_np, [2, 98])
    diff_lim = max(1, np.percentile(np.abs(diff_np), 98))

    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True, sharey=True)
    full_var_name = variable_name.split(" ")[0]

    if not publication:
        fig.suptitle(
            f"{title_prefix}\n{_get_name(full_var_name, language)} ({lang_text['case']} #{case_id})"
        )

    unit_str = _get_unit(full_var_name)
    panel_labels = ["(A)", "(B)", "(C)"]

    plots_data = [
        (
            axes[0],
            targ_np,
            lang_text["truth"],
            "turbo",
            dict(vmin=vmin, vmax=vmax),
            unit_str,
        ),
        (
            axes[1],
            pred_np,
            lang_text["pred"],
            "turbo",
            dict(vmin=vmin, vmax=vmax),
            unit_str,
        ),
        (
            axes[2],
            diff_np,
            lang_text["diff"],
            "coolwarm",
            dict(norm=TwoSlopeNorm(vcenter=0, vmin=-diff_lim, vmax=diff_lim)),
            " [%]",
        ),
    ]

    for ax, data, title, cmap, norm_kwargs, cbar_unit in plots_data:
        im = ax.imshow(data, aspect="auto", cmap=cmap, **norm_kwargs)
        if not publication:
            ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="20%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", format="%.2f")
        cbar.set_label(cbar_unit)
        if publication:
            ax.text(
                0.02,
                0.94,
                panel_labels.pop(0),
                transform=ax.transAxes,
                fontsize=plot_manager.font_sizes["base"],  # Changed from 12
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    filename = f"field_comp_{variable_name}_{case_type}"
    plot_manager.save_plot(fig, filename)


# ------------------------------------------------------------------------------
#                               ERROR ANALYSIS
# ------------------------------------------------------------------------------
def plot_error_analysis(
    predictions,
    targets,
    field_names,
    plot_manager,
    title_prefix,
    language,
    publication=False,
):
    lang_text = TRANSLATIONS[language]
    field_preds, _ = predictions
    field_targs, _ = targets
    if field_preds is None:
        return

    for i, name in enumerate(field_names):
        pred_np, targ_np = field_preds[:, i].numpy(), field_targs[:, i].numpy()

        # --- Error Histogram ---
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        sample_size = min(pred_np.size, 2_000_000)
        indices = np.random.choice(pred_np.size, sample_size, replace=False)
        errors_sample = targ_np.flatten()[indices] - pred_np.flatten()[indices]

        ax_hist.hist(errors_sample, bins=100, density=True, color=plt.cm.turbo(0.6))
        mu, sigma = np.mean(errors_sample), np.std(errors_sample)
        stats_text = f"$\\mu = {mu:.3f}$\n$\\sigma = {sigma:.3f}$"
        ax_hist.text(
            0.05,
            0.95,
            stats_text,
            transform=ax_hist.transAxes,
            fontsize=plot_manager.font_sizes["small"],  # Changed from 10
            va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

        ax_hist.axvline(mu, color="red", linestyle="--", label=f"mean ({mu:.3f})")
        ax_hist.axvline(
            mu + sigma, color="gray", linestyle=":", label=f"±1σ ({sigma:.3f})"
        )
        ax_hist.axvline(mu - sigma, color="gray", linestyle=":")
        ax_hist.legend(frameon=False)

        if not publication:
            ax_hist.set_title(
                f"{title_prefix}\n{_get_name(name, language)} - Error Histogram"
            )
        ax_hist.set_xlabel(f"{lang_text['error']}{_get_unit(name)}")
        ax_hist.set_ylabel(lang_text["frequency"])
        ax_hist.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plot_manager.save_plot(fig_hist, f"error_hist_{name}")

        # --- Spatial Error Distribution ---
        fig_spatial, ax_spatial = plt.subplots(figsize=(15, 4))
        spatial_mae_sum = np.zeros_like(targ_np[0], dtype=np.float32)
        for j in range(targ_np.shape[0]):
            spatial_mae_sum += np.abs(targ_np[j] - pred_np[j])
        spatial_mae = spatial_mae_sum / targ_np.shape[0]

        im = ax_spatial.imshow(spatial_mae, aspect="auto", cmap="turbo")
        if not publication:
            ax_spatial.set_title(
                f"{title_prefix}\n{_get_name(name, language)} - Spatial Mean Absolute Error"
            )

        divider = make_axes_locatable(ax_spatial)
        cax = divider.append_axes("bottom", size="20%", pad=0.7)
        cbar = fig_spatial.colorbar(
            im, cax=cax, orientation="horizontal", format="%.2f"
        )
        cbar.set_label(f"Mean Absolute Error{_get_unit(name)}")

        ax_spatial.set_xlabel(lang_text["x_axis"])
        ax_spatial.set_ylabel(lang_text["y_axis"])
        plt.tight_layout()
        plot_manager.save_plot(fig_spatial, f"error_spatial_{name}")
