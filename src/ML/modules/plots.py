# src/ML/modules/plots.py
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- PlotManager Class (No changes needed here) ---
class PlotManager:
    """Enhanced plot manager for saving figures."""

    def __init__(self, base_dir: str = "plots"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(self, fig, plot_name: str, subdirectory: str | None = None):
        save_dir = self.base_dir
        if subdirectory:
            save_dir = self.base_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        sanitized_name = re.sub(r'[<>:"/\\|?*\0]', "_", plot_name).strip(". ")
        for fmt in ["png", "pdf"]:
            filepath = save_dir / f"{sanitized_name}.{fmt}"
            try:
                fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
            except Exception as e:
                print(f"Failed to save {filepath}: {e}")
        plt.close(fig)


# --- Translation and Naming Dictionaries (No changes needed) ---
UNITS = {
    "H": "m",
    "U": "m/s",
    "V": "m/s",
    "B": "m",
    "H*": "-",
    "U*": "-",
    "V*": "-",
    "H0": "m",
    "Q0": "m³/s",
    "n": r"$s \cdot m^{-1/3}$",
    "nut": r"$m^2/s$",
    "Hr": "-",
    "Fr": "-",
    "M": "-",
    "Re": "-",
    "B*": "-",
    "Ar": "-",
    "Vr": "-",
}
SYMBOLS = {
    "H": "$H$",
    "U": "$U$",
    "V": "$V$",
    "B": "$B$",
    "H*": "$H^*$",
    "U*": "$U^*$",
    "V*": "$V^*$",
    "H0": "$H_0$",
    "Q0": "$Q_0$",
    "n": "$n$",
    "nut": "$\\nu_t$",
    "Hr": "$H_r$",
    "Fr": "$F_r$",
    "M": "$M$",
    "Re": "$Re_t$",
    "B*": "$B^*$",
    "Ar": "$A_r$",
    "Vr": "$V_r$",
}
LONG_NAMES = {
    "H": {"en": "Water Depth", "es": "Profundidad del Agua"},
    "U": {"en": "X-Velocity", "es": "Velocidad en X"},
    "V": {"en": "Y-Velocity", "es": "Velocidad en Y"},
    "B": {"en": "Bed Geometry", "es": "Geometría del Lecho"},
    "H*": {"en": "Dimensionless Depth", "es": "Profundidad Adimensional"},
    "U*": {"en": "Dimensionless X-Velocity", "es": "Velocidad Adim. en X"},
    "V*": {"en": "Dimensionless Y-Velocity", "es": "Velocidad Adim. en Y"},
    "B*": {"en": "Dimensionless Bed", "es": "Lecho Adimensional"},
    "H0": {"en": "Initial Height", "es": "Altura Inicial"},
    "Q0": {"en": "Inflow", "es": "Caudal"},
    "n": {"en": "Manning's n", "es": "n de Manning"},
    "nut": {"en": "Turb. Viscosity", "es": "Visc. Turb."},
    "Hr": {"en": "Height Ratio", "es": "Relación de Altura"},
    "Fr": {"en": "Froude Number", "es": "Número de Froude"},
    "M": {"en": "Manning Param.", "es": "Parám. de Manning"},
    "Re": {"en": "Reynolds Number", "es": "Número de Reynolds"},
    "Ar": {"en": "Aspect Ratio", "es": "Relación de Aspecto"},
    "Vr": {"en": "Velocity Ratio", "es": "Relación de Velocidad"},
}
TRANSLATIONS = {
    "en": {
        "truth": "Ground Truth",
        "pred": "Prediction",
        "diff": "Difference",
        "identity": "Identity",
        "case": "Case",
        "error": "Error",
        "frequency": "Frequency",
        "count": "Point Density",
        "x_axis": "X Position",
        "y_axis": "Y Position",
        "scatter_title": "Prediction vs. Target Scatter Plots",
    },
    "es": {
        "truth": "Valor Real",
        "pred": "Predicción",
        "diff": "Diferencia",
        "identity": "Identidad",
        "case": "Caso",
        "error": "Error",
        "frequency": "Frecuencia",
        "count": "Densidad de Puntos",
        "x_axis": "Posición X",
        "y_axis": "Posición Y",
        "scatter_title": "Gráficos de Dispersión",
    },
}


def _get_name(var, lang):
    return f"{LONG_NAMES.get(var, {}).get(lang, var)} ({SYMBOLS.get(var, var)})"


def _get_unit(var):
    return f" [{UNITS.get(var, '')}]"


# --- plot_scatter_predictions (No changes needed, it already subsamples) ---
def plot_scatter_predictions(
    predictions,
    targets,
    field_names,
    scalar_names,
    plot_manager,
    metrics,
    title_prefix,
    language,
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
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5.5, rows * 4.5), squeeze=False
    )
    fig.suptitle(f"{title_prefix}\n{lang_text['scatter_title']}", fontsize=18)

    for i, ax in enumerate(axes.flat):
        if i >= num_plots:
            ax.axis("off")
            continue
        pred, targ, name = (
            all_preds[i].numpy().flatten(),
            all_targs[i].numpy().flatten(),
            all_names[i],
        )

        if len(pred) > 50000:
            indices = np.random.choice(len(pred), 50000, replace=False)
            pred, targ = pred[indices], targ[indices]

        hb = ax.hexbin(targ, pred, gridsize=50, cmap="turbo", mincnt=1, norm=LogNorm())
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label(lang_text["count"])

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "r--", linewidth=2, label=lang_text["identity"])
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"{_get_name(name, language)}{_get_unit(name)}")
        ax.set_xlabel(lang_text["truth"])
        ax.set_ylabel(lang_text["pred"])

        var_metrics = metrics.get(name, {})
        r2, rmse, mae, smape = (
            var_metrics.get("r2", float("nan")),
            var_metrics.get("rmse", float("nan")),
            var_metrics.get("mae", float("nan")),
            var_metrics.get("smape", float("nan")),
        )
        stats_text = f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nSMAPE = {smape:.2f}%"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_manager.save_plot(fig, "scatter_predictions")


# --- plot_field_comparison (No changes needed) ---
def plot_field_comparison(
    prediction, target, variable_name, plot_manager, case_id, title_prefix, language
):
    lang_text = TRANSLATIONS[language]
    pred_np, targ_np = prediction.numpy(), target.numpy()
    diff_np = targ_np - pred_np

    vmin, vmax = np.percentile(targ_np, 2), np.percentile(targ_np, 98)
    diff_lim = np.percentile(np.abs(diff_np), 98)

    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True, sharey=True)
    full_var_name = variable_name.split(" ")[0]
    fig.suptitle(
        f"{title_prefix}\n{_get_name(full_var_name, language)} ({lang_text['case']} #{case_id})"
    )

    plots_data = [
        (axes[0], targ_np, lang_text["truth"], "turbo", {"vmin": vmin, "vmax": vmax}),
        (axes[1], pred_np, lang_text["pred"], "turbo", {"vmin": vmin, "vmax": vmax}),
        (
            axes[2],
            diff_np,
            lang_text["diff"],
            "seismic",
            {"vmin": -diff_lim, "vmax": diff_lim},
        ),
    ]

    for ax, data, title, cmap, norm_kwargs in plots_data:
        im = ax.imshow(data, aspect="auto", cmap=cmap, **norm_kwargs)
        ax.set_title(title)
        ax.set_ylabel(lang_text["y_axis"])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="25%", pad=0.6)
        fig.colorbar(im, cax=cax, orientation="horizontal")

    axes[-1].set_xlabel(lang_text["x_axis"])
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plot_manager.save_plot(
        fig, f"field_comparison_{variable_name.replace(' ', '_')}_case_{case_id}"
    )


# --- plot_error_analysis (THE CRITICAL CHANGE IS HERE) ---
def plot_error_analysis(
    predictions, targets, field_names, plot_manager, title_prefix, language
):
    lang_text = TRANSLATIONS[language]
    field_preds, _ = predictions
    field_targs, _ = targets
    if field_preds is None:
        return

    for i, name in enumerate(field_names):
        pred_np, targ_np = field_preds[:, i].numpy(), field_targs[:, i].numpy()

        # 1. Error Histogram (Memory-Efficient Subsampling)
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))

        # --- MEMORY FIX 1: Subsample for Histogram ---
        # Instead of flattening the entire error array, which could be huge,
        # we calculate errors on a large but manageable random sample.
        num_total_points = pred_np.size
        sample_size = min(
            num_total_points, 2_000_000
        )  # Cap at 2 million points for histogram
        indices = np.random.choice(num_total_points, sample_size, replace=False)
        errors_sample = targ_np.flatten()[indices] - pred_np.flatten()[indices]

        ax_hist.hist(errors_sample, bins=100, density=True, color=plt.cm.turbo(0.3))
        ax_hist.set_title(
            f"{title_prefix}\n{_get_name(name, language)} - Error Histogram"
        )
        ax_hist.set_xlabel(f"{lang_text['error']}{_get_unit(name)}")
        ax_hist.set_ylabel(lang_text["frequency"])
        ax_hist.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plot_manager.save_plot(fig_hist, f"error_histogram_{name}")

        # 2. Spatial Error Distribution (Memory-Efficient Iterative Calculation)
        fig_spatial, ax_spatial = plt.subplots(figsize=(15, 4))

        # --- MEMORY FIX 2: Iterative Mean Calculation ---
        # Instead of creating a massive `errors_np` array, we initialize a sum array
        # and add the absolute error of each case one by one.
        spatial_mae_sum = np.zeros_like(targ_np[0], dtype=np.float32)
        for j in range(targ_np.shape[0]):
            spatial_mae_sum += np.abs(targ_np[j] - pred_np[j])
        spatial_mae = spatial_mae_sum / targ_np.shape[0]

        im = ax_spatial.imshow(spatial_mae, aspect="auto", cmap="turbo")
        ax_spatial.set_title(
            f"{title_prefix}\n{_get_name(name, language)} - Spatial Mean Absolute Error"
        )

        divider = make_axes_locatable(ax_spatial)
        cax = divider.append_axes("bottom", size="20%", pad=0.7)
        cbar = fig_spatial.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(f"Mean Absolute Error{_get_unit(name)}")

        ax_spatial.set_xlabel(lang_text["x_axis"])
        ax_spatial.set_ylabel(lang_text["y_axis"])
        plt.tight_layout()
        plot_manager.save_plot(fig_spatial, f"spatial_error_{name}")
