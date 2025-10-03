# src/ML/plotting_style.py

import matplotlib.pyplot as plt
from cycler import cycler

# Define a consistent color palette
COLORS = {
    "blue": "#3498db",
    "green": "#2ecc71",
    "red": "#e74c3c",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "grey": "#7f8c8d",
}


def set_plotting_style():
    """
    Applies a consistent, publication-quality style to all matplotlib plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    style_updates = {
        # --- Font ---
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        # --- Figure ---
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        # --- Lines & Markers ---
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        # --- Axes & Ticks ---
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlepad": 15,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "black",
        "ytick.color": "black",
        # --- Legend ---
        "legend.frameon": True,
        "legend.framealpha": 0.8,
        "legend.facecolor": "white",
        "legend.edgecolor": "gray",
        # --- Color Cycle ---
        "axes.prop_cycle": cycler(color=list(COLORS.values())),
    }

    plt.rcParams.update(style_updates)
