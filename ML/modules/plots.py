from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from config import *
from cmap import Colormap

def plot_difference_im(outputs: torch.Tensor, targets: torch.Tensor, step: int, name: str, fold_n: int) -> str:
    """
    Plot and save the difference between outputs and targets for each channel in a single column.
    
    Args:
    outputs (torch.Tensor): Model outputs (batch_size, channels, numpoints_x, numpoints_y)
    targets (torch.Tensor): Ground truth targets (batch_size, channels, numpoints_x, numpoints_y)
    step (int): Current step (for filename)
    name (str): Name of the run (for filename)
    fold_n (int): Fold number (for filename)
    
    Returns:
    str: Path to the saved plot image
    """
    diff = (outputs - targets).abs()
    diff_np = diff.cpu().detach().numpy()
    
    num_channels = diff_np.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(6, 3))
    fig.suptitle(f'Absolute Difference (Output - Target) at Step {step}')
    
    for i in range(num_channels):
        ax = axes[i] if num_channels > 1 else axes
        im = ax.imshow(diff_np[0, i], cmap='viridis', vmin=-1, vmax=1)
        # ax.set_title(f'Channel {i+1}')
        ax.set_ylabel(f'{VARIABLES[i]}')
        # plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    filename = f"plots/{name}_fold_{fold_n}_step_{step}_diff.png"
    plt.savefig(filename)
    plt.close(fig)
    
    return filename

def plot_difference_scatter(outputs: torch.Tensor, targets: torch.Tensor, step: int, name: str, fold_n: int) -> str:
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
        ax.set_title(f'{VARIABLES[i]}')
    # Remove any unused subplots
    for i in range(num_channels, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    plt.tight_layout()
    filename = f"plots/{name}_fold_{fold_n}_step_{step}_scatter.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

def plot_difference_hex(outputs: torch.Tensor, targets: torch.Tensor, name: str, step: int, fold_n: int) -> str:
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
    outputs_np = outputs.cpu().detach().numpy()
    targets_np = targets.cpu().detach().numpy()
    
    num_channels = outputs_np.shape[1]
    cols = int(math.ceil(math.sqrt(num_channels)))
    rows = int(math.ceil(num_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'Hexbin Plot (Output vs Target) at Step {step}')

    cm = Colormap('google:turbo').to_mpl()  # case insensitive

    for i in range(num_channels):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        # hb = ax.hexbin(targets_np[:, i].flatten(), outputs_np[:, i].flatten(), gridsize=50, cmap=cm, extent=(-2,2,-2,2))
        hb = ax.hist2d(targets_np[:, i].flatten(), outputs_np[:, i].flatten(), bins=100, cmap=cm, range=[[-2, 2], [-2, 2]])
        ax.set_aspect('equal')
        ax.set_xlabel('Target')
        ax.set_ylabel('Output')
        ax.set_title(f'{VARIABLES[i]}')

    # Remove any unused subplots
    for i in range(num_channels, rows*cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col] if rows > 1 else axes[col])

    plt.tight_layout()
    filename = f"plots/{name}_fold_{fold_n}_step_{step}_hexbin.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename