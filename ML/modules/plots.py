from typing import Dict, List, Tuple, Type
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import get_config
from cmap import Colormap

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