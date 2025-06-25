#utils.py
import os
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import yaml
from loguru import logger
from torcheval.metrics.functional import mean_squared_error, r2_score


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, save_path="savepoints/best_model.pth"
    ):
        """
        Args:
            patience (int): How long to wait after last improvement.
            verbose (bool): Whether to print messages.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model across all folds.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.reset()  # Initialize/reset per fold
        self.global_best_loss = np.Inf  # Track the best loss across all folds

    def reset(self):
        """Resets parameters for a new fold."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.epoch = 0

    def __call__(self, val_loss, model, epoch):
        """
        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model to save if improvement is seen.
            epoch (int): The current epoch number.
        """
        score = -val_loss
        self.epoch = epoch

        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_loss, model)  # Save best model for this fold
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Saves the best model across all folds if the current one is the best."""
        if val_loss < self.global_best_loss:
            self.global_best_loss = val_loss
            if self.verbose:
                logger.info(
                    f"New global best model saved at epoch {self.epoch+1} with loss {val_loss}"
                )
            torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss

def seed_worker(worker_id):
    """Seeds DataLoader workers for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    print(f"Using {seed} as seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def setup_experiment(config: dict) -> None:
    """Set up the experiment environment."""
    os.environ["WANDB_SILENT"] = "true"


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_hparams(config: dict) -> dict:
    """Extract hyperparameters from the configuration."""
    return {
        "n_layers": config.model.n_layers,
        "n_modes_x": config.model.n_modes_x,
        "n_modes_y": config.model.n_modes_y,
        "hidden_channels": config.model.hidden_channels,
        "lifting_channels": config.model.lifting_channels,
        "projection_channels": config.model.projection_channels,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "accumulation_steps": config.training.accumulation_steps,
    }


def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":  # IPython terminal
            return False
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter


def setup_logger():
    logger.remove()
    logger.add(
        "logs/file_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {function} | {line} | {message}",
        level="INFO",
        rotation="00:00",  # Rotate the log file at midnight
        retention= 1,  # Keep only the most recent log file
    )
    # logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
    return logger


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    variable_names: List[str],
) -> Dict[str, torch.Tensor]:
    metrics = {}
    epsilon = 1e-10  # Small value to avoid division by zero

    # Determine if we're dealing with scalar or field data
    is_scalar = len(outputs.shape) == 2  # Scalars are [batch_size, n_vars]

    # Get the number of variables in the current tensors
    batch_size = outputs.shape[0]
    n_vars = outputs.shape[1]

    # Check if there are enough variable names
    assert len(variable_names) >= n_vars, "Insufficient variable names provided."

    # Select the relevant variable names for this batch
    current_var_names = variable_names[:n_vars]

    # Reshape tensors based on scalar or field data
    if is_scalar:
        outputs_reshaped = outputs  # Already in shape [batch_size, n_vars]
        targets_reshaped = targets
    else:
        # Reshape field data to [batch_size, n_vars, height * width]
        outputs_reshaped = outputs.view(batch_size, n_vars, -1)
        targets_reshaped = targets.view(batch_size, n_vars, -1)

    # Overall metrics across all variables
    mse = torch.mean((targets_reshaped - outputs_reshaped) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(targets_reshaped - outputs_reshaped))

    # Overall R2 score
    targets_mean = torch.mean(targets_reshaped, dim=0, keepdim=True)  # Mean across batch
    ss_tot = torch.sum((targets_reshaped - targets_mean) ** 2)
    ss_res = torch.sum((targets_reshaped - outputs_reshaped) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    # Add overall metrics without prefix
    metrics.update({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    })

    # Compute per-variable metrics
    for i, var_name in enumerate(current_var_names):
        var_outputs = outputs_reshaped[:, i]  # [batch_size, height * width] or [batch_size]
        var_targets = targets_reshaped[:, i]

        # Flatten spatial dimensions for field data if not scalar
        if not is_scalar:
            var_outputs = var_outputs.view(batch_size, -1)  # [batch_size, height * width]
            var_targets = var_targets.view(batch_size, -1)

        # Per-variable metrics
        var_mse = torch.mean((var_targets - var_outputs) ** 2)
        var_rmse = torch.sqrt(var_mse)
        var_mae = torch.mean(torch.abs(var_targets - var_outputs))

        # Per-variable R2 score
        var_targets_mean = torch.mean(var_targets, dim=0, keepdim=True)
        var_ss_tot = torch.sum((var_targets - var_targets_mean) ** 2)
        var_ss_res = torch.sum((var_targets - var_outputs) ** 2)
        var_r2 = 1 - (var_ss_res / (var_ss_tot + epsilon))

        # MAPE calculation, ignoring values near zero
        mask = var_targets.abs() > epsilon  # Mask for safe division
        var_mape = torch.mean(torch.abs((var_targets[mask] - var_outputs[mask]) / var_targets[mask]))

        metrics.update({
            f"{var_name}_mse": var_mse,
            f"{var_name}_rmse": var_rmse,
            f"{var_name}_mae": var_mae,
            f"{var_name}_r2": var_r2,
            f"{var_name}_mape": var_mape,
        })

    return metrics

def denormalize_outputs_and_targets(
    outputs: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    targets: Tuple[List[torch.Tensor], List[torch.Tensor]],
    dataset,
    config,
    normalize_output_setting: bool
) -> Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], 
           Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """
    Denormalize model outputs and targets.

    Args:
        outputs: Tuple of field and scalar predictions
        targets: Tuple of field and scalar targets
        dataset: Dataset object with _denormalize method
        output_vars: List of output variable names
        normalize_output: Whether output normalization is enabled

    Returns:
        Tuple of denormalized outputs and targets
    """
    output_vars = config.data.outputs

    normalize_output = normalize_output_setting 
    
    # If normalization is not enabled, return original data
    if not normalize_output:
        return outputs, targets

    # --- Denormalization Logic (Only runs if normalize_output_setting is True) ---
    field_outputs, scalar_outputs = outputs
    field_targets, scalar_targets = targets # These are lists

    # Get variable names (already done in original code, keeping it)
    scalar_vars = [var for var in output_vars if var in config.data.scalars]
    non_scalar_vars = [var for var in output_vars if var in config.data.non_scalars]

    # Denormalize field outputs (if they exist)
    field_outputs_denorm = None
    if field_outputs is not None:
         # Ensure field_outputs is B x C x H x W before unbinding
        if field_outputs.dim() == 4 and len(non_scalar_vars) == field_outputs.shape[1]:
            field_outputs_denorm = torch.stack([
                dataset._denormalize(p, var)
                # unbind splits along dim 1 (channel/variable dim)
                for p, var in zip(torch.unbind(field_outputs, dim=1), non_scalar_vars)
            ], dim=1) # Stack back along dim 1
        else:
            # Handle potential shape mismatch or log a warning
            print(f"Warning: Shape mismatch or unexpected field_outputs shape {field_outputs.shape} in denormalize. Expected Bx{len(non_scalar_vars)}xHxW.")
            field_outputs_denorm = field_outputs # Return original as fallback

    # Denormalize field targets (if they exist)
    field_targets_denorm = [] # Initialize as list
    if field_targets: # Check if list is not empty
         # Ensure list length matches non_scalar_vars
         if len(field_targets) == len(non_scalar_vars):
             field_targets_denorm = [
                 dataset._denormalize(t, var)
                 for t, var in zip(field_targets, non_scalar_vars)
             ]
         else:
             # Handle potential mismatch
             print(f"Warning: Mismatch between len(field_targets)={len(field_targets)} and len(non_scalar_vars)={len(non_scalar_vars)}.")
             field_targets_denorm = field_targets # Fallback

    # Denormalize scalar outputs (if they exist)
    scalar_outputs_denorm = None
    if scalar_outputs is not None:
         # Ensure scalar_outputs is B x C before unbinding
        if scalar_outputs.dim() == 2 and len(scalar_vars) == scalar_outputs.shape[1]:
            scalar_outputs_denorm = torch.stack([
                dataset._denormalize(p, var)
                # unbind splits along dim 1 (variable dim for scalars)
                for p, var in zip(torch.unbind(scalar_outputs, dim=1), scalar_vars)
            ], dim=1) # Stack back along dim 1
        else:
            # Handle potential shape mismatch
            print(f"Warning: Shape mismatch or unexpected scalar_outputs shape {scalar_outputs.shape} in denormalize. Expected Bx{len(scalar_vars)}.")
            scalar_outputs_denorm = scalar_outputs # Fallback

    # Denormalize scalar targets (if they exist)
    scalar_targets_denorm = [] # Initialize as list
    if scalar_targets: # Check if list is not empty
         # Ensure list length matches scalar_vars
         if len(scalar_targets) == len(scalar_vars):
             scalar_targets_denorm = [
                 dataset._denormalize(t, var)
                 for t, var in zip(scalar_targets, scalar_vars)
             ]
         else:
             # Handle potential mismatch
             print(f"Warning: Mismatch between len(scalar_targets)={len(scalar_targets)} and len(scalar_vars)={len(scalar_vars)}.")
             scalar_targets_denorm = scalar_targets # Fallback

    # Return the potentially denormalized data
    return (field_outputs_denorm, scalar_outputs_denorm), (field_targets_denorm, scalar_targets_denorm)