import os
import random
from typing import List, Dict

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


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_experiment(config: dict) -> None:
    """Set up the experiment environment."""
    os.environ["WANDB_SILENT"] = "true"
    set_seed(config.seed)


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_hparams(config: dict) -> dict:
    """Extract hyperparameters from the configuration."""
    return {
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "accumulation_steps": config["training"]["accumulation_steps"],
        "n_layers": config["model"]["n_layers"],
        "hidden_channels": config["model"]["hidden_channels"],
        "n_modes_x": config["model"]["n_modes_x"],
        "n_modes_y": config["model"]["n_modes_y"],
        "lifting_channels": config["model"]["lifting_channels"],
        "projection_channels": config["model"]["projection_channels"],
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
    logger.add("logs/file_{time}.log", rotation="500 MB")
    # logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
    return logger


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    variable_names: List[str],
) -> Dict[str, torch.Tensor]:
    assert outputs.shape == targets.shape, "Outputs and targets must have the same shape"
    assert outputs.shape[1] == len(variable_names), "Number of variables doesn't match tensor shape"

    metrics = {}
    epsilon = 1e-8  # Small value to avoid division by zero

    # Compute overall metrics
    mse = torch.mean((targets - outputs) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(targets - outputs))
    
    # R2 score calculation
    ss_tot = torch.sum((targets - torch.mean(targets, dim=0)) ** 2)
    ss_res = torch.sum((targets - outputs) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    metrics.update({
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    })

    # Compute metrics for each variable
    for i, var_name in enumerate(variable_names):
        var_outputs = outputs[:, i]
        var_targets = targets[:, i]
        
        # If the variable is a 2D field, flatten it for metric computation
        if var_outputs.dim() > 1:
            var_outputs = var_outputs.flatten(1)
            var_targets = var_targets.flatten(1)
        
        var_mse = torch.mean((var_targets - var_outputs) ** 2)
        var_rmse = torch.sqrt(var_mse)
        var_mae = torch.mean(torch.abs(var_targets - var_outputs))
        
        # R2 score calculation
        var_ss_tot = torch.sum((var_targets - torch.mean(var_targets, dim=0)) ** 2)
        var_ss_res = torch.sum((var_targets - var_outputs) ** 2)
        var_r2 = 1 - (var_ss_res / (var_ss_tot + epsilon))
        
        var_mape = torch.mean(torch.abs((var_targets - var_outputs) / (var_targets + epsilon))) * 100

        metrics.update({
            f"{var_name}_mse": var_mse,
            f"{var_name}_rmse": var_rmse,
            f"{var_name}_mae": var_mae,
            f"{var_name}_r2": var_r2,
            f"{var_name}_mape": var_mape,
        })

    return metrics