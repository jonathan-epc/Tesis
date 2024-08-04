import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from config import CONFIG
from loguru import logger
from neuralop import LpLoss
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from modules.data import HDF5Dataset
from modules.logging import setup_logger
from modules.models import *
from modules.utils  
from modules.training import cross_validate, test_model

def cross_validation_procedure(
    name: str,
    data_path: str,
    model_class: Type[nn.Module],
    kfolds: int,
    hparams: Optional[Dict[str, Any]] = None,
    use_wandb: bool = False,
    is_sweep: bool = False,
    architecture: Optional[str] = None,
    plot_enabled: bool = False,
    already_normalized: bool = True,
) -> float:
    """
    Perform cross-validation procedure for model training and evaluation.

    Parameters
    ----------
    name : str
        Name of the experiment.
    data_path : str
        Path to the HDF5 dataset file.
    model_class : Type[nn.Module]
        PyTorch model class to be instantiated.
    kfolds : int
        Number of folds for cross-validation.
    hparams : Optional[Dict[str, Any]], optional
        Dictionary of hyperparameters. If None, default values will be used.
    use_wandb : bool, optional
        Whether to use Weights & Biases for logging (default is False).
    is_sweep : bool, optional
        Whether this is part of a hyperparameter sweep (default is False).
    architecture : Optional[str], optional
        Name of the model architecture.
    plot_enabled : bool, optional
        Whether to enable plotting (default is False).
    already_normalized : bool, optional
        Whether the dataset is already normalized (default is True).

    Returns
    -------
    float
        Test loss after cross-validation and final testing.
    """
    logger.info("Starting cross-validation procedure")
    
    # Load and split dataset
    full_dataset = HDF5Dataset(
        file_path=data_path,
        variables=CONFIG['data']['variables'],
        parameters=CONFIG['data']['parameters'],
        numpoints_x=CONFIG['data']['numpoints_x'],
        numpoints_y=CONFIG['data']['numpoints_y'],
        already_normalized=already_normalized,
        device=CONFIG['device'],
    )

    test_size = int(CONFIG['training']['test_frac'] * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size]
    )

    logger.info(
        f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)"
    )

    # Set up training components
    criterion = nn.MSELoss()

    # Perform cross-validation
    logger.info("Starting cross-validation")
    avg_results, avg_metrics = cross_validate(
        name,
        model_class,
        train_val_dataset,
        k_folds=kfolds,
        num_epochs=CONFIG['training']['num_epochs'],
        accumulation_steps=hparams["accumulation_steps"],
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        model_kwargs = hparams,
        use_wandb=use_wandb,
        is_sweep=is_sweep,
        architecture=architecture,
        plot_enabled=plot_enabled
    )
    logger.info(f"Cross-validation completed. Avg results: Loss={avg_results:.4f}")

    # Test the model on the test dataset
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=CONFIG['training']['num_workers'],
    )
    model = model_class(len(CONFIG['data']['parameters']), len(CONFIG['data']['variables']), CONFIG['data']['numpoints_x'], CONFIG['data']['numpoints_y'], **model_kwargs).to(CONFIG['device'])
    logger.info("Testing model on the test dataset")
    test_loss = test_model(name, model, test_dataloader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # Log final metrics and hyperparameters
    metrics = {"Test/loss": test_loss, "CrossValidation/Loss": avg_results}
    
    # Log to Weights & Biases if enabled
    if use_wandb:
        wandb.init(
            project="Tesis",
            name=name,
            config=hparams,
            group=name,
            job_type="Sweep" if is_sweep else "Run"
        )
        wandb.config.update({
            "Architecture": architecture,
            "Test_Loss": test_loss,
            "Cross_Loss": avg_results
        })
        wandb.finish()

    logger.info("Cross-validation procedure completed")
    return test_loss


if __name__ == "__main__":
    # Add any main execution code here if needed
    pass
