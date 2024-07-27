# cross_validation.py
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
from config import *
from loguru import logger
from neuralop import LpLoss
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from modules.data import HDF5Dataset
from modules.logging import setup_logger
from modules.models import *
from modules.training import cross_validate, test_model


def setup_writer_and_hparams(
    comment: str, hparams: Dict[str, Any] = None
) -> Tuple[SummaryWriter, Dict[str, Any]]:
    writer = SummaryWriter(comment=comment)
    if hparams is None:
        hparams = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "n_layers": N_LAYERS,
            "hidden_channels": HIDDEN_CHANNELS,
            "n_modes_x": N_MODES_X,
            "n_modes_y": N_MODES_Y,
            "lifting_channels": LIFTING_CHANNELS,
            "projection_channels": PROJECTION_CHANNELS,
        }
    return writer, hparams


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_validation_procedure(
    name: str,
    data_path: str,
    model_class: Type[nn.Module],
    kfolds: int,
    hparams: Dict[str, Any] = None,
    use_wandb: bool = False,
    is_sweep: bool = False,
    architecture: str = None,
) -> float:
    logger.info("Starting cross-validation procedure")
    full_dataset = HDF5Dataset(
        data_path,
        VARIABLES,
        PARAMETERS,
        NUMPOINTS_X,
        NUMPOINTS_Y,
        DEVICE,
    )

    test_size = int(TEST_FRAC * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size]
    )

    logger.info(
        f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)"
    )

    criterion = nn.SmoothL1Loss()
    writer, hparams = setup_writer_and_hparams(comment=name, hparams=hparams)

    logger.info("Starting cross-validation")
    avg_results, avg_metrics = cross_validate(
        name,
        model_class,
        train_val_dataset,
        k_folds=kfolds,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=hparams["accumulation_steps"],
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        use_wandb=use_wandb,
        is_sweep=is_sweep,
        architecture=architecture,
        model_kwargs={
            "n_modes": (
                hparams.get("n_modes_x", N_MODES_X),
                hparams.get("n_modes_y", N_MODES_Y),
            ),
            "hidden_channels": hparams.get("hidden_channels", HIDDEN_CHANNELS),
            "n_layers": hparams.get("n_layers", N_LAYERS),
            "lifting_channels": hparams.get("lifting_channels", LIFTING_CHANNELS),
            "projection_channels": hparams.get(
                "projection_channels", PROJECTION_CHANNELS
            ),
        },
    )
    logger.info(f"Cross-validation completed. Avg results: Loss={avg_results:.4f}")

    for metric in avg_metrics:
        writer.add_scalar(f"CrossValidation/{metric}", avg_metrics[metric])

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    model = model_class(len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y).to(
        DEVICE
    )

    logger.info("Testing model on the test dataset")
    test_loss = test_model(name, model, test_dataloader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    metrics = {"Test/loss": test_loss, "CrossValidation/Loss": avg_results}
    writer.add_hparams(hparams, metrics, run_name=".")
    writer.close()
    if use_wandb:
        if is_sweep:
            wandb.init(
                project="Tesis", name=name, config=hparams, group=name, job_type="Sweep"
            )
        else:
            wandb.init(project="Tesis", name=name, config=hparams, group=name)
        wandb.config["Architecture"] = architecture
        wandb.config["Test_Loss"] = test_loss
        wandb.config["Cross_Loss"] = avg_results
        wandb.finish()

    logger.info("Cross-validation procedure completed")
    return test_loss