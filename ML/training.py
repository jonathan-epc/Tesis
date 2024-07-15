import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from config import *
from modules.data import HDF5Dataset, create_dataloaders
from modules.logging import setup_logger
from modules.models import FNOnet2
from modules.training import cross_validate, test_model


def setup_writer_and_hparams(comment: str) -> Tuple[SummaryWriter, Dict[str, Any]]:
    writer = SummaryWriter(comment=comment)
    hparams = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
    }
    return writer, hparams


def cross_validation_procedure(name: str, data_path: str, model_class: Type[nn.Module], kfolds:int) -> float:
    logger.info("Starting cross-validation procedure")

    full_dataset = HDF5Dataset(
        data_path,
        VARIABLES,
        PARAMETERS,
        NUMPOINTS_X,
        NUMPOINTS_Y,
        DEVICE,
    )

    test_size = int(0.2 * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size])

    logger.info(f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)")

    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer, hparams = setup_writer_and_hparams(comment=name)

    logger.info("Starting cross-validation")
    avg_results = cross_validate(
        name,
        model_class,
        train_val_dataset,
        k_folds=kfolds,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        writer=writer
    )

    logger.info(f"Cross-validation completed. Avg results: Loss={avg_results[0]:.4f}")

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    model = model_class(len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y).to(DEVICE)

    logger.info("Testing model on the test dataset")
    test_loss = test_model(name, model, test_dataloader, criterion)
    logger.info(f"Test Loss: {test_loss:.4f}")

    metrics = {"Loss/test": test_loss, "Loss/Cross": avg_results}
    writer.add_hparams(hparams, metrics, run_name=".")
    writer.close()

    logger.info("Cross-validation procedure completed")
    return test_loss


def main(name: str, model_class: Type[nn.Module]) -> float:
    try:
        logger.info("Starting cross-validation procedure")
        test_loss = cross_validation_procedure(name, "simulation_data_normalized_flat.hdf5", model_class, kfolds=2)
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise
    logger.info("Training process completed successfully")
    return test_loss


if __name__ == "__main__":
    logger = setup_logger()
    try:
        main("FNO2", FNOnet2)
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)

