# main_training.py
import os
import sys
from typing import Type

import torch.nn as nn

from modules.cross_validation import (
    cross_validation_procedure,
    set_seed,
    setup_logger
)
from modules.models import *
from config import CONFIG


def main(name: str, architecture: str, model_class: Type[nn.Module]) -> float:
    try:
        logger.info("Starting cross-validation procedure")
        hparams = {
            "learning_rate": CONFIG['training']['learning_rate'],
            "batch_size": CONFIG['training']['batch_size'],
            "accumulation_steps": CONFIG['training']['accumulation_steps'],
            "n_layers": CONFIG['model']['n_layers'],
            "hidden_channels": CONFIG['model']['hidden_channels'],
            "n_modes_x": CONFIG['model']['n_modes_x'],
            "n_modes_y": CONFIG['model']['n_modes_y'],
            "lifting_channels": CONFIG['model']['lifting_channels'],
            "projection_channels": CONFIG['model']['projection_channels'],
        }
        test_loss = cross_validation_procedure(
            name,
            "simulation_data_noise.hdf5",
            model_class,
            kfolds=5,
            hparams=hparams,
            use_wandb=True,
            architecture=architecture,
            plot_enabled=False,
            already_normalized=False,
        )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise
    logger.info("Training process completed successfully")
    return test_loss


if __name__ == "__main__":
    logger = setup_logger()
    os.environ["WANDB_SILENT"] = "true"
    set_seed(43)
    try:
        main("FNOv2nn", "FNO", FNOnet)
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
