# main_training.py
import os
import sys
from typing import Type

import torch.nn as nn

from modules.cross_validation import (
    cross_validation_procedure,
    set_seed,
    setup_logger,
    setup_writer_and_hparams,
)
from modules.models import *


def main(name: str, model_class: Type[nn.Module]) -> float:
    try:
        logger.info("Starting cross-validation procedure")
        test_loss = cross_validation_procedure(
            name,
            "simulation_data_normalized_noise.hdf5",
            model_class,
            kfolds=5,
            use_wandb=True,
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
        main("FNOv3-MLP", FNOv3net)
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
