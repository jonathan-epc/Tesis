# main_training.py

import argparse
import sys
from typing import Type

import torch.nn as nn
from neuralop import LpLoss

from modules.cross_validation import cross_validation_procedure
from modules.logging import setup_logger
from modules.models import *
from modules.utils import get_hparams, load_config, set_seed, setup_experiment


def main(config_path: str) -> float:
    """
    Main function to run the cross-validation procedure for model training.

    ... [rest of the docstring remains the same] ...
    """
    config = load_config(config_path)
    setup_experiment(config)
    try:
        logger.info("Starting cross-validation procedure")
        hparams = get_hparams(config)
        model_class = globals()[config["model"]["class"]]
        test_loss = cross_validation_procedure(
            config["model"]["name"],
            config["data"]["file_name"],
            model_class,
            nn.HuberLoss(),
            kfolds=config["training"]["kfolds"],
            hparams=hparams,
            use_wandb=config["logging"]["use_wandb"],
            architecture=config["model"]["architecture"],
            plot_enabled=config["logging"]["plot_enabled"],
        )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise

    logger.info("Training process completed successfully")
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model training with cross-validation."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file"
    )
    args = parser.parse_args("")

    logger = setup_logger()

    try:
        main(args.config)
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
