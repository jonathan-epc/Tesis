# main_hypertuning.py
import os

import optuna

from modules.cross_validation import (
    cross_validation_procedure,
    set_seed,
    setup_logger,
    setup_writer_and_hparams,
)
from modules.models import *


def objective(trial):
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
        "accumulation_steps": trial.suggest_int("accumulation_steps", 1, 16),
        "n_layers": trial.suggest_int("n_layers", 1, 16),
        "hidden_channels": trial.suggest_int("hidden_channels", 1, 64),
        "n_modes_x": trial.suggest_int("n_modes_x", 1, 400),
        "n_modes_y": trial.suggest_int("n_modes_y", 1, 400),
        "lifting_channels": trial.suggest_int("lifting_channels", 1, 64),
        "projection_channels": trial.suggest_int("projection_channels", 1, 64),
    }
    architecture = "FNO"
    name = f"{architecture}_trial_{trial.number}"

    try:
        test_loss = cross_validation_procedure(
            name,
            "simulation_data_normalized_noise.hdf5",
            FNOnet,
            kfolds=5,
            hparams=hparams,
            use_wandb=True,
            is_sweep=True,
            architecture=architecture,
        )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise optuna.exceptions.TrialPruned()

    return test_loss


def main():
    study = optuna.create_study(direction="minimize", study_name="tuning")
    study.optimize(objective, n_trials=100)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    logger = setup_logger()
    set_seed(43)
    os.environ["WANDB_SILENT"] = "true"
    try:
        main()
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)
