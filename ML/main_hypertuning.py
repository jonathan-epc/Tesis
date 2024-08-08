# main_hypertuning.py

import argparse
import sys

import optuna

from modules.cross_validation import cross_validation_procedure
from modules.logging import setup_logger
from modules.models import *
from modules.utils import load_config, set_seed, setup_experiment


def objective(trial, config: dict):
    """
    Objective function for Optuna optimization.

    This function defines the hyperparameter space and runs the cross-validation
    procedure for each trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial object for suggesting hyperparameters.
    config : dict
        Configuration dictionary containing experiment settings.

    Returns
    -------
    float
        The test loss after cross-validation.

    Raises
    ------
    optuna.exceptions.TrialPruned
        If an error occurs during the training process.
    """
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "accumulation_steps": trial.suggest_int("accumulation_steps", 1, 16),
        "n_layers": trial.suggest_int("n_layers", 1, 64),
        "hidden_channels": trial.suggest_int("hidden_channels", 1, 8),
        "n_modes_x": trial.suggest_int("n_modes_x", 1, 200),
        "n_modes_y": trial.suggest_int("n_modes_y", 1, 200),
        "lifting_channels": trial.suggest_int("lifting_channels", 1, 64),
        "projection_channels": trial.suggest_int("projection_channels", 1, 64),
    }

    architecture = config["model"]["architecture"]
    name = f"{architecture}_trial_{trial.number}"
    model_class = globals()[config["model"]["class"]]

    try:
        test_loss = cross_validation_procedure(
            name,
            config["data"]["file_name"],
            model_class,
            nn.HuberLoss(),
            kfolds=config["training"]["kfolds"],
            hparams=hparams,
            use_wandb=config["logging"]["use_wandb"],
            is_sweep=True,
            architecture=architecture,
            plot_enabled=config["logging"]["plot_enabled"],
        )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise optuna.exceptions.TrialPruned()

    return test_loss


def main(config_path: str):
    """
    Main function to run the hyperparameter tuning process.

    This function sets up the Optuna study and runs the optimization process.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    config = load_config(config_path)
    setup_experiment(config)

    study = optuna.create_study(direction="minimize", study_name="tuning")
    study.optimize(
        lambda trial: objective(trial, config), n_trials=config["optuna"]["n_trials"]
    )

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning for model training."
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
