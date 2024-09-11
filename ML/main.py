import argparse
import pickle
import sys
from typing import Type

import optuna
import torch.nn as nn
from neuralop import LpLoss

from config import get_config

from modules.loss import FluidDynamicsLoss
from modules.models import *
from modules.training import cross_validation_procedure
from modules.utils import is_jupyter, set_seed, setup_experiment, setup_logger


class TrialSkippedException(Exception):
    pass


def create_hparams(trial, config):
    """Create hyperparameters based on Optuna trial suggestions."""
    hparams = {}
    for param, space in config.optuna.hyperparameter_space.items():
        if space["type"] == "categorical":
            hparams[param] = trial.suggest_categorical(param, space["choices"])
        else:
            hparams[param] = getattr(trial, f"suggest_{space['type']}")(
                param, space["low"], space["high"], log=space.get("log", False)
            )
    return hparams


def save_trial_params(trial, study_name):
    """Save trial parameters to a JSON file."""
    os.makedirs(f"studies/{study_name}", exist_ok=True)
    trial_params = {
        "number": trial.number,
        "params": trial.params,
        "value": trial.value,
        "state": trial.state.name,
    }
    with open(f"studies/{study_name}/trial_{trial.number}.json", "w") as f:
        json.dump(trial_params, f, indent=2)


def objective(trial, config):
    """Objective function for hyperparameter optimization."""
    hparams = create_hparams(trial, config)
    model_class = getattr(sys.modules[__name__], config.model.class_name)
    name = (
        f"{config.optuna.study_name}_{config.model.architecture}_trial_{trial.number}"
    )

    # Log the parameters at the start of each trial
    logger.info(f"Starting trial {trial.number} with parameters:")
    for key, value in hparams.items():
        logger.info(f"  {key}: {value}")

    # If hypertuning, skip using pretrained models
    config.training.pretrained_model_name = None

    try:
        result = cross_validation_procedure(
            name,
            config.data.file_name,
            model_class,
            nn.HuberLoss(),
            kfolds=config.training.kfolds,
            hparams=hparams,
            is_sweep=True,
            trial=trial,
            config=config,
        )
        # Save trial parameters after successful completion
        save_trial_params(trial, config.optuna.study_name)
        return result
    except KeyboardInterrupt:
        logger.info(f"Skipping trial {trial.number} due to keyboard interrupt.")
        raise TrialSkippedException()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise optuna.exceptions.TrialPruned()


def save_study_artifacts(study, study_name):
    """Save Optuna study artifacts."""
    for artifact in ["sampler", "pruner"]:
        with open(f"studies/{study_name}_{artifact}.pkl", "wb") as f:
            pickle.dump(getattr(study, artifact), f)
    logger.info(f"Study artifacts saved for {study_name}")


def log_best_trial(study):
    """Log the best trial details after hypertuning."""
    best_trial = study.best_trial
    logger.info(f"Best trial:\n  Value: {best_trial.value}")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")


def run_hypertuning(config):
    """Run Optuna hypertuning."""
    study = optuna.create_study(
        study_name=config.optuna.study_name,
        load_if_exists=True,
        direction="minimize",
        storage=config.optuna.storage,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config.optuna.n_trials,
            catch=(TrialSkippedException,),
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    finally:
        save_study_artifacts(study, config.optuna.study_name)
        log_best_trial(study)


def get_default_hparams(config):
    """Helper function to get default hyperparameters for training."""
    return {
        "n_layers": config.model.n_layers,
        "n_modes_x": config.model.n_modes_x,
        "n_modes_y": config.model.n_modes_y,
        "hidden_channels": config.model.hidden_channels,
        "lifting_channels": config.model.lifting_channels,
        "projection_channels": config.model.projection_channels,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.learning_rate,
        "learning_rate": config.training.weight_decay,
        "accumulation_steps": config.training.accumulation_steps,
    }


def run_single_training(config):
    """Run a single training procedure."""
    model_class = getattr(sys.modules[__name__], config.model.class_name)
    hparams = get_default_hparams(config)

    test_loss = cross_validation_procedure(
        config.model.name,
        config.data.file_name,
        model_class,
        nn.HuberLoss(),
        kfolds=config.training.kfolds,
        hparams=hparams,
        config=config,
    )
    logger.info(f"Final test loss: {test_loss}")
    return test_loss


def main(mode: str):
    config = get_config()
    setup_experiment(config)
    set_seed(config.seed)

    try:
        if mode == "hypertuning":
            logger.info("Starting hyperparameter tuning.")
            run_hypertuning(config)
        elif mode == "training":
            logger.info("Starting single training run.")
            run_single_training(config)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    except Exception as e:
        logger.error(f"Process error: {e}")
        raise

    logger.info("Process completed successfully.")


if __name__ == "__main__" or is_jupyter():
    logger = setup_logger()
    if is_jupyter():
        mode = "hypertuning"
        logger.info(f"Running in Jupyter environment. Default mode: {mode}")
    else:
        parser = argparse.ArgumentParser(
            description="Run hyperparameter tuning or single training for model."
        )
        parser.add_argument(
            "--mode",
            choices=["hypertuning", "training"],
            required=True,
            help="Mode of operation",
        )
        args = parser.parse_args()
        mode = args.mode

    try:
        main(mode)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        if not is_jupyter():
            sys.exit(1)
        else:
            raise  # Re-raise the exception in Jupyter for better traceback
