import argparse
import pickle
import sys
from typing import Type

import optuna
import torch.nn as nn
from neuralop import LpLoss

from modules.cross_validation import cross_validation_procedure
from modules.logging import setup_logger
from modules.models import *
from modules.utils import get_hparams, load_config, set_seed, setup_experiment


class TrialSkippedException(Exception):
    pass


def create_hparams(trial, config):
    return {
        param: getattr(trial, f"suggest_{space['type']}")(
            param, space["low"], space["high"], log=space.get("log", False)
        )
        for param, space in config["optuna"]["hyperparameter_space"].items()
    }


def objective(trial, config):
    hparams = create_hparams(trial, config)
    model_class = globals()[config["model"]["class"]]
    name = f"{config['optuna']['study_name']}_{config['model']['architecture']}_trial_{trial.number}"

    try:
        return cross_validation_procedure(
            name,
            config["data"]["file_name"],
            model_class,
            nn.HuberLoss(),
            kfolds=config["training"]["kfolds"],
            hparams=hparams,
            use_wandb=config["logging"]["use_wandb"],
            is_sweep=True,
            trial=trial,
            architecture=config["model"]["architecture"],
            plot_enabled=config["logging"]["plot_enabled"],
        )
    except KeyboardInterrupt:
        logger.info(f"Skipping trial {trial.number} due to keyboard interrupt.")
        raise TrialSkippedException()
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise optuna.exceptions.TrialPruned()


def save_study_artifacts(study, study_name):
    for artifact in ["sampler", "pruner"]:
        with open(f"studies/{study_name}_{artifact}.pkl", "wb") as f:
            pickle.dump(getattr(study, artifact), f)
    logger.info(f"Study artifacts saved for {study_name}")


def log_best_trial(study):
    best_trial = study.best_trial
    logger.info(f"Best trial:\n  Value: {best_trial.value}")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")


def run_hypertuning(config):
    study = optuna.create_study(
        study_name=config["optuna"]["study_name"],
        load_if_exists=True,
        direction="minimize",
        storage=config["optuna"]["storage"],
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
    )

    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config["optuna"]["n_trials"],
            catch=(TrialSkippedException,),
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")
    finally:
        save_study_artifacts(study, config["optuna"]["study_name"])
        log_best_trial(study)


def run_single_training(config):
    model_class = globals()[config["model"]["class"]]
    test_loss = cross_validation_procedure(
        config["model"]["name"],
        config["data"]["file_name"],
        model_class,
        nn.HuberLoss(),
        kfolds=config["training"]["kfolds"],
        hparams=get_hparams(config),
        use_wandb=config["logging"]["use_wandb"],
        architecture=config["model"]["architecture"],
        plot_enabled=config["logging"]["plot_enabled"],
    )
    logger.info(f"Final test loss: {test_loss}")
    return test_loss


def main(config_path: str, mode: str):
    config = load_config(config_path)
    setup_experiment(config)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning or single training for model."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["hypertuning", "training"],
        required=True,
        help="Mode of operation",
    )

    args = parser.parse_args("")
    args.mode = 'hypertuning'
    logger = setup_logger()

    try:
        main(args.config, args.mode)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)

logger = setup_logger()
try:
    main('config.yaml', 'hypertuning')
except Exception as e:
    logger.error(f"Unhandled exception: {e}")
    sys.exit(1)
