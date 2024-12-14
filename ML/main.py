import argparse
import json
import os
import pickle
import sys
import traceback
from typing import Any, Dict, Type

import optuna
import torch.nn as nn

from config import get_config
from torch.cuda import empty_cache

from modules.models import *
from modules.training import cross_validation_procedure
from modules.utils import is_jupyter, set_seed, setup_experiment, setup_logger


class TrialSkippedException(Exception):
    """Exception raised when a trial is skipped due to user interruption."""

    pass


class HyperparameterOptimizer:
    """Class to manage hyperparameter optimization."""

    def __init__(self, config):
        self.config = config

    def create_hparams(self, trial) -> Dict[str, Any]:
        """Generate hyperparameters based on Optuna trial suggestions."""
        hparams = {}
        for param, space in self.config.optuna.hyperparameter_space.items():
            suggest = getattr(trial, f"suggest_{space['type']}")
            if space["type"] == "categorical":
                hparams[param] = suggest(param, space["choices"])
            elif "step" in space and space["step"] == 2:
                low, high = [
                    x if x % 2 == 0 else x + 1 for x in (space["low"], space["high"])
                ]
                hparams[param] = suggest(param, low, high, step=2)
            else:
                hparams[param] = suggest(
                    param, space["low"], space["high"], log=space.get("log", False)
                )
        return hparams

    def save_trial_params(self, trial):
        """Save trial parameters to a JSON file."""
        study_name = self.config.optuna.study_name
        os.makedirs(f"studies/{study_name}", exist_ok=True)
        trial_data = {
            "number": trial.number,
            "params": trial.params,
        }
        with open(f"studies/{study_name}/trial_{trial.number}.json", "w") as f:
            json.dump(trial_data, f, indent=2)

    def objective(self, trial):
        """Objective function for hyperparameter optimization."""
        hparams = self.create_hparams(trial)
        model_class = getattr(sys.modules[__name__], self.config.model.class_name)
        name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{trial.number}"

        logger.info(f"Starting trial {trial.number} with parameters: {hparams}")

        self.config.training.pretrained_model_name = None
        try:
            result = cross_validation_procedure(
                name,
                self.config.data.file_name,
                model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                is_sweep=True,
                trial=trial,
                config=self.config,
            )
            # self.save_trial_params(trial)
            return result
        except KeyboardInterrupt:
            logger.info(f"Skipping trial {trial.number} due to interruption.")
            raise TrialSkippedException()
        except Exception as e:
            # Capture and truncate the traceback
            full_traceback = traceback.format_exception(type(e), e, e.__traceback__)
            truncated_traceback = "".join(full_traceback[-5:])  # Keep last 5 frames
            logger.error(
                f"Error during trial {trial.number}: {e}\nTruncated Traceback:\n{truncated_traceback}"
            )
            raise optuna.exceptions.TrialPruned()
        finally:
            empty_cache()
            logger.info(f"GPU cache cleared after trial {trial.number}.")

    def save_study_artifacts(self, study):
        """Save Optuna study artifacts."""
        for artifact in ["sampler", "pruner"]:
            with open(
                f"studies/{self.config.optuna.study_name}_{artifact}.pkl", "wb"
            ) as f:
                pickle.dump(getattr(study, artifact), f)
        logger.info(f"Artifacts saved for {self.config.optuna.study_name}")

    def log_best_trial(self, study):
        """Log details of the best trial."""
        best_trial = study.best_trial
        logger.info(
            f"Best trial value: {best_trial.value}\nParameters: {best_trial.params}"
        )

    def run_hypertuning(self):
        """Run hyperparameter optimization with Optuna."""
        study = optuna.create_study(
            study_name=self.config.optuna.study_name,
            load_if_exists=True,
            direction="minimize",
            storage=self.config.optuna.storage,
        )

        try:
            study.optimize(
                self.objective,
                n_trials=self.config.optuna.n_trials,
                catch=(TrialSkippedException,),
            )
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user.")
        finally:
            self.save_study_artifacts(study)
            self.log_best_trial(study)


def get_default_hparams(config) -> Dict[str, Any]:
    """Retrieve default hyperparameters for training."""
    return {
        key: getattr(
            config.model if key in config.model.__dict__ else config.training, key
        )
        for key in [
            "n_layers",
            "n_modes_x",
            "n_modes_y",
            "hidden_channels",
            "lifting_channels",
            "projection_channels",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "accumulation_steps",
            "lambda_physics",
        ]
    }


def run_single_training(config):
    """Execute a single training procedure."""
    model_class = getattr(sys.modules[__name__], config.model.class_name)
    hparams = get_default_hparams(config)

    test_loss = cross_validation_procedure(
        config.model.name,
        config.data.file_name,
        model_class,
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

    optimizer = HyperparameterOptimizer(config)

    try:
        if mode == "hypertuning":
            logger.info("Starting hyperparameter tuning.")
            optimizer.run_hypertuning()
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
        main("hypertuning")
    else:
        parser = argparse.ArgumentParser(description="Run model tuning or training.")
        parser.add_argument(
            "--mode",
            choices=["hypertuning", "training"],
            required=True,
            help="Mode of operation",
        )
        args = parser.parse_args()
        main(args.mode)
