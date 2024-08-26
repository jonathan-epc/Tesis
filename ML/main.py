import argparse
import pickle
import sys
from typing import Type

import optuna
import torch.nn as nn
from neuralop import LpLoss

from modules.training import cross_validation_procedure
from modules.logging import setup_logger
from modules.models import *
from modules.utils import set_seed, setup_experiment
from config import get_config  # Import the new config function


class TrialSkippedException(Exception):
    pass


def create_hparams(trial, config):
    return {
        param: getattr(trial, f"suggest_{space.type}")(
            param, space.low, space.high, log=space.log if hasattr(space, 'log') else False
        )
        for param, space in config.optuna.hyperparameter_space.items()
    }


def objective(trial, config):
    hparams = create_hparams(trial, config)
    model_class = globals()[config.model.class_name]
    name = f"{config.optuna.study_name}_{config.model.architecture}_trial_{trial.number}"

    try:
        return cross_validation_procedure(
            name,
            config.data.file_name,
            model_class,
            nn.HuberLoss(),
            kfolds=config.training.kfolds,
            hparams=hparams,
            use_wandb=config.logging.use_wandb,
            is_sweep=True,
            trial=trial,
            architecture=config.model.architecture,
            plot_enabled=config.logging.plot_enabled,
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
        study_name=config.optuna.study_name,
        load_if_exists=True,
        direction="minimize",
        storage=config.optuna.storage,
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
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


def run_single_training(config):
    model_class = globals()[config.model.class_name]
    
    test_loss = cross_validation_procedure(
        config.model.name,
        config.data.file_name,
        model_class,
        nn.HuberLoss(),
        kfolds=config.training.kfolds,
        hparams={
            'n_layers': config.model.n_layers,
            'n_modes_x': config.model.n_modes_x,
            'n_modes_y': config.model.n_modes_y,
            'hidden_channels': config.model.hidden_channels,
            'lifting_channels': config.model.lifting_channels,
            'projection_channels': config.model.projection_channels,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'accumulation_steps': config.training.accumulation_steps,
        },
        use_wandb=config.logging.use_wandb,
        architecture=config.model.architecture,
        plot_enabled=config.logging.plot_enabled,
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


def is_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter


if __name__ == "__main__" or is_jupyter():
    logger = setup_logger()
    if is_jupyter():
        # Default mode when running in Jupyter
        mode = "training"
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
<<<<<<< HEAD
        if not is_jupyter():
            sys.exit(1)
        else:
            raise  # Re-raise the exception in Jupyter for better traceback
=======
        sys.exit(1)

logger = setup_logger()
try:
    main('config.yaml', 'training')
except Exception as e:
    logger.error(f"Unhandled exception: {e}")
    sys.exit(1)
>>>>>>> 84d7ef7df9b1cc45d29242968ba9fe7ba1b33b43
