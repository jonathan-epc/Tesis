import argparse
import json
import os
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
from torch.cuda import empty_cache

from config import get_config
from modules.models import *
from modules.training import cross_validation_procedure
from modules.utils import is_jupyter, set_seed, setup_experiment, setup_logger


class OptimizerMode(Enum):
    HYPERTUNING = auto()
    TRAINING = auto()
    REPEAT = auto()  # New mode added

    @classmethod
    def from_string(cls, s: str) -> "OptimizerMode":
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid mode: {s}. Must be one of {[m.name.lower() for m in cls]}"
            )


@dataclass
class OptimizationResults:
    best_value: float
    best_params: Dict[str, Any]
    study_name: str
    n_trials: int

    def save(self, path: Path):
        path.write_text(
            json.dumps(
                {
                    "best_value": self.best_value,
                    "best_params": self.best_params,
                    "study_name": self.study_name,
                    "n_trials": self.n_trials,
                },
                indent=2,
            )
        )


class HyperparameterOptimizer:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()
        self.study_dir = Path(f"studies/{self.config.optuna.study_name}")
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self):
        """Validate configuration before starting optimization"""
        required_fields = ["hyperparameter_space", "study_name", "n_trials", "storage"]
        missing = [f for f in required_fields if not hasattr(self.config.optuna, f)]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

    def create_hparams(self, trial) -> Dict[str, Any]:
        hparams = {}
        for param, space in self.config.optuna.hyperparameter_space.items():
            try:
                suggest = getattr(trial, f"suggest_{space['type']}")
                if space["type"] == "categorical":
                    hparams[param] = suggest(param, space["choices"])
                else:
                    step = space.get("step", None)
                    if step == 2:
                        low = space["low"] + (space["low"] % 2)
                        high = space["high"] + (space["high"] % 2)
                        hparams[param] = suggest(param, low, high, step=2)
                    else:
                        hparams[param] = suggest(
                            param,
                            space["low"],
                            space["high"],
                            log=space.get("log", False),
                        )
            except AttributeError as e:
                raise ValueError(
                    f"Invalid hyperparameter space type for {param}: {space['type']}"
                ) from e
            except KeyError as e:
                raise ValueError(
                    f"Missing required field in hyperparameter space for {param}"
                ) from e
        return hparams

    def _get_model_class(self):
        try:
            return globals()[self.config.model.class_name]
        except KeyError:
            raise ValueError(f"Model class {self.config.model.class_name} not found")

    def objective(self, trial):
        hparams = self.create_hparams(trial)
        model_class = self._get_model_class()
        name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{trial.number}"

        self.logger.info(f"Starting trial {trial.number} with parameters: {hparams}")

        try:
            result = cross_validation_procedure(
                name=name,
                data_path=self.config.data.file_name,
                model_class=model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                is_sweep=True,
                trial=trial,
                config=self.config,
            )

            # Save trial results for analysis
            trial_results = {"number": trial.number, "params": hparams, "value": result, "config": self.config}
            (self.study_dir / f"trial_{trial.number}.json").write_text(
                json.dumps(trial_results, indent=2)
            )

            return result

        except KeyboardInterrupt:
            self.logger.info("Trial interrupted by user")
            raise optuna.exceptions.TrialPruned()
        except Exception as e:
            self.logger.error(f"Trial failed with error: {str(e)}")
            raise optuna.exceptions.TrialPruned()
        finally:
            empty_cache()

    def run_optimization(self) -> OptimizationResults:
        self._validate_config()

        study = optuna.create_study(
            study_name=self.config.optuna.study_name,
            load_if_exists=True,
            direction="minimize",
            storage=self.config.optuna.storage,
        )

        try:
            study.optimize(self.objective, n_trials=self.config.optuna.n_trials)
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        finally:
            self._save_artifacts(study)
            return self._create_results(study)

    def _save_artifacts(self, study: optuna.Study):
        """Save study artifacts for later analysis"""
        for artifact in ["sampler", "pruner"]:
            artifact_path = self.study_dir / f"{artifact}.pkl"
            with artifact_path.open("wb") as f:
                pickle.dump(getattr(study, artifact), f)

    def _create_results(self, study: optuna.Study) -> OptimizationResults:
        """Create and save optimization results"""
        results = OptimizationResults(
            best_value=study.best_value,
            best_params=study.best_trial.params,
            study_name=study.study_name,
            n_trials=len(study.trials),
        )

        results.save(self.study_dir / "results.json")

        self.logger.info(
            f"Best trial value: {results.best_value}\n"
            f"Parameters: {results.best_params}"
        )

        return results


class ModelTrainer:
    """Handles single training runs with fixed hyperparameters"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()

    def _get_default_hparams(self) -> Dict[str, Any]:
        param_keys = [
            "n_layers",
            "n_modes_x",
            "n_modes_y",
            "hidden_channels",
            "lifting_channels",
            "projection_channels",
            "batch_size",
            "use_physics_loss",
            "normalize_output",  # This should come from the 'data' section
            "learning_rate",
            "weight_decay",
            "accumulation_steps",
            "lambda_physics",
        ]

        hparams = {}
        for key in param_keys:
            if key == "normalize_output":
                hparams[key] = getattr(self.config.data, key)
            elif hasattr(self.config.model, key):
                hparams[key] = getattr(self.config.model, key)
            elif hasattr(self.config.training, key):
                hparams[key] = getattr(self.config.training, key)
            else:
                raise AttributeError(f"Missing attribute: {key}")

        return hparams

    def train(self) -> float:
        model_class = globals()[self.config.model.class_name]
        hparams = self._get_default_hparams()

        test_loss = cross_validation_procedure(
            name=self.config.model.name,
            data_path=self.config.data.file_name,
            model_class=model_class,
            kfolds=self.config.training.kfolds,
            hparams=hparams,
            config=self.config,
        )

        self.logger.info(f"Final test loss: {test_loss}")
        return test_loss


def repeat_trial_from_study(trial_id: int, config) -> float:
    """
    Load the Optuna study, retrieve the trial with the given trial_id,
    and re-run the training procedure using its parameters.
    """
    study = optuna.load_study(
        study_name=config.optuna.study_name,
        storage=config.optuna.storage
    )
    trial = next((t for t in study.trials if t.number == trial_id), None)
    if trial is None:
        raise ValueError(f"Trial number {trial_id} not found in study {config.optuna.study_name}")

    hparams = trial.params
    model_class = globals()[config.model.class_name]
    name = f"{config.optuna.study_name}_{config.model.architecture}_repeated_trial_{trial_id}"

    result = cross_validation_procedure(
        name=name,
        data_path=config.data.file_name,
        model_class=model_class,
        kfolds=config.training.kfolds,
        hparams=hparams,
        config=config,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run model optimization, training, or repeat a trial")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.name.lower() for m in OptimizerMode],
        required=not is_jupyter(),
        help="Mode of operation",
    )
    parser.add_argument("--trial_id", type=int, help="Trial number to repeat (used in repeat mode)")

    args = (
        parser.parse_args()
        if not is_jupyter()
        else argparse.Namespace(mode="hypertuning", trial_id=None)
    )
    args.mode = "training"
    args.trial_id = 5
    mode = OptimizerMode.from_string(args.mode)

    config = get_config()
    setup_experiment(config)
    set_seed(config.seed)
    logger = setup_logger()

    try:
        if mode == OptimizerMode.HYPERTUNING:
            logger.info("Starting hyperparameters tuning")
            optimizer = HyperparameterOptimizer(config)
            results = optimizer.run_optimization()
            logger.info(f"Optimization completed with best value: {results.best_value}")
        elif mode == OptimizerMode.TRAINING:
            logger.info("Starting single training run")
            trainer = ModelTrainer(config)
            test_loss = trainer.train()
            logger.info(f"Training completed with test loss: {test_loss}")
        elif mode == OptimizerMode.REPEAT:
            if args.trial_id is None:
                raise ValueError("Trial ID must be provided in repeat mode using --trial_id")
            logger.info(f"Repeating trial number: {args.trial_id}")
            result = repeat_trial_from_study(args.trial_id, config)
            logger.info(f"Repeated trial {args.trial_id} result: {result}")
    except Exception as e:
        logger.error(f"Process error: {str(e)}")
        raise

    logger.info("Process completed successfully")


if __name__ == "__main__" or is_jupyter():
    main()
