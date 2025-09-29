# ML/core/optimizer.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch
import wandb
from torch.cuda import empty_cache
from torch.utils.data import DataLoader, random_split

from common.utils import setup_logger
from ML.modules.data import HDF5Dataset
from ML.modules.loss import PhysicsInformedLoss
from ML.modules.models import FNOnet
from ML.modules.training import Trainer, cross_validation_procedure
from ML.modules.utils import seed_worker
from nconfig import Config


@dataclass
class OptimizationResults:
    best_value: float
    best_params: dict[str, Any]
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
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger()
        self.study_dir = Path(config.paths.studies_dir) / self.config.optuna.study_name
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.full_dataset_for_stats: HDF5Dataset | None = None
        self.train_val_dataset: torch.utils.data.Dataset | None = None
        self.test_dataset: torch.utils.data.Dataset | None = None

    def create_hparams(self, trial: optuna.Trial) -> dict[str, Any]:
        hparams = {}
        for param, space in self.config.optuna.hyperparameter_space.items():
            suggest_type = space["type"]
            suggest_func = getattr(trial, f"suggest_{suggest_type}")

            if suggest_type == "categorical":
                hparams[param] = suggest_func(param, space["choices"])

            elif suggest_type == "int":
                # Integer suggestions can have a step
                step = space.get("step", 1)
                hparams[param] = suggest_func(
                    param, space["low"], space["high"], step=step
                )

            elif suggest_type == "float":
                # Float suggestions should NOT have a step if log=True
                log = space.get("log", False)
                hparams[param] = suggest_func(
                    param, space["low"], space["high"], log=log
                )

            else:
                raise ValueError(
                    f"Unsupported suggestion type: {suggest_type} for param {param}"
                )

        return hparams

    def _get_model_class(self):
        # This assumes FNOnet is the primary model. Can be made more dynamic if needed.
        return FNOnet

    def objective(self, trial: optuna.Trial) -> float:
        hparams = self.create_hparams(trial)
        model_class = self._get_model_class()
        name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{trial.number}"
        self.logger.info(f"Starting trial {trial.number} with parameters: {hparams}")

        try:
            if self.train_val_dataset is None or self.full_dataset_for_stats is None:
                self.logger.error("Datasets not loaded before calling objective.")
                self._load_and_split_data()

            avg_cv_loss = cross_validation_procedure(
                name=name,
                model_class=model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                is_sweep=True,
                trial=trial,
                config=self.config,
                train_val_dataset=self.train_val_dataset,
                full_dataset_for_stats=self.full_dataset_for_stats,
            )
            return avg_cv_loss

        except optuna.exceptions.TrialPruned as e:
            self.logger.warning(f"Trial {trial.number} pruned: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Trial {trial.number} failed with error: {str(e)}")
            raise optuna.exceptions.TrialPruned(f"Trial failed: {str(e)}") from e
        finally:
            empty_cache()

    def _load_and_split_data(self):
        self.logger.info("Loading and splitting dataset...")
        self.full_dataset_for_stats = HDF5Dataset.from_config(
            self.config, file_path=self.config.data.file_path
        )
        test_size = int(
            self.config.training.test_frac * len(self.full_dataset_for_stats)
        )
        train_val_size = len(self.full_dataset_for_stats) - test_size
        g_split = torch.Generator().manual_seed(self.config.seed)
        self.train_val_dataset, self.test_dataset = random_split(
            self.full_dataset_for_stats, [train_val_size, test_size], generator=g_split
        )
        self.logger.info(
            f"Dataset split: Train/Val={len(self.train_val_dataset)}, Test={len(self.test_dataset)}"
        )

    def run_optimization(self) -> OptimizationResults | None:
        try:
            self._load_and_split_data()
            study = optuna.create_study(
                study_name=self.config.optuna.study_name,
                load_if_exists=True,
                direction="minimize",
                storage=self.config.optuna_storage_url,
                pruner=optuna.pruners.MedianPruner(),
            )
            study.set_user_attr("config", self.config.to_dict())
            self.logger.info(f"Starting/Resuming Optuna study '{study.study_name}'...")
            study.optimize(
                self.objective,
                n_trials=self.config.optuna.n_trials,
                timeout=self.config.optuna.time_limit_per_trial,
            )

            results = self._process_results(study)
            if results and results.best_params and self.test_dataset:
                self._evaluate_best_model_on_test_set(results.best_params)
            return results

        except Exception as e:
            self.logger.exception(f"Optimization process failed: {e}")
            return None

    def _process_results(self, study: optuna.Study) -> OptimizationResults | None:
        if not study.best_trial:
            self.logger.warning("No completed trials in the study.")
            return None

        results = OptimizationResults(
            best_value=study.best_trial.value,
            best_params=study.best_trial.params,
            study_name=study.study_name,
            n_trials=len(study.trials),
        )
        results_path = self.study_dir / "best_params.json"
        results.save(results_path)
        self.logger.info(f"Optimization results saved to {results_path}")
        self.logger.info(
            f"Best trial #{study.best_trial.number}: Value={results.best_value}, Params={results.best_params}"
        )
        return results

    def _evaluate_best_model_on_test_set(self, best_hparams: dict[str, Any]):
        """
        Loads the best model from the optimization study, evaluates it on the held-out
        test set, and logs the final performance metrics.
        """
        if self.test_dataset is None or len(self.test_dataset) == 0:
            self.logger.error("Test dataset is empty. Skipping final evaluation.")
            return

        self.logger.info("--- Starting Final Evaluation on Test Set ---")
        try:
            model_class = self._get_model_class()
            study = optuna.load_study(
                study_name=self.config.optuna.study_name,
                storage=self.config.optuna_storage_url,
            )
            best_trial_number = study.best_trial.number

            # Construct the path to the best model saved by EarlyStopping across all folds
            best_trial_base_name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{best_trial_number}"
            best_model_path = (
                Path(self.config.paths.savepoints_dir)
                / f"{best_trial_base_name}_best_model.pth"
            )

            if not best_model_path.exists():
                self.logger.error(
                    f"Best model checkpoint not found at: {best_model_path}. Cannot perform final test evaluation."
                )
                return

            self.logger.info(
                f"Loading best model from {best_model_path} for final testing."
            )

            # Instantiate the model with the best hyperparameters
            final_model = model_class(
                field_inputs_n=len(self.config.data.input_fields),
                scalar_inputs_n=len(self.config.data.input_scalars),
                field_outputs_n=len(self.config.data.output_fields),
                scalar_outputs_n=len(self.config.data.output_scalars),
                **best_hparams,
            ).to(self.config.device)

            # Load the state dict robustly
            checkpoint = None
            try:
                checkpoint = torch.load(
                    best_model_path, map_location=self.config.device, weights_only=True
                )
            except Exception:
                self.logger.warning(
                    "Loading best model with weights_only=True failed. Retrying with weights_only=False."
                )
                checkpoint = torch.load(
                    best_model_path, map_location=self.config.device, weights_only=False
                )

            if isinstance(checkpoint, dict):
                checkpoint.pop("_metadata", None)
            final_model.load_state_dict(checkpoint)

            # Create Criterion for evaluation
            criterion = PhysicsInformedLoss(
                input_vars=self.config.data.inputs,
                output_vars=self.config.data.outputs,
                config=self.config,
                dataset=self.full_dataset_for_stats,
                use_physics_loss=best_hparams.get(
                    "use_physics_loss", self.config.training.use_physics_loss
                ),
                normalize_output=best_hparams.get(
                    "normalize_output", self.config.data.normalize_output
                ),
            )

            # Create a temporary Trainer for the validation logic
            eval_trainer = Trainer(
                model=final_model,
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=self.config.device,
                accumulation_steps=1,
                config=self.config,
                full_dataset=self.full_dataset_for_stats,
                hparams=best_hparams,
            )

            # Create Test DataLoader
            g_test = torch.Generator().manual_seed(self.config.seed + 1)
            eval_batch_size = best_hparams.get(
                "batch_size", self.config.training.batch_size
            )
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=g_test,
            )

            self.logger.info(f"Evaluating on {len(self.test_dataset)} test samples...")
            test_metrics = eval_trainer.validate(
                test_loader,
                name=f"{self.config.optuna.study_name}_Final_Test",
                step=-1,
                fold_n=-1,
            )

            test_metrics_str = ", ".join(
                [f"{k}={v:.4f}" for k, v in test_metrics.items()]
            )
            self.logger.info(f"Final Test Set Metrics: {test_metrics_str}")

            # Save test metrics to a JSON file
            test_results_path = self.study_dir / "test_set_results.json"
            test_summary = {
                "best_hyperparameters": best_hparams,
                "test_metrics": {
                    k: float(v) for k, v in test_metrics.items()
                },  # Ensure JSON serializable
                "best_cv_value": study.best_value,
                "best_trial_number": best_trial_number,
            }
            test_results_path.write_text(json.dumps(test_summary, indent=2))
            self.logger.info(f"Test set results saved to {test_results_path}")

            # Log final results to WandB
            if self.config.logging.use_wandb:
                self._log_final_results_to_wandb(study, best_hparams, test_metrics)

        except Exception as e:
            self.logger.exception(
                f"An error occurred during final test set evaluation: {e}"
            )
        finally:
            empty_cache()

    def _log_final_results_to_wandb(
        self,
        study: optuna.Study,
        best_hparams: dict[str, Any],
        test_metrics: dict[str, float],
    ):
        """Logs the final evaluation results to a new WandB run."""
        self.logger.info("Logging final test results to WandB.")
        run_name = f"{self.config.optuna.study_name}_Final_Test_Eval"
        try:
            with wandb.init(
                project=self.config.logging.wandb_project,
                name=run_name,
                group=self.config.optuna.study_name,
                job_type="Final_Evaluation",
                config=best_hparams,
                notes=f"Final test set evaluation for study {self.config.optuna.study_name}.",
                reinit=True,
            ) as run:
                # Log test metrics with a distinct prefix
                wandb_test_metrics = {
                    f"Final_Test/{k}": v for k, v in test_metrics.items()
                }
                run.log(wandb_test_metrics)

                # Update summary for easy access in the WandB UI
                run.summary.update(
                    {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                )
                run.summary["Best_CV_Value"] = study.best_value
                run.summary["Best_Trial_Number"] = study.best_trial.number

            self.logger.info(
                f"Successfully logged final test results to WandB: {run_name}"
            )
        except Exception as wandb_e:
            self.logger.error(f"Failed to log final test results to WandB: {wandb_e}")
