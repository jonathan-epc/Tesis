# main.py
import argparse
import json
import os
import sys

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

import optuna
import torch
from torch.cuda import empty_cache
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split

from config import get_config

from modules.data import HDF5Dataset
from modules.loss import PhysicsInformedLoss
from modules.models import FNOnet
from modules.training import Trainer, cross_validation_procedure
from modules.utils import (
    EarlyStopping,
    is_jupyter,
    seed_worker,
    set_seed,
    setup_experiment,
    setup_logger,
)

import wandb

class OptimizerMode(Enum):
    HYPERTUNING = auto()
    TRAINING = auto()
    REPEAT = auto()

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
        # Initialize datasets to None, they will be loaded in run_optimization
        self.full_dataset_for_stats = None
        self.train_val_dataset = None
        self.test_dataset = None

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
                    # Handle integer steps correctly, especially step=2
                    if space["type"] == "int" and space.get("step", 1) > 1:
                        step = space["step"]
                        low = space["low"]
                        if step == 2:  # Keep specific even logic if needed
                            low_adj = low + (low % 2)  # Start from first even >= low
                            high_adj = space["high"] - (
                                space["high"] % 2
                            )  # End at last even <= high
                            if low_adj > high_adj:
                                high_adj = low_adj  # Handle edge case
                            hparams[param] = suggest(
                                param, low_adj, high_adj, step=step
                            )
                        else:
                            # General step for integers
                            num_steps = (space["high"] - low) // step
                            high_adj = low + num_steps * step
                            hparams[param] = suggest(param, low, high_adj, step=step)

                    else:
                        hparams[param] = suggest(
                            param,
                            space["low"],
                            space["high"],
                            log=space.get("log", False),
                            # step=space.get("step") # Pass step for float if present
                        )
                        # For float with step, ensure it's handled if needed (often not needed like int step)
                        if space["type"] == "float" and space.get("step"):
                            # Optuna's suggest_float doesn't have a direct step like suggest_int
                            # If discrete float steps are needed, consider suggest_discrete_uniform
                            # or adjust the value after suggestion
                            pass  # Assuming continuous float suggestion is sufficient

            except AttributeError as e:
                raise ValueError(
                    f"Invalid hyperparameter space type for {param}: {space['type']}"
                ) from e
            except KeyError as e:
                raise ValueError(
                    f"Missing required field in hyperparameter space for {param}"
                ) from e

        # Log the suggested hyperparameters for inspection
        self.logger.debug(
            "Suggested hyperparameters for trial %s: %s", trial.number, hparams
        )
        return hparams

    def _get_model_class(self):
        try:
            return globals()[self.config.model.class_name]
        except KeyError:
            raise ValueError(
                f"Model class {self.config.model.class_name} not found in global scope."
            )

    def objective(self, trial):
        hparams = self.create_hparams(trial)
        model_class = self._get_model_class()
        name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{trial.number}"

        self.logger.info(f"Starting trial {trial.number} with parameters: {hparams}")
        print(f"Starting trial {trial.number} with parameters: {hparams}")

        try:
            # Ensure datasets are loaded before calling objective
            if self.train_val_dataset is None or self.full_dataset_for_stats is None:
                self.logger.error(
                    "Datasets not loaded before calling objective function."
                )
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

            # Save trial results (including the CV loss)
            trial_results_path = self.study_dir / f"trial_{trial.number}.json"
            try:
                trial_results = {
                    "number": trial.number,
                    "params": hparams,
                    "value": avg_cv_loss,
                    "config": self.config.to_dict(),
                }
                trial_results_path.write_text(
                    json.dumps(trial_results, indent=2, default=str)
                )
            except Exception as json_e:
                self.logger.error(
                    f"Failed to serialize trial results to JSON for trial {trial.number}: {json_e}"
                )

            return avg_cv_loss
        except KeyboardInterrupt:
            self.logger.info(f"Trial {trial.number} interrupted by user")
            raise optuna.exceptions.TrialPruned("User interruption")
        except optuna.exceptions.TrialPruned as e:
            self.logger.warning(f"Trial {trial.number} pruned: {e}")
            raise e
        except Exception as e:
            self.logger.exception(f"Trial {trial.number} failed with error: {str(e)}")
            self.logger.debug("Trial parameters: %s", hparams)
            self.logger.debug("Stack trace: %s", traceback.format_exc())
            # Pruning on failure prevents Optuna from getting stuck on a bad parameter set
            raise optuna.exceptions.TrialPruned(f"Trial failed: {str(e)}")
        finally:
            empty_cache()

    def _load_and_split_data(self):
        """Loads the HDF5 dataset and performs the train/val + test split."""
        self.logger.info("Loading and splitting dataset...")
        try:
            # Load the full dataset (needed for stats and splitting)
            self.full_dataset_for_stats = HDF5Dataset.from_config(
                self.config, file_path=self.config.data.file_name
            )
            self.logger.info(
                f"Full dataset loaded with {len(self.full_dataset_for_stats)} samples."
            )

            # Perform the train/val + test split
            test_frac = self.config.training.test_frac
            if not (0 < test_frac < 1):
                raise ValueError(
                    f"Invalid test_frac: {test_frac}. Must be between 0 and 1."
                )

            test_size = int(test_frac * len(self.full_dataset_for_stats))
            train_val_size = len(self.full_dataset_for_stats) - test_size

            if (
                train_val_size <= 0 or test_size < 0
            ):  # test_size can be 0 if test_frac is very small
                self.logger.error(
                    f"Dataset split resulted in invalid sizes. Train/Val={train_val_size}, Test={test_size}. Check total samples and test_frac."
                )
                raise ValueError("Invalid dataset split sizes.")

            # Use a generator for reproducible splitting
            g_split = torch.Generator().manual_seed(self.config.seed)
            self.train_val_dataset, self.test_dataset = random_split(
                self.full_dataset_for_stats,
                [train_val_size, test_size],
                generator=g_split,
            )
            self.logger.info(
                f"Dataset split: Train/Validation={len(self.train_val_dataset)}, Test={len(self.test_dataset)}"
            )

        except FileNotFoundError:
            self.logger.exception(
                f"Dataset file not found at {self.config.data.file_name}"
            )
            raise
        except Exception as e:
            self.logger.exception(f"Error during dataset loading or splitting: {e}")
            raise

    def run_optimization(self) -> Optional[OptimizationResults]:
        try:
            self._validate_config()
            self._load_and_split_data()  # Load data before starting study

            # --- Optuna Study ---
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(
                study_name=self.config.optuna.study_name,
                load_if_exists=True,
                direction="minimize",
                storage=self.config.optuna.storage,
                pruner=pruner,
            )

            # Add config as a study-level attribute
            study.set_user_attr("config", self.config.to_dict())

            self.logger.info(
                f"Starting/Resuming Optuna study '{study.study_name}' with {self.config.optuna.n_trials} trials."
            )
            study.optimize(
                self.objective,
                n_trials=self.config.optuna.n_trials,
            )  # Add timeout if defined in config

            # --- Process Results ---
            if not study.trials:
                self.logger.warning("No trials completed in the study.")
                return None

            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                self.logger.warning("No trials completed successfully.")
                # Optionally log pruned/failed trials
                pruned_count = len(
                    [
                        t
                        for t in study.trials
                        if t.state == optuna.trial.TrialState.PRUNED
                    ]
                )
                failed_count = len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                )
                self.logger.info(
                    f"Trial summary: Pruned={pruned_count}, Failed={failed_count}"
                )
                return None

            self.logger.info(
                f"Optimization finished. Best trial: #{study.best_trial.number}"
            )
            results = self._create_results(study)
            
            # --- !! AGGRESSIVE CLEANUP HERE !! ---
            self.logger.info("Attempting explicit GPU cleanup immediately after Optuna study...")
            # Optional: Delete large objects if safe
            # if hasattr(self, 'train_val_dataset'): del self.train_val_dataset
            import gc
            gc.collect()
            self.logger.info("Python garbage collection triggered.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("torch.cuda.empty_cache() called successfully.")
                    self.logger.info(f"GPU Memory Allocated (Post-Cleanup): {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
                    self.logger.info(f"GPU Memory Reserved (Post-Cleanup): {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
                else:
                    self.logger.info("CUDA not available, skipping empty_cache.")
            except Exception as e_cl:
                self.logger.warning(f"Exception during GPU cleanup: {e_cl}")
            # --- END OF CLEANUP ---
            
            # --- Final Test Set Evaluation ---
            if results and results.best_params:
                self.logger.info(
                    "Performing final evaluation on test set with best hyperparameters..."
                )
                self._evaluate_best_model_on_test_set(
                    results.best_params, self.test_dataset
                )
            else:
                self.logger.warning(
                    "Could not determine best parameters or no successful trials. Skipping final test set evaluation."
                )

            return results

        except KeyboardInterrupt:
            self.logger.info("Optimization process interrupted by user.")
            # Optionally save current study state or results here
            return None  # Or return partial results if possible
        except Exception as e:
            self.logger.exception(
                f"An error occurred during the optimization process: {e}"
            )
            # Re-raise or handle as appropriate
            raise

    def _create_results(self, study: optuna.Study) -> Optional[OptimizationResults]:
        """Create and save optimization results"""
        try:
            # Ensure there's a best trial (it might not exist if all trials failed/pruned)
            if study.best_trial is None:
                self.logger.warning(
                    "No best trial found in the study. Cannot create results."
                )
                return None

            best_trial = study.best_trial
            results = OptimizationResults(
                best_value=best_trial.value,
                best_params=best_trial.params,
                study_name=study.study_name,
                n_trials=len(study.trials),
            )

            results_path = self.study_dir / "results.json"
            results.save(results_path)
            self.logger.info(f"Optimization results saved to {results_path}")

            self.logger.info(f"--- Optuna Study Summary ---")
            self.logger.info(f"Study Name: {study.study_name}")
            self.logger.info(f"Direction: {study.direction}")
            self.logger.info(f"Number of Trials: {len(study.trials)}")
            self.logger.info(f"Best Trial Number: {best_trial.number}")
            self.logger.info(f"Best Value ({study.direction}): {best_trial.value:.6f}")
            self.logger.info(f"Best Hyperparameters:")
            for key, value in best_trial.params.items():
                self.logger.info(f"  {key}: {value}")

            return results
        except Exception as e:
            self.logger.exception(f"Failed to create or save optimization results: {e}")
            return None

    def _evaluate_best_model_on_test_set(self, best_hparams, test_dataset):
        """Loads the best model (found during CV) and evaluates it on the test set."""
        if test_dataset is None or len(test_dataset) == 0:
            self.logger.error(
                "Test dataset is empty or not provided. Skipping final evaluation."
            )
            return

        try:
            model_class = self._get_model_class()

            # Determine the path to the best model checkpoint
            # This relies on EarlyStopping having saved the best model across all folds for the best trial
            # Find the best trial number first
            study = optuna.load_study(
                study_name=self.config.optuna.study_name,
                storage=self.config.optuna.storage,
            )
            if study.best_trial is None:
                self.logger.error(
                    "No best trial found in the study. Cannot determine best model path."
                )
                return
            best_trial_number = study.best_trial.number

            # Construct the name used during the best trial's cross-validation run
            best_trial_base_name = f"{self.config.optuna.study_name}_{self.config.model.architecture}_trial_{best_trial_number}"
            # Path where EarlyStopping saved the overall best model for that trial
            best_model_path = (
                Path("savepoints") / f"{best_trial_base_name}_best_model.pth"
            )

            if not best_model_path.exists():
                self.logger.error(
                    f"Best model checkpoint not found at expected path: {best_model_path}. This checkpoint should have been saved by EarlyStopping during the best trial's cross-validation. Cannot perform final test evaluation."
                )
                # Potential reasons: Early stopping never triggered save, path mismatch, file deleted.
                return

            self.logger.info(
                f"Loading best model checkpoint from {best_model_path} for final testing."
            )

            # Instantiate the model with the best hyperparameters
            # Ensure dataset statistics are available if needed for model init (e.g., input/output dims)
            if self.full_dataset_for_stats is None:
                # Reload if necessary, though it should be loaded already
                self.logger.warning(
                    "Full dataset for stats not available. Reloading to determine model dims."
                )
                # Avoid reloading if possible, pass dims directly or ensure dataset persists
                temp_full_dataset = HDF5Dataset.from_config(
                    self.config, file_path=self.config.data.file_name
                )
            else:
                temp_full_dataset = self.full_dataset_for_stats  # Use existing

            final_model = model_class(
                field_inputs_n=len(
                    [
                        p
                        for p in self.config.data.inputs
                        if p in temp_full_dataset.non_scalar_input_indices
                    ]
                ),  # Use actual dataset dims
                scalar_inputs_n=len(
                    [
                        p
                        for p in self.config.data.inputs
                        if p in temp_full_dataset.scalar_input_indices
                    ]
                ),
                field_outputs_n=len(
                    [
                        p
                        for p in self.config.data.outputs
                        if p in temp_full_dataset.non_scalar_output_indices
                    ]
                ),
                scalar_outputs_n=len(
                    [
                        p
                        for p in self.config.data.outputs
                        if p in temp_full_dataset.scalar_output_indices
                    ]
                ),
                **best_hparams,  # Use best hyperparameters found by Optuna
            ).to(self.config.device)

            # Load the state dict
            try:
                # Try loading with weights_only=True first for security
                checkpoint = torch.load(
                    best_model_path, map_location=self.config.device, weights_only=True
                )
                final_model.load_state_dict(checkpoint)
                self.logger.info("Loaded best model state dict (weights_only=True).")
            except Exception as e_load:
                self.logger.warning(
                    f"Loading best model with weights_only=True failed ({e_load}). Trying weights_only=False."
                )
                try:
                    # Fallback to weights_only=False (less secure, use with caution)
                    checkpoint = torch.load(
                        best_model_path,
                        map_location=self.config.device,
                        weights_only=False,
                    )
                    final_model.load_state_dict(checkpoint)
                    self.logger.info(
                        "Loaded best model state dict (weights_only=False)."
                    )
                except Exception as e_fallback:
                    self.logger.exception(
                        f"Fallback loading (weights_only=False) also failed: {e_fallback}. Cannot evaluate model."
                    )
                    return

            # Create Criterion for evaluation (re-use PhysicsInformedLoss or a simpler one if physics not needed for test eval)
            # Ensure the full dataset is available for the criterion if it needs stats
            if self.full_dataset_for_stats is None:
                self.logger.error(
                    "Full dataset required for criterion initialization is missing."
                )
                return

            criterion = PhysicsInformedLoss(
                input_vars=self.config.data.inputs,
                output_vars=self.config.data.outputs,
                config=self.config,
                dataset=self.full_dataset_for_stats,
                # Use .get() with defaults from config for robustness
                use_physics_loss=best_hparams.get(
                    "use_physics_loss", self.config.training.use_physics_loss
                ),
                normalize_output=best_hparams.get(
                    "normalize_output", self.config.data.normalize_output
                ),
                normalize_input=self.config.data.normalize_input,
            )
            self.logger.info(
                f"Evaluation criterion initialized with: "
                f"UsePhysics={criterion.use_physics_loss}, "
                f"NormalizeOutput={criterion.normalize_output}"
            )

            # Create a temporary Trainer instance purely for the validation logic
            # No optimizer/scheduler needed for evaluation
            eval_trainer = Trainer(
                model=final_model,  # Use the loaded final_model
                criterion=criterion,
                optimizer=None,  # No optimizer needed for eval
                scheduler=None,  # No scheduler needed for eval
                scaler=None,  # No scaler needed for eval
                device=self.config.device,
                accumulation_steps=1,  # Not relevant for eval
                config=self.config,
                full_dataset=self.full_dataset_for_stats,
                hparams=best_hparams,  # Pass the best hyperparameters
            )

            # Create Test DataLoader
            # Use a consistent generator seed if reproducibility is desired for test evaluation
            g_test = torch.Generator().manual_seed(
                self.config.seed + 1
            )  # Use a different seed than train/val split

            # Determine batch size: Use best trial's batch size or a default evaluation batch size
            eval_batch_size = best_hparams.get(
                "batch_size", self.config.training.batch_size
            )
            # Optionally override with a specific eval batch size from config if defined
            # eval_batch_size = self.config.training.get("eval_batch_size", eval_batch_size)

            test_loader = DataLoader(
                test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,  # No shuffling for test set evaluation
                num_workers=self.config.training.num_workers,
                pin_memory=False,  # Keep as False based on user feedback
                worker_init_fn=seed_worker,  # Seed workers if needed
                generator=g_test,  # Use generator for worker seeding consistency
            )

            self.logger.info(
                f"Starting final evaluation on test set ({len(test_dataset)} samples)..."
            )

            # Perform validation call on the test set
            # The 'validate' method calculates metrics without training
            test_metrics = eval_trainer.validate(
                test_loader,
                name=f"{self.config.optuna.study_name}_Final_Test",  # Unique name for logging/outputs
                step=-1,  # Indicate this is a final evaluation step
                fold_n=-1,  # Indicate not part of a fold
            )
            # test_loss = test_metrics.get("loss", float('inf')) # Get overall loss if needed

            test_metrics_str = ", ".join(
                f"{k.replace('_', ' ').title()}={v:.4f}"
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            )
            self.logger.info(f"--- Final Test Set Evaluation Results ---")
            self.logger.info(f"Test Set Metrics: {test_metrics_str}")
            self.logger.info(f"Evaluated with Best Hyperparameters: {best_hparams}")

            # --- Save Test Metrics ---
            test_results_path = self.study_dir / "test_set_results.json"
            test_summary = {
                "best_hyperparameters": best_hparams,
                "test_metrics": test_metrics,
                "best_cv_value": study.best_value,  # Include the CV score that led to this model
                "best_trial_number": best_trial_number,
            }
            try:
                test_results_path.write_text(
                    json.dumps(test_summary, indent=2, default=str)
                )
                self.logger.info(
                    f"Test set evaluation results saved to {test_results_path}"
                )
            except Exception as json_e:
                self.logger.error(f"Failed to serialize test results to JSON: {json_e}")

            # --- Log Final Test Results to WandB ---
            if self.config.logging.use_wandb:
                self.logger.info("Logging final test results to WandB.")
                run_name = f"{self.config.optuna.study_name}_Final_Test_Eval"
                try:
                    wandb.init(
                        project=self.config.logging.wandb_project,
                        name=run_name,
                        group=self.config.optuna.study_name,  # Group with the optimization study
                        job_type="Final_Evaluation",
                        config=best_hparams,  # Log the hyperparameters used
                        notes=f"Final test set evaluation for study {self.config.optuna.study_name}. Best trial: {best_trial_number}, CV score: {study.best_value:.4f}",
                        reinit=True,  # Ensure a new run is initialized
                    )
                    # Log test metrics with a distinct prefix
                    wandb_test_metrics = {
                        f"Final_Test/{k.replace('_', ' ').title()}": v
                        for k, v in test_metrics.items()
                    }
                    wandb.log(wandb_test_metrics)

                    # Update summary for easy access in WandB UI
                    wandb.summary.update(
                        {
                            f"Final_Test_Summary/{k.replace('_', ' ').title()}": v
                            for k, v in test_metrics.items()
                        }
                    )
                    wandb.summary["Best_CV_Value"] = (
                        study.best_value
                    )  # Add the best CV score to summary
                    wandb.summary["Best_Trial_Number"] = best_trial_number

                    wandb.finish()
                    self.logger.info("Successfully logged final test results to WandB.")
                except Exception as wandb_e:
                    self.logger.error(
                        f"Failed to log final test results to WandB: {wandb_e}"
                    )

        except Exception as e:
            self.logger.exception(
                f"An unexpected error occurred during final test set evaluation: {e}"
            )
        finally:
            empty_cache()  # Clean up memory after evaluation

class ModelTrainer:
    """Handles single training runs with fixed hyperparameters"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()

    def _get_default_hparams(self) -> Dict[str, Any]:
        """Extracts default hyperparameters from config for model and training."""
        hparams = {}
        # Combine model and training parameters, giving precedence to model if overlaps occur
        # Use .get() to handle potentially missing sections gracefully if config structure varies
        model_params = self.config.model.dict() if hasattr(self.config, "model") else {}
        training_params = (
            self.config.training.dict() if hasattr(self.config, "training") else {}
        )
        data_params = self.config.data.dict() if hasattr(self.config, "data") else {}

        # Merge dictionaries - later dictionaries overwrite earlier ones if keys conflict
        hparams.update(training_params)
        hparams.update(model_params)

        # Explicitly add data params that might be needed as hparams (like normalize_output)
        # Check if these are actually used as hyperparameters during model init or training setup
        if "normalize_output" in data_params:
            hparams["normalize_output"] = data_params["normalize_output"]

        # Remove parameters that are definitely not hyperparameters
        # Example: 'num_workers', 'kfolds', 'test_frac' are training setup, not model hparams
        hparams.pop("num_workers", None)
        hparams.pop("kfolds", None)
        hparams.pop("test_frac", None)
        hparams.pop("validation_frac", None)
        hparams.pop("early_stopping_patience", None)
        hparams.pop("early_stopping_delta", None)
        hparams.pop("clip_grad_value", None)
        hparams.pop("pretrained_model_name", None)
        hparams.pop("time_limit", None)
        hparams.pop("num_epochs", None)

        # Example: You might want to ensure essential hparams like learning_rate are present
        if "learning_rate" not in hparams:
            self.logger.warning(
                "learning_rate not found in combined hparams, check config."
            )

        self.logger.debug(f"Default hyperparameters for single training run: {hparams}")
        return hparams

    def train(self) -> Optional[float]:
        """Loads data, splits it, and runs cross_validation_procedure for a single run."""
        self.logger.info(
            "Starting single training run with default hyperparameters from config."
        )
        try:
            # --- 1. Load and Split Data ---
            self.logger.info("Loading and splitting data for single training run...")
            full_dataset_for_stats = HDF5Dataset.from_config(
                self.config, file_path=self.config.data.file_name
            )
            self.logger.info(
                f"Full dataset loaded with {len(full_dataset_for_stats)} samples."
            )

            test_frac = self.config.training.test_frac
            if not (0 < test_frac < 1):
                raise ValueError(
                    f"Invalid test_frac: {test_frac}. Must be between 0 and 1."
                )

            test_size = int(test_frac * len(full_dataset_for_stats))
            train_val_size = len(full_dataset_for_stats) - test_size

            if train_val_size <= 0 or test_size < 0:
                self.logger.error(
                    f"Dataset split resulted in invalid sizes. Train/Val={train_val_size}, Test={test_size}."
                )
                raise ValueError("Invalid dataset split sizes.")

            g_split = torch.Generator().manual_seed(self.config.seed)
            train_val_dataset, test_dataset = random_split(
                full_dataset_for_stats, [train_val_size, test_size], generator=g_split
            )
            self.logger.info(
                f"Dataset split: Train/Validation={len(train_val_dataset)}, Test={len(test_dataset)}"
            )

            # --- 2. Get Model and Hyperparameters ---
            model_class = globals()[
                self.config.model.class_name
            ]  # Assumes model class is in global scope
            hparams = self._get_default_hparams()

            # --- 3. Run Cross-Validation Procedure ---
            # Note: cross_validation_procedure returns the average validation loss across folds.
            # It also handles saving the best model based on validation performance within the CV run.
            avg_cv_loss = cross_validation_procedure(
                name=self.config.model.name,  # Use the model name from config
                model_class=model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                config=self.config,
                train_val_dataset=train_val_dataset,  # Pass the correct dataset split
                full_dataset_for_stats=full_dataset_for_stats,  # Pass the full dataset
                is_sweep=False,  # Not part of an Optuna sweep
                trial=None,  # No Optuna trial involved
            )

            self.logger.info(
                f"Training run finished. Average cross-validation loss: {avg_cv_loss:.4f}"
            )

            # --- 4. Optional: Evaluate on Test Set ---
            # You might want to load the best model saved during the CV run
            # and evaluate it on the 'test_dataset' similar to _evaluate_best_model_on_test_set
            self.logger.info(
                "Performing evaluation on the test set using the best model from the training run..."
            )
            best_model_path = (
                Path("savepoints") / f"{self.config.model.name}_best_model.pth"
            )
            if best_model_path.exists():
                # Instantiate model with default hparams again
                eval_model = model_class(
                    field_inputs_n=len(
                        [
                            p
                            for p in self.config.data.inputs
                            if p in full_dataset_for_stats.non_scalar_input_indices
                        ]
                    ),
                    scalar_inputs_n=len(
                        [
                            p
                            for p in self.config.data.inputs
                            if p in full_dataset_for_stats.scalar_input_indices
                        ]
                    ),
                    field_outputs_n=len(
                        [
                            p
                            for p in self.config.data.outputs
                            if p in full_dataset_for_stats.non_scalar_output_indices
                        ]
                    ),
                    scalar_outputs_n=len(
                        [
                            p
                            for p in self.config.data.outputs
                            if p in full_dataset_for_stats.scalar_output_indices
                        ]
                    ),
                    **hparams,
                ).to(self.config.device)

                # Load state dict
                try:
                    checkpoint = torch.load(
                        best_model_path,
                        map_location=self.config.device,
                        weights_only=True,
                    )
                    eval_model.load_state_dict(checkpoint)
                except Exception:
                    self.logger.warning(
                        "Loading best model (training mode) with weights_only=True failed. Trying weights_only=False."
                    )
                    checkpoint = torch.load(
                        best_model_path,
                        map_location=self.config.device,
                        weights_only=False,
                    )
                    eval_model.load_state_dict(checkpoint)

                # Create criterion and eval trainer
                criterion = PhysicsInformedLoss(
                    input_vars=self.config.data.inputs,
                    output_vars=self.config.data.outputs,
                    config=self.config,
                    dataset=full_dataset_for_stats,
                    use_physics_loss=hparams.get(
                        "use_physics_loss", self.config.training.use_physics_loss
                    ),
                    normalize_output=hparams.get(
                        "normalize_output", self.config.data.normalize_output
                    ),
                    normalize_input=self.config.data.normalize_input,
                )
                self.logger.info(
                    f"Single run evaluation criterion initialized with: "
                    f"UsePhysics={criterion.use_physics_loss}, "
                    f"NormalizeOutput={criterion.normalize_output}"
                )

                # Pass the hparams dictionary to the Trainer
                eval_trainer = Trainer(
                    model=eval_model,  # Use the loaded eval_model
                    criterion=criterion,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    device=self.config.device,
                    accumulation_steps=1,
                    config=self.config,
                    full_dataset=full_dataset_for_stats,
                    hparams=hparams,  # Pass the default hyperparameters used for this run
                )

                # Test loader
                g_test = torch.Generator().manual_seed(self.config.seed + 1)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=hparams.get(
                        "batch_size", self.config.training.batch_size
                    ),
                    shuffle=False,
                    num_workers=self.config.training.num_workers,
                    pin_memory=False,
                    worker_init_fn=seed_worker,
                    generator=g_test,
                )

                # Validate on test set
                test_metrics = eval_trainer.validate(
                    test_loader,
                    name=f"{self.config.model.name}_SingleRun_Test",
                    step=-1,
                    fold_n=-1,
                )
                test_metrics_str = ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in test_metrics.items()
                    if isinstance(v, (float, int))
                )
                self.logger.info(
                    f"Final Test Set Evaluation (Single Run): {test_metrics_str}"
                )

                # Log to WandB if enabled
                if self.config.logging.use_wandb:
                    run_name = f"{self.config.model.name}_Single_Run_Test_Eval"
                    wandb.init(
                        project=self.config.logging.wandb_project,
                        name=run_name,
                        group=self.config.model.name,
                        job_type="Single_Run_Evaluation",
                        config=hparams,
                        reinit=True,
                    )
                    wandb.log({f"Final_Test/{k}": v for k, v in test_metrics.items()})
                    wandb.summary.update(
                        {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                    )
                    wandb.summary["Avg_CV_Loss"] = avg_cv_loss
                    wandb.finish()

            else:
                self.logger.warning(
                    f"Best model checkpoint not found at {best_model_path}. Skipping test set evaluation for single training run."
                )

            return avg_cv_loss  # Return the average CV loss from the run

        except FileNotFoundError:
            self.logger.exception(
                f"Dataset file not found: {self.config.data.file_name}"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error during single training run: {e}")
            return None
        finally:
            empty_cache()


def repeat_trial_from_study(trial_id: int, config) -> Optional[float]:
    """
    Load an Optuna study, retrieve a specific trial, load/split data,
    and re-run the training procedure using the trial's parameters.
    """
    logger = setup_logger()  # Get logger instance
    logger.info(
        f"Attempting to repeat Optuna trial {trial_id} from study '{config.optuna.study_name}'."
    )

    try:
        # --- 1. Load Study and Trial ---
        study = optuna.load_study(
            study_name=config.optuna.study_name, storage=config.optuna.storage
        )
        trial_to_repeat = next((t for t in study.trials if t.number == trial_id), None)

        if trial_to_repeat is None:
            logger.error(
                f"Trial number {trial_id} not found in study '{config.optuna.study_name}'."
            )
            raise ValueError(f"Trial {trial_id} not found.")
        elif trial_to_repeat.state != optuna.trial.TrialState.COMPLETE:
            logger.warning(
                f"Trial {trial_id} was not completed successfully (State: {trial_to_repeat.state}). Repeating it may reproduce failure or pruning."
            )
            # Allow repeating non-completed trials, but log a warning.

        hparams = trial_to_repeat.params
        logger.info(f"Found trial {trial_id} with parameters: {hparams}")

        # --- 2. Load and Split Data ---
        logger.info(f"Loading and splitting data for repeating trial {trial_id}...")
        full_dataset_for_stats = HDF5Dataset.from_config(
            config, file_path=config.data.file_name
        )
        logger.info(f"Full dataset loaded with {len(full_dataset_for_stats)} samples.")

        test_frac = config.training.test_frac
        if not (0 < test_frac < 1):
            raise ValueError(
                f"Invalid test_frac: {test_frac}. Must be between 0 and 1."
            )

        test_size = int(test_frac * len(full_dataset_for_stats))
        train_val_size = len(full_dataset_for_stats) - test_size

        if train_val_size <= 0 or test_size < 0:
            logger.error(
                f"Dataset split resulted in invalid sizes. Train/Val={train_val_size}, Test={test_size}."
            )
            raise ValueError("Invalid dataset split sizes.")

        g_split = torch.Generator().manual_seed(
            config.seed
        )  # Use same seed for consistent split
        train_val_dataset, test_dataset = random_split(
            full_dataset_for_stats, [train_val_size, test_size], generator=g_split
        )
        logger.info(
            f"Dataset split: Train/Validation={len(train_val_dataset)}, Test={len(test_dataset)}"
        )

        # --- 3. Get Model Class ---
        model_class = globals()[
            config.model.class_name
        ]  # Assumes model class is in global scope

        # --- 4. Run Cross-Validation Procedure ---
        # Use a distinct name for the repeated run
        name = f"{config.optuna.study_name}_{config.model.architecture}_repeat_trial_{trial_id}"

        avg_cv_loss = cross_validation_procedure(
            name=name,  # Use the specific name for the repeat run
            model_class=model_class,
            kfolds=config.training.kfolds,
            hparams=hparams,  # Use hyperparameters from the specific trial
            config=config,
            train_val_dataset=train_val_dataset,  # Pass the correct dataset split
            full_dataset_for_stats=full_dataset_for_stats,  # Pass the full dataset
            is_sweep=False,  # This is not part of the original sweep
            trial=None,  # Not linked to an active Optuna trial object
        )

        logger.info(
            f"Finished repeating trial {trial_id}. Average cross-validation loss: {avg_cv_loss:.4f}"
        )
        logger.info(f"Original trial {trial_id} value was: {trial_to_repeat.value}")

        # --- 5. Optional: Evaluate on Test Set ---
        # Similar to ModelTrainer.train, evaluate the best model saved during this *repeated* CV run.
        logger.info(
            f"Performing evaluation on the test set using the best model from the repeated trial {trial_id} run..."
        )
        best_model_path = (
            Path("savepoints") / f"{name}_best_model.pth"
        )  # Path based on the repeat run's name
        if best_model_path.exists():
            # Instantiate model with the trial's hparams
            eval_model = model_class(
                field_inputs_n=len(
                    [
                        p
                        for p in config.data.inputs
                        if p in full_dataset_for_stats.non_scalar_input_indices
                    ]
                ),
                scalar_inputs_n=len(
                    [
                        p
                        for p in config.data.inputs
                        if p in full_dataset_for_stats.scalar_input_indices
                    ]
                ),
                field_outputs_n=len(
                    [
                        p
                        for p in config.data.outputs
                        if p in full_dataset_for_stats.non_scalar_output_indices
                    ]
                ),
                scalar_outputs_n=len(
                    [
                        p
                        for p in config.data.outputs
                        if p in full_dataset_for_stats.scalar_output_indices
                    ]
                ),
                **hparams,
            ).to(config.device)

            # Load state dict
            try:
                checkpoint = torch.load(
                    best_model_path, map_location=config.device, weights_only=True
                )
                eval_model.load_state_dict(checkpoint)
            except Exception:
                logger.warning(
                    f"Loading best model (repeat trial {trial_id}) with weights_only=True failed. Trying weights_only=False."
                )
                checkpoint = torch.load(
                    best_model_path, map_location=config.device, weights_only=False
                )
                eval_model.load_state_dict(checkpoint)

                # Create criterion and eval trainer
                criterion = PhysicsInformedLoss(
                    input_vars=config.data.inputs,
                    output_vars=config.data.outputs,
                    config=config,
                    dataset=full_dataset_for_stats,
                    use_physics_loss=hparams.get(
                        "use_physics_loss", config.training.use_physics_loss
                    ),
                    normalize_output=hparams.get(
                        "normalize_output", config.data.normalize_output
                    ),
                    normalize_input=config.data.normalize_input,  # Assuming fixed by config
                )
            logger.info(
                f"Repeat trial evaluation criterion initialized with: "
                f"UsePhysics={criterion.use_physics_loss}, "
                f"NormalizeOutput={criterion.normalize_output}"
            )

            # Pass the loaded trial's hparams dictionary to the Trainer
            eval_trainer = Trainer(
                model=eval_model,  # Use the loaded eval_model
                criterion=criterion,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=config.device,
                accumulation_steps=1,
                config=config,
                full_dataset=full_dataset_for_stats,
                hparams=hparams,  # Pass the trial's hyperparameters
            )

            # Test loader
            g_test = torch.Generator().manual_seed(config.seed + 1)
            test_loader = DataLoader(
                test_dataset,
                batch_size=hparams.get("batch_size", config.training.batch_size),
                shuffle=False,
                num_workers=config.training.num_workers,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=g_test,
            )

            # Validate on test set
            test_metrics = eval_trainer.validate(
                test_loader, name=f"{name}_Test", step=-1, fold_n=-1
            )
            test_metrics_str = ", ".join(
                f"{k}={v:.4f}"
                for k, v in test_metrics.items()
                if isinstance(v, (float, int))
            )
            logger.info(
                f"Final Test Set Evaluation (Repeat Trial {trial_id}): {test_metrics_str}"
            )

            # Log to WandB if enabled
            if config.logging.use_wandb:
                run_name = f"{name}_Test_Eval"
                wandb.init(
                    project=config.logging.wandb_project,
                    name=run_name,
                    group=f"{config.optuna.study_name}_Repeats",
                    job_type="Repeat_Trial_Evaluation",
                    config=hparams,
                    reinit=True,
                )
                wandb.log({f"Final_Test/{k}": v for k, v in test_metrics.items()})
                wandb.summary.update(
                    {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                )
                wandb.summary["Repeated_Trial_ID"] = trial_id
                wandb.summary["Avg_CV_Loss_Repeat"] = avg_cv_loss
                wandb.summary["Avg_CV_Loss_Original"] = trial_to_repeat.value
                wandb.finish()

        else:
            logger.warning(
                f"Best model checkpoint not found at {best_model_path}. Skipping test set evaluation for repeated trial {trial_id}."
            )

        return avg_cv_loss  # Return the average CV loss from the repeated run

    except FileNotFoundError:
        logger.exception(
            f"Dataset file not found during repeat trial: {config.data.file_name}"
        )
        return None
    except optuna.exceptions.OptunaError as e:
        logger.exception(
            f"Optuna error while trying to load study/trial {trial_id}: {e}"
        )
        return None
    except Exception as e:
        logger.exception(f"Error during repetition of trial {trial_id}: {e}")
        return None
    finally:
        empty_cache()


def main():
    # --- 1. Define the Argument Parser ---
    # Define all arguments and their types/defaults as they would be used from the CLI
    parser = argparse.ArgumentParser(
        description="Run model optimization, training, or repeat a trial"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.name.lower() for m in OptimizerMode],
        default="hypertuning",  # Default value if not provided on CLI
        help="Mode of operation: hypertuning, training, repeat",
    )
    parser.add_argument(
        "--trial_id",
        type=int,
        default=None,  # Default to None if not provided
        help="Trial number to repeat (required only in repeat mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )

    # --- 2. Handle Argument Parsing based on Environment ---
    if is_jupyter():
        # --- Jupyter Environment ---
        print("INFO: Running in Jupyter environment. Using manually defined args.")
        args = parser.parse_args(args=[])  # Parse an empty list to get defaults

        # Modify 'args' object here as needed for your specific Jupyter run
        # Example: Test 'repeat' mode
        # args.mode = "repeat"
        # args.trial_id = 197
        # args.config = 'config/special_test_config.yaml'
        print(
            f"INFO: Jupyter using - Mode: {args.mode}, Trial ID: {args.trial_id}, Config: {args.config}"
        )

    else:
        # --- Command Line Environment ---
        # Check if arguments were passed (beyond just the script name)
        # This prevents errors if the script is run with no arguments when some are expected
        # (though defaults usually handle this, it adds robustness)
        if len(sys.argv) > 1:
            print(f"INFO: Running from CLI with args: {' '.join(sys.argv[1:])}")
        else:
            print("INFO: Running from CLI with default args.")
        args = parser.parse_args()

    # --- 3. Load Configuration ---
    try:
        config = get_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {args.config}")
        sys.exit(1)  # Exit if config is crucial
    except Exception as e:
        print(f"ERROR: Failed to load configuration from {args.config}: {e}")
        sys.exit(1)

    # --- 4. Validate Arguments ---
    try:
        mode = OptimizerMode.from_string(args.mode)
        if mode == OptimizerMode.REPEAT and args.trial_id is None:
            # Raise error consistently, let calling environment handle it
            raise ValueError("--trial_id is required when mode is 'repeat'")
    except ValueError as e:
        print(f"ERROR: Invalid arguments: {e}")
        sys.exit(1)  # Exit on validation failure

    # --- 5. Setup Experiment ---
    # Setup logging, directories etc. *after* loading config
    setup_experiment(config)
    set_seed(config.seed)
    logger = setup_logger()  # Logger setup might depend on experiment setup

    logger.info(f"--- Starting Process ---")
    logger.info(f"Mode: {mode.name}")
    if mode == OptimizerMode.REPEAT:
        logger.info(f"Target Trial ID: {args.trial_id}")
    logger.info(f"Configuration Path: {args.config}")

    # --- 6. Execute Selected Mode ---
    try:
        if mode == OptimizerMode.HYPERTUNING:
            logger.info("Initiating HyperparameterOptimizer...")
            optimizer = HyperparameterOptimizer(config)
            results = optimizer.run_optimization()
            if results:
                logger.info(
                    f"Optimization completed. Best CV value: {results.best_value:.6f}"
                )
            else:
                logger.warning("Optimization process did not yield results.")

        elif mode == OptimizerMode.TRAINING:
            logger.info("Initiating ModelTrainer...")
            trainer = ModelTrainer(config)
            avg_cv_loss = trainer.train()
            if avg_cv_loss is not None:
                logger.info(
                    f"Single training run completed. Average CV loss: {avg_cv_loss:.6f}"
                )
            else:
                logger.error("Single training run failed.")

        elif mode == OptimizerMode.REPEAT:
            logger.info(f"Initiating repeat of trial {args.trial_id}...")
            result = repeat_trial_from_study(args.trial_id, config)
            if result is not None:
                logger.info(
                    f"Repeated trial {args.trial_id} finished. Average CV loss: {result:.6f}"
                )
            else:
                logger.error(f"Repeating trial {args.trial_id} failed.")

    except Exception as e:
        logger.exception(f"An unhandled error occurred during execution: {str(e)}")
        # Re-raise the exception to see the full traceback in the console/notebook
        raise

    logger.info("--- Process Completed ---")


if __name__ == "__main__" or is_jupyter():
    main()
