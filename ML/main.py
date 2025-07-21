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
from typing import Any, Dict, Optional, Tuple

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

# --- MODIFICATION 1: Add a new mode ---
class OptimizerMode(Enum):
    HYPERTUNING = auto()
    TRAINING = auto()
    REPEAT = auto()
    FINETUNING = auto()
    CV_RERUN = auto()  # New mode for 5-fold cross-validation rerun

    @classmethod
    def from_string(cls, s: str) -> "OptimizerMode":
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid mode: {s}. Must be one of {[m.name.lower() for m in cls]}"
            )

# ... (The existing classes: OptimizationResults, FineTuningConfigArgs, ModelFineTuner, HyperparameterOptimizer, ModelTrainer remain unchanged)
# ... (Paste the existing classes here)
# START OF EXISTING CLASSES TO PASTE
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

@dataclass
class FineTuningConfigArgs:  # To group fine-tuning specific CLI args
    base_study_name: str
    base_trial_id: int
    finetune_data_file: str
    finetune_config_path: Optional[str] = (
        None  # Path to a YAML for fine-tuning specific training params
    )
    finetune_exp_name_suffix: Optional[str] = None  # e.g., "_finetune_barsa"


class ModelFineTuner:
    # Modify the constructor
    def __init__(
        self,
        config,
        source_model_info: dict,
        target_dataset_path: str,
        finetune_run_name: Optional[str] = None,
    ):
        self.base_config = config
        self.logger = setup_logger()
        self.source_model_info = source_model_info  # Store the whole dictionary
        self.source_study_name = self.source_model_info["study_name"]
        self.source_trial_id = self.source_model_info["trial_number"]
        self.target_dataset_path = target_dataset_path

        if not Path(self.target_dataset_path).exists():
            raise FileNotFoundError(
                f"Target dataset for fine-tuning not found: {self.target_dataset_path}"
            )

        self.pretrained_model_name_stem = f"{self.source_study_name}_{self.base_config.model.architecture}_trial_{self.source_trial_id}"

        target_geom_name = Path(self.target_dataset_path).stem
        self.run_name = (
            finetune_run_name
            or f"{self.pretrained_model_name_stem}_finetuned_on_{target_geom_name}"
        )

        self.logger.info(f"--- Initializing Fine-Tuning ---")
        self.logger.info(f"Source Model Info: {self.source_model_info}")
        self.logger.info(f"Target Dataset: {self.target_dataset_path}")
        self.logger.info(f"Fine-tuning Run Name: {self.run_name}")

    def _load_source_model_config_and_hparams(
        self,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """
        Loads the hparams of the source model and constructs a config object
        reflecting the source model's I/O structure from TRAINED_MODELS_INFO.
        """
        self.logger.info(
            f"Loading Optuna study '{self.source_study_name}' for hparams."
        )

        # --- 1. Load Optuna Hparams ---
        source_optuna_storage_url = f"{self.base_config.optuna.base_storage_path.rstrip('/')}/{self.source_study_name}.db"
        try:
            source_study = optuna.load_study(
                study_name=self.source_study_name, storage=source_optuna_storage_url
            )
            source_trial = next(
                (t for t in source_study.trials if t.number == self.source_trial_id),
                None,
            )
            if source_trial is None:
                raise ValueError(f"Trial {self.source_trial_id} not found.")
            source_hparams = source_trial.params
            self.logger.info(f"Loaded source model hparams: {source_hparams}")
        except Exception as e:
            self.logger.error(
                f"Failed to load hparams for trial {self.source_trial_id} from study {self.source_study_name}: {e}"
            )
            raise

        # --- 2. Construct source config object using authoritative I/O info ---
        from config import Config  # Ensure Pydantic Config class is available

        source_model_config_dict = self.base_config.dict(exclude_unset=True)

        # Override the inputs and outputs with the values from our authoritative dictionary
        source_model_config_dict["data"]["inputs"] = self.source_model_info["inputs"]
        source_model_config_dict["data"]["outputs"] = self.source_model_info["outputs"]

        # Re-create a Pydantic Config object from this modified dictionary.
        try:
            source_model_config_for_io = Config.model_validate(source_model_config_dict)
        except AttributeError:
            source_model_config_for_io = Config(**source_model_config_dict)

        self.logger.info(
            f"Using authoritative I/O for source model '{self.pretrained_model_name_stem}':"
        )
        self.logger.info(f"  Inputs: {source_model_config_for_io.data.inputs}")
        self.logger.info(f"  Outputs: {source_model_config_for_io.data.outputs}")

        # --- 3. Calculate source_io_counts based on this robust config ---
        # Use base_config lists of scalars/non_scalars to categorize the I/O variables
        source_io_counts = {
            "field_inputs_n": len(
                [
                    p
                    for p in source_model_config_for_io.data.inputs
                    if p in self.base_config.data.non_scalars
                ]
            ),
            "scalar_inputs_n": len(
                [
                    p
                    for p in source_model_config_for_io.data.inputs
                    if p in self.base_config.data.scalars
                ]
            ),
            "field_outputs_n": len(
                [
                    p
                    for p in source_model_config_for_io.data.outputs
                    if p in self.base_config.data.non_scalars
                ]
            ),
            "scalar_outputs_n": len(
                [
                    p
                    for p in source_model_config_for_io.data.outputs
                    if p in self.base_config.data.scalars
                ]
            ),
        }
        self.logger.info(f"Calculated source model io_counts: {source_io_counts}")

        # Final check against the error message
        if (
            source_io_counts["field_inputs_n"] + source_io_counts["scalar_inputs_n"]
        ) != 9 and self.source_model_info["study_name"] == "study32dan":
            self.logger.warning(
                f"For 'dan' model, expected 9 input channels from error, but calculated {source_io_counts['field_inputs_n'] + source_io_counts['scalar_inputs_n']}. Check the 'inputs' list in TRAINED_MODELS_INFO."
            )
        if (
            source_io_counts["field_outputs_n"] + source_io_counts["scalar_outputs_n"]
        ) != 3 and self.source_model_info["study_name"] == "study32dan":
            self.logger.warning(
                f"For 'dan' model, expected 3 output channels from error, but calculated {source_io_counts['field_outputs_n'] + source_io_counts['scalar_outputs_n']}. Check the 'outputs' list in TRAINED_MODELS_INFO."
            )

        return source_model_config_for_io, source_hparams, source_io_counts

    def finetune(self) -> Optional[float]:
        self.logger.info(f"Starting fine-tuning process for {self.run_name}")
        try:
            # 1. Load HPARAMS and IO_COUNTS of the SOURCE model
            source_model_original_config, source_hparams, source_io_counts = (
                self._load_source_model_config_and_hparams()
            )

            # 2. Create a new config for this fine-tuning run
            # Start with a copy of the base config, then override
            ft_config = self.base_config.copy(deep=True)

            # Override with source model's relevant architectural and training params
            # We need to be careful here. We want the source model's architecture.
            # Training params like learning rate might be reset or adjusted for fine-tuning.
            # For now, let's assume we use source_hparams for model instantiation.
            # And ft_config.training for new training loop.

            # Set the pretrained_model_name in the config for Trainer to pick up
            ft_config.training.pretrained_model_name = self.pretrained_model_name_stem

            # Set the target dataset for fine-tuning
            ft_config.data.file_name = self.target_dataset_path

            # The inputs/outputs for the *dataset loading* should match the *source model's structure*.
            # This is because the model architecture is fixed.
            ft_config.data.inputs = source_model_original_config.data.inputs
            ft_config.data.outputs = source_model_original_config.data.outputs

            # Potentially adjust learning rate for fine-tuning (e.g., smaller LR)
            # ft_config.training.learning_rate = source_hparams.get('learning_rate', ft_config.training.learning_rate) / 10 # Example
            # Or use a new LR from config if desired for fine-tuning. For simplicity, let's use the one from base_config or allow override via CLI later.
            self.logger.info(
                f"Fine-tuning learning rate: {ft_config.training.learning_rate}"
            )

            # 3. Load target dataset for fine-tuning
            self.logger.info(
                f"Loading and splitting target dataset: {ft_config.data.file_name}"
            )
            # The HDF5Dataset will use ft_config.data.inputs/outputs which are now set to source model's
            full_dataset_for_stats = HDF5Dataset.from_config(
                ft_config, file_path=ft_config.data.file_name
            )
            self.logger.info(
                f"Target dataset loaded with {len(full_dataset_for_stats)} samples."
            )

            test_frac = ft_config.training.test_frac
            test_size = int(test_frac * len(full_dataset_for_stats))
            train_val_size = len(full_dataset_for_stats) - test_size

            g_split = torch.Generator().manual_seed(
                ft_config.seed
            )  # Use consistent seed
            train_val_dataset, test_dataset = random_split(
                full_dataset_for_stats, [train_val_size, test_size], generator=g_split
            )
            self.logger.info(
                f"Target dataset split: Train/Val={len(train_val_dataset)}, Test={len(test_dataset)}"
            )

            # 4. Get Model Class
            model_class = globals()[ft_config.model.class_name]

            # 5. Run Cross-Validation (or single training run) for fine-tuning
            # The Trainer will instantiate the model using source_hparams and source_io_counts
            # then load weights from ft_config.training.pretrained_model_name

            # The hparams passed to cross_validation_procedure should be the source_hparams for model architecture,
            # but also include relevant training params for the new loop (batch_size, accumulation_steps).
            # The Trainer will use ft_config for optimizer, scheduler, epochs etc.
            # The hparams for cross_validation_procedure are primarily for model instantiation and some criterion settings.

            cv_hparams = {
                **source_hparams
            }  # Start with source model's architectural hparams
            # Add/override necessary training hparams if they are expected by cross_validate/Trainer from hparams dict
            cv_hparams["batch_size"] = (
                ft_config.training.batch_size
            )  # Use batch size from current (fine-tuning) config
            cv_hparams["accumulation_steps"] = ft_config.training.accumulation_steps
            # Ensure 'learning_rate' and 'weight_decay' are in cv_hparams if model_class or FNO uses them from there.
            # FNOnet itself takes them via **kwargs, so source_hparams covers this.
            # Optimizer in Trainer will use ft_config.training.learning_rate.
            # The `hparams` dict passed to Trainer in `cross_validate` is used for `normalize_output` in `denormalize_outputs_and_targets`
            # and potentially by criterion. So, ensure `use_physics_loss` and `normalize_output` from `source_hparams` are there.
            cv_hparams["use_physics_loss"] = source_hparams.get(
                "use_physics_loss", ft_config.training.use_physics_loss
            )
            cv_hparams["normalize_output"] = source_hparams.get(
                "normalize_output", ft_config.data.normalize_output
            )

            avg_cv_loss = cross_validation_procedure(
                name=self.run_name,  # Use the new fine-tuning run name
                model_class=model_class,  # Source model's class
                kfolds=ft_config.training.kfolds,
                hparams=cv_hparams,  # Source model's ARCHITECTURAL hparams + relevant training ones
                config=ft_config,  # Current fine-tuning config (for epochs, LR, dataset path etc.)
                train_val_dataset=train_val_dataset,
                full_dataset_for_stats=full_dataset_for_stats,
                is_sweep=False,
                trial=None,
            )
            self.logger.info(
                f"Fine-tuning run '{self.run_name}' finished. Avg CV loss: {avg_cv_loss:.4f}"
            )

            # 6. Optional: Evaluate on the test set of the TARGET dataset
            best_model_path_ft = Path("savepoints") / f"{self.run_name}_best_model.pth"
            if best_model_path_ft.exists() and test_dataset and len(test_dataset) > 0:
                self.logger.info(
                    f"Evaluating fine-tuned model '{self.run_name}' on its test set..."
                )

                # Instantiate model with SOURCE hparams and IO counts
                eval_model = model_class(
                    field_inputs_n=source_io_counts["field_inputs_n"],
                    scalar_inputs_n=source_io_counts["scalar_inputs_n"],
                    field_outputs_n=source_io_counts["field_outputs_n"],
                    scalar_outputs_n=source_io_counts["scalar_outputs_n"],
                    **source_hparams,
                ).to(ft_config.device)

                # Load the fine-tuned weights
                try:
                    checkpoint = torch.load(
                        best_model_path_ft,
                        map_location=ft_config.device,
                        weights_only=True,
                    )
                except:
                    checkpoint = torch.load(
                        best_model_path_ft,
                        map_location=ft_config.device,
                        weights_only=False,
                    )
                if isinstance(checkpoint, dict) and "_metadata" in checkpoint:
                    checkpoint.pop("_metadata")
                eval_model.load_state_dict(checkpoint)

                criterion = PhysicsInformedLoss(
                    input_vars=ft_config.data.inputs,  # Should be source model's inputs
                    output_vars=ft_config.data.outputs,  # Should be source model's outputs
                    config=ft_config,
                    dataset=full_dataset_for_stats,  # Stats from TARGET dataset
                    use_physics_loss=cv_hparams.get(
                        "use_physics_loss", ft_config.training.use_physics_loss
                    ),
                    normalize_output=cv_hparams.get(
                        "normalize_output", ft_config.data.normalize_output
                    ),
                )
                # The hparams for eval_trainer should be the source_hparams for consistency with model
                eval_trainer_hparams = cv_hparams
                eval_trainer = Trainer(
                    model=eval_model,
                    criterion=criterion,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    device=ft_config.device,
                    accumulation_steps=1,
                    config=ft_config,
                    full_dataset=full_dataset_for_stats,
                    hparams=eval_trainer_hparams,
                )
                g_test = torch.Generator().manual_seed(ft_config.seed + 1)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=eval_trainer_hparams["batch_size"],
                    shuffle=False,
                    num_workers=ft_config.training.num_workers,
                    pin_memory=False,
                    worker_init_fn=seed_worker,
                    generator=g_test,
                )
                test_metrics = eval_trainer.validate(
                    test_loader, name=f"{self.run_name}_Final_Test", step=-1, fold_n=-1
                )
                self.logger.info(f"Fine-tuned model test metrics: {test_metrics}")
                # Log to WandB if enabled for fine-tuning run
                if ft_config.logging.use_wandb:
                    wandb.init(
                        project=ft_config.logging.wandb_project,
                        name=f"{self.run_name}_Test_Eval",
                        group=self.source_study_name + "_Finetuning",
                        job_type="Finetune_Evaluation",
                        config={
                            **source_hparams,
                            "target_dataset": self.target_dataset_path,
                            "source_model": self.pretrained_model_name_stem,
                        },
                        reinit=True,
                    )
                    wandb.log({f"Final_Test/{k}": v for k, v in test_metrics.items()})
                    wandb.summary.update(
                        {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                    )
                    wandb.summary["Avg_CV_Loss_Finetune"] = avg_cv_loss
                    wandb.finish()
            else:
                self.logger.warning(
                    "Skipping test set evaluation for fine-tuned model (no checkpoint or empty test set)."
                )

            return avg_cv_loss

        except Exception as e:
            self.logger.exception(
                f"Error during fine-tuning of {self.pretrained_model_name_stem} on {self.target_dataset_path}: {e}"
            )
            return None
        finally:
            empty_cache()

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
            self.logger.info(
                "Attempting explicit GPU cleanup immediately after Optuna study..."
            )
            # Optional: Delete large objects if safe
            # if hasattr(self, 'train_val_dataset'): del self.train_val_dataset
            import gc

            gc.collect()
            self.logger.info("Python garbage collection triggered.")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("torch.cuda.empty_cache() called successfully.")
                    self.logger.info(
                        f"GPU Memory Allocated (Post-Cleanup): {torch.cuda.memory_allocated(0)/1024**2:.2f} MB"
                    )
                    self.logger.info(
                        f"GPU Memory Reserved (Post-Cleanup): {torch.cuda.memory_reserved(0)/1024**2:.2f} MB"
                    )
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

                    # Check if it's a dictionary and remove the unexpected key before loading
                    if isinstance(checkpoint, dict):
                        # Remove the metadata key if it exists
                        removed_key = checkpoint.pop(
                            "_metadata", None
                        )  # Use pop with default None
                        if removed_key is not None:
                            self.logger.debug(
                                "Removed '_metadata' key from loaded checkpoint dictionary."
                            )
                        # Now load the cleaned dictionary
                        final_model.load_state_dict(checkpoint)
                    else:
                        # If it wasn't a dict (unexpected), raise an error or handle differently
                        raise TypeError(
                            f"Loaded checkpoint is not a dictionary (type: {type(checkpoint)}). Cannot load state_dict."
                        )
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


# END OF EXISTING CLASSES TO PASTE

# ... (The existing function: repeat_trial_from_study remains unchanged)
# ... (Paste the existing function here)
# START OF EXISTING FUNCTION TO PASTE
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

# END OF EXISTING FUNCTION TO PASTE

# ... (The existing TRAINED_MODELS_INFO, GEOMETRY_FILES, GEOM_NAMES dictionaries remain unchanged)
# ... (Paste the existing dictionaries here)
# START OF EXISTING DICTIONARIES TO PASTE
TRAINED_MODELS_INFO = {
    "ddb": {
        "trial_number": 161,
        "study_name": "study28ddb",
        "source_geom": "b",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idb": {
        "trial_number": 57,
        "study_name": "study25idb",
        "source_geom": "b",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "dab": {
        "trial_number": 35,
        "study_name": "study26dab",
        "source_geom": "b",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "iab": {
        "trial_number": 157,
        "study_name": "study36iab",
        "source_geom": "b",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "dds": {
        "trial_number": 93,
        "study_name": "study30dds",
        "source_geom": "s",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "ids": {
        "trial_number": 63,
        "study_name": "study34ids",
        "source_geom": "s",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "das": {
        "trial_number": 26,
        "study_name": "study31das",
        "source_geom": "s",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "ias": {
        "trial_number": 36,
        "study_name": "study35ias",
        "source_geom": "s",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "ddn": {
        "trial_number": 194,
        "study_name": "study29ddb",
        "source_geom": "n",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idn": {
        "trial_number": 94,
        "study_name": "study33ids",
        "source_geom": "n",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "dan": {
        "trial_number": 195,
        "study_name": "study32dan",
        "source_geom": "n",
        "inputs": [
            "Hr",
            "Fr",
            "M",
            "Re",
            "B*",
            "Ar",
            "Vr",
        ],
        "outputs": [
            "H*",
            "U*",
            "V*",
        ],
    },
    "ian": {
        "trial_number": 88,
        "study_name": "study37ian",
        "source_geom": "n",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
}

GEOMETRY_FILES = {
    "b": "data/barsa.hdf5",
    "s": "data/slopea.hdf5",
    "n": "data/noisea.hdf5",
}
GEOM_NAMES = {"b": "barsa", "s": "slopea", "n": "noisea"}

# END OF EXISTING DICTIONARIES TO PASTE


# ... (The existing function: run_all_fine_tuning_jobs remains unchanged)
# ... (Paste the existing function here)
# START OF EXISTING FUNCTION TO PASTE
def run_all_fine_tuning_jobs(base_config_path="config.yaml"):
    # Setup logger ONCE for this overarching function.
    # Child processes/classes can get their own logger instance from setup_logger()
    # but this one is for the main loop.
    main_loop_logger = setup_logger()  # Get a logger instance
    main_loop_logger.info("Starting the 'run_all_fine_tuning_jobs' process.")

    try:
        base_config_template = get_config(base_config_path)
        main_loop_logger.info(f"Base configuration loaded from {base_config_path}")
    except Exception as e:
        main_loop_logger.error(
            f"CRITICAL: Failed to load base configuration from {base_config_path}. Aborting. Error: {e}"
        )
        main_loop_logger.debug(traceback.format_exc())
        return

    # Counter for processed jobs
    jobs_attempted = 0
    jobs_succeeded = 0

    for model_key, model_info in TRAINED_MODELS_INFO.items():
        main_loop_logger.info(
            f"\n{'='*20} Processing Model Series: {model_key.upper()} {'='*20}"
        )
        source_study = model_info["study_name"]
        source_trial = model_info["trial_number"]
        source_geom_char = model_info["source_geom"]

        main_loop_logger.info(
            f"Source Model Details: Key='{model_key}', Study='{source_study}', Trial='{source_trial}', OriginalGeom='{GEOM_NAMES[source_geom_char]}'"
        )

        for target_geom_char, target_data_file in GEOMETRY_FILES.items():
            if target_geom_char == source_geom_char:
                main_loop_logger.info(
                    f"Skipping fine-tuning {model_key} on its original geometry '{GEOM_NAMES[target_geom_char]}'."
                )
                continue

            jobs_attempted += 1
            main_loop_logger.info(
                f"\n--- Attempting Job #{jobs_attempted}: Fine-tune {model_key} on {GEOM_NAMES[target_geom_char].upper()} ---"
            )
            main_loop_logger.info(f"Target data file: {target_data_file}")

            # Construct a descriptive run name
            # Example: ft_ian_from_study37ian_trial88_on_barsa
            base_model_name_for_run = (
                f"{model_info['study_name']}_trial{model_info['trial_number']}"
            )
            finetune_run_name = f"ft_{model_key}_from_{base_model_name_for_run}_on_{GEOM_NAMES[target_geom_char]}"
            main_loop_logger.info(f"Generated fine-tune run name: {finetune_run_name}")

            # Create a DEEP COPY of the base_config_template for each fine-tuning job
            current_job_config = base_config_template.copy(deep=True)

            # Optionally, override specific fine-tuning training parameters here if needed for all series runs
            # current_job_config.training.learning_rate = 1e-5
            # current_job_config.training.num_epochs = 50
            # current_job_config.logging.use_wandb = True # Ensure WandB is on if desired for these runs

            fine_tuner = None  # Initialize to ensure it's cleared or properly scoped
            try:
                main_loop_logger.info(
                    f"Instantiating ModelFineTuner for {finetune_run_name}."
                )
                fine_tuner = ModelFineTuner(
                    config=current_job_config,
                    source_model_info=model_info,
                    target_dataset_path=target_data_file,
                    finetune_run_name=finetune_run_name,
                )
                main_loop_logger.info(
                    f"Calling fine_tuner.finetune() for {finetune_run_name}."
                )
                result = fine_tuner.finetune()

                if result is not None:
                    main_loop_logger.info(
                        f"SUCCESS: Fine-tuning job '{finetune_run_name}' completed. Avg CV Loss: {result:.6f}"
                    )
                    jobs_succeeded += 1
                else:
                    main_loop_logger.error(
                        f"FAILURE: Fine-tuning job '{finetune_run_name}' returned None (likely failed)."
                    )

            except Exception as e:
                main_loop_logger.error(
                    f"CRITICAL ERROR during fine-tuning job '{finetune_run_name}': {e}"
                )
                main_loop_logger.debug(traceback.format_exc())
            finally:
                # Explicitly clean up to free resources, especially GPU memory
                del fine_tuner  # Remove reference to the tuner object
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                main_loop_logger.info(
                    f"Cleaned up resources for job '{finetune_run_name}'."
                )
                main_loop_logger.info(
                    f"--- Finished Job #{jobs_attempted}: {finetune_run_name} ---"
                )

    main_loop_logger.info(f"\n{'='*20} ALL FINE-TUNING JOBS PROCESSED {'='*20}")
    main_loop_logger.info(f"Total jobs attempted: {jobs_attempted}")
    main_loop_logger.info(f"Total jobs succeeded: {jobs_succeeded}")

# END OF EXISTING FUNCTION TO PASTE

# --- MODIFICATION 2: Add the new main function for the requested task ---
def run_all_best_trials_with_5_folds(base_config_path="config.yaml"):
    """
    Iterates through all 12 best models defined in TRAINED_MODELS_INFO,
    and reruns their training from scratch using their best hyperparameters,
    but with 5-fold cross-validation.
    """
    main_loop_logger = setup_logger()
    main_loop_logger.info(
        "--- Starting 5-Fold Cross-Validation Rerun for all Best Trials ---"
    )

    try:
        base_config_template = get_config(base_config_path)
    except Exception as e:
        main_loop_logger.error(
            f"CRITICAL: Failed to load base configuration from {base_config_path}. Aborting. Error: {e}"
        )
        main_loop_logger.debug(traceback.format_exc())
        return

    # Iterate through each of the 12 model configurations
    for model_key, model_info in TRAINED_MODELS_INFO.items():
        main_loop_logger.info(
            f"\n{'='*20} Rerunning Best Trial for: {model_key.upper()} with 5-Fold CV {'='*20}"
        )

        # Create a deep copy of the base config to avoid modifying it in place
        current_job_config = base_config_template.copy(deep=True)

        # Get the specific details for this model run
        source_study = model_info["study_name"]
        source_trial = model_info["trial_number"]
        source_geom_char = model_info["source_geom"]

        # Dynamically update the configuration for this specific run
        current_job_config.optuna.study_name = source_study
        current_job_config.data.inputs = model_info["inputs"]
        current_job_config.data.outputs = model_info["outputs"]
        current_job_config.data.file_name = GEOMETRY_FILES[source_geom_char]

        # This is the key change requested: override kfolds to 5
        current_job_config.training.kfolds = 5

        main_loop_logger.info(f"Target Study: {source_study}, Trial ID: {source_trial}")
        main_loop_logger.info(f"Inputs: {current_job_config.data.inputs}")
        main_loop_logger.info(f"Outputs: {current_job_config.data.outputs}")
        main_loop_logger.info(f"Data File: {current_job_config.data.file_name}")
        main_loop_logger.info(f"K-Folds: {current_job_config.training.kfolds}")

        try:
            # Call the existing 'repeat_trial_from_study' function.
            # It will now use the modified config with kfolds=5.
            # A new, distinct run name will be generated inside the function.
            repeat_trial_from_study(trial_id=source_trial, config=current_job_config)

            main_loop_logger.info(
                f"--- Successfully completed 5-fold run for {model_key.upper()} ---"
            )

        except Exception as e:
            main_loop_logger.error(
                f"--- FAILED 5-fold run for {model_key.upper()}: {e} ---"
            )
            main_loop_logger.debug(traceback.format_exc())
        finally:
            # Clean up GPU memory between major runs to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    main_loop_logger.info("--- All 5-Fold Cross-Validation Reruns Completed ---")


def main():
    # --- 1. Define the Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run model optimization, training, repeat, or fine-tuning"
    )
    # --- MODIFICATION 3: Update parser choices and help text ---
    parser.add_argument(
        "--mode",
        type=str,
        choices=[m.name.lower() for m in OptimizerMode],
        default="hypertuning",
        help="Mode of operation: hypertuning, training, repeat, finetune, cv_rerun",
    )
    parser.add_argument(
        "--trial_id",
        type=int,
        default=None,
        help="Trial number to repeat (repeat mode) or source trial for fine-tuning (finetune mode)",
    )
    parser.add_argument(
        "--source_study_name",
        type=str,
        default=None,
        help="Name of the Optuna study containing the source model for fine-tuning (finetune mode)",
    )
    parser.add_argument(
        "--target_dataset_path",
        type=str,
        default=None,
        help="Path to the target HDF5 dataset for fine-tuning (finetune mode, e.g., 'data/barsa.hdf5')",
    )
    parser.add_argument(
        "--finetune_run_name",
        type=str,
        default=None,
        help="Specific name for the fine-tuning run/model, e.g., 'ddb_on_slopea' (finetune mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the base configuration file",
    )

    # --- 2. Handle Argument Parsing based on Environment ---
    if is_jupyter():
        print("INFO: Running in Jupyter environment. Using manually defined args.")
        args = parser.parse_args(args=[])
        print(
            f"INFO: Jupyter using - Mode: {args.mode}, Trial ID: {args.trial_id}, Config: {args.config}"
        )
    else:
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
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load configuration from {args.config}: {e}")
        sys.exit(1)

    # --- 4. Validate Arguments ---
    try:
        mode = OptimizerMode.from_string(args.mode)
        if mode == OptimizerMode.REPEAT and args.trial_id is None:
            raise ValueError("--trial_id is required when mode is 'repeat'")
        if mode == OptimizerMode.FINETUNE:
            if (
                args.source_study_name is None
                or args.trial_id is None
                or args.target_dataset_path is None
            ):
                raise ValueError(
                    "--source_study_name, --trial_id, and --target_dataset_path are required for finetune mode."
                )
    except ValueError as e:
        print(f"ERROR: Invalid arguments: {e}")
        sys.exit(1)

    # --- 5. Setup Experiment ---
    setup_experiment(config)
    set_seed(config.seed)
    logger = setup_logger()

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

        elif mode == OptimizerMode.FINETUNE:
            logger.info("Initiating ModelFineTuner...")
            fine_tuner = ModelFineTuner(
                config=config,
                source_study_name=args.source_study_name,
                source_trial_id=args.trial_id,
                target_dataset_path=args.target_dataset_path,
                finetune_run_name=args.finetune_run_name,
            )
            avg_ft_loss = fine_tuner.finetune()
            if avg_ft_loss is not None:
                logger.info(
                    f"Fine-tuning completed. Average CV loss on target data: {avg_ft_loss:.6f}"
                )
            else:
                logger.error("Fine-tuning process failed.")

        # --- MODIFICATION 4: Add the new mode to the execution block ---
        elif mode == OptimizerMode.CV_RERUN:
            logger.info("Initiating 5-fold CV rerun for all best models...")
            run_all_best_trials_with_5_folds(args.config)

    except Exception as e:
        logger.exception(f"An unhandled error occurred during execution: {str(e)}")
        raise

    logger.info("--- Process Completed ---")


if __name__ == "__main__" or is_jupyter():
    # --- MODIFICATION 5: Change the default execution to call main() ---
    # main()
    # # Example of running a specific function directly (for debugging)
    run_all_best_trials_with_5_folds()
    # run_all_fine_tuning_jobs()
