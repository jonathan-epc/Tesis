# ML/core/finetuner.py

from pathlib import Path
from typing import Any

import optuna
import torch
from nconfig import Config
from torch.cuda import empty_cache
from torch.utils.data import DataLoader, random_split

import wandb

from ..modules.data import HDF5Dataset
from ..modules.loss import PhysicsInformedLoss
from ..modules.models import FNOnet
from ..modules.training import Trainer, cross_validation_procedure
from ..modules.utils import seed_worker, setup_logger

# --- Authoritative Information for Pre-trained Models and Datasets ---

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
        "study_name": "study29ddn",
        "source_geom": "n",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idn": {
        "trial_number": 94,
        "study_name": "study33idn",
        "source_geom": "n",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "dan": {
        "trial_number": 195,
        "study_name": "study32dan",
        "source_geom": "n",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
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
    "b": "data/BARSa.hdf5",
    "s": "data/SLOPEa.hdf5",
    "n": "data/NOISEa.hdf5",
}
GEOM_NAMES = {"b": "BARSa", "s": "SLOPEa", "n": "NOISEa"}


class ModelFineTuner:
    def __init__(
        self,
        config: Config,
        source_model_info: dict,
        target_dataset_path: str,
        finetune_run_name: str | None = None,
    ):
        self.base_config = config
        self.logger = setup_logger()
        self.source_model_info = source_model_info
        self.source_study_name = self.source_model_info["study_name"]
        self.source_trial_id = self.source_model_info["trial_number"]
        self.target_dataset_path = target_dataset_path

        if not Path(self.target_dataset_path).exists():
            raise FileNotFoundError(
                f"Target dataset not found: {self.target_dataset_path}"
            )

        self.pretrained_model_name_stem = f"{self.source_study_name}_{self.base_config.model.architecture}_trial_{self.source_trial_id}"
        target_geom_name = Path(self.target_dataset_path).stem
        self.run_name = (
            finetune_run_name
            or f"{self.pretrained_model_name_stem}_finetuned_on_{target_geom_name}"
        )

        self.logger.info("--- Initializing Fine-Tuning ---")
        self.logger.info(f"Source Model Info: {self.source_model_info}")
        self.logger.info(f"Target Dataset: {self.target_dataset_path}")
        self.logger.info(f"Fine-tuning Run Name: {self.run_name}")

    def _load_source_model_hparams(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.logger.info(
            f"Loading Optuna study '{self.source_study_name}' for hyperparameters."
        )
        source_optuna_storage_url = f"sqlite:///{self.base_config.paths.studies_dir}/{self.source_study_name}.db"

        try:
            source_study = optuna.load_study(
                study_name=self.source_study_name, storage=source_optuna_storage_url
            )
            source_trial = source_study.trials[self.source_trial_id]
            source_hparams = source_trial.params
            self.logger.info(f"Loaded source model hparams: {source_hparams}")
        except Exception as e:
            self.logger.error(
                f"Failed to load hparams for trial {self.source_trial_id} from study {self.source_study_name}: {e}"
            )
            raise

        # Calculate the I/O dimensions of the source model
        source_io_counts = {
            "field_inputs_n": len(
                [
                    p
                    for p in self.source_model_info["inputs"]
                    if p in self.base_config.data.all_field_vars
                ]
            ),
            "scalar_inputs_n": len(
                [
                    p
                    for p in self.source_model_info["inputs"]
                    if p not in self.base_config.data.all_field_vars
                ]
            ),
            "field_outputs_n": len(
                [
                    p
                    for p in self.source_model_info["outputs"]
                    if p in self.base_config.data.all_field_vars
                ]
            ),
            "scalar_outputs_n": len(
                [
                    p
                    for p in self.source_model_info["outputs"]
                    if p not in self.base_config.data.all_field_vars
                ]
            ),
        }
        self.logger.info(f"Calculated source model I/O counts: {source_io_counts}")
        return source_hparams, source_io_counts

    def finetune(self) -> float | None:
        self.logger.info(f"Starting fine-tuning process for {self.run_name}")
        try:
            source_hparams, source_io_counts = self._load_source_model_hparams()

            ft_config = self.base_config.copy(deep=True)
            ft_config.training.pretrained_model_name = self.pretrained_model_name_stem
            ft_config.data.file_path = self.target_dataset_path

            # The model architecture is fixed, so the data loader must provide data
            # with the same I/O structure as the original pre-trained model.
            ft_config.data.inputs = self.source_model_info["inputs"]
            ft_config.data.outputs = self.source_model_info["outputs"]

            # Load target dataset for fine-tuning
            full_dataset_for_stats = HDF5Dataset.from_config(
                ft_config, file_path=ft_config.data.file_path
            )
            test_size = int(ft_config.training.test_frac * len(full_dataset_for_stats))
            train_val_size = len(full_dataset_for_stats) - test_size
            g_split = torch.Generator().manual_seed(ft_config.seed)
            train_val_dataset, test_dataset = random_split(
                full_dataset_for_stats, [train_val_size, test_size], generator=g_split
            )

            # The trainer will instantiate the model using the source model's I/O counts,
            # but using the training parameters (LR, batch size, etc.) from the fine-tuning config.
            # We merge the source architecture hparams with the fine-tuning training hparams.
            cv_hparams = source_hparams.copy()
            cv_hparams["batch_size"] = ft_config.training.batch_size
            cv_hparams["accumulation_steps"] = ft_config.training.accumulation_steps
            cv_hparams["learning_rate"] = (
                ft_config.training.learning_rate
            )  # Use FT learning rate

            avg_cv_loss = cross_validation_procedure(
                name=self.run_name,
                model_class=FNOnet,
                kfolds=ft_config.training.kfolds,
                hparams=cv_hparams,
                config=ft_config,
                train_val_dataset=train_val_dataset,
                full_dataset_for_stats=full_dataset_for_stats,
                is_sweep=False,
                trial=None,
            )
            self.logger.info(
                f"Fine-tuning run '{self.run_name}' finished. Avg CV loss: {avg_cv_loss:.4f}"
            )

            # --- Final Test Set Evaluation on Target Dataset ---
            best_model_path_ft = (
                Path(ft_config.paths.savepoints_dir) / f"{self.run_name}_best_model.pth"
            )
            if not best_model_path_ft.exists():
                self.logger.warning(
                    f"Best fine-tuned model not found at {best_model_path_ft}. Skipping test set evaluation."
                )
                return avg_cv_loss

            if not test_dataset or len(test_dataset) == 0:
                self.logger.warning(
                    "Target test dataset is empty. Skipping evaluation."
                )
                return avg_cv_loss

            self.logger.info(
                f"Evaluating fine-tuned model '{self.run_name}' on its test set..."
            )

            # Instantiate a fresh model with the SOURCE architecture
            eval_model = FNOnet(
                field_inputs_n=source_io_counts["field_inputs_n"],
                scalar_inputs_n=source_io_counts["scalar_inputs_n"],
                field_outputs_n=source_io_counts["field_outputs_n"],
                scalar_outputs_n=source_io_counts["scalar_outputs_n"],
                **source_hparams,
            ).to(ft_config.device)

            # Load the newly fine-tuned weights
            try:
                checkpoint = torch.load(
                    best_model_path_ft, map_location=ft_config.device, weights_only=True
                )
            except Exception:
                self.logger.warning(
                    "Loading fine-tuned model with weights_only=True failed. Retrying with weights_only=False."
                )
                checkpoint = torch.load(
                    best_model_path_ft,
                    map_location=ft_config.device,
                    weights_only=False,
                )

            if isinstance(checkpoint, dict):
                checkpoint.pop("_metadata", None)
            eval_model.load_state_dict(checkpoint)

            # Criterion for evaluation (using target dataset stats)
            criterion = PhysicsInformedLoss(
                input_vars=ft_config.data.inputs,  # Source model's I/O
                output_vars=ft_config.data.outputs,  # Source model's I/O
                config=ft_config,
                dataset=full_dataset_for_stats,  # Stats from TARGET dataset
                use_physics_loss=cv_hparams.get(
                    "use_physics_loss", ft_config.training.use_physics_loss
                ),
                normalize_output=cv_hparams.get(
                    "normalize_output", ft_config.data.normalize_output
                ),
            )

            # Create a temporary Trainer for validation logic
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
                hparams=cv_hparams,
            )

            # Test DataLoader for the target dataset's test split
            g_test = torch.Generator().manual_seed(ft_config.seed + 1)
            test_loader = DataLoader(
                test_dataset,
                batch_size=cv_hparams["batch_size"],
                shuffle=False,
                num_workers=ft_config.training.num_workers,
                pin_memory=False,
                worker_init_fn=seed_worker,
                generator=g_test,
            )

            # Run evaluation
            test_metrics = eval_trainer.validate(
                test_loader, name=f"{self.run_name}_Final_Test", step=-1, fold_n=-1
            )
            self.logger.info(f"Fine-tuned model test metrics: {test_metrics}")

            # Log to WandB
            if ft_config.logging.use_wandb:
                with wandb.init(
                    project=ft_config.logging.wandb_project,
                    name=f"{self.run_name}_Test_Eval",
                    group=f"{self.source_study_name}_Finetuning",
                    job_type="Finetune_Evaluation",
                    config={
                        **cv_hparams,
                        "target_dataset": self.target_dataset_path,
                        "source_model": self.pretrained_model_name_stem,
                    },
                    reinit=True,
                ) as run:
                    run.log({f"Final_Test/{k}": v for k, v in test_metrics.items()})
                    run.summary.update(
                        {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                    )
                    run.summary["Avg_CV_Loss_Finetune"] = avg_cv_loss

            return avg_cv_loss

        except Exception as e:
            self.logger.exception(
                f"Error during fine-tuning of {self.pretrained_model_name_stem}: {e}"
            )
            return None
        finally:
            empty_cache()
