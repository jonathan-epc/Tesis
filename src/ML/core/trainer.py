# ML/core/trainer.py

from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader, random_split

from common.utils import setup_logger
from ML.modules.data import HDF5Dataset
from ML.modules.loss import PhysicsInformedLoss
from ML.modules import models as model_zoo
from ML.modules.training import Trainer, cross_validation_procedure
from ML.modules.utils import seed_worker
from nconfig import Config

logger = setup_logger()


class ModelTrainer:
    """Handles the process of a single training run using fixed hyperparameters.

    This class orchestrates the data loading, model instantiation, training
    via the cross-validation procedure, and final evaluation on a held-out
    test set. It reads all its parameters from a single, unified config object.

    Args:
        config (Config): The main Pydantic configuration object for the project.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logger

    def _get_default_hparams(self) -> dict[str, Any]:
        """Extracts hyperparameters from the config for the model and training.

        Returns:
            Dict[str, Any]: A flat dictionary of hyperparameters required by the
                            model and training procedure.
        """
        hparams = self.config.model.model_dump()
        hparams.update(self.config.training.model_dump())
        hparams["normalize_output"] = self.config.data.normalize_output
        return hparams

    def _get_model_class(self):
        """Gets the model class from the config file using its name."""
        model_name = self.config.model.class_name
        try:
            # getattr will find the class by its string name in the models.py module
            model_class = getattr(model_zoo, model_name)
            return model_class
        except AttributeError:
            raise ValueError(f"Model class '{model_name}' not found in 'src/ML/modules/models.py'")   

    def train(self) -> float | None:
        """Loads data, splits it, and runs the full training and evaluation procedure.

        Returns:
            Optional[float]: The average cross-validation loss from the run,
                             or None if the training failed.
        """
        self.logger.info("Starting single training run with fixed hyperparameters.")
        try:
            # 1. Load Data
            full_dataset = HDF5Dataset.from_config(
                self.config, self.config.data.file_path
            )
            test_size = int(self.config.training.test_frac * len(full_dataset))
            train_val_size = len(full_dataset) - test_size
            g_split = torch.Generator().manual_seed(self.config.seed)
            train_val_dataset, test_dataset = random_split(
                full_dataset, [train_val_size, test_size], generator=g_split
            )
            self.logger.info(
                f"Data split: Train/Val={len(train_val_dataset)}, Test={len(test_dataset)}"
            )

            # 2. Get Model and Hyperparameters
            model_class = self._get_model_class()
            hparams = self._get_default_hparams()

            # 3. Run Training
            avg_cv_loss = cross_validation_procedure(
                name=self.config.model.name,
                model_class=model_class,
                kfolds=self.config.training.kfolds,
                hparams=hparams,
                config=self.config,
                train_val_dataset=train_val_dataset,
                full_dataset_for_stats=full_dataset,
                is_sweep=False,
                trial=None,
            )
            self.logger.info(f"Training finished. Average CV loss: {avg_cv_loss:.4f}")

            # 4. Evaluate on Test Set
            self._evaluate_on_test_set(
                avg_cv_loss, test_dataset, full_dataset, model_class, hparams
            )

            return avg_cv_loss

        except Exception as e:
            self.logger.exception(f"Error during single training run: {e}")
            return None

    def _evaluate_on_test_set(
        self,
        avg_cv_loss: float,
        test_dataset: torch.utils.data.Dataset,
        full_dataset: HDF5Dataset,
        model_class: torch.nn.Module,
        hparams: dict[str, Any],
    ) -> None:
        """Evaluates the best saved model on the held-out test set."""
        self.logger.info("Performing final evaluation on the test set...")
        best_model_path = (
            Path(self.config.paths.savepoints_dir)
            / f"{self.config.model.name}_best_model.pth"
        )

        if not best_model_path.exists():
            self.logger.warning(
                f"Best model not found at {best_model_path}. Skipping test evaluation."
            )
            return

        eval_model = model_class(
            field_inputs_n=len(self.config.data.input_fields),
            scalar_inputs_n=len(self.config.data.input_scalars),
            field_outputs_n=len(self.config.data.output_fields),
            scalar_outputs_n=len(self.config.data.output_scalars),
            **hparams,
        ).to(self.config.device)

        try:
            checkpoint = torch.load(
                best_model_path, map_location=self.config.device, weights_only=True
            )
        except Exception:
            self.logger.warning(
                "Loading with weights_only=True failed. Retrying with weights_only=False."
            )
            checkpoint = torch.load(
                best_model_path, map_location=self.config.device, weights_only=False
            )

        if isinstance(checkpoint, dict):
            checkpoint.pop("_metadata", None)
        eval_model.load_state_dict(checkpoint)

        criterion = PhysicsInformedLoss(
            input_vars=self.config.data.inputs,
            output_vars=self.config.data.outputs,
            config=self.config,
            dataset=full_dataset,
            use_physics_loss=hparams.get(
                "use_physics_loss", self.config.training.use_physics_loss
            ),
            normalize_output=hparams.get(
                "normalize_output", self.config.data.normalize_output
            ),
        )

        eval_trainer = Trainer(
            model=eval_model,
            criterion=criterion,
            optimizer=None,
            scheduler=None,
            scaler=None,
            device=self.config.device,
            accumulation_steps=1,
            config=self.config,
            full_dataset=full_dataset,
            hparams=hparams,
        )

        g_test = torch.Generator().manual_seed(self.config.seed + 1)
        test_loader = DataLoader(
            test_dataset,
            batch_size=hparams.get("batch_size", self.config.training.batch_size),
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g_test,
        )

        test_metrics = eval_trainer.validate(
            test_loader,
            name=f"{self.config.model.name}_SingleRun_Test",
            step=-1,
            fold_n=-1,
        )
        test_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        self.logger.info(f"Final Test Set Evaluation (Single Run): {test_metrics_str}")

        if self.config.logging.use_wandb:
            run_name = f"{self.config.model.name}_Test_Eval"
            with wandb.init(
                project=self.config.logging.wandb_project,
                name=run_name,
                group=self.config.model.name,
                job_type="Single_Run_Evaluation",
                config=hparams,
                reinit=True,
            ) as run:
                run.log({f"Final_Test/{k}": v for k, v in test_metrics.items()})
                run.summary.update(
                    {f"Final_Test_Summary/{k}": v for k, v in test_metrics.items()}
                )
                run.summary["Avg_CV_Loss"] = avg_cv_loss
