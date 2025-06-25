# --- Imports ---
import os
import pickle  # For potential fallback loading issues
import random  # For seeding workers
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import h5py
import numpy as np
import optuna
import pandas as pd  # Check if actually used, can be removed if not
import torch
import torch.nn as nn
import wandb

# --- Local Imports ---
# Assume config provides Pydantic models for configuration
from config import Config, get_config  # Example: Import base Config model
from loguru import logger  # For structured logging
from sklearn.model_selection import KFold, train_test_split
from torch.amp import GradScaler, autocast  # For mixed-precision training
from torch.nn.utils import clip_grad_norm_  # For gradient clipping
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from tqdm.autonotebook import tqdm  # For progress bars

# --- Custom Modules ---
# Assuming these modules contain relevant classes/functions
from modules.data import HDF5Dataset
from modules.loss import PhysicsInformedLoss
from modules.models import FNOnet
from modules.utils import (
    EarlyStopping,
    compute_metrics,
    denormalize_outputs_and_targets,
    seed_worker
)

# --- Trainer Class ---

class Trainer:
    """
    Manages the training and validation process for a neural network model.

    Handles epoch loops, optimization steps, mixed-precision training,
    validation metric calculation, logging, and model loading/saving.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[GradScaler],
        device: torch.device,
        accumulation_steps: int,
        config: Config,  # Use the specific Pydantic Config type
        full_dataset: HDF5Dataset,
        hparams: Dict[str, Any]
    ):
        """
        Initializes the Trainer instance.

        Args:
            model (nn.Module): The neural network model to train.
            criterion (nn.Module): The loss function.
            optimizer (Optional[torch.optim.Optimizer]): The optimization algorithm.
            scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
            scaler (Optional[GradScaler]): Gradient scaler for mixed-precision training.
            device (torch.device): The device to run training on (e.g., 'cuda', 'cpu').
            accumulation_steps (int): Number of steps to accumulate gradients over.
            config (Config): Pydantic configuration object containing parameters.
            full_dataset (HDF5Dataset): The complete dataset instance, used for obtaining
                                       normalization statistics if needed by denormalization.
        """
        self.config = config
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.pretrained_model_path = None
        self.full_dataset = full_dataset
        self.hparams = hparams

        # Check for complex parameters affecting autocast
        self.model_is_complex = self._model_uses_complex()
        if self.model_is_complex and self.scaler is not None:
            logger.warning(
                "Model uses complex parameters. Disabling GradScaler as autocast might not be fully compatible."
            )
            # self.scaler = None # Optionally disable scaler

        # Load pretrained model if specified in config (using Pydantic access)
        # Ensure pretrained_model_name is Optional in the Pydantic model or has a default
        if self.config.training.pretrained_model_name:
            self.pretrained_model_path = os.path.join(
                "savepoints",
                f"{self.config.training.pretrained_model_name}_best_model.pth",
            )
            self._load_pretrained_model()

    def _model_uses_complex(self) -> bool:
        """Checks if any parameters in the model are complex types."""
        return any(p.is_complex() for p in self.model.parameters())

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Performs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)

        batch_iterator = tqdm(
            dataloader, desc="Training Epoch", leave=False, position=1
        )
        for idx, batch in enumerate(batch_iterator):
            # Assuming batch format is (inputs, targets) where inputs/targets are tuples of lists
            inputs, targets = self._move_to_device(batch[0], batch[1])

            # Mixed Precision Context
            with autocast(
                device_type=self.device,
                enabled=(self.device == "cuda" and not self.model_is_complex),
            ):
                outputs = self.model(inputs)
                # Ensure criterion handles the structure of inputs/outputs/targets
                loss, _, _, _, _ = self.criterion(inputs, outputs, targets)
                # Normalize loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward Pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimization Step
            if (idx + 1) % self.accumulation_steps == 0:
                self._step_optimization()

            # Accumulate loss (rescaled for proper averaging)
            total_loss += loss.item() * self.accumulation_steps
            batch_iterator.set_postfix(
                {"Batch Loss": f"{loss.item() * self.accumulation_steps:.4f}"}
            )

        # Handle final step if dataset size isn't multiple of accumulation steps * batch_size
        if len(dataloader) % self.accumulation_steps != 0:
            self._step_optimization()

        # Return average loss per sample
        # Use dataloader.dataset to get the underlying dataset size if sampler is used
        avg_loss = (
            total_loss / len(dataloader.dataset)
            if dataloader.dataset
            else total_loss / (len(dataloader) * dataloader.batch_size)
        )
        return avg_loss

    @torch.no_grad()
    def validate(
        self, dataloader: DataLoader, name: str = "", step: int = -1, fold_n: int = -1
    ) -> Dict[str, float]:
        """Performs validation on the given dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_data_loss = 0.0
        total_continuity_loss = 0.0
        total_momentum_x_loss = 0.0
        total_momentum_y_loss = 0.0

        all_field_outputs, all_scalar_outputs = [], []
        all_field_targets, all_scalar_targets = [], []

        normalize_output_setting = self.hparams.get('normalize_output', self.config.data.normalize_output)
        logger.debug(f"Validation using normalize_output_setting: {normalize_output_setting}")

        batch_iterator = tqdm(dataloader, desc="Validation", leave=False, position=1)
        for batch in batch_iterator:
            inputs, targets = self._move_to_device(batch[0], batch[1])

            with autocast(
                device_type=self.device,
                enabled=(self.device == "cuda" and not self.model_is_complex),
            ):
                outputs = self.model(inputs)
                loss, data_loss, continuity_loss, momentum_x_loss, momentum_y_loss = (
                    self.criterion(inputs, outputs, targets)
                )

            total_loss += loss.item()
            total_data_loss += data_loss.item()
            total_continuity_loss += continuity_loss.item()
            total_momentum_x_loss += momentum_x_loss.item()
            total_momentum_y_loss += momentum_y_loss.item()

            # Denormalize outputs and targets USING THE TRIAL'S SETTING
            outputs_denorm, targets_denorm = denormalize_outputs_and_targets(
                outputs,              # Pass the raw model outputs
                targets,              # Pass the original targets (potentially normalized)
                self.full_dataset,    # Pass the dataset for stats
                self.config,          # Pass config for var names
                normalize_output_setting # Pass the trial's setting
            )

            # Separate field and scalar data
            target_fields, target_scalars = targets_denorm
            field_outputs, scalar_outputs = outputs_denorm

            # Collect denormalized outputs/targets (moved to CPU)
            if field_outputs is not None:
                all_field_outputs.append(field_outputs.cpu())
                if target_fields:
                    all_field_targets.append(torch.stack(target_fields, dim=1).cpu())

            if scalar_outputs is not None:
                all_scalar_outputs.append(scalar_outputs.cpu())
                if target_scalars:
                    all_scalar_targets.append(torch.stack(target_scalars, dim=1).cpu())

        # Concatenate results
        all_field_outputs = (
            torch.cat(all_field_outputs, dim=0) if all_field_outputs else None
        )
        all_field_targets = (
            torch.cat(all_field_targets, dim=0) if all_field_targets else None
        )
        all_scalar_outputs = (
            torch.cat(all_scalar_outputs, dim=0) if all_scalar_outputs else None
        )
        all_scalar_targets = (
            torch.cat(all_scalar_targets, dim=0) if all_scalar_targets else None
        )

        # Compute metrics
        metrics = {}
        field_var_names = [
            var
            for var in self.config.data.outputs
            if var in self.config.data.non_scalars
        ]
        scalar_var_names = [
            var for var in self.config.data.outputs if var in self.config.data.scalars
        ]

        if all_field_outputs is not None and all_field_targets is not None:
            metrics.update(
                compute_metrics(
                    all_field_outputs, all_field_targets, variable_names=field_var_names
                )
            )

        if all_scalar_outputs is not None and all_scalar_targets is not None:
            metrics.update(
                compute_metrics(
                    all_scalar_outputs,
                    all_scalar_targets,
                    variable_names=scalar_var_names,
                )
            )

        # --- Plotting / Saving Tensors (Optional) ---
        # Ensure plot_enabled exists in your LoggingConfig Pydantic model
        if self.config.logging.plot_enabled and step != -1:
            save_path = os.path.join(
                self.config.logging.save_dir,
                f"{name}_fold{fold_n}_validation_step{step}",
            )
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Saving validation tensors for plotting to {save_path}")
            if all_field_outputs is not None:
                torch.save(
                    all_field_outputs, os.path.join(save_path, "field_outputs.pt")
                )
            if all_field_targets is not None:
                torch.save(
                    all_field_targets, os.path.join(save_path, "field_targets.pt")
                )
            if all_scalar_outputs is not None:
                torch.save(
                    all_scalar_outputs, os.path.join(save_path, "scalar_outputs.pt")
                )
            if all_scalar_targets is not None:
                torch.save(
                    all_scalar_targets, os.path.join(save_path, "scalar_targets.pt")
                )

        # Calculate Average Losses per batch
        num_batches = len(dataloader)
        if num_batches > 0:
            metrics["loss"] = total_loss / num_batches
            metrics["data_loss"] = total_data_loss / num_batches
            # Only report physics losses if they were actually used in criterion
            if hasattr(self.criterion, 'use_physics_loss') and self.criterion.use_physics_loss:
                 metrics["continuity_loss"] = total_continuity_loss / num_batches
                 metrics["momentum_x_loss"] = total_momentum_x_loss / num_batches
                 metrics["momentum_y_loss"] = total_momentum_y_loss / num_batches
            else: # Report as zero if not used
                 metrics["continuity_loss"] = 0.0
                 metrics["momentum_x_loss"] = 0.0
                 metrics["momentum_y_loss"] = 0.0

        else:
            logger.warning("Validation dataloader is empty. Metrics cannot be averaged.")
            metrics["loss"] = total_loss
            metrics["data_loss"] = total_data_loss
            metrics["continuity_loss"] = 0.0
            metrics["momentum_x_loss"] = 0.0
            metrics["momentum_y_loss"] = 0.0

        return metrics

    def _move_to_device(
        self,
        inputs: Tuple[List[torch.Tensor], List[torch.Tensor]],
        targets: Tuple[List[torch.Tensor], List[torch.Tensor]],
    ) -> Tuple[
        Tuple[List[torch.Tensor], List[torch.Tensor]],
        Tuple[List[torch.Tensor], List[torch.Tensor]],
    ]:
        """Moves input and target tensors (structured as tuple of lists) to the specified device."""
        inputs_on_device = tuple(
            [
                [tensor.to(self.device, non_blocking=True) for tensor in tensor_list]
                for tensor_list in inputs
            ]
        )
        targets_on_device = tuple(
            [
                [tensor.to(self.device, non_blocking=True) for tensor in tensor_list]
                for tensor_list in targets
            ]
        )
        return inputs_on_device, targets_on_device

    def _step_optimization(self):
        """Performs a single optimization step, including gradient clipping and scheduler update."""
        if not self.optimizer:
            logger.warning("Optimizer is None, skipping optimization step.")
            return

        # Use clip_grad_value directly from Pydantic config
        max_norm = self.config.training.clip_grad_value

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler:
            self.scheduler.step()

    def train(
        self,
        name: str,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        fold_n: int,
        is_sweep: bool,
        trial: Optional[optuna.Trial],
        early_stopping: EarlyStopping,
    ) -> None:
        """Runs the main training loop for a specified number of epochs."""
        # --- WandB Initialization (if enabled) ---
        # Ensure use_wandb exists in your LoggingConfig Pydantic model
        if self.config.logging.use_wandb:
            run_name = f"{name}_fold_{fold_n}"
            wandb_id = wandb.util.generate_id()
            wandb_config = {**self.config.dict(), **self.hparams}
            # Ensure wandb_project and study_notes exist in config
            wandb.init(
                project=self.config.logging.wandb_project,  # Use config value
                name=run_name,
                group=name,
                id=wandb_id,
                job_type="Sweep" if is_sweep else "Run",
                config=wandb_config,
                notes=self.config.optuna.study_notes,
                resume="allow",
                reinit=True,
            )

        # --- Epoch Loop ---
        epoch_pbar = tqdm(
            range(num_epochs), desc=f"Fold {fold_n} Training", position=0, leave=True
        )
        best_val_loss = float("inf")  # Track best loss *within this fold* for logging

        for epoch in epoch_pbar:
            # --- Time Limit Check (Optuna Pruning) ---
            if epoch_pbar.format_dict["rate"] and epoch_pbar.total:
                rate = epoch_pbar.format_dict["rate"]
                remaining_iterations = epoch_pbar.total - epoch_pbar.n
                estimated_remaining_time = (
                    remaining_iterations / rate if rate > 0 else 0
                )
            else:
                estimated_remaining_time = 0

            # Use time_limit directly from Pydantic config (Optional field)
            if (
                self.config.training.time_limit is not None
                and estimated_remaining_time > self.config.training.time_limit
            ):
                logger.warning(
                    f"Fold {fold_n}, Epoch {epoch}: Estimated remaining time ({estimated_remaining_time:.0f}s) "
                    f"exceeds limit ({self.config.training.time_limit}s). Pruning trial."
                )
                if self.config.logging.use_wandb:
                    wandb.finish(exit_code=1)
                raise optuna.TrialPruned(
                    f"Estimated time exceeded limit at epoch {epoch}"
                )

            # --- Train & Validate ---
            train_loss = self.train_epoch(train_dataloader)
            val_metrics = self.validate(val_dataloader, name, epoch, fold_n)
            val_loss = val_metrics["loss"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # --- Update Progress Bar ---
            lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler
                else (self.optimizer.param_groups[0]["lr"] if self.optimizer else 0)
            )
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Best Val Loss": f"{best_val_loss:.4f}",  # Best within this fold
                    "LR": f"{lr:.6f}",
                }
            )

            # --- Logging ---
            if self.config.logging.use_wandb:
                log_data = {
                    "Epoch": epoch,
                    "Train_loss": train_loss,
                    "Learning_Rate": lr,
                    **{f"Validation_{k}": v for k, v in val_metrics.items()},
                }
                wandb.log(log_data)

            # --- Optuna Reporting & Pruning ---
            if is_sweep and trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logger.info(
                        f"Optuna pruning trial at epoch {epoch} based on intermediate value: {val_loss:.4f}"
                    )
                    if self.config.logging.use_wandb:
                        wandb.finish(exit_code=1)
                    raise optuna.TrialPruned(
                        f"Pruned at epoch {epoch} with val_loss {val_loss}"
                    )

            # --- Early Stopping ---
            # Pass model for potential saving via save_checkpoint if global best improves
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                epoch_pbar.update()  # Ensure pbar finishes cleanly
                epoch_pbar.set_postfix(
                    {
                        "Status": "Early Stopped",
                        "Best Fold Epoch": early_stopping.best_epoch,
                        "Best Fold Val Loss": -early_stopping.best_score,  # Negate score back to loss
                    }
                )
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. Best fold validation loss {-early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}."
                )
                break

        # --- Cleanup ---
        epoch_pbar.close()
        if self.config.logging.use_wandb:
            # Log the best score achieved during *this fold's* training
            # Note: best_score is negative loss, best_epoch is from this fold
            if early_stopping.best_score is not None:
                wandb.summary["best_fold_validation_loss"] = -early_stopping.best_score
                wandb.summary["best_fold_epoch"] = early_stopping.best_epoch
            wandb.finish()

    def _load_pretrained_model(self):
        """Loads model weights from the path specified in self.pretrained_model_path."""
        if not self.pretrained_model_path or not os.path.exists(self.pretrained_model_path): # Make sure os is imported
            logger.warning(
                f"Pretrained model path not specified or not found: {self.pretrained_model_path}. Skipping loading."
            )
            return

        checkpoint_to_load = None
        try:
            logger.info(f"Attempting to load pretrained model weights (weights_only=True) from: {self.pretrained_model_path}")
            checkpoint_to_load = torch.load(
                self.pretrained_model_path, map_location=self.device, weights_only=True
            )
            logger.info("Successfully loaded checkpoint with weights_only=True.")
        
        # Catching a broad Exception for the first attempt, then specifically for the fallback.
        # The original RuntimeError from weights_only=True is often due to pickle issues (like GELU).
        except Exception as e_true: 
            logger.warning(
                f"Loading pretrained model with weights_only=True failed: {e_true}. "
                "Attempting fallback with weights_only=False."
            )
            try:
                checkpoint_to_load = torch.load(
                    self.pretrained_model_path,
                    map_location=self.device,
                    weights_only=False, # Fallback
                )
                logger.info("Successfully loaded checkpoint with weights_only=False.")
            except Exception as e_false:
                logger.error(
                    f"Fallback loading (weights_only=False) also failed for {self.pretrained_model_path}: {e_false}"
                )
                raise # Re-raise if both attempts fail

        # Now, process the loaded checkpoint (if successful)
        if checkpoint_to_load is not None:
            try:
                # Check if it's a dictionary and remove the unexpected key before loading
                final_state_dict = checkpoint_to_load
                if isinstance(checkpoint_to_load, dict):
                    if '_metadata' in checkpoint_to_load:
                        logger.debug("Removing '_metadata' key from loaded checkpoint dictionary for Trainer.")
                        # Create a new dict if you want to be safe, or pop from the original
                        final_state_dict = {k: v for k, v in checkpoint_to_load.items() if k != '_metadata'}
                        # Or, if pop is okay: final_state_dict.pop('_metadata', None)
                
                self.model.load_state_dict(final_state_dict, strict=True)
                logger.info(f"Successfully applied state_dict to model from {self.pretrained_model_path}.")

            except Exception as e_load_state:
                logger.error(f"Error applying loaded state_dict to model: {e_load_state}")
                logger.debug(f"Keys in loaded checkpoint: {list(checkpoint_to_load.keys()) if isinstance(checkpoint_to_load, dict) else 'Not a dict'}")
                logger.debug(f"Model keys: {list(self.model.state_dict().keys())}")
                raise # Re-raise the error after logging details
        else:
            # This case should not be reached if the above logic raises on failure
            logger.error(f"Checkpoint for {self.pretrained_model_path} could not be loaded (is None).")


# --- Cross-Validation Function ---


def cross_validate(
    name: str,
    model_class: Type[nn.Module],
    criterion: nn.Module,
    dataset: Dataset,
    k_folds: int,
    num_epochs: int,
    hparams: Dict[str, Any],  # Note: hparams often passed via Optuna or directly
    is_sweep: bool,
    trial: Optional[optuna.Trial],
    config: Config,  # Use the specific Pydantic Config type
    full_dataset: HDF5Dataset,
) -> Tuple[float, Dict[str, float]]:
    """
    Performs K-fold cross-validation or a single train/validation split.

    Uses a single EarlyStopping instance to track the best model across all folds.
    """
    # --- Setup K-Fold or Single Split ---
    if k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.seed)
        splits = list(kf.split(np.arange(len(dataset))))  # Materialize splits
        desc = f"{k_folds}-fold Cross Validation"
        logger.info(f"Starting {k_folds}-fold cross-validation.")
    else:
        indices = np.arange(len(dataset))
        # Use validation_frac directly from Pydantic config
        train_idx, val_idx = train_test_split(
            indices,
            test_size=config.training.validation_frac,
            shuffle=True,
            random_state=config.seed,
        )
        splits = [(train_idx, val_idx)]
        desc = "Single Train/Validation Split"
        logger.info(
            f"Starting single train/validation split (Validation fraction: {config.training.validation_frac})."
        )

    fold_results_loss = []
    # Define metrics to aggregate across folds
    fold_metrics_keys = [
        "mse",
        "rmse",
        "r2",
        "mae",
        "loss",
        "data_loss",
        "continuity_loss",
        "momentum_x_loss",
        "momentum_y_loss",
    ]
    fold_metrics = {metric: [] for metric in fold_metrics_keys}

    # --- Early Stopping Initialization ---
    # Instantiated ONCE to track the best model across all folds.
    save_dir = "savepoints"
    os.makedirs(save_dir, exist_ok=True)
    early_stopping_path = os.path.join(save_dir, f"{name}_best_model.pth")
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        verbose=True,
        save_path=early_stopping_path,
        delta=config.training.early_stopping_delta,
    )
    logger.info(
        f"Early stopping enabled: Patience={config.training.early_stopping_patience}, Save Path='{early_stopping_path}'"
    )
    # The early_stopping instance keeps track of global_best_loss internally.

    # Use a shared generator for reproducibility in DataLoader shuffling/sampling
    g = torch.Generator()
    g.manual_seed(config.seed)

    # --- Fold Loop ---
    fold_pbar = tqdm(enumerate(splits), desc=desc, total=len(splits), leave=True)
    for fold, (train_idx, val_idx) in fold_pbar:
        fold_num = fold + 1
        fold_pbar.set_description(f"Fold {fold_num}/{len(splits)}")
        logger.info(f"--- Starting Fold {fold_num}/{len(splits)} ---")

        # --- Create DataLoaders for the current fold ---
        train_sampler = SubsetRandomSampler(train_idx, generator=g)  # Use generator
        val_sampler = SubsetRandomSampler(val_idx, generator=g)  # Use generator

        train_loader = DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=False,  # pin_memory=False if config.device == 'cuda' else True,
            worker_init_fn=seed_worker,  # Seed workers
            generator=g,  # Ensure sampler uses the generator
        )
        val_loader = DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            sampler=val_sampler,
            num_workers=config.training.num_workers,
            pin_memory=False,  # pin_memory=False if config.device == 'cuda' else True,
            worker_init_fn=seed_worker,  # Seed workers
            generator=g,  # Ensure sampler uses the generator
        )
        logger.info(
            f"Fold {fold_num}: Train samples={len(train_idx)}, Val samples={len(val_idx)}"
        )

        # --- Initialize Model, Optimizer, Scheduler, Scaler for the fold ---
        model = model_class(
            field_inputs_n=len(
                [p for p in config.data.inputs if p in config.data.non_scalars]
            ),
            scalar_inputs_n=len(
                [p for p in config.data.inputs if p in config.data.scalars]
            ),
            field_outputs_n=len(
                [p for p in config.data.outputs if p in config.data.non_scalars]
            ),
            scalar_outputs_n=len(
                [p for p in config.data.outputs if p in config.data.scalars]
            ),
            **hparams,  # Pass model-specific hparams
        ).to(config.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        steps_per_epoch = ceil(len(train_loader) / hparams["accumulation_steps"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hparams["learning_rate"],
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
        )

        use_scaler = config.device == "cuda" and not any(
            p.is_complex() for p in model.parameters()
        )
        scaler = GradScaler(enabled=use_scaler)
        if not use_scaler:
            logger.info(
                f"Fold {fold_num}: GradScaler disabled (Device: {config.device}, Complex Params: {any(p.is_complex() for p in model.parameters())})"
            )

        # --- Initialize Trainer for the fold ---
        trainer = Trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config.device,
            hparams["accumulation_steps"],
            config,
            full_dataset,
            hparams=hparams
        )

        # --- Reset Early Stopping State for the Fold ---
        # This resets the fold's counter, best_score etc., but crucially KEEPS
        # the global_best_loss and save_path from the __init__ call.
        early_stopping.reset()
        logger.debug(f"Fold {fold_num}: Early stopping state reset for the fold.")

        # --- Train the model for the current fold ---
        try:
            trainer.train(
                name,
                train_loader,
                val_loader,
                num_epochs,
                fold_num,
                is_sweep,
                trial,
                early_stopping,
            )
        except optuna.TrialPruned as e:
            logger.warning(f"Fold {fold_num} pruned by Optuna: {e}")
            raise e  # Re-raise to stop the cross-validation for this trial

        # --- Evaluate the *final state* model of this fold for reporting ---
        # We don't necessarily need to reload the best model *from this fold* here,
        # because EarlyStopping handles saving the overall best model across folds.
        # We just need metrics from this fold's validation set to average later.
        # Using the model state at the end of training for this fold's validation report.
        logger.info(f"Validating final model state for Fold {fold_num}...")
        val_metrics = trainer.validate(
            val_loader, name=f"{name}_fold{fold_num}", step=-1, fold_n=fold_num
        )

        # --- Store results for the fold ---
        fold_loss = val_metrics.get("loss", float("inf"))
        fold_results_loss.append(fold_loss)
        for key in fold_metrics_keys:
            fold_metrics[key].append(val_metrics.get(key, float("nan")))

        metrics_str = ", ".join(
            f"{k.replace('_', ' ').title()}={v:.4f}"
            for k, v in val_metrics.items()
            if isinstance(v, (int, float))
        )
        logger.info(f"Fold {fold_num} final validation results: {metrics_str}")

        # --- Clean up GPU memory ---
        del (
            model,
            optimizer,
            scheduler,
            scaler,
            trainer,
            train_loader,
            val_loader,
        )  # Keep criterion, dataset, config etc.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug(f"Fold {fold_num}: Cleaned up model and training components.")

    # --- Aggregate Results Across Folds ---
    avg_loss = np.nanmean(fold_results_loss)
    avg_metrics = {
        metric: np.nanmean(values)
        for metric, values in fold_metrics.items()
        if metric != "loss"  # Keep overall loss separate if desired
    }
    # Add the specific average loss metric back if needed
    avg_metrics["loss"] = avg_loss

    logger.info(f"Cross-validation finished. Average Validation Loss: {avg_loss:.4f}")
    avg_metrics_str = ", ".join(
        f"Avg {k.replace('_', ' ').title()}={v:.4f}"
        for k, v in avg_metrics.items()
        if k != "loss"  # Exclude avg loss here if already printed
    )
    logger.info(f"Average Validation Metrics: {avg_metrics_str}")

    return avg_loss, avg_metrics


# --- Main Cross-Validation Procedure ---


def cross_validation_procedure(
    name: str,
    model_class: Type[nn.Module],
    kfolds: int = 1,
    hparams: Optional[Dict[str, Any]] = None,
    is_sweep: bool = False,
    trial: Optional[optuna.Trial] = None,
    config: Optional[Config] = None,
    train_val_dataset: Optional[Dataset] = None,
    full_dataset_for_stats: Optional[HDF5Dataset] = None,
) -> float:
    """
    Orchestrates K-fold cross-validation on the provided train_val_dataset.

    Returns:
        float: The average validation loss (or other primary metric) across all folds.
               This value should be used by Optuna for optimization.
    """
    # --- Setup ---
    if config is None:
        config = get_config()
    if hparams is None:
        # Handle default case if needed, ensure hparams exists
        hparams = {**config.model.dict(), **config.training.dict(), **config.data.dict()} # Include data for normalize_output
        logger.info("No hyperparameters provided, using defaults derived from config.")

    # --- Dataset Loading and Splitting ---
    if train_val_dataset is None:
        logger.error(
            "train_val_dataset must be provided to cross_validation_procedure."
        )
        raise ValueError("train_val_dataset not provided.")
    if full_dataset_for_stats is None:
         logger.warning(
             "full_dataset_for_stats not provided. Denormalization or Criterion stats might fail."
         )
         raise ValueError("full_dataset_for_stats not provided.")
    logger.info(
        f"Starting cross-validation procedure for '{name}' with {len(train_val_dataset)} train/val samples."
    )
    logger.debug(f"Using hyperparameters for this trial: {hparams}") 
    
    # --- Criterion Definition ---
    criterion = PhysicsInformedLoss(
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        config=config,
        dataset=full_dataset_for_stats,
        use_physics_loss=hparams.get('use_physics_loss', config.training.use_physics_loss),
        normalize_output=hparams.get('normalize_output', config.data.normalize_output)
    )
    logger.info(
         f"Criterion initialized: PhysicsInformedLoss (Use Physics: {criterion.use_physics_loss}, Initial Lambda: 'ReLoBRaLo Enabled')"
    )

    # --- Cross-Validation ---
    logger.info("Starting cross-validation phase...")
    avg_cv_loss, avg_cv_metrics = cross_validate(
        name=name,
        model_class=model_class,
        criterion=criterion,
        dataset=train_val_dataset,
        k_folds=kfolds,
        num_epochs=config.training.num_epochs,
        hparams=hparams,
        is_sweep=is_sweep,
        trial=trial,
        config=config,
        full_dataset=full_dataset_for_stats,
    )
    logger.info(f"Cross-validation finished. Average CV Loss: {avg_cv_loss:.4f}")
    logger.info("Cross-validation procedure completed for this trial.")
    return avg_cv_loss