from math import ceil
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import wandb
import h5py
from config import get_config
from loguru import logger
from sklearn.model_selection import KFold, train_test_split
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from tqdm.autonotebook import tqdm

from modules.data import HDF5Dataset
from modules.loss import PhysicsInformedLoss
from modules.models import *
from modules.plots import plot_hist as plot_difference
from modules.plots import plot_im as plot_difference_im
from modules.utils import EarlyStopping, compute_metrics, setup_logger


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        accumulation_steps,
        config,
    ):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.pretrained_model_path = None
        
        if self.config.training.pretrained_model_name:
            self.pretrained_model_path = f"savepoints/{self.config.training.pretrained_model_name}_best_model.pth"
            self._load_pretrained_model()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = self._move_to_device(inputs, targets)
            
            # Check if inputs contain complex tensors
            if any(isinstance(i, torch.Tensor) and i.dtype == torch.complex64 for i in inputs[0]):
                # Disable AMP if complex tensors are present
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.model(inputs)
                    loss = self.criterion(inputs, outputs, targets)
            else:
                with autocast(device_type=self.device, enabled=(self.device == "cuda"), dtype=torch.float32):
                    outputs = self.model(inputs)
                    loss = self.criterion(inputs, outputs, targets)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (idx + 1) % self.accumulation_steps == 0:
                self._step_optimization()

            total_loss += loss.item()

        if (idx + 1) % self.accumulation_steps != 0:
            self._step_optimization()

        return total_loss / len(dataloader)


    @torch.no_grad()
    def validate(self, dataloader, name="", step=-1, fold_n=-1):
        self.model.eval()
        total_loss = 0.0
        all_field_outputs, all_scalar_outputs = [], []
        all_field_targets, all_scalar_targets = [], []
        all_field_inputs, all_scalar_inputs = [], []
        has_scalar_data = False
        
        for batch in dataloader:
            inputs, targets = batch
            input_fields, input_scalars = inputs
            target_fields, target_scalars = targets
            
            # Move inputs and targets to device
            input_fields = [field.to(self.device) for field in input_fields]
            input_scalars = [scalar.to(self.device) for scalar in input_scalars] if input_scalars else []
            target_fields = [field.to(self.device) for field in target_fields]
            target_scalars = [scalar.to(self.device) for scalar in target_scalars] if target_scalars else []
            
            # Reconstruct the tuples
            inputs = (input_fields, input_scalars)
            targets = (target_fields, target_scalars)
            
            with autocast(
                device_type=self.device,
                enabled=self.device == "cuda",
                dtype=torch.float32,
            ):
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(inputs, outputs, targets)
                    
            total_loss += loss.item()
    
            # Collect field and scalar outputs and targets
            field_outputs, scalar_outputs = outputs

            if field_outputs is not None:
                all_field_outputs.append(field_outputs)
                if target_fields:
                    all_field_targets.append(torch.stack(target_fields, dim=1))
    
            # Only collect scalar outputs and targets if they exist
            if scalar_outputs is not None:
                all_scalar_outputs.append(scalar_outputs)
                if target_scalars:
                    all_scalar_targets.append(torch.stack(target_scalars, dim=1))  # Stack to preserve batch structure
    
        # After processing all batches, concatenate outputs and targets along batch dimension
        all_field_outputs = torch.cat(all_field_outputs, dim=0)  if all_field_outputs else None
        all_field_targets = torch.cat(all_field_targets, dim=0)  if all_field_targets else None
    
         # Only concatenate scalar outputs if they exist
        all_scalar_outputs = torch.cat(all_scalar_outputs, dim=0) if all_scalar_outputs else None
        all_scalar_targets = torch.cat(all_scalar_targets, dim=0) if all_scalar_targets else None
       
        # Compute metrics
        metrics = {}
        if all_field_outputs is not None:  # If field outputs are present
            metrics.update(compute_metrics(all_field_outputs, all_field_targets, variable_names=[var for var in self.config.data.outputs if var in self.config.data.non_scalars]))
    
        if all_scalar_outputs is not None:  # If scalar outputs are present
            metrics.update(compute_metrics(all_scalar_outputs, all_scalar_targets, variable_names=[var for var in self.config.data.outputs if var in self.config.data.scalars]))
    
        # Generate plots if enabled
        if self.config.logging.plot_enabled:
            plot_difference(
                all_field_outputs, all_field_targets, f"{name}_validation", step, fold_n
            )
            plot_difference_im(
                all_field_outputs, all_field_targets, f"{name}_validation", step, fold_n
            )
    
        # Calculate average loss
        metrics["loss"] = total_loss / len(dataloader)
    
        return metrics


    def _move_to_device(self, inputs, targets):
        # inputs and targets are tuples containing two lists of tensors
        inputs = tuple([
            [tensor.to(self.device) for tensor in input_list] 
            for input_list in inputs
        ])
        targets = tuple([
            [tensor.to(self.device) for tensor in target_list]
            for target_list in targets
        ])
        return inputs, targets
        
    def _step_optimization(self):
        if self.scaler is not None and not any(isinstance(p, torch.Tensor) and p.dtype == torch.complex64 for p in self.model.parameters()):
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(self.model.parameters(), self.config.training.clip_grad_value)
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    def train(
        self,
        name,
        train_dataloader,
        val_dataloader,
        num_epochs,
        fold_n,
        is_sweep,
        trial,
        early_stopping,
    ):
        if self.config.logging.use_wandb:
            wandb.init(
                project="Tesis",
                name=f"{name}_fold_{fold_n}",
                group=name,
                job_type="Sweep" if is_sweep else "Run",
                config={"Architecture": self.config.model.architecture},
            )

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc=f"Fold {fold_n}", position=0)

        best_val_loss = float("inf")
        for epoch in epoch_pbar:
            # Check estimated remaining time from tqdm
            estimated_remaining_time = (
                (epoch_pbar.total - epoch_pbar.n) / epoch_pbar.format_dict["rate"]
                if epoch_pbar.format_dict["rate"] and epoch_pbar.total
                else 0
            )
            # Check if the time limit is exceeded
            if (
                self.config.training.time_limit
                and estimated_remaining_time > self.config.training.time_limit
            ):
                logger.info(
                    f"Time limit exceeded for trial in fold {fold_n}. Pruning the trial."
                )
                raise optuna.TrialPruned()

            # Train epoch
            train_loss = self.train_epoch(train_dataloader)

            # Validate
            val_metrics = self.validate(val_dataloader, name, epoch, fold_n)
            val_loss = val_metrics["loss"]

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Update epoch progress bar
            epoch_pbar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Val Loss": f"{val_loss:.4f}",
                    "Best Val Loss": f"{best_val_loss:.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.8f}",
                }
            )

            if is_sweep:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if self.config.logging.use_wandb:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Train_loss": train_loss,
                        "Learning_Rate": self.scheduler.get_last_lr()[0],
                        **{f"Validation_{k}": v for k, v in val_metrics.items()},
                    }
                )

            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                epoch_pbar.set_postfix(
                    {
                        "Status": "Early Stopped",
                        "Best Epoch": f"{early_stopping.best_epoch}",
                        "Best Val Loss": f"{early_stopping.best_score:.4f}",
                    }
                )
                logger.info(
                    f"Early stopping triggered. Best epoch: {early_stopping.best_epoch}"
                )
                break

        if self.config.logging.use_wandb:
            wandb.finish()

    def _load_pretrained_model(self):
        try:
            checkpoint = torch.load(
                self.pretrained_model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded pretrained model from {self.pretrained_model_path}")
        except Exception as e:
            logger.error(
                f"Failed to load pretrained model from {self.pretrained_model_path}: {e}"
            )
            raise e


def cross_validate(
    name,
    model_class,
    criterion,
    dataset,
    k_folds,
    num_epochs,
    hparams,
    is_sweep,
    trial,
    config,
):
    if k_folds > 1:
        splits = KFold(n_splits=k_folds, shuffle=True, random_state=config.seed).split(
            dataset
        )
        desc = f"{k_folds}-fold Cross Validation"
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)),
            test_size=0.2,
            shuffle=True,
            random_state=config.seed,
        )
        splits = [(train_idx, val_idx)]
        desc = "Single Train-Test Split"

    results = []
    all_metrics = {metric: [] for metric in ["mse", "rmse", "r2", "mae"]}

    # Instantiate EarlyStopping once, to be shared across folds
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        verbose=True,
        save_path=f"savepoints/{name}_best_model.pth",
    )

    for fold, (train_idx, val_idx) in enumerate(tqdm(splits, desc=desc, total=k_folds)):
        train_loader = DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            sampler=SubsetRandomSampler(train_idx),
            num_workers=config.training.num_workers,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            sampler=SubsetRandomSampler(val_idx),
            num_workers=config.training.num_workers,
        )

        model = model_class(
            field_inputs_n=len([param for param in config.data.inputs if param in config.data.non_scalars]),
            scalar_inputs_n=len([param for param in config.data.inputs if param in config.data.scalars]),
            field_outputs_n=len([param for param in config.data.outputs if param in config.data.non_scalars]),
            scalar_outputs_n=len([param for param in config.data.outputs if param in config.data.scalars]),
            **hparams,
        ).to(config.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=hparams["learning_rate"],
            steps_per_epoch=ceil(
                len(train_loader) / hparams["accumulation_steps"]
            ),
            epochs=num_epochs,
        )
        scaler = GradScaler(enabled=(config.device == "cuda"))
        # scaler = None

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config.device,
            hparams["accumulation_steps"],
            config,
        )

        # Reset early stopping for each fold, but keep the best model across folds
        early_stopping.reset()

        trainer.train(
            name,
            train_loader,
            val_loader,
            num_epochs,
            fold,
            is_sweep,
            trial,
            early_stopping,  # Pass EarlyStopping instance to each fold
        )

        val_metrics = trainer.validate(val_loader, name, -1, fold)
        results.append(val_metrics["loss"])
        for key in all_metrics:
            all_metrics[key].append(val_metrics[key])

        logger.info(
            f"Fold {fold + 1} results: {', '.join(f'{k.capitalize()}={v:.4f}' for k, v in val_metrics.items())}"
        )
        torch.cuda.empty_cache()

    avg_metrics = {
        metric: torch.mean(torch.tensor(values)).item()
        for metric, values in all_metrics.items()
    }
    avg_loss = np.mean(results)
    return avg_loss, avg_metrics


def cross_validation_procedure(
    name,
    data_path,
    model_class,
    kfolds=1,
    hparams=None,
    is_sweep=False,
    trial=None,
    config=None,
):
    logger.info("Starting cross-validation procedure")

    full_dataset = HDF5Dataset(
        file_path=data_path,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize_input=config.data.normalize_input,
        normalize_output=config.data.normalize_output,
        device=config.device,
        preload=config.data.preload,
    )

    test_size = int(config.training.test_frac * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    logger.info(
        f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)"
    )

    criterion = PhysicsInformedLoss(
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        dataset=full_dataset,
        config=config,
        lambda_physics=hparams["lambda_physics"]
    )
    
    avg_loss, avg_metrics = cross_validate(
        name,
        model_class,
        criterion,
        train_val_dataset,
        kfolds,
        config.training.num_epochs,
        hparams,
        is_sweep,
        trial,
        config,
    )

    logger.info(f"Cross-validation completed. Avg loss: {avg_loss:.4f}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model = model_class(
            field_inputs_n=len([param for param in config.data.inputs if param in config.data.non_scalars]),
            scalar_inputs_n=len([param for param in config.data.inputs if param in config.data.scalars]),
            field_outputs_n=len([param for param in config.data.outputs if param in config.data.non_scalars]),
            scalar_outputs_n=len([param for param in config.data.outputs if param in config.data.scalars]),
            **hparams,
        ).to(config.device)
    model.load_state_dict(
        torch.load(f"savepoints/{name}_best_model.pth", weights_only=True)
    )

    trainer = Trainer(
        model, criterion, None, None, None, config.device, 1, config=config
    )
    test_metrics = trainer.validate(test_loader)
    logger.info(
        f"Test metrics: {', '.join(f'{k.capitalize()}={v:.4f}' for k, v in test_metrics.items())}"
    )

    if config.logging.use_wandb:
        wandb.init(
            project="Tesis",
            name=name,
            config=hparams,
            group=name,
            job_type="Sweep" if is_sweep else "Run",
        )
        wandb.config.update(
            {
                "Test_Loss": test_metrics["loss"],
                "Cross_Loss": avg_loss,
                **avg_metrics,
                "architecture": config.model.architecture,
                "study": config.optuna.study_name,
            }
        )
        wandb.finish()

    logger.info("Cross-validation procedure completed")
    return test_metrics["loss"]