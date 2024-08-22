import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import wandb
from config import get_config
from loguru import logger
from sklearn.model_selection import KFold, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torcheval.metrics.functional import mean_squared_error, r2_score
from tqdm.auto import tqdm

from modules.data import HDF5Dataset
from modules.logging import setup_logger
from modules.models import *
from modules.plots import plot_im as plot_difference_im
from modules.plots import plot_hist as plot_difference
from modules.utils import EarlyStopping

def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    variable_names: List[str],
    variable_units: List[str],
) -> Dict[str, torch.Tensor]:
    outputs, targets = (
        outputs.view(-1, len(variable_names)),
        targets.view(-1, len(variable_names)),
    )

    metrics = {}

    # Overall metrics
    mse = mean_squared_error(targets, outputs)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(targets - outputs))
    r2 = r2_score(targets, outputs)

    metrics.update(
        {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }
    )

    # Normalized Root Mean Square Error (NRMSE)
    nrmse = rmse / (targets.max() - targets.min())
    metrics["nrmse"] = nrmse

    # Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-8  # Small value to avoid division by zero
    mape = torch.mean(torch.abs((targets - outputs) / (targets + epsilon))) * 100
    metrics["mape"] = mape

    # Compute metrics for each variable
    for i, (var_name, var_unit) in enumerate(zip(variable_names, variable_units)):
        var_outputs = outputs[:, i]
        var_targets = targets[:, i]

        var_mse = mean_squared_error(var_targets, var_outputs)
        var_rmse = torch.sqrt(var_mse)
        var_mae = torch.mean(torch.abs(var_targets - var_outputs))
        var_r2 = r2_score(var_targets, var_outputs)

        metrics.update(
            {
                f"{var_name}_mse": var_mse,
                f"{var_name}_rmse": var_rmse,
                f"{var_name}_mae": var_mae,
                f"{var_name}_r2": var_r2,
            }
        )

        # For variables with units, add interpretable metrics
        if var_unit != "dimensionless":
            var_mean_error = torch.mean(var_outputs - var_targets)
            var_std_error = torch.std(var_outputs - var_targets)

            metrics.update(
                {
                    f"{var_name}_mean_error_{var_unit}": var_mean_error,
                    f"{var_name}_std_error_{var_unit}": var_std_error,
                }
            )

    return metrics

class Trainer:
    def __init__(
        self,
        model,
        architecture,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        accumulation_steps,
        clip_grad_value,
        variable_names,
        variable_units,
    ):
        self.model = model
        self.architecture = architecture
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.clip_grad_value = clip_grad_value
        self.variable_names = variable_names
        self.variable_units = variable_units

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = [input.to(self.device) for input in inputs]
            targets = targets.to(self.device)
            with autocast(enabled=self.device == "cuda", dtype=torch.float32):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / self.accumulation_steps
            self.scaler.scale(loss).backward()
            if (idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.clip_grad_value)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            total_loss += loss.item() * self.accumulation_steps
        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader, name="", step=-1, fold_n=-1, plot_enabled=True):
        self.model.eval()
        total_loss = 0.0
        all_outputs, all_targets = [], []
        for inputs, targets in dataloader:
            inputs = [input.to(self.device) for input in inputs]
            targets = targets.to(self.device)
            with autocast(enabled=self.device == "cuda", dtype=torch.float32):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            all_outputs.append(outputs)
            all_targets.append(targets)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        if plot_enabled:
            plot_difference(
                all_outputs, all_targets, f"{name}_validation", step, fold_n
            )
            plot_difference_im(
                all_outputs, all_targets, f"{name}_validation", step, fold_n
            )

        metrics = compute_metrics(
            all_outputs, all_targets, self.variable_names, self.variable_units
        )
        metrics["loss"] = total_loss / len(dataloader)
        return metrics

    def train(
        self,
        name,
        train_dataloader,
        val_dataloader,
        num_epochs,
        fold_n,
        use_wandb,
        is_sweep,
        trial,
        plot_enabled
    ):
        config = get_config()
        early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            verbose=True,
            save_path=f"savepoints/{name}_best_model.pth",
        )
        if use_wandb:
            wandb.init(
                project="Tesis",
                name=f"{name}_fold_{fold_n}",
                group=name,
                job_type="Sweep" if is_sweep else "Run",
                config={"architecture": self.architecture}
            )

        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc=f"Fold {fold_n}", position=0)
        
        best_val_loss = float('inf')
        for epoch in epoch_pbar:
            # Train epoch
            train_loss = self.train_epoch(train_dataloader)
            
            # Validate
            val_metrics = self.validate(
                val_dataloader, name, epoch, fold_n, plot_enabled
            )
            val_loss = val_metrics["loss"]

            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Best Val Loss': f'{best_val_loss:.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

            if is_sweep:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if use_wandb:
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
                epoch_pbar.set_postfix({
                    'Status': 'Early Stopped',
                    'Best Epoch': f'{early_stopping.best_epoch}',
                    'Best Val Loss': f'{early_stopping.best_score:.4f}'
                })
                logger.info(f"Early stopping triggered. Best epoch: {early_stopping.best_epoch}")
                break

        if use_wandb:
            wandb.finish()

def cross_validate(
    name,
    model_class,
    dataset,
    k_folds,
    num_epochs,
    hparams,
    use_wandb,
    is_sweep,
    trial,
    plot_enabled,
    architecture,
):
    if k_folds > 1:
        splits = KFold(n_splits=k_folds, shuffle=True, random_state=42).split(dataset)
        desc = f"{k_folds}-fold Cross Validation"
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), test_size=0.2, shuffle=True, random_state=42
        )
        splits = [(train_idx, val_idx)]
        desc = "Single Train-Test Split"

    results = []
    all_metrics = {metric: [] for metric in ["mse", "rmse", "r2", "mae"]}

    config = get_config()

    for fold, (train_idx, val_idx) in enumerate(tqdm(splits, desc=desc)):
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
            len(config.data.parameters),
            len(config.data.variables),
            config.data.numpoints_x,
            config.data.numpoints_y,
            **hparams,
        ).to(config.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["learning_rate"])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader)
        )
        scaler = GradScaler()

        trainer = Trainer(
            model,
            architecture,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config.device,
            hparams["accumulation_steps"],
            1.0,
            config.data.variables,
            config.data.variable_units,
        )
        trainer.train(
            name,
            train_loader,
            val_loader,
            num_epochs,
            fold + 1,
            use_wandb,
            is_sweep,
            trial,
            plot_enabled,
        )

        val_metrics = trainer.validate(val_loader, plot_enabled=plot_enabled)
        results.append(val_metrics["loss"])
        for metric in all_metrics:
            all_metrics[metric].append(val_metrics[metric])

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
    criterion,
    kfolds=1,
    hparams=None,
    use_wandb=False,
    is_sweep=False,
    trial=None,
    plot_enabled=False,
    architecture=None,
):
    logger.info("Starting cross-validation procedure")

    config = get_config()

    full_dataset = HDF5Dataset(
        file_path=data_path,
        variables=config.data.variables,
        parameters=config.data.parameters,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        normalize=config.data.normalize,
        device=config.device,
    )

    test_size = int(config.training.test_frac * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(config.seed)
    )

    logger.info(
        f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)"
    )

    avg_loss, avg_metrics = cross_validate(
        name,
        model_class,
        train_val_dataset,
        kfolds,
        config.training.num_epochs,
        hparams,
        use_wandb,
        is_sweep,
        trial,
        plot_enabled,
        architecture,
    )

    logger.info(f"Cross-validation completed. Avg loss: {avg_loss:.4f}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model = model_class(
        len(config.data.parameters),
        len(config.data.variables),
        config.data.numpoints_x,
        config.data.numpoints_y,
        **hparams,
    ).to(config.device)
    model.load_state_dict(torch.load(f"savepoints/{name}_best_model.pth"))

    trainer = Trainer(
        model,
        architecture,
        criterion,
        None,
        None,
        None,
        config.device,
        1,
        1.0,
        variable_names=config.data.variables,
        variable_units=config.data.variable_units,
    )
    test_metrics = trainer.validate(test_loader)
    logger.info(
        f"Test metrics: {', '.join(f'{k.capitalize()}={v:.4f}' for k, v in test_metrics.items())}"
    )

    if use_wandb:
        wandb.init(
            project="Tesis",
            name=name,
            config=hparams,
            group=name,
            job_type="Sweep" if is_sweep else "Run",
        )
        wandb.config.update(
            {"Test_Loss": test_metrics["loss"], "Cross_Loss": avg_loss, **avg_metrics, "architecture": architecture}
        )
        wandb.finish()

    logger.info("Cross-validation procedure completed")
    return test_metrics["loss"]