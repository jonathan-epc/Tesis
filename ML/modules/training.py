import gc
import inspect
from typing import Dict, List, Tuple, Type, Optional, Any

import numpy as np
import torch
import wandb
from config import CONFIG
from loguru import logger
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torcheval.metrics.functional import mean_squared_error, r2_score
from tqdm.autonotebook import tqdm

from modules.plots import plot_hex as plot_difference
from modules.plots import plot_im as plot_difference_im
from modules.utils import EarlyStopping

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute regression metrics: MSE, RMSE, R2, and MAE.

    CONFIG['data']['parameters']
    ----------
    outputs : torch.Tensor
        The predicted outputs from the model.
    targets : torch.Tensor
        The true targets.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        MSE, RMSE, R2, and MAE metrics.
    """
    outputs = outputs.view(-1)
    targets = targets.view(-1)

    mse = mean_squared_error(targets, outputs)
    rmse = torch.sqrt(mse)
    r2 = r2_score(targets, outputs)
    mae = torch.mean(torch.abs(targets - outputs))

    return mse, rmse, r2, mae

def train_model(
    name: str,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    fold_n: int,
    accumulation_steps: int = 1,
    validate_every: int = 1000,
    clip_grad_value: float = 1.0,
    use_wandb: bool = False,
    is_sweep: bool = False,
    architecture: str = None,
    plot_every: int = 100,
    plot_enabled: bool = True,
) -> None:
    """
    Train the model for a specified number of epochs.

    CONFIG['data']['parameters']
    ----------
    name : str
        The name of the experiment.
    model : torch.nn.Module
        The model to train.
    train_dataloader : DataLoader
        DataLoader for the training data.
    val_dataloader : DataLoader
        DataLoader for the validation data.
    num_epochs : int
        Number of epochs to train.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    scaler : GradScaler
        Gradient scaler for mixed precision training.
    fold_n : int
        Fold number for cross-validation.
    accumulation_steps : int, optional
        Gradient accumulation steps, by default 1.
    validate_every : int, optional
        Validation frequency, by default 1000.
    clip_grad_value : float, optional
        Gradient clipping value, by default 1.0.
    use_wandb : bool, optional
        Whether to use Weights & Biases for logging, by default False.
    is_sweep : bool, optional
        Whether this is a sweep run for hyperparameter tuning, by default False.
    architecture : str, optional
        Model architecture name, by default None.
    plot_every : int, optional
        Plotting frequency, by default 100.
    plot_enabled : bool, optional
        Whether plotting is enabled, by default True.

    Returns
    -------
    None
    """
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(
        patience=CONFIG['training']['early_stopping_patience'],
        verbose=True,
        save_path=f"savepoints/{name}_best_model.pth",
    )
    logger.info(f"Starting training for {num_epochs} epochs")
    if use_wandb:
        if is_sweep:
            fold_run = wandb.init(
                project="Tesis",
                name=f"{name}_fold_{fold_n}",
                group=f"{name}",
                job_type="Sweep",
            )
        else:
            fold_run = wandb.init(
                project="Tesis", name=f"{name}_fold_{fold_n}", group=f"{name}"
            )

    global_step = 0
    with tqdm(total=num_epochs, desc=f"Fold {fold_n}") as pbar:
        try:
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()
                for idx, (inputs, targets) in enumerate(train_dataloader):
                    inputs = [input.to(CONFIG['device']) for input in inputs]
                    targets = targets.to(CONFIG['device'])
                    with autocast(enabled=CONFIG['device'] == "cuda", dtype=torch.float32):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        if accumulation_steps > 1:
                            loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                    if accumulation_steps == 1 or (idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), clip_grad_value)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    train_loss += loss.item() * (
                        accumulation_steps if accumulation_steps > 1 else 1
                    )

                    # Plot and save difference image if enabled
                    if (
                        plot_enabled
                        and plot_every > 0
                        and global_step % plot_every == 0
                    ):
                        plot_difference(outputs, targets, name, global_step, fold_n)
                        plot_difference_im(outputs, targets, name, global_step, fold_n)

                    global_step += 1

                train_loss /= len(train_dataloader)
                val_loss, val_metrics = validate_model(
                    name,
                    model,
                    val_dataloader,
                    criterion,
                    global_step,
                    fold_n,
                    plot_enabled=plot_enabled,
                )
                step = epoch
                if use_wandb:
                    fold_run.log(
                        {
                            "Train_loss": train_loss,
                            "Validation_loss": val_loss,
                            "Validation_mse": val_metrics["mse"],
                            "Validation_rmse": val_metrics["rmse"],
                            "Validation_r2": val_metrics["r2"],
                            "Validation_mae": val_metrics["mae"],
                        }
                    )
                early_stopping(val_loss, model, epoch)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break
                pbar.update(1)
                pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)
            if use_wandb:
                fold_run.finish()
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()
        logger.info("Training completed")

def validate_model(
    name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    step: int = -1,
    fold_n: int = -1,
    plot_enabled: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model on a validation set.

    CONFIG['data']['parameters']
    ----------
    name : str
        The name of the experiment.
    model : torch.nn.Module
        The model to validate.
    dataloader : DataLoader
        DataLoader for the validation data.
    criterion : torch.nn.Module
        Loss function.
    step : int, optional
        Current training step, by default -1.
    fold_n : int, optional
        Fold number for cross-validation, by default -1.
    plot_enabled : bool, optional
        Whether plotting is enabled, by default True.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Validation loss and metrics.
    """
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [input.to(CONFIG['device']) for input in inputs]
            targets = targets.to(CONFIG['device'])
            with autocast(enabled=CONFIG['device'] == "cuda", dtype=torch.float32):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)

    if plot_enabled:
        plot_difference(outputs, targets, name + "_validation", step, fold_n)
        plot_difference_im(outputs, targets, name + "_validation", step, fold_n)

    val_loss /= len(dataloader)
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    mse, rmse, r2, mae = compute_metrics(all_outputs, all_targets)

    metrics = {"loss": val_loss, "mse": mse, "rmse": rmse, "r2": r2, "mae": mae}
    return val_loss, metrics

def cross_validate(
    name: str,
    model_class: Type[torch.nn.Module],
    dataset: Dataset,
    k_folds: int,
    num_epochs: int,
    accumulation_steps: int,
    criterion: torch.nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler],
    hparams: Optional[Dict[str, Any]] = None,
    use_wandb: bool = False,
    is_sweep: bool = False,
    architecture: str = None,
    plot_enabled: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Perform k-fold cross-validation.

    CONFIG['data']['parameters']
    ----------
    name : str
        The name of the experiment.
    model_class : Type[torch.nn.Module]
        The class of the model to instantiate.
    dataset : Dataset
        The dataset for cross-validation.
    k_folds : int
        Number of folds for cross-validation.
    num_epochs : int
        Number of epochs to train for each fold.
    accumulation_steps : int
        Gradient accumulation steps.
    criterion : torch.nn.Module
        Loss function.
    optimizer_class : Type[torch.optim.Optimizer]
        Optimizer class.
    scheduler_class : Type[torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler class.
    use_wandb : bool, optional
        Whether to use Weights & Biases for logging, by default False.
    is_sweep : bool, optional
        Whether this is a sweep run for hyperparameter tuning, by default False.
    architecture : str, optional
        Model architecture name, by default None.
    plot_enabled : bool, optional
        Whether plotting is enabled, by default False.

    Returns
    -------
    Tuple[float, Dict[str, float]]
        Average validation loss and metrics.
    """
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = []
    all_metrics = {"mse": [], "rmse": [], "r2": [], "mae": []}

    with tqdm(total=k_folds, desc=f"{k_folds} folds Cross Validation") as pbar:
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Starting fold {fold + 1}/{k_folds}")
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)

            train_dataloader = DataLoader(
                dataset,
                batch_size=hparams['batch_size'],
                sampler=train_subsampler,
                num_workers=CONFIG['training']['num_workers'],
            )
            val_dataloader = DataLoader(
                dataset,
                batch_size=hparams['batch_size'],
                sampler=val_subsampler,
                num_workers=CONFIG['training']['num_workers'],
            )

            model = model_class(len(CONFIG['data']['parameters']), len(CONFIG['data']['variables']), CONFIG['data']['numpoints_x'], CONFIG['data']['numpoints_y'], **hparams).to(CONFIG['device'])
            optimizer = optimizer_class(
                model.parameters(), lr=hparams['learning_rate'], weight_decay=0.0
            )
            scheduler = scheduler_class(
                optimizer,
                max_lr=0.01,
                epochs=num_epochs,
                steps_per_epoch=len(train_dataloader),
            )
            scaler = GradScaler()

            train_model(
                name,
                model,
                train_dataloader,
                val_dataloader,
                num_epochs,
                criterion,
                optimizer,
                scheduler,
                scaler,
                fold + 1,
                accumulation_steps,
                use_wandb=use_wandb,
                is_sweep=is_sweep,
                architecture=architecture,
                plot_enabled=plot_enabled,
            )

            val_loss, val_metrics = validate_model(
                name, model, val_dataloader, criterion, plot_enabled=plot_enabled
            )
            results.append(val_loss)

            all_metrics["mse"].append(val_metrics["mse"])
            all_metrics["rmse"].append(val_metrics["rmse"])
            all_metrics["r2"].append(val_metrics["r2"])
            all_metrics["mae"].append(val_metrics["mae"])

            logger.info(
                f"Fold {fold + 1} results: Loss={val_loss:.4f}, MSE={val_metrics['mse']:.4f}, RMSE={val_metrics['rmse']:.4f}, R2={val_metrics['r2']:.4f}, MAE={val_metrics['mae']:.4f}"
            )
            pbar.update(1)
            torch.cuda.empty_cache()

    avg_metrics = {
        metric: torch.mean(torch.tensor(all_metrics[metric])) for metric in all_metrics
    }
    avg_results = np.array(results).mean()

    return avg_results, avg_metrics

def test_model(
    name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
) -> float:
    """
    Test the model on a test set.

    CONFIG['data']['parameters']
    ----------
    name : str
        The name of the experiment.
    model : torch.nn.Module
        The model to test.
    dataloader : DataLoader
        DataLoader for the test data.
    criterion : torch.nn.Module
        Loss function.

    Returns
    -------
    float
        Test loss.
    """
    model.load_state_dict(torch.load(f"savepoints/{name}_best_model.pth"))
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [input.to(CONFIG['device']) for input in inputs]
            targets = targets.to(CONFIG['device'])
            with autocast(enabled=CONFIG['device'] == "cuda", dtype=torch.float32):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

    test_loss /= len(dataloader)
    return test_loss
