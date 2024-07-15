import gc
from typing import List, Tuple, Type

import numpy as np
import torch
from config import *
from loguru import logger
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from modules.utils import EarlyStopping


def train_model(
    name: str,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    accumulation_steps: int,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    writer: SummaryWriter,
    fold_n: int,
    validate_every: int = 1000,
    clip_grad_value: float = 1.0,
) -> None:
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(
        patience=20, verbose=True, save_path=f"savepoints/{name}_best_model.pth"
    )

    logger.info(f"Starting training for {num_epochs} epochs")
    with tqdm(total=num_epochs, desc="Epochs") as pbar:
        try:
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()

                for idx, (inputs, targets) in enumerate(train_dataloader):
                    inputs = [input.to(DEVICE) for input in inputs]
                    targets = targets.to(DEVICE)

                    with autocast(enabled=DEVICE == "cuda"):
                        outputs = model(inputs).view(targets.shape)
                        loss = criterion(outputs, targets)
                        loss = loss / accumulation_steps

                    scaler.scale(loss).backward()

                    if (idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), clip_grad_value)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()

                    train_loss += loss.item() * accumulation_steps

                    if (idx + 1) % validate_every == 0:
                        val_loss = validate_model(model, val_dataloader, criterion)
                        writer.add_scalar(f'Loss/train', train_loss / (idx + 1), epoch * len(train_dataloader) + idx)
                        writer.add_scalar(f'Loss/val', val_loss, epoch * len(train_dataloader) + idx)

                train_loss /= len(train_dataloader.dataset)
                val_loss = validate_model(model, val_dataloader, criterion)

                writer.add_scalar(f'Loss/train', train_loss, epoch)
                writer.add_scalar(f'Loss/val', val_loss, epoch)

                early_stopping(val_loss, model, epoch)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

                pbar.update(1)
                pbar.set_postfix(train_loss=train_loss, val_loss=val_loss)

        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            raise

        finally:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("Training completed")


def validate_model(
    model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module
) -> float:
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [input.to(DEVICE) for input in inputs]
            targets = targets.to(DEVICE)
            outputs = model(inputs).view(targets.shape)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(dataloader.dataset)
    return val_loss


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
    writer: SummaryWriter,
) -> np.ndarray:
    kfold = KFold(n_splits=k_folds, shuffle=True)
    results = []

    with tqdm(total=k_folds, desc=f"{k_folds} folds Cross Validation") as pbar:
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Starting fold {fold + 1}/{k_folds}")
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)

            train_dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=4
            )
            val_dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=4
            )

            model = model_class(
                len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y
            ).to(DEVICE)
            optimizer = optimizer_class(model.parameters(), lr=LEARNING_RATE)
            scheduler = scheduler_class(
                optimizer,
                max_lr=0.01,
                steps_per_epoch=len(train_dataloader),
                epochs=num_epochs,
            )
            scaler = GradScaler()

            train_model(
                name,
                model,
                train_dataloader,
                val_dataloader,
                num_epochs,
                accumulation_steps,
                criterion,
                optimizer,
                scheduler,
                scaler,
                writer,
                fold + 1
            )

            val_loss = validate_model(model, val_dataloader, criterion)
            results.append(val_loss)

            logger.info(f"Fold {fold + 1} results: Loss={val_loss:.4f}")
            pbar.update(1)

    return np.array(results)


def test_model(
    name: str,
    model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module
) -> float:
    model.load_state_dict(torch.load(f"savepoints/{name}_best_model.pth"))
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [input.to(DEVICE) for input in inputs]
            targets = targets.to(DEVICE)
            outputs = model(inputs).view(targets.shape)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(dataloader.dataset)
    return test_loss