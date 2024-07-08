import os
from timeit import default_timer
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from typing import List, Tuple, Dict, Any, Optional, Callable, Type, Union
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from loguru import logger
import sys
import numpy as np
from time import sleep
from neuralop.models import SFNO

# Constants
NUMPOINTS_X = 401
NUMPOINTS_Y = 11
VARIABLES = ["F", "H", "Q", "S", "U", "V"]
PARAMETERS = ["H0", "Q0", "SLOPE", "n"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 256
ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20


def compute_metrics(
    outputs: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Computes mean squared error, mean absolute error, and R2 score.

    Args:
        outputs (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.

    Returns:
        tuple: MSE, MAE, and R2 score.
    """
    outputs = outputs.view(-1).detach().cpu().numpy()
    targets = targets.view(-1).detach().cpu().numpy()
    mse = mean_squared_error(targets, outputs)
    mae = mean_absolute_error(targets, outputs)
    r2 = r2_score(targets, outputs)
    return mse, mae, r2


class HDF5Dataset(Dataset):
    """
    Custom Dataset for loading HDF5 files.

    Args:
        file_path (str): Path to the HDF5 file.
        variables (list): List of variable names.
        parameters (list): List of parameter names.
        numpoints_x (int): Number of points in the x dimension.
        numpoints_y (int): Number of points in the y dimension.
        normalized (bool): Whether to normalize data.
        swap (bool): Whether to swap outputs and inputs.
        transform (callable, optional): Optional transform to apply to inputs.
        target_transform (callable, optional): Optional transform to apply to targets.
    """


class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        variables: List[str],
        parameters: List[str],
        numpoints_x: int,
        numpoints_y: int,
        normalized: bool = True,
        swap: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.file_path = file_path
        try:
            with h5py.File(self.file_path, "r") as data:
                self.keys = [key for key in data.keys() if key != "statistics"]
                self.stat_means = {
                    param: data["statistics"].attrs[f"{param}_mean"]
                    for param in parameters + ["B"] + variables
                }
                self.stat_stds = {
                    param: np.sqrt(data["statistics"].attrs[f"{param}_variance"])
                    for param in parameters + ["B"] + variables
                }
        except IOError as e:
            logger.error(f"Error opening file {self.file_path}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Error accessing data in file {self.file_path}: {e}")
            raise
        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.normalized = normalized
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform
        self.data = {}
        try:
            with h5py.File(self.file_path, "r") as data:
                for key in self.keys:
                    self.data[key] = {
                        param: data[key].attrs[param] for param in self.parameters
                    }
                    self.data[key]["B"] = data[key]["B"][()]
                    for var in self.variables:
                        self.data[key][var] = data[key][var][()]
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {e}")
            raise

    def __len__(self):
        return len(self.keys)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        try:
            key = self.keys[idx]
            case = self.data[key]

            parameters = [case[param] for param in self.parameters]
            B = case["B"].reshape(self.numpoints_y, self.numpoints_x)
            output = [
                case[var].reshape(self.numpoints_y, self.numpoints_x)
                for var in self.variables
            ]

            if not self.normalized:
                parameters = [
                    self.normalize(p, param)
                    for p, param in zip(parameters, self.parameters)
                ]
                B = self.normalize(B, "B")
                output = [
                    self.normalize(o, var) for o, var in zip(output, self.variables)
                ]

            parameters_normalized = torch.tensor(parameters).double()
            B_normalized = torch.tensor(B).double()
            output = torch.stack([torch.tensor(o).double() for o in output])

            if self.transform:
                parameters_normalized = self.transform(parameters_normalized)
                B_normalized = self.transform(B_normalized)
            if self.target_transform:
                output = self.target_transform(output)

            if self.swap:
                return output, [parameters_normalized, B_normalized]
            else:
                return [parameters_normalized, B_normalized], output

        except IndexError:
            logger.error(f"Index {idx} out of range for dataset")
            raise
        except KeyError as e:
            logger.error(f"Error accessing data for key {key}: {e}")
            raise

    def normalize(self, data: np.ndarray, prefix: str) -> np.ndarray:
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return (data - mean) / std

    def denormalize(self, data: np.ndarray, prefix: str) -> np.ndarray:
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return data * std + mean


def create_dataloaders(
    file_path: str,
    variables: List[str],
    parameters: List[str],
    numpoints_x: int,
    numpoints_y: int,
    batch_size: int,
    do_cross_validation: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    try:
        dataset = HDF5Dataset(
            file_path, variables, parameters, numpoints_x, numpoints_y
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

    try:
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    except ValueError as e:
        logger.error(f"Error splitting dataset: {e}")
        raise

    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        raise

    return train_dataloader, val_dataloader, test_dataloader


def cross_validate(
    model_class: Type[nn.Module],
    dataset: Dataset,
    k_folds: int,
    num_epochs: int,
    accumulation_steps: int,
    criterion: nn.Module,
    optimizer_class: Type[torch.optim.Optimizer],
    scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    writer: SummaryWriter,
) -> np.ndarray:
    kfold = KFold(n_splits=k_folds, shuffle=True)
    with tqdm(total=k_folds, desc=f"{k_folds} folds Cross Validation") as pbar:
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            fold_results = []
            logger.info(f"Starting fold {fold+1}/{k_folds}")
            # Create a new writer for each fold
            fold_writer = SummaryWriter(comment=f"- fold-{fold+1}")
            train_subsampler = data_utils.SubsetRandomSampler(train_idx)
            val_subsampler = data_utils.SubsetRandomSampler(val_idx)

            train_dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, num_workers=4
            )
            val_dataloader = DataLoader(
                dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, num_workers=4
            )

            model = (
                model_class(len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y)
                .to(torch.double)
                .to(DEVICE)
            )
            optimizer = optimizer_class(model.parameters(), lr=LEARNING_RATE)
            scheduler = scheduler_class(
                optimizer,
                max_lr=0.01,
                steps_per_epoch=len(train_dataloader),
                epochs=num_epochs,
            )
            logger.info(f"Training model for fold {fold+1}")
            train_model(
                model,
                train_dataloader,
                val_dataloader,
                num_epochs,
                accumulation_steps,
                criterion,
                optimizer,
                scheduler,
                scaler,
                fold_writer,
            )

            val_loss, val_mse, val_mae, val_r2 = validate_model(
                model, val_dataloader, criterion
            )
            fold_results.append((val_loss, val_mse, val_mae, val_r2))

            logger.log(
                "METRIC",
                f"Fold {fold+1} results: Loss={val_loss:.4f}, MSE={val_mse:.4f}, MAE={val_mae:.4f}, R2={val_r2:.4f}",
            )
            pbar.update(1)

            # Close the writer for the current fold
            fold_writer.close()

    avg_results = np.mean(fold_results, axis=0)
    # Add the avg_results to TensorBoard
    writer.add_scalar("CrossVal/Loss", avg_results[0])
    writer.add_scalar("CrossVal/MSE", avg_results[1])
    writer.add_scalar("CrossVal/MAE", avg_results[2])
    writer.add_scalar("CrossVal/R2", avg_results[3])
    logger.log(
        "METRIC",
        f"Cross-validation results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}",
    )
    return avg_results


def validate_model(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module
) -> Tuple[float, float, float, float]:
    model.eval()
    val_loss, val_mse, val_mae, val_r2 = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            mse, mae, r2 = compute_metrics(outputs, targets)
            val_mse += mse
            val_mae += mae
            val_r2 += r2
    val_loss /= len(dataloader)
    val_mse /= len(dataloader)
    val_mae /= len(dataloader)
    val_r2 /= len(dataloader)
    return val_loss, val_mse, val_mae, val_r2


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int,
    accumulation_steps: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    writer: SummaryWriter,
) -> None:
    best_val_loss = float("inf")
    early_stopping_counter = 0

    logger.info(f"Starting training for {num_epochs} epochs")
    with tqdm(total=num_epochs, desc="Fold") as pbar:
        for epoch in range(num_epochs):
            model.train()
            train_loss, train_mse, train_mae, train_r2 = 0, 0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            logger.debug(f"Starting epoch {epoch+1}")
            for idx, (inputs, targets) in enumerate(train_dataloader):
                with autocast(enabled=DEVICE == "cuda"):
                    outputs = model(inputs)
                    outputs = outputs.view(targets.shape)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()

                if (idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item()
                mse, mae, r2 = compute_metrics(outputs, targets)
                train_mse += mse
                train_mae += mae
                train_r2 += r2

            train_loss /= len(train_dataloader)
            train_mse /= len(train_dataloader)
            train_mae /= len(train_dataloader)
            train_r2 /= len(train_dataloader)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("MSE/train", train_mse, epoch)
            writer.add_scalar("MAE/train", train_mae, epoch)
            writer.add_scalar("R2/train", train_r2, epoch)
            # logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
            val_loss, val_mse, val_mae, val_r2 = validate_model(
                model, val_dataloader, criterion
            )
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("MSE/val", val_mse, epoch)
            writer.add_scalar("MAE/val", val_mae, epoch)
            writer.add_scalar("R2/val", val_r2, epoch)
            # logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "models/best_model.pth")
                logger.info(f"New best model saved at epoch {epoch+1}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                logger.warning(f"Early stopping triggered at epoch {epoch+1}")
                break

            scheduler.step()
            pbar.update(1)
            pbar.set_postfix(
                {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
            )
    logger.info("Training completed")


def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    writer: SummaryWriter,
    hparams: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.load_state_dict(torch.load("models/best_model.pth"))
    model.eval()
    test_loss, test_mse, test_mae, test_r2 = 0, 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            mse, mae, r2 = compute_metrics(outputs, targets)
            test_mse += mse
            test_mae += mae
            test_r2 += r2

    test_loss /= len(dataloader)
    test_mse /= len(dataloader)
    test_mae /= len(dataloader)
    test_r2 /= len(dataloader)

    logger.info(
        f"Test Loss: {test_loss:.4f}\nTest MSE: {test_mse:.4f}\nTest MAE: {test_mae:.4f}\nTest R2: {test_r2:.4f}"
    )

    metrics = {
        "test/Loss": test_loss,
        "test/MSE": test_mse,
        "test/MAE": test_mae,
        "test/R2": test_r2,
    }
    writer.add_hparams(hparams, metrics, run_name=".")
    inputs_for_plot = inputs
    targets_for_plot = targets
    outputs_for_plot = outputs
    return inputs_for_plot, targets_for_plot, outputs_for_plot


def setup_writer_and_hparams(comment: str) -> Tuple[SummaryWriter, Dict[str, Any]]:
    writer = SummaryWriter(comment=comment)
    hparams = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
    }
    return writer, hparams


def cross_validation_procedure(
    name: str, neural_network: Type[nn.Module]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("Starting cross-validation procedure")

    # Load the complete dataset
    logger.info("Loading dataset")
    full_dataset = HDF5Dataset(
        "simulation_data_normalized.hdf5",
        VARIABLES,
        PARAMETERS,
        NUMPOINTS_X,
        NUMPOINTS_Y,
    )

    # Split the dataset into training/validation (80%) and test (20%)
    test_size = int(0.2 * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    train_val_dataset, test_dataset = random_split(
        full_dataset, [train_val_size, test_size]
    )

    logger.info(
        f"Dataset split into training/validation ({train_val_size} samples) and test ({test_size} samples)"
    )

    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer, hparams = setup_writer_and_hparams(comment=name)

    logger.info("Starting cross-validation")
    avg_results = cross_validate(
        neural_network,
        train_val_dataset,
        k_folds=5,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        criterion=criterion,
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        scaler=scaler,
        writer=writer,
    )

    logger.log(
        "METRIC",
        f"Cross-validation completed.\n Avg results: Loss={avg_results[0]:.4f}, MSE={avg_results[1]:.4f}, MAE={avg_results[2]:.4f}, R2={avg_results[3]:.4f}",
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    model = (
        neural_network(len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y)
        .to(torch.double)
        .to(DEVICE)
    )

    logger.info("Testing model on the test dataset")
    inputs_for_plot, targets_for_plot, outputs_for_plot = test_model(
        model, test_dataloader, criterion, writer, hparams
    )
    writer.close()

    logger.info("Cross-validation procedure completed")
    return inputs_for_plot, targets_for_plot, outputs_for_plot


def standard_training_procedure(
    name: str, neural_network: Type[nn.Module]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("Starting standard procedure")
    logger.info("Loading dataset")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        "simulation_data_normalized.hdf5",
        VARIABLES,
        PARAMETERS,
        NUMPOINTS_X,
        NUMPOINTS_Y,
        BATCH_SIZE,
    )

    model = (
        neural_network(len(PARAMETERS), len(VARIABLES), NUMPOINTS_X, NUMPOINTS_Y)
        .to(torch.double)
        .to(DEVICE)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS
    )
    criterion = nn.MSELoss()
    scaler = GradScaler()
    writer, hparams = setup_writer_and_hparams(comment=name)

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        NUM_EPOCHS,
        ACCUMULATION_STEPS,
        criterion,
        optimizer,
        scheduler,
        scaler,
        writer,
    )

    logger.info("Testing model on the test dataset")
    inputs_for_plot, targets_for_plot, outputs_for_plot = test_model(
        model, test_dataloader, criterion, writer, hparams
    )
    logger.info("Procedure completed")
    writer.close()
    return inputs_for_plot, targets_for_plot, outputs_for_plot


def main(
    name: str, neural_network: Type[nn.Module], do_cross_validation: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        if do_cross_validation:
            logger.info("Starting cross-validation procedure")
            inputs_for_plot, targets_for_plot, outputs_for_plot = (
                cross_validation_procedure(name, neural_network)
            )
        else:
            logger.info("Starting standard training procedure")
            inputs_for_plot, targets_for_plot, outputs_for_plot = (
                standard_training_procedure(name, neural_network)
            )
    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
        raise

    logger.info("Training process completed successfully")
    return inputs_for_plot, targets_for_plot, outputs_for_plot


class SimpleNN(nn.Module):
    """
    Simple Neural Network with batch normalization and dropout.

    Args:
        input_size (int): Size of input layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(parameters_n + numpoints_x * numpoints_y, 2)
        self.fc2 = nn.Linear(2, variables_n * numpoints_x * numpoints_y)
        self.batch_norm1 = nn.BatchNorm1d(2)

    def forward(self, x):
        x = torch.cat([x[0], x[1].view(x[1].size(0), -1)], 1)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleNN(nn.Module):
    """
    Simple Neural Network with batch normalization and dropout.

    Args:
        input_size (int): Size of input layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, variables_n, 3)
        self.fc1 = nn.Linear(
            6 * (numpoints_x - 3 + 1) * (numpoints_y - 3 + 1) + parameters_n, 32
        )
        self.fc2 = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

    def forward(self, x):
        x0, x1 = x
        x1 = self.conv1(x1.unsqueeze(1))
        x1 = F.relu(x1)
        x1 = x1.view(x1.size(0), -1)
        x = torch.cat([x0, x1], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ComplexNN(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexNN, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.fc1_params = nn.Linear(parameters_n, 64)
        self.fc2_params = nn.Linear(64, 128)

        # Branch 2: Convolutional layers for 2D field
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected layers for the combined output
        self.fc1_combined = nn.Linear(
            64 * (numpoints_x // 2 // 2) * (numpoints_y // 2 // 2) + 128, 256
        )
        self.fc2_combined = nn.Linear(256, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.batch_norm_conv = nn.BatchNorm2d(64)
        self.batch_norm_combined = nn.BatchNorm1d(256)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for parameters branch
        x_params = F.relu(self.fc1_params(x_params))
        x_params = F.relu(self.batch_norm_fc(self.fc2_params(x_params)))

        # Forward pass for 2D field branch
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = F.relu(self.conv1(x_field))
        x_field = self.pool(F.relu(self.conv2(x_field)))
        x_field = self.pool(F.relu(self.batch_norm_conv(self.conv3(x_field))))
        x_field = x_field.view(x_field.size(0), -1)  # Flatten

        # Combine both branches
        x_combined = torch.cat([x_params, x_field], dim=1)
        x_combined = F.relu(self.batch_norm_combined(self.fc1_combined(x_combined)))
        x_combined = self.fc2_combined(x_combined)

        return x_combined


class ComplexSFNO(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexSFNO, self).__init__()

        # Branch 2: SFNO for 2D field
        self.sfno = SFNO(
            n_modes=(16, 16),  # Adjust based on your data
            hidden_channels=16,
            in_channels=1,
            out_channels=8,
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
        )

        # Fully Connected layers for the combined output
        self.fc1 = nn.Linear(8 * numpoints_x * numpoints_y, 32)
        self.fc2 = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm = nn.BatchNorm1d(32)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for 2D field branch using SFNO
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = self.sfno(x_field)

        # Flatten the output of SFNO
        x_field = x_field.view(x_field.size(0), -1)

        # Pass through fully connected layers
        x_field = self.fc1(x_field)
        x_field = self.batch_norm(x_field)
        x_field = F.relu(x_field)
        x_field = self.fc2(x_field)

        return x_field


class ComplexSFNO(nn.Module):
    def __init__(self, parameters_n, variables_n, numpoints_x, numpoints_y):
        super(ComplexSFNO, self).__init__()

        # Branch 1: Fully Connected layers for parameters
        self.fc1_params = nn.Linear(parameters_n, 16)
        self.fc2_params = nn.Linear(16, 32)

        # Branch 2: SFNO for 2D field
        self.sfno = SFNO(
            n_modes=(16, 16),  # Adjust based on your data
            hidden_channels=16,
            in_channels=1,
            out_channels=8,
            n_layers=4,
            lifting_channels=16,
            projection_channels=16,
        )

        # Fully Connected layers for the combined output
        self.fc1_combined = nn.Linear(32+8 * numpoints_x * numpoints_y, 32)
        self.fc2_combined = nn.Linear(32, variables_n * numpoints_x * numpoints_y)

        # Batch normalization layers
        self.batch_norm_fc = nn.BatchNorm1d(32)
        self.batch_norm_combined = nn.BatchNorm1d(32)

    def forward(self, x):
        x_params, x_field = x

        # Forward pass for parameters branch
        x_params = F.relu(self.fc1_params(x_params))
        x_params = F.relu(self.batch_norm_fc(self.fc2_params(x_params)))

        # Forward pass for 2D field branch using SFNO
        x_field = x_field.unsqueeze(1)  # Add a channel dimension
        x_field = self.sfno(x_field)
        x_field = x_field.view(x_field.size(0), -1)  # Flatten

        # Combine both branches
        x_combined = torch.cat([x_params, x_field], dim=1)
        x_combined = F.relu(self.batch_norm_combined(self.fc1_combined(x_combined)))
        x_combined = self.fc2_combined(x_combined)

        return x_combined


if __name__ == "__main__":
    # Remove default logger and add custom one
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add("logs/file_{time}.log", rotation="500 MB")
    try:
        inputs_for_plot, targets_for_plot, outputs_for_plot = main(
            "SFNO", ComplexSFNO, do_cross_validation=True
        )
    except Exception as e:
        logger.error(f"An unhandled exception occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logger.remove()
    logger.add(
        "training.log",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {message}",
        rotation="10 MB",
        compression="zip",
        mode="a",
    )
    logger.add(lambda msg: tqdm.write(msg, end=""))
    try:
        logger.level("METRIC", no=15, color="<white>")
    except:
        pass
    inputs_for_plot, targets_for_plot, outputs_for_plot = main(
        "SFNO", ComplexSFNO, do_cross_validation=True
    )

reshaped_target = target0[0].reshape(NUMPOINTS_Y * 6, NUMPOINTS_X)
reshaped_output = output0[0].reshape(NUMPOINTS_Y * 6, NUMPOINTS_X).detach().numpy()
reshaped_target_flat = reshaped_target.flatten()
reshaped_output_flat = reshaped_output.flatten()

fig, axs = plt.subplots(
    2, 1, figsize=(10, 5)
)  # Create a figure with 1 row and 2 columns

# Display the first image
axs[0].imshow(reshaped_target)
axs[0].axis("off")  # This will remove the axis labels and ticks

# Display the second image
axs[1].imshow(reshaped_output)
axs[1].axis("off")  # This will remove the axis labels and ticks
plt.savefig("resultados.pdf")
plt.show()

fig, axs = plt.subplots(
    nrows=2, ncols=3, figsize=(15, 10)
)  # Create a 2x3 grid of subplots

# Flatten the tensors for plotting
tensor1_flat = target0[0].reshape(6, 401 * 11)
tensor2_flat = output0[0].reshape(6, 401 * 11).detach().numpy()

# Iterate over each subplot and each pair of flattened tensors
for i, (ax, (t1, t2)) in enumerate(zip(axs.flat, zip(tensor1_flat, tensor2_flat))):
    sns.kdeplot(
        x=t1,
        y=t2,
        fill=True,
        thresh=0,
        levels=100,
        cmap="turbo",
        ax=ax,  # Specify the subplot to draw on
    )
    ax.plot(
        [min(t1), max(t1)],
        [min(t2), max(t2)],
        "w--",
    )
    ax.set_xlabel("Valor esperado")
    ax.set_ylabel("Predicción")
    ax.set_title(f"{VARIABLES[i]}")  # Set a title for each subplot

plt.tight_layout()  # Improve spacing between subplots
plt.savefig("density_plots.pdf")
plt.show()
