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
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


def compute_metrics(outputs, targets):
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


numpoints_x = 401
numpoints_y = 11
variables = ["F", "H", "Q", "S", "U", "V"]
parameters = ["H0", "Q0", "SLOPE", "n"]

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    def __init__(
        self,
        file_path: str,
        variables: list[str],
        parameters: list[str],
        numpoints_x: int,
        numpoints_y: int,
        normalized: bool = True,
        swap: bool = False,
        transform=None,
        target_transform=None,
    ):
        self.file_path = file_path
        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.normalized = normalized
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform

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
        self.data = {}
        with h5py.File(self.file_path, "r") as data:
            for key in self.keys:
                self.data[key] = {
                    param: data[key].attrs[param] for param in self.parameters
                }
                self.data[key]["B"] = data[key]["B"][()]
                for var in self.variables:
                    self.data[key][var] = data[key][var][()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
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
            output = [self.normalize(o, var) for o, var in zip(output, self.variables)]

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

    def normalize(self, data, prefix):
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return (data - mean) / std

    def denormalize(self, data, prefix):
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return data * std + mean


class SimpleNN(nn.Module):
    """
    Simple Neural Network with batch normalization and dropout.

    Args:
        input_size (int): Size of input layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = torch.cat([x[0], x[1].view(x[1].size(0), -1)], 1)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.fc4(x)
        return x


class SimpleNN(nn.Module):
    """
    Simple Neural Network with batch normalization and dropout.

    Args:
        input_size (int): Size of input layer.
        output_size (int): Size of output layer.
    """

    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.cat([x[0], x[1].view(x[1].size(0), -1)], 1)
        x = F.relu(self.fc1(x))
        return x


# Create an instance of the dataset and dataloaders
dataset = HDF5Dataset(
    "simulation_data_normalized.hdf5", variables, parameters, numpoints_x, numpoints_y
)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the model
input_size = len(parameters) + numpoints_y * numpoints_x
output_size = len(variables) * numpoints_y * numpoints_x
model = SimpleNN(input_size, output_size).to(torch.double).to(device)

# Training settings
num_epochs = 256
accumulation_steps = 4
os.makedirs("models", exist_ok=True)
learning_rate = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs
)
criterion = nn.MSELoss()
scaler = GradScaler()
best_val_loss = float("inf")
early_stopping_patience = 20
early_stopping_counter = 0
writer = SummaryWriter()
writer.add_text("Structure", "1 linear layers")

# Training loop
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    train_mse, train_mae, train_r2 = 0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for idx, (inputs, targets) in enumerate(train_dataloader):
        with autocast(enabled=device == "CUDA"):
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

    # Validation loop
    model.eval()
    val_loss = 0
    val_mse, val_mae, val_r2 = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            mse, mae, r2 = compute_metrics(outputs, targets)
            val_mse += mse
            val_mae += mae
            val_r2 += r2

    val_loss /= len(val_dataloader)
    val_mse /= len(val_dataloader)
    val_mae /= len(val_dataloader)
    val_r2 /= len(val_dataloader)

    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MSE/val", val_mse, epoch)
    writer.add_scalar("MAE/val", val_mae, epoch)
    writer.add_scalar("R2/val", val_r2, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"models/best_model.pth")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

    scheduler.step()
# Testing phase
model.load_state_dict(torch.load(f"models/best_model.pth"))
model.eval()
test_loss = 0
test_mse, test_mae, test_r2 = 0, 0, 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        outputs = outputs.view(targets.shape)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        mse, mae, r2 = compute_metrics(outputs, targets)
        test_mse += mse
        test_mae += mae
        test_r2 += r2

test_loss /= len(test_dataloader)
test_mse /= len(test_dataloader)
test_mae /= len(test_dataloader)
test_r2 /= len(test_dataloader)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2: {test_r2:.4f}")

writer.add_scalar("Loss/test", test_loss, epoch)
writer.add_scalar("MSE/test", test_mse, epoch)
writer.add_scalar("MAE/test", test_mae, epoch)
writer.add_scalar("R2/test", test_r2, epoch)
hparams = {"learning_rate": learning_rate, "batch_size": batch_size, "accumulation_steps": accumulation_steps}
metrics = {"Loss": test_loss, "MSE": test_mse, "MAE": test_mae, "R2": test_r2}
writer.add_hparams(hparams, metrics)
writer.close()

# Load the best model
model = SimpleNN(input_size, output_size).to(torch.double).to(device)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

input0, target0 = next(iter(test_dataloader))

output0 = model(input0)

plt.imshow(target0[0].reshape(numpoints_y * 6, numpoints_x))

plt.imshow(output0[0].reshape(numpoints_y * 6, numpoints_x).detach().numpy())
