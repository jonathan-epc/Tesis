# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Import the necessary libraries
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

numpoints_x = 401
numpoints_y = 11

variables = list(dict(file["results_0"]).keys())

variables = ["F", "H", "Q", "S", "U", "V"]

parameters = list(file["results_0"].attrs.keys())

parameters = ["H0", "Q", "S", "n"]

data = h5py.File("simulation_data.hdf5", "r")

keys = list(data.keys())

statistical_group = keys.pop()

statistical_data = data[statistical_group]

parameters

statistical_data.attrs.keys()

statistical_data.attrs['Q_count']

[statistical_data.attrs[parameter+'_count'] for parameter in parameters]


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.file_path = file_path
        self.data = h5py.File(self.file_path, "r")
        self.keys = list(self.data.keys())
        # Separate the statistical group from the rest of the data
        self.statistical_group = self.keys.pop()  # Assuming the statistical group is the last group in the file
        self.statistical_data = self.data[self.statistical_group]
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        case = self.data[key]
        input = torch.cat(
            [
                torch.tensor([case.attrs[parameter] for parameter in parameters]),
                torch.tensor(case["B"][()].reshape(numpoints_y, numpoints_x)).view(-1),
            ]
        )
        output = torch.stack(
            [
                torch.from_numpy(var)
                for var in [
                    case[var][()].reshape(numpoints_y, numpoints_x) for var in variables
                ]
            ]
        )
        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            output = self.target_transform(output)
        return input, output

    def normalize(self, data, prefix):
        mean = torch.tensor(self.statistical_data[f"{prefix}_mean"])
        std = torch.sqrt(torch.tensor(self.statistical_data[f"{prefix}_var"]))
        return (data - mean) / std

    def denormalize(self, data, prefix):
        mean = torch.tensor(self.statistical_data[f"{prefix}_mean"])
        std = torch.sqrt(torch.tensor(self.statistical_data[f"{prefix}_var"]))
        return data * std + mean


# Create an instance of the dataloader
dataset = HDF5Dataset("simulation_data.hdf5")

# Split the dataset into training, validation, and testing sets
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create dataloaders for the training, validation, and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.to(torch.float)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_epochs = 10

# Create an instance of the model
input_size = len(parameters) + numpoints_y * numpoints_x
output_size = len(variables) * numpoints_y * numpoints_x
model = SimpleNN(input_size, output_size)


# Create a directory to save the models
if not os.path.exists("models"):
    os.makedirs("models")

# Create a SummaryWriter for logging
writer = SummaryWriter()

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
for epoch in range(num_epochs):
    # Training loop
    model.train()
    train_loss = 0
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(targets.shape)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Log the training loss
    writer.add_scalar("Loss/train", train_loss / len(train_dataloader), epoch)

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    # Log the validation loss
    writer.add_scalar("Loss/val", val_loss / len(val_dataloader), epoch)

    # Save the model
    torch.save(model.state_dict(), f"models/model_{epoch}.pth")

# Close the SummaryWriter
writer.close()

# Load the model
model = SimpleNN(input_size, output_size)
model.load_state_dict(torch.load("models/model_9.pth"))
model.eval()

# Compute the test loss
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        outputs = outputs.view(targets.shape)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
test_loss /= len(test_dataset)

# Log the test loss
writer.add_scalar("Loss/test", test_loss, epoch)

# Print the test loss
print(f"Test loss: {test_loss:.4f}")
