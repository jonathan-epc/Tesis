---
jupytext:
  formats: ipynb,py:nomarker,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# Import the necessary libraries
import os
from timeit import default_timer
```

```{code-cell} ipython3
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
from torch.cuda.amp import GradScaler
from torch.cpu.amp import autocast
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
```

```{code-cell} ipython3
def compute_metrics(outputs, targets):
    outputs = outputs.view(-1).detach().numpy()
    targets = targets.view(-1).detach().numpy()
    mse = mean_squared_error(targets, outputs)
    mae = mean_absolute_error(targets, outputs)
    r2 = r2_score(targets, outputs)
    return mse, mae, r2
```

```{code-cell} ipython3
numpoints_x = 401
numpoints_y = 11
```

```{code-cell} ipython3
device = "cuda" if torch.cuda.is_available() else "cpu"
```

```{code-cell} ipython3
variables = ["F", "H", "Q", "S", "U", "V"]
```

```{code-cell} ipython3
parameters = ["H0", "Q0", "SLOPE", "n"]
```

```{code-cell} ipython3
data = h5py.File("simulation_data.hdf5", "r")
```

```{code-cell} ipython3
keys = list(data.keys())
```

```{code-cell} ipython3
statistical_group = keys.pop()
```

```{code-cell} ipython3
statistical_data = data[statistical_group]
```

```{code-cell} ipython3
statistical_data.attrs.keys()
```

```{code-cell} ipython3
class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path,
        variables,
        parameters,
        numpoints_x,
        numpoints_y,
        transform=None,
        target_transform=None,
    ):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as data:
            self.keys = [key for key in data.keys() if key != "statistics"]
            self.stat_means = {
                param: torch.tensor(data["statistics"].attrs[f"{param}_mean"])
                for param in parameters + ["B"] + variables
            }
            self.stat_stds = {
                param: torch.sqrt(
                    torch.tensor(data["statistics"].attrs[f"{param}_variance"])
                )
                for param in parameters + ["B"] + variables
            }

        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as data:
            key = self.keys[idx]
            case = data[key]
            parameters_normalized = [
                self.normalize(case.attrs[param], param) for param in self.parameters
            ]
            B_normalized = (
                self.normalize(case["B"][()], "B")
                .reshape(self.numpoints_y, self.numpoints_x)
                .view(-1)
            )
            input = torch.cat(parameters_normalized + [B_normalized])
            output = torch.stack(
                [
                    self.normalize(case[var][()], var).reshape(
                        self.numpoints_y, self.numpoints_x
                    )
                    for var in self.variables
                ]
            )

        if self.transform:
            input = self.transform(input)
        if self.target_transform:
            output = self.target_transform(output)

        return input, output

    def normalize(self, data, prefix):
        data = torch.tensor(data)
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return (data - mean) / std

    def denormalize(self, data, prefix):
        data = torch.tensor(data)
        mean = self.stat_means[prefix]
        std = self.stat_stds[prefix]
        return data * std + mean
```

```{code-cell} ipython3
# Create an instance of the dataloader
dataset = HDF5Dataset(
    "simulation_data.hdf5", variables, parameters, numpoints_x, numpoints_y
)
```

```{code-cell} ipython3
# Split the dataset into training, validation, and testing sets
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

```{code-cell} ipython3
# Create dataloaders for the training, validation, and testing sets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

```{code-cell} ipython3
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.double()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

```{code-cell} ipython3
num_epochs = 256
```

```{code-cell} ipython3
# Create an instance of the model
input_size = len(parameters) + numpoints_y * numpoints_x
output_size = len(variables) * numpoints_y * numpoints_x
model = SimpleNN(input_size, output_size)
```

```{code-cell} ipython3
# Create a directory to save the models
os.makedirs("models", exist_ok=True)
```

```{code-cell} ipython3
# Create a SummaryWriter for logging
writer = SummaryWriter()
```

```{code-cell} ipython3
# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

```{code-cell} ipython3
scaler = GradScaler()
best_val_loss = float("inf")
```

```{code-cell} ipython3
accumulation_steps=4
```

```{code-cell} ipython3
for epoch in tqdm(range(num_epochs)):
    # Training loop
    model.train()
    train_loss = 0
    train_mse, train_mae, train_r2 = 0, 0, 0
    for idx, (inputs, targets) in enumerate(train_dataloader):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(inputs)
            outputs = outputs.view(targets.shape)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        # Accumulate gradients over multiple mini-batches
        if (idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

        train_loss += loss.item()
        mse, mae, r2 = compute_metrics(outputs, targets)
        train_mse += mse
        train_mae += mae
        train_r2 += r2

    # Perform a weight update if the number of mini-batches is not a multiple of accumulation_steps
    if idx % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()    

    train_loss /= len(train_dataloader)
    train_mse /= len(train_dataloader)
    train_mae /= len(train_dataloader)
    train_r2 /= len(train_dataloader)

    # Log the training loss and metrics
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

    # Log the validation loss and metrics
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MSE/val", val_mse, epoch)
    writer.add_scalar("MAE/val", val_mae, epoch)
    writer.add_scalar("R2/val", val_r2, epoch)

    # Save the model if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"models/best_model.pth")
# Close the SummaryWriter
writer.close()
```

```{code-cell} ipython3
# Load the best model
model = SimpleNN(input_size, output_size)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()
```

```{code-cell} ipython3
# Compute the test loss and metrics
test_loss = 0
test_mse, test_mae, test_r2 = 0, 0, 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        outputs = outputs.view(targets.shape)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
        mse, mae, r2 = compute_metrics(outputs, targets)
        test_mse += mse * inputs.size(0)
        test_mae += mae * inputs.size(0)
        test_r2 += r2 * inputs.size(0)
```

```{code-cell} ipython3
test_loss /= len(test_dataset)
test_mse /= len(test_dataset)
test_mae /= len(test_dataset)
test_r2 /= len(test_dataset)
```

```{code-cell} ipython3
# Log the test loss and metrics
writer.add_scalar("Loss/test", test_loss, epoch)
writer.add_scalar("MSE/test", test_mse, epoch)
writer.add_scalar("MAE/test", test_mae, epoch)
writer.add_scalar("R2/test", test_r2, epoch)
```

```{code-cell} ipython3
# Print the test loss and metrics
print(f"Test loss: {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R2: {test_r2:.4f}")
```

```{code-cell} ipython3
input0, target0 = next(iter(test_dataloader))
output0 = model(input0)
```

```{code-cell} ipython3
plt.imshow(target0[0].reshape(numpoints_y * 6, numpoints_x))
```

```{code-cell} ipython3
plt.imshow(output0[0].reshape(numpoints_y * 6, numpoints_x).detach().numpy())
```
