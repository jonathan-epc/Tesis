from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for loading and processing data from an HDF5 file.
    
    This dataset supports loading data, normalization, denormalization, and optional 
    data transformations, and it can work with both CPU and CUDA devices.
    """

    def __init__(
        self,
        file_path: str,
        variables: List[str],
        parameters: List[str],
        numpoints_x: int,
        numpoints_y: int,
        device: torch.device,
        normalize: List[bool] = [True, True],
        swap: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = True,
    ):
        """
        Initialize the HDF5Dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            variables (List[str]): List of variable names to load.
            parameters (List[str]): List of parameter names to load.
            numpoints_x (int): Number of points in the x-dimension.
            numpoints_y (int): Number of points in the y-dimension.
            device (torch.device): Device to load the data onto ('cpu' or 'cuda').
            normalize (List[bool]): Whether to normalize input and output data. 
            swap (bool): Whether to swap inputs and outputs.
            transform (Optional[Callable]): Optional transform for the input.
            target_transform (Optional[Callable]): Optional transform for the target.
            augment (bool): Whether to apply augmentation (random flipping).
        """
        self.file_path = file_path
        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.device = device
        self.normalize_input, self.normalize_output = normalize
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

        # Load statistics for normalization
        self._load_statistics()

        # Preload all data into memory
        self.data = self._load_data()

    def _load_statistics(self) -> None:
        """Load normalization statistics from the HDF5 file."""
        with h5py.File(self.file_path, "r") as f:
            self.keys = [key for key in f.keys() if key != "statistics"]
            self.stat_means = {
                param: f["statistics"].attrs[f"{param}_mean"]
                for param in self.parameters + ["B"] + self.variables
            }
            self.stat_stds = {
                param: np.sqrt(f["statistics"].attrs[f"{param}_variance"])
                for param in self.parameters + ["B"] + self.variables
            }

    def _load_data(self) -> dict:
        """Load all data from the HDF5 file into memory."""
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                case_data = {param: f[key].attrs[param] for param in self.parameters}
                case_data["B"] = torch.from_numpy(f[key]["B"][()]).float()
                for var in self.variables:
                    case_data[var] = torch.from_numpy(f[key][var][()]).float()
                data[key] = case_data
        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.keys)

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: 
                A tuple containing the input data and target data.
        """
        key = self.keys[idx]
        case = self.data[key]

        # Load and process parameters
        parameters = torch.tensor([case[param] for param in self.parameters], dtype=torch.float32)
        B_field = case["B"].reshape(self.numpoints_y, self.numpoints_x)
        outputs = torch.stack(
            [case[var].reshape(self.numpoints_y, self.numpoints_x) for var in self.variables]
        )

        # Apply normalization if specified
        if self.normalize_input:
            parameters = self._normalize(parameters, self.parameters)
            B_field = self._normalize(B_field, "B")

        if self.normalize_output:
            outputs = self._normalize(outputs, self.variables)

        # Apply data augmentation (random flipping)
        if self.augment and torch.rand(1).item() > 0.5:
            B_field = torch.flip(B_field, [0])
            outputs = torch.flip(outputs, [1])

        # Apply any specified transformations
        if self.transform:
            parameters = self.transform(parameters)
            B_field = self.transform(B_field)

        if self.target_transform:
            outputs = self.target_transform(outputs)

        # Return swapped inputs and outputs if specified
        if self.swap:
            return outputs.to(self.device), [parameters.to(self.device), B_field.to(self.device)]
        else:
            return [parameters.to(self.device), B_field.to(self.device)], outputs.to(self.device)

    def _normalize(self, data: torch.Tensor, keys: Union[str, List[str]]) -> torch.Tensor:
        """
        Normalize data using precomputed mean and standard deviation.

        Args:
            data (torch.Tensor): Data to normalize.
            keys (Union[str, List[str]]): Key(s) to retrieve corresponding statistics.

        Returns:
            torch.Tensor: Normalized data.
        """
        if isinstance(keys, str):
            mean = torch.tensor(self.stat_means[keys], dtype=torch.float32)
            std = torch.tensor(self.stat_stds[keys], dtype=torch.float32)
        else:
            mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32)
            std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32)

        # Adjust mean and std for multi-channel data
        if data.dim() == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return (data - mean) / std

    def _denormalize(self, data: torch.Tensor, keys: Union[str, List[str]]) -> torch.Tensor:
        """
        Denormalize data using precomputed mean and standard deviation.

        Args:
            data (torch.Tensor): Data to denormalize.
            keys (Union[str, List[str]]): Key(s) to retrieve corresponding statistics.

        Returns:
            torch.Tensor: Denormalized data.
        """
        if isinstance(keys, str):
            mean = torch.tensor(self.stat_means[keys], dtype=torch.float32)
            std = torch.tensor(self.stat_stds[keys], dtype=torch.float32)
        else:
            mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32)
            std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32)

        # Adjust mean and std for multi-channel data
        if data.dim() == 3:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return data * std + mean

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"HDF5Dataset(file_path='{self.file_path}', "
            f"variables={self.variables}, parameters={self.parameters}, "
            f"numpoints_x={self.numpoints_x}, numpoints_y={self.numpoints_y}, "
            f"device='{self.device}', normalize_input={self.normalize_input}, "
            f"normalize_output={self.normalize_output}, swap={self.swap})"
        )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the shape of the dataset.

        Returns:
            Tuple[int, int, int]: (num_samples, numpoints_y, numpoints_x)
        """
        return len(self), self.numpoints_y, self.numpoints_x