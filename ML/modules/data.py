from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """
    PyTorch Dataset for loading and processing data from an HDF5 file.
    
    Variables (2D arrays): ['B', 'F', 'H', 'Q', 'S', 'U', 'V']
    Parameters (scalars): ['BOTTOM', 'H0', 'Q0', 'SLOPE', 'direction', 'id', 'n', 'subcritical', 'yc', 'yn']
    """

    # Fixed lists of variables (2D arrays) and parameters (scalars)
    VARIABLES = ['B', 'F', 'H', 'Q', 'S', 'U', 'V']
    NUMERIC_PARAMETERS = ['H0', 'Q0', 'SLOPE', 'n']
    NON_NUMERIC_PARAMETERS = ['BOTTOM', 'direction', 'id', 'subcritical', 'yc', 'yn']
    PARAMETERS = NUMERIC_PARAMETERS + NON_NUMERIC_PARAMETERS

    def __init__(
        self,
        file_path: str,
        input_vars: List[str],
        output_vars: List[str],
        numpoints_x: int,
        numpoints_y: int,
        device: torch.device,
        normalize: Tuple[bool, bool] = (True, True),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = True,
    ):
        """
        Initialize the HDF5Dataset.

        Args:
            file_path (str): Path to the HDF5 file.
            input_vars (List[str]): List of variable/parameter names to use as inputs.
            output_vars (List[str]): List of variable names to use as outputs.
            numpoints_x (int): Number of points in the x-dimension.
            numpoints_y (int): Number of points in the y-dimension.
            device (torch.device): Device to load the data onto ('cpu' or 'cuda').
            normalize (Tuple[bool, bool]): Whether to normalize (inputs, outputs).
            transform (Optional[Callable]): Optional transform for the input.
            target_transform (Optional[Callable]): Optional transform for the target.
            augment (bool): Whether to apply augmentation (random flipping).
        """
        # Validate inputs and outputs
        for var in input_vars + output_vars:
            if var not in self.VARIABLES + self.PARAMETERS:
                raise ValueError(f"Unknown variable/parameter: {var}")

        self.file_path = file_path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.device = device
        self.normalize_input, self.normalize_output = normalize
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

        # Load statistics and data
        self._load_statistics()
        self.data = self._load_data()

    def _load_statistics(self) -> None:
        """Load normalization statistics from the HDF5 file."""
        with h5py.File(self.file_path, "r") as f:
            self.keys = [key for key in f.keys() if key != "statistics"]
            self.stat_means = {
                param: f["statistics"].attrs[f"{param}_mean"]
                for param in self.NUMERIC_PARAMETERS + self.VARIABLES
            }
            self.stat_stds = {
                param: np.sqrt(f["statistics"].attrs[f"{param}_variance"])
                for param in self.NUMERIC_PARAMETERS + self.VARIABLES
            }

    def _load_data(self) -> dict:
        """Load all data from the HDF5 file into memory."""
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                case_data = {param: f[key].attrs[param] for param in self.PARAMETERS}
                for var in self.VARIABLES:
                    case_data[var] = torch.from_numpy(f[key][var][()]).float()
                data[key] = case_data
        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.keys)

    def _process_var(self, case: dict, var: str, normalize: bool) -> torch.Tensor:
        """
        Process a single variable or parameter into a tensor.
        
        Args:
            case (dict): Dictionary containing the case data.
            var (str): Variable or parameter name to process.
            normalize (bool): Whether to normalize the data.
            
        Returns:
            torch.Tensor: Processed tensor.
        """
        if var in self.VARIABLES:
            # Handle 2D variable
            data = case[var].reshape(self.numpoints_y, self.numpoints_x)
            if normalize:
                data = self._normalize(data, [var])
        else:
            # Handle scalar parameter
            data = torch.tensor(case[var], dtype=torch.float32)
            if normalize and var in self.NUMERIC_PARAMETERS:
                data = self._normalize(data, [var])
        
        return data

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
                Lists of input and output tensors.
        """
        key = self.keys[idx]
        case = self.data[key]

        # Process each input and output separately
        inputs = []
        outputs = []
        
        # Process inputs
        for var in self.input_vars:
            tensor = self._process_var(case, var, self.normalize_input)
            if self.transform:
                tensor = self.transform(tensor)
            inputs.append(tensor.to(self.device))
        
        # Process outputs
        for var in self.output_vars:
            tensor = self._process_var(case, var, self.normalize_output)
            if self.target_transform:
                tensor = self.target_transform(tensor)
            outputs.append(tensor.to(self.device))

        # Apply data augmentation (random flipping) for 2D data
        if self.augment and torch.rand(1).item() > 0.5:
            for i, var in enumerate(self.input_vars):
                if var in self.VARIABLES:
                    inputs[i] = torch.flip(inputs[i], [0])
            for i, var in enumerate(self.output_vars):
                if var in self.VARIABLES:
                    outputs[i] = torch.flip(outputs[i], [0])

        return inputs, outputs

    def _normalize(self, data: torch.Tensor, keys: List[str]) -> torch.Tensor:
        """
        Normalize data using precomputed mean and standard deviation.

        Args:
            data (torch.Tensor): Data to normalize.
            keys (List[str]): Keys to retrieve corresponding statistics.

        Returns:
            torch.Tensor: Normalized data.
        """
        mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32)
        std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32)

        if data.dim() == 2:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        
        return (data - mean) / std

    def _denormalize(self, data: torch.Tensor, keys: List[str]) -> torch.Tensor:
        """
        Denormalize data using precomputed mean and standard deviation.

        Args:
            data (torch.Tensor): Data to denormalize.
            keys (List[str]): Keys to retrieve corresponding statistics.

        Returns:
            torch.Tensor: Denormalized data.
        """
        mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32)
        std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32)

        if data.dim() == 2:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return data * std + mean