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
        # Initialize parameters (same as before)
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

        # Validate variables
        for var in input_vars + output_vars:
            if var not in self.VARIABLES + self.PARAMETERS:
                raise ValueError(f"Unknown variable/parameter: {var}")

        # Load data and statistics
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
        return len(self.keys)

    def _process_var(self, case: dict, var: str, normalize: bool) -> torch.Tensor:
        """
        Process a single variable or parameter into a tensor.
        """
        if var in self.VARIABLES:
            # Process 2D field variables
            data = case[var].reshape(self.numpoints_y, self.numpoints_x)
            if normalize:
                data = self._normalize(data, [var])
        else:
            # Process scalar parameters
            data = torch.tensor(case[var], dtype=torch.float32, device=self.device)
            if normalize and var in self.NUMERIC_PARAMETERS:
                data = self._normalize(data, [var])
        
        return data

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Retrieve a sample from the dataset by index.

        Returns:
            Tuple[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
                List of input tensors (both 2D fields and scalars), 
                Tuple of (field outputs, scalar outputs).
        """
        key = self.keys[idx]
        case = self.data[key]

        # Process inputs
        input_fields = []
        input_scalars = []
        for var in self.input_vars:
            tensor = self._process_var(case, var, self.normalize_input)
            if self.transform:
                tensor = self.transform(tensor)
            if var in self.VARIABLES:
                input_fields.append(tensor.to(self.device))
            else:
                input_scalars.append(tensor.to(self.device))

        # Process outputs
        output_fields = []
        output_scalars = []
        for var in self.output_vars:
            tensor = self._process_var(case, var, self.normalize_output)
            if self.target_transform:
                tensor = self.target_transform(tensor)
            if var in self.VARIABLES:
                output_fields.append(tensor.to(self.device))
            else:
                output_scalars.append(tensor.to(self.device))

        # Data augmentation (flip) applied only to 2D fields
        if self.augment and torch.rand(1).item() > 0.5:
            input_fields = [torch.flip(f, [0]) for f in input_fields]
            output_fields = [torch.flip(f, [0]) for f in output_fields]

        return (input_fields, input_scalars), (output_fields, output_scalars)

    def _normalize(self, data: torch.Tensor, keys: List[str]) -> torch.Tensor:
        """
        Normalize data using precomputed mean and standard deviation.
        """
        mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32, device=self.device)
        std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32, device=self.device)
        data = data.to(self.device)

        if data.dim() == 2:  # 2D field
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return (data - mean) / std

    def _denormalize(self, data: torch.Tensor, keys: List[str]) -> torch.Tensor:
        """
        Denormalize data using precomputed mean and standard deviation.
        """
        mean = torch.tensor([self.stat_means[key] for key in keys], dtype=torch.float32, device=self.device)
        std = torch.tensor([self.stat_stds[key] for key in keys], dtype=torch.float32, device=self.device)
        data = data.to(self.device)

        if data.dim() == 2:  # 2D field
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)

        return data * std + mean
