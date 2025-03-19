import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Callable, Optional, Tuple


from scipy.stats import boxcox
from scipy.special import inv_boxcox
import numpy as np
import torch

class HDF5Dataset(Dataset):
    """
    Optimized PyTorch Dataset for FNO training with HDF5 files.
    Includes Box-Cox transformation functionality.
    """

    # Class constants
    FIELDS = ['B', 'F', 'H', 'Q', 'S', 'U', 'V', 'D', 'B*', 'H*', 'U*', 'V*']
    NUMERIC_SCALARS = ['H0', 'Q0', 'SLOPE', 'n', 'nut', 'Vr', 'Fr', 'Re', 'Ar', 'Hr', 'M']
    NON_NUMERIC_SCALARS = ['BOTTOM', 'direction', 'id', 'subcritical', 'yc', 'yn']
    ADIMENSIONAL_FIELDS = ['B*', 'H*', 'U*', 'V*']
    ADIMENSIONAL_SCALARS = ['Ar', 'Vr', 'Fr', 'Hr', 'Re', 'M']
    SCALARS = NUMERIC_SCALARS + NON_NUMERIC_SCALARS

    def __init__(
        self,
        file_path: str,
        input_vars: List[str],
        output_vars: List[str],
        numpoints_x: int,
        numpoints_y: int,
        channel_length: float,
        channel_width: float,
        device: torch.device = torch.device('cpu'),
        normalize_input: bool = True,
        normalize_output: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = True,
        preload: bool = False,
        chunk_size: Optional[int] = None,
        boxcox_transform: bool = False
    ):
        self.file_path = file_path
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.nx, self.ny = numpoints_x, numpoints_y
        self.channel_length, self.channel_width = channel_length, channel_width
        self.device = device
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
        self.preload = preload
        self.chunk_size = chunk_size
        self.boxcox_transform = boxcox_transform
        self.boxcox_lambdas = {}
        self.boxcox_shifts = {}

        with h5py.File(self.file_path, 'r', swmr=True) as f:
            self.keys = [key for key in f.keys() if key != 'statistics']
            
            self.len = len(self.keys)

            if self.keys:
                first_key = self.keys[0]
                group = f[first_key]
                available_fields = {var for var in self.FIELDS if var in group}
                available_scalars = {scalar for scalar in self.SCALARS if scalar in group.attrs}
            else:
                available_fields = set()
                available_scalars = set()

            self.FIELDS = list(available_fields)
            self.SCALARS = list(available_scalars)
            self.stats = self._load_stats(f) if any([normalize_input, normalize_output, boxcox_transform]) else ({}, {}, {})
            if self.boxcox_transform:
                variables = set(input_vars + output_vars)
                for var in variables:
                    if var in self.stats[2]:  # minima dict
                        min_val = self.stats[2][var]
                        # Calculate shift to ensure positivity
                        self.boxcox_shifts[var] = max(0.0, -min_val) + 1e-6
                        self.boxcox_lambdas[var] = 0.0

        self.data = self._preload_data() if preload else None
        self.h5_file = None
        self.current_chunk = None
        self.current_chunk_idx = -1 if chunk_size else None

        self._validate_variables(input_vars + output_vars)
    @classmethod
    def from_config(cls, config, file_path: str):
        """
        Create an HDF5Dataset instance from a configuration object.
        """
        return cls(
            file_path=file_path,
            input_vars=config.data.inputs,
            output_vars=config.data.outputs,
            numpoints_x=config.data.numpoints_x,
            numpoints_y=config.data.numpoints_y,
            channel_length=config.channel.length,
            channel_width=config.channel.width,
            normalize_input=getattr(config.data, "normalize_input", True),
            normalize_output=getattr(config.data, "normalize_output", True),
            device=getattr(config, "device", "cpu"),
            preload=getattr(config.data, "preload", False),
            chunk_size=getattr(config.data, "chunk_size", None),
            boxcox_transform=getattr(config.data, "boxcox_transform", False),
        )

    def _process_variable(self, case: dict, var: str, normalize: bool) -> torch.Tensor:
        tensor = torch.tensor(case[var], dtype=torch.float32) if not isinstance(case[var], torch.Tensor) else case[var]
        tensor = tensor.clone().detach().to(dtype=torch.float32)
        tensor = tensor.reshape(self.ny, self.nx) if var in self.FIELDS else tensor
        if self.boxcox_transform and var in self.boxcox_lambdas:
            tensor = self._apply_boxcox(tensor, var)
        if normalize:
            tensor = self._normalize(tensor, var)
        return tensor.to(self.device)


    def _validate_variables(self, vars_to_check: List[str]):
        # Combine FIELDS, SCALARS, and ADIMENSIONAL_VARIABLES for validation
        valid_variables = set(self.FIELDS + self.SCALARS)
        
        # Check for invalid variables
        invalid_vars = set(vars_to_check) - valid_variables
        if invalid_vars:
            raise ValueError(f"Unknown variables/parameters: {invalid_vars}")   

    def _load_stats(self, f) -> Tuple[dict, dict, dict]:
        """Load means, stds, and minima from dataset statistics"""
        if 'statistics' not in f:
            return {}, {}, {}
            
        stats_group = f['statistics']
        fields = self.FIELDS + [s for s in self.NUMERIC_SCALARS if s in self.SCALARS]
        
        return (
            {k: stats_group.attrs[f"{k}_mean"] for k in fields},      # Means
            {k: np.sqrt(stats_group.attrs[f"{k}_variance"]) for k in fields},  # Stds
            {k: stats_group.attrs[f"{k}_min"] for k in fields}        # Minima
        )

    def _preload_data(self):
        with h5py.File(self.file_path, 'r', swmr=True) as f:
            return {key: self._load_case(f[key]) for key in self.keys}

    def _load_case(self, group) -> dict:
        case = {param: group.attrs[param] for param in self.SCALARS if param in group.attrs}
        case.update({var: torch.from_numpy(group[var][()]).float() for var in self.FIELDS if var in group})
        return case

    def _get_case(self, idx: int) -> dict:
        if self.preload:
            return self.data[self.keys[idx]]
    
        if self.chunk_size:
            chunk_idx = idx // self.chunk_size
            if chunk_idx != self.current_chunk_idx:
                self._load_chunk(chunk_idx)
            return self.current_chunk[self.keys[idx]]
    
        # Ensure file is open before accessing it
        if self.h5_file is None or not bool(self.h5_file):
            self.h5_file = h5py.File(self.file_path, 'r', swmr=True)
            
        return self._load_case(self.h5_file[self.keys[idx]])

    def _load_chunk(self, chunk_idx: int):
        start, end = chunk_idx * self.chunk_size, min((chunk_idx + 1) * self.chunk_size, self.len)
        if self.h5_file is None:
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                self.h5_file = f
        self.current_chunk = {key: self._load_case(self.h5_file[key]) for key in self.keys[start:end]}
        self.current_chunk_idx = chunk_idx

    def _normalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        mean, std = self.stats[0][var], self.stats[1][var]
        return (tensor - mean) / std

    def _denormalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        mean, std = self.stats[0][var], self.stats[1][var]
        return tensor * std + mean

    def _apply_boxcox(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Apply Box-Cox transformation to a tensor (field or scalar)."""
        lambda_val = self.boxcox_lambdas.get(var)
        shift = self.boxcox_shifts.get(var, 0)
    
        # Convert tensor to numpy array
        numpy_data = tensor.cpu().numpy()
    
        # Ensure numpy_data is an array
        if numpy_data.ndim == 0:
            numpy_data = numpy_data[None]  # Convert scalar to 1D array
    
        shifted_data = numpy_data + shift
        transformed_data = boxcox(shifted_data, lmbda=lambda_val)
    
        # Convert back to torch tensor
        return torch.from_numpy(transformed_data).float()

    def _inverse_boxcox(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Apply inverse Box-Cox transformation to a tensor (field or scalar)."""
        lambda_val = self.boxcox_lambdas.get(var)
        shift = self.boxcox_shifts.get(var, 0)
    
        # Convert tensor to numpy array
        numpy_data = tensor.cpu().numpy()
    
        # Ensure numpy_data is an array
        if numpy_data.ndim == 0:
            numpy_data = numpy_data[None]  # Convert scalar to 1D array
    
        original_data = inv_boxcox(numpy_data, lambda_val) - shift
    
        # Convert back to torch tensor
        return torch.from_numpy(original_data).float()


    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        case = self._get_case(idx)

        input_fields = [self._process_variable(case, var, self.normalize_input) 
                        for var in self.input_vars if var not in self.SCALARS]
        input_scalars = [self._process_variable(case, var, self.normalize_input) 
                         for var in self.input_vars if var in self.SCALARS]

        output_fields = [self._process_variable(case, var, self.normalize_output) 
                         for var in self.output_vars if var not in self.SCALARS]
        output_scalars = [self._process_variable(case, var, self.normalize_output) 
                          for var in self.output_vars if var in self.SCALARS]

        if self.augment and torch.rand(1).item() > 0.5:
            input_fields = [torch.flip(f, [0]) for f in input_fields]
            output_fields = [torch.flip(f, [0]) for f in output_fields]
            
        return (input_fields, input_scalars), (output_fields, output_scalars)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
