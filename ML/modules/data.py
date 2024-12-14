import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Callable, Optional, Tuple


class HDF5Dataset(Dataset):
    """
    Optimized PyTorch Dataset for FNO training with HDF5 files.
    """
    
    FIELDS = ['B', 'F', 'H', 'Q', 'S', 'U', 'V', 'D', 'Vr', 'Fr', 'Re', 'B*', 'H*', 'U*', 'V*']
    NUMERIC_SCALARS = ['H0', 'Q0', 'SLOPE', 'n', 'nut', 'Ar', 'Hr', 'M']
    NON_NUMERIC_SCALARS = ['BOTTOM', 'direction', 'id', 'subcritical', 'yc', 'yn']
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
        chunk_size: Optional[int] = None
    ):
        self._validate_variables(input_vars + output_vars)

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

        combined_vars = set(input_vars + output_vars)

        #Update variables and parameters
        self.FIELDS = ['B', 'F', 'H', 'Q', 'S', 'U', 'V', 'D', 'B*', 'H*', 'U*', 'V*']
        self.NUMERIC_SCALARS = ['H0', 'Q0', 'SLOPE', 'n', 'nut', 'Ar', 'Hr', 'M', 'Vr', 'Fr', 'Re']
        self.NON_NUMERIC_SCALARS = ['BOTTOM', 'direction', 'id', 'subcritical', 'yc', 'yn']
        self.SCALARS = self.NUMERIC_SCALARS + self.NON_NUMERIC_SCALARS

        with h5py.File(self.file_path, 'r') as f:
            self.keys = [key for key in f.keys() if key != 'statistics']
            self.len = len(self.keys)
            if self.normalize_input or self.normalize_output:
                self.stats = self._load_stats(f)

        self.preload = preload
        if preload:
            self.data = self._preload_data()
        else:
            self.chunk_size = chunk_size
            self.h5_file = None
            self.current_chunk = None
            self.current_chunk_idx = -1 if chunk_size else None

    def _validate_variables(self, vars_to_check: List[str]):
        invalid_vars = set(vars_to_check) - set(self.FIELDS + self.SCALARS)
        if invalid_vars:
            raise ValueError(f"Unknown variables/parameters: {invalid_vars}")

    def _load_stats(self, f) -> Tuple[dict, dict]:
        """Load normalization statistics, if available."""
        if 'statistics' in f:
            means = {k: f['statistics'].attrs[f"{k}_mean"] for k in self.FIELDS + self.NUMERIC_SCALARS}
            stds = {k: np.sqrt(f['statistics'].attrs[f"{k}_variance"]) for k in self.FIELDS + self.NUMERIC_SCALARS}
            return means, stds
        return {}, {}

    def _preload_data(self):
        """Preload all data into memory."""
        with h5py.File(self.file_path, 'r') as f:
            return {key: self._load_case(f[key]) for key in self.keys}
       
    def _load_case(self, group) -> dict:
        """Load a single data case and compute adimensional numbers."""
        case = {
            **{param: group.attrs[param] for param in self.SCALARS if param in group.attrs},
            **{var: torch.from_numpy(group[var][()]).float() for var in self.FIELDS if var in group}
        }
        adimensional_numbers = self._compute_adimensional_numbers(case)
        case.update(adimensional_numbers)
        return case
        
    def _get_case(self, idx: int) -> dict:
        """Load data for a given index, using chunking if configured."""
        if self.preload:
            return self.data[self.keys[idx]]
        
        if self.chunk_size:
            chunk_idx = idx // self.chunk_size
            if chunk_idx != self.current_chunk_idx:
                self._load_chunk(chunk_idx)
            return self.current_chunk[self.keys[idx]]
        
        # Direct file access for slow loading (fallback)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
        return self._load_case(self.h5_file[self.keys[idx]])

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk of data into memory."""
        start = chunk_idx * self.chunk_size
        end = min((chunk_idx + 1) * self.chunk_size, self.len)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'r')
        self.current_chunk = {key: self._load_case(self.h5_file[key]) for key in self.keys[start:end]}
        self.current_chunk_idx = chunk_idx

    def _normalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Normalize tensor if statistics are available."""
        mean = self.stats[0][var]
        std = self.stats[1][var]
        return (tensor - mean) / std

    def _denormalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Denormalize tensor if statistics are available."""
        mean = self.stats[0][var]
        std = self.stats[1][var]
        return (tensor *std + mean)

    def _process_variable(self, case: dict, var: str, normalize: bool, transform: Optional[Callable]) -> torch.Tensor:
        """Apply normalization and transformation to a variable."""
        if var in self.FIELDS:
            if isinstance(case[var], torch.Tensor):
                tensor = case[var].clone().detach().to(dtype=torch.float32).reshape(self.ny, self.nx)
            else:
                tensor = torch.tensor(case[var], dtype=torch.float32).reshape(self.ny, self.nx)
        else:
            if isinstance(case[var], torch.Tensor):
                tensor = case[var].clone().detach().to(dtype=torch.float32)
            else:
                tensor = torch.tensor(case[var], dtype=torch.float32)
        if normalize:
            tensor = self._normalize(tensor, var)
        if transform and var in self.FIELDS:
            tensor = transform(tensor)
        return tensor.to(self.device)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int):
        case = self._get_case(idx)
        
        input_fields = [self._process_variable(case, var, self.normalize_input, self.transform) for var in self.input_vars if var not in self.SCALARS]
        input_scalars = [self._process_variable(case, var, self.normalize_input, None) for var in self.input_vars if var in self.SCALARS]
        
        output_fields = [self._process_variable(case, var, self.normalize_output, self.target_transform) for var in self.output_vars if var not in self.SCALARS]
        output_scalars = [self._process_variable(case, var, self.normalize_output, None) for var in self.output_vars if var in self.SCALARS]

        if self.augment and torch.rand(1).item() > 0.5:
            input_fields = [torch.flip(f, [0]) for f in input_fields]
            output_fields = [torch.flip(f, [0]) for f in output_fields]

        return (input_fields, input_scalars), (output_fields, output_scalars)

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()

    def _compute_adimensional_numbers(self, case: dict) -> dict:
        """Compute adimensional numbers based on the case attributes."""
        g = 9.81  # Gravity constant

        H0 = case['H0']
        Q0 = case['Q0']
        n = case['n']
        nut = case['nut']
        SLOPE = case['SLOPE']
        xc = self.channel_length  # Channel length from configuration
        yc = self.channel_width  # Channel width from configuration
        bc = SLOPE * xc  # Channel width scaled by slope and length
        hc = H0  # Critical depth is just H0
        uc = Q0 / (H0 * yc)  # Velocity derived from discharge, depth, and width
        vc = uc  # Same as uc
    
        # Compute adimensional numbers
        adimensional_numbers = {
            'Ar': torch.tensor(xc / yc),                      # Aspect ratio
            'Vr': torch.tensor(1),                            # Velocity ratio uc / vc (always 1 here)
            'Fr': torch.tensor(uc / (g * hc)**0.5),           # Froude number
            'Hr': torch.tensor(bc / hc),                      # Hydraulic depth ratio
            'Re': torch.tensor((uc * xc) / nut),              # Reynolds number
            'M': torch.tensor(g * n**2 * xc / (hc**(4 / 3)))  # M factor
        }
        
        # Adimensionalize inputs
        adimensional_numbers.update({
            'H*': case['H'] / hc,
            'U*': case['U'] / uc,
            'V*': case['V'] / vc,
            'B*': case['B'] / bc,
        })
        
        return adimensional_numbers



