from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split


class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        variables: List[str],
        parameters: List[str],
        numpoints_x: int,
        numpoints_y: int,
        device: str,
        normalized: bool = True,
        swap: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.file_path = file_path
        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.already_normalized = normalized
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

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

        # Pre-load all data into memory
        self.data = self._load_data()

    def _load_data(self):
        data = {}
        with h5py.File(self.file_path, "r") as file:
            for key in self.keys:
                data[key] = {param: file[key].attrs[param] for param in self.parameters}
                data[key]["B"] = torch.from_numpy(file[key]["B"][()]).float()
                for var in self.variables:
                    data[key][var] = torch.from_numpy(file[key][var][()]).float()
        return data

    def __len__(self) -> int:
        return len(self.keys)

    @lru_cache(maxsize=None)
    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        key = self.keys[idx]
        case = self.data[key]

        parameters = torch.tensor(
            [case[param] for param in self.parameters], dtype=torch.float32
        )
        B = case["B"].reshape(self.numpoints_y, self.numpoints_x)
        output = torch.stack(
            [
                case[var].reshape(self.numpoints_y, self.numpoints_x)
                for var in self.variables
            ]
        )

        if self.already_normalized:
            if self.swap:
                parameters = self.denormalize(parameters, self.parameters)
                B = self.denormalize(B, "B")
            else:
                output = self.denormalize(output, self.variables)                
        else:
            if self.swap:
                output = self.normalize(output, self.variables)
            else:
                parameters = self.normalize(parameters, self.parameters)
                B = self.normalize(B, "B")

        if self.transform:
            parameters = self.transform(parameters)
            B = self.transform(B)
        if self.target_transform:
            output = self.target_transform(output)

        if self.swap:
            return output.to(self.device), [
                parameters.to(self.device),
                B.to(self.device),
            ]
        else:
            return [parameters.to(self.device), B.to(self.device)], output.to(
                self.device
            )

    def normalize(
        self, data: torch.Tensor, prefixes: Union[str, List[str]]
    ) -> torch.Tensor:
        if isinstance(prefixes, str):
            mean = self.stat_means[prefixes].float()
            std = self.stat_stds[prefixes].float()
        else:
            mean = torch.tensor([self.stat_means[prefix] for prefix in prefixes])
            std = torch.tensor([self.stat_stds[prefix] for prefix in prefixes])
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        return (data - mean) / std

    def denormalize(
        self, data: torch.Tensor, prefixes: Union[str, List[str]]
    ) -> torch.Tensor:
        if isinstance(prefixes, str):
            mean = self.stat_means[prefixes].float()
            std = self.stat_stds[prefixes].float()
        else:
            mean = torch.tensor([self.stat_means[prefix] for prefix in prefixes],dtype=torch.float32)
            std = torch.tensor([self.stat_stds[prefix] for prefix in prefixes],dtype=torch.float32)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        return data * std + mean