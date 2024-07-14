from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger


class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        variables: List[str],
        parameters: List[str],
        numpoints_x: int,
        numpoints_y: int,
        device:str,
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
        self.device = device
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
                return output.to(self.device), [parameters_normalized.to(self.device), B_normalized.to(self.device)]
            else:
                return [parameters_normalized.to(self.device), B_normalized.to(self.device)], output.to(self.device)

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