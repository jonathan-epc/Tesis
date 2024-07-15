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
        self.normalized = normalized
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

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
        except (IOError, KeyError) as e:
            logger.error(f"Error accessing data in file {self.file_path}: {e}")
            raise

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

    def __len__(self) -> int:
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

            if self.normalized:
                parameters = [
                    self.normalize(p, param)
                    for p, param in zip(parameters, self.parameters)
                ]
                B = self.normalize(B, "B")
                output = [
                    self.normalize(o, var) for o, var in zip(output, self.variables)
                ]

            parameters_normalized = torch.tensor(parameters, dtype=torch.float32)
            B_normalized = torch.tensor(B, dtype=torch.float32)
            output = torch.stack([torch.tensor(o, dtype=torch.float32) for o in output])

            if self.transform:
                parameters_normalized = self.transform(parameters_normalized)
                B_normalized = self.transform(B_normalized)
            if self.target_transform:
                output = self.target_transform(output)

            if self.swap:
                return output.to(self.device), [
                    parameters_normalized.to(self.device),
                    B_normalized.to(self.device),
                ]
            else:
                return [
                    parameters_normalized.to(self.device),
                    B_normalized.to(self.device),
                ], output.to(self.device)

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


def create_dataloaders(
    file_path: str,
    variables: List[str],
    parameters: List[str],
    numpoints_x: int,
    numpoints_y: int,
    device: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    try:
        dataset = HDF5Dataset(
            file_path, variables, parameters, numpoints_x, numpoints_y, device
        )
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise

    try:
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    except ValueError as e:
        logger.error(f"Error splitting dataset: {e}")
        raise

    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        raise
    return train_dataloader, val_dataloader, test_dataloader