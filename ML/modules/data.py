from functools import lru_cache
from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """
    A PyTorch Dataset for loading and processing HDF5 data.

    This dataset class is designed to work with HDF5 files containing
    parameter data, B fields, and variable data. It supports normalization,
    denormalization, and optional data transformations.

    Attributes:
        file_path (str): Path to the HDF5 file.
        variables (List[str]): List of variable names to load.
        parameters (List[str]): List of parameter names to load.
        numpoints_x (int): Number of points in the x-dimension.
        numpoints_y (int): Number of points in the y-dimension.
        device (str): Device to load the data onto ('cpu' or 'cuda').
        already_normalized (bool): Whether the data is already normalized.
        swap (bool): Whether to swap inputs and outputs.
        transform (Optional[Callable]): Optional transform to be applied on the input.
        target_transform (Optional[Callable]): Optional transform to be applied on the target.
        keys (List[str]): List of keys in the HDF5 file.
        stat_means (dict): Mean values for normalization.
        stat_stds (dict): Standard deviation values for normalization.
        data (dict): Pre-loaded data from the HDF5 file.
    """

    def __init__(
        self,
        file_path: str,
        variables: List[str],
        parameters: List[str],
        numpoints_x: int,
        numpoints_y: int,
        device: str,
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
            device (str): Device to load the data onto ('cpu' or 'cuda').
            normalized (bool, optional): Whether the data is already normalized. Defaults to True.
            swap (bool, optional): Whether to swap inputs and outputs. Defaults to False.
            transform (Optional[Callable], optional): Optional transform to be applied on the input. Defaults to None.
            target_transform (Optional[Callable], optional): Optional transform to be applied on the target. Defaults to None.
        """
        self.file_path = file_path
        self.variables = variables
        self.parameters = parameters
        self.numpoints_x = numpoints_x
        self.numpoints_y = numpoints_y
        self.normalize_input, self.normalize_output = normalize
        self.swap = swap
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.augment = augment

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

    def _load_data(self) -> dict:
        """
        Load all data from the HDF5 file into memory.

        Returns:
            dict: Dictionary containing all loaded data.
        """
        data = {}
        with h5py.File(self.file_path, "r") as file:
            for key in self.keys:
                data[key] = {param: file[key].attrs[param] for param in self.parameters}
                data[key]["B"] = torch.from_numpy(file[key]["B"][()]).float()
                for var in self.variables:
                    data[key][var] = torch.from_numpy(file[key][var][()]).float()
        return data

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.keys)

    @lru_cache(maxsize=None)
    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]: 
                A tuple containing the input data and target data.
        """
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
        
        if self.normalize_input:
            parameters = self.normalize(parameters, self.parameters)
            B = self.normalize(B, "B")

        if self.normalize_output:
            output = self.normalize(output, self.variables)       

        if self.augment and torch.rand(1).item() > 0.5:
            B = torch.flip(B, [0])  
            output = torch.flip(output, [1])  

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
        """
        Normalize the given data using pre-computed statistics.

        Args:
            data (torch.Tensor): Data to be normalized.
            prefixes (Union[str, List[str]]): Prefix(es) for selecting normalization statistics.

        Returns:
            torch.Tensor: Normalized data.
        """
        if isinstance(prefixes, str):
            mean = torch.tensor(self.stat_means[prefixes], dtype=torch.float32)
            std = torch.tensor(self.stat_stds[prefixes], dtype=torch.float32)
        else:
            mean = torch.tensor([self.stat_means[prefix] for prefix in prefixes], dtype=torch.float32)
            std = torch.tensor([self.stat_stds[prefix] for prefix in prefixes], dtype=torch.float32)

        if data.dim() ==3:
            mean = mean.view(-1,1,1)
            std = std.view(-1,1,1)
            
        return (data - mean) / std


    def denormalize(
        self, data: torch.Tensor, prefixes: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Denormalize the given data using pre-computed statistics.

        Args:
            data (torch.Tensor): Data to be denormalized.
            prefixes (Union[str, List[str]]): Prefix(es) for selecting denormalization statistics.

        Returns:
            torch.Tensor: Denormalized data.
        """
        if isinstance(prefixes, str):
            mean = torch.tensor(self.stat_means[prefixes], dtype=torch.float32)
            std = torch.tensor(self.stat_stds[prefixes], dtype=torch.float32)
        else:
            mean = torch.tensor([self.stat_means[prefix] for prefix in prefixes], dtype=torch.float32)
            std = torch.tensor([self.stat_stds[prefix] for prefix in prefixes], dtype=torch.float32)

        if data.dim() ==3:
            mean = mean.view(-1,1,1)
            std = std.view(-1,1,1)
        return data * std + mean

    def __repr__(self) -> str:
        """
        Return a string representation of the HDF5Dataset.

        Returns:
            str: String representation of the HDF5Dataset.
        """
        return (f"HDF5Dataset(file_path='{self.file_path}', "
                f"variables={self.variables}, "
                f"parameters={self.parameters}, "
                f"numpoints_x={self.numpoints_x}, "
                f"numpoints_y={self.numpoints_y}, "
                f"device='{self.device}', "
                f"normalized={self.already_normalized}, "
                f"swap={self.swap})")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the dataset.

        Returns:
            Tuple[int, int, int]: A tuple containing (num_samples, numpoints_y, numpoints_x).
        """
        return len(self), self.numpoints_y, self.numpoints_x