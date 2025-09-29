from collections.abc import Callable

import h5py
import numpy as np
import torch
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Optimized PyTorch Dataset for FNO training with HDF5 files.

    This dataset provides efficient loading and preprocessing of hydraulic simulation
    data stored in HDF5 format. It includes support for Box-Cox transformations,
    normalization, data augmentation, and various loading strategies (preloading,
    chunked loading, or on-demand loading).

    The dataset handles both field data (spatial arrays) and scalar parameters,
    with configurable input and output variable selection. All data is automatically
    converted to PyTorch tensors and moved to the specified device.

    Attributes:
        FIELDS: List of available field variable names in the dataset.
        NUMERIC_SCALARS: List of numeric scalar parameter names.
        NON_NUMERIC_SCALARS: List of non-numeric scalar parameter names.
        ADIMENSIONAL_FIELDS: List of dimensionless field variable names.
        ADIMENSIONAL_SCALARS: List of dimensionless scalar parameter names.
        SCALARS: Combined list of all scalar parameter names.
        file_path: Path to the HDF5 data file.
        input_vars: List of variable names to use as model inputs.
        output_vars: List of variable names to use as model outputs.
        nx: Number of grid points in x direction.
        ny: Number of grid points in y direction.
        channel_length: Physical length of the channel domain.
        channel_width: Physical width of the channel domain.
        device: PyTorch device for tensor operations.
        normalize_input: Whether to normalize input variables.
        normalize_output: Whether to normalize output variables.
        transform: Optional transform function for inputs.
        target_transform: Optional transform function for outputs.
        augment: Whether to apply data augmentation (horizontal flipping).
        preload: Whether to preload all data into memory.
        chunk_size: Size of data chunks for chunked loading strategy.
        boxcox_transform: Whether to apply Box-Cox transformations.
        boxcox_lambdas: Dictionary of Box-Cox lambda parameters per variable.
        boxcox_shifts: Dictionary of shift values to ensure positivity.
        keys: List of case identifiers in the dataset.
        len: Total number of cases in the dataset.
        stats: Tuple containing mean, std, and minimum statistics.
        scalar_input_indices: List of scalar variables in input_vars.
        non_scalar_input_indices: List of field variables in input_vars.
        scalar_output_indices: List of scalar variables in output_vars.
        non_scalar_output_indices: List of field variables in output_vars.
    """

    # Class constants
    FIELDS: list[str] = ["B", "F", "H", "Q", "S", "U", "V", "D", "B*", "H*", "U*", "V*"]
    NUMERIC_SCALARS: list[str] = [
        "H0",
        "Q0",
        "SLOPE",
        "n",
        "nut",
        "Vr",
        "Fr",
        "Re",
        "Ar",
        "Hr",
        "M",
    ]
    NON_NUMERIC_SCALARS: list[str] = [
        "BOTTOM",
        "direction",
        "id",
        "subcritical",
        "yc",
        "yn",
    ]
    ADIMENSIONAL_FIELDS: list[str] = ["B*", "H*", "U*", "V*"]
    ADIMENSIONAL_SCALARS: list[str] = ["Ar", "Vr", "Fr", "Hr", "Re", "M"]
    SCALARS: list[str] = NUMERIC_SCALARS + NON_NUMERIC_SCALARS

    def __init__(
        self,
        file_path: str,
        input_vars: list[str],
        output_vars: list[str],
        numpoints_x: int,
        numpoints_y: int,
        channel_length: float,
        channel_width: float,
        device: torch.device,
        normalize_input: bool = True,
        normalize_output: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        augment: bool = True,
        preload: bool = False,
        chunk_size: int | None = None,
        boxcox_transform: bool = False,
    ) -> None:
        """Initialize the HDF5Dataset.

        Args:
            file_path: Path to the HDF5 file containing the dataset.
            input_vars: List of variable names to use as model inputs.
            output_vars: List of variable names to use as model outputs.
            numpoints_x: Number of grid points in the x direction.
            numpoints_y: Number of grid points in the y direction.
            channel_length: Physical length of the channel domain.
            channel_width: Physical width of the channel domain.
            device: PyTorch device where tensors should be moved.
            normalize_input: Whether to apply normalization to input variables.
            normalize_output: Whether to apply normalization to output variables.
            transform: Optional callable to transform input data.
            target_transform: Optional callable to transform output data.
            augment: Whether to apply data augmentation (horizontal flipping).
            preload: Whether to load all data into memory at initialization.
            chunk_size: Number of cases to load per chunk. If None, loads on-demand.
            boxcox_transform: Whether to apply Box-Cox transformation to variables.

        Raises:
            ValueError: If any variables in input_vars or output_vars are not found
                in the dataset.
        """
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
        self.boxcox_lambdas: dict[str, float] = {}
        self.boxcox_shifts: dict[str, float] = {}

        # Load dataset metadata and statistics
        with h5py.File(self.file_path, "r", swmr=True) as f:
            self.keys = [key for key in f if key != "statistics"]
            self.len = len(self.keys)

            if self.keys:
                first_key = self.keys[0]
                group = f[first_key]
                available_fields = {var for var in self.FIELDS if var in group}
                available_scalars = {
                    scalar for scalar in self.SCALARS if scalar in group.attrs
                }
            else:
                available_fields = set()
                available_scalars = set()

            self.FIELDS = list(available_fields)
            self.SCALARS = list(available_scalars)

            self.stats = (
                self._load_stats(f)
                if any([normalize_input, normalize_output, boxcox_transform])
                else ({}, {}, {})
            )

            if self.boxcox_transform:
                variables = set(input_vars + output_vars)
                for var in variables:
                    if var in self.stats[2]:  # minima dict
                        min_val = self.stats[2][var]
                        # Calculate shift to ensure positivity
                        self.boxcox_shifts[var] = max(0.0, -min_val) + 1e-6
                        self.boxcox_lambdas[var] = 0.0

        # Calculate variable indices for efficient processing
        self.scalar_input_indices = [v for v in self.input_vars if v in self.SCALARS]
        self.non_scalar_input_indices = [v for v in self.input_vars if v in self.FIELDS]
        self.scalar_output_indices = [v for v in self.output_vars if v in self.SCALARS]
        self.non_scalar_output_indices = [
            v for v in self.output_vars if v in self.FIELDS
        ]

        self.data: dict[str, dict] | None = self._preload_data() if preload else None
        self.h5_file: h5py.File | None = None
        self.current_chunk: dict[str, dict] | None = None
        self.current_chunk_idx = -1 if chunk_size else None

        self._validate_variables(input_vars + output_vars)

    @classmethod
    def from_config(cls, config, file_path: str) -> "HDF5Dataset":
        """Create an HDF5Dataset instance from a configuration object.

        Args:
            config: Configuration object containing dataset parameters.
            file_path: Path to the HDF5 file.

        Returns:
            Configured HDF5Dataset instance.
        """
        return cls(
            file_path=file_path,
            input_vars=config.data.inputs,
            output_vars=config.data.outputs,
            numpoints_x=config.mesh.num_points_x,
            numpoints_y=config.mesh.num_points_y,
            channel_length=config.channel.length,
            channel_width=config.channel.width,
            normalize_input=config.data.normalize_input,
            normalize_output=config.data.normalize_output,
            device=config.device,
            preload=config.data.preload_hdf5,
            chunk_size=config.data.chunk_size_hdf5,
            boxcox_transform=config.data.boxcox_transform,
        )

    def _process_variable(self, case: dict, var: str, normalize: bool) -> torch.Tensor:
        """Process a single variable from a case dictionary.

        Converts the variable to a PyTorch tensor, applies transformations,
        and moves to the appropriate device.

        Args:
            case: Dictionary containing case data.
            var: Variable name to process.
            normalize: Whether to apply normalization to this variable.

        Returns:
            Processed tensor ready for model input/output.
        """
        tensor = (
            torch.tensor(case[var], dtype=torch.float32)
            if not isinstance(case[var], torch.Tensor)
            else case[var]
        )
        tensor = tensor.clone().detach().to(dtype=torch.float32)
        tensor = tensor.reshape(self.ny, self.nx) if var in self.FIELDS else tensor

        if self.boxcox_transform and var in self.boxcox_lambdas:
            tensor = self._apply_boxcox(tensor, var)

        if normalize:
            tensor = self._normalize(tensor, var)

        return tensor.to(self.device)

    def _validate_variables(self, vars_to_check: list[str]) -> None:
        """Validate that all requested variables exist in the dataset.

        Args:
            vars_to_check: List of variable names to validate.

        Raises:
            ValueError: If any variables are not found in the dataset.
        """
        valid_variables = set(self.FIELDS + self.SCALARS)
        invalid_vars = set(vars_to_check) - valid_variables
        if invalid_vars:
            raise ValueError(f"Unknown variables/parameters: {invalid_vars}")

    def _load_stats(
        self, f: h5py.File
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Load statistical information from the HDF5 file.

        Args:
            f: Open HDF5 file handle.

        Returns:
            Tuple containing dictionaries of means, standard deviations, and minima
            for each variable in the dataset.
        """
        if "statistics" not in f:
            return {}, {}, {}

        stats_group = f["statistics"]
        fields = self.FIELDS + [s for s in self.NUMERIC_SCALARS if s in self.SCALARS]
        return (
            {k: stats_group.attrs[f"{k}_mean"] for k in fields},  # Means
            {k: np.sqrt(stats_group.attrs[f"{k}_variance"]) for k in fields},  # Stds
            {k: stats_group.attrs[f"{k}_min"] for k in fields},  # Minima
        )

    def _preload_data(self) -> dict[str, dict]:
        """Preload all dataset cases into memory.

        Returns:
            Dictionary mapping case keys to loaded case data.
        """
        with h5py.File(self.file_path, "r", swmr=True) as f:
            return {key: self._load_case(f[key]) for key in self.keys}

    def _load_case(self, group: h5py.Group) -> dict:
        """Load a single case from an HDF5 group.

        Args:
            group: HDF5 group containing case data.

        Returns:
            Dictionary containing loaded case data with scalars and fields.
        """
        case = {
            param: group.attrs[param] for param in self.SCALARS if param in group.attrs
        }
        case.update(
            {
                var: torch.from_numpy(group[var][()]).float()
                for var in self.FIELDS
                if var in group
            }
        )
        return case

    def _get_case(self, idx: int) -> dict:
        """Retrieve a case by index using the appropriate loading strategy.

        Args:
            idx: Case index.

        Returns:
            Dictionary containing case data.
        """
        if self.preload:
            return self.data[self.keys[idx]]

        if self.chunk_size:
            chunk_idx = idx // self.chunk_size
            if chunk_idx != self.current_chunk_idx:
                self._load_chunk(chunk_idx)
            return self.current_chunk[self.keys[idx]]

        # On-demand loading
        if self.h5_file is None or not bool(self.h5_file):
            self.h5_file = h5py.File(self.file_path, "r", swmr=True)
        return self._load_case(self.h5_file[self.keys[idx]])

    def _load_chunk(self, chunk_idx: int) -> None:
        """Load a chunk of cases into memory.

        Args:
            chunk_idx: Index of the chunk to load.
        """
        start, end = (
            chunk_idx * self.chunk_size,
            min((chunk_idx + 1) * self.chunk_size, self.len),
        )

        if self.h5_file is None:
            with h5py.File(self.file_path, "r", swmr=True) as f:
                self.h5_file = f

        self.current_chunk = {
            key: self._load_case(self.h5_file[key]) for key in self.keys[start:end]
        }
        self.current_chunk_idx = chunk_idx

    def _normalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Apply normalization to a tensor using dataset statistics.

        Args:
            tensor: Input tensor to normalize.
            var: Variable name for retrieving statistics.

        Returns:
            Normalized tensor with zero mean and unit variance.
        """
        mean, std = self.stats[0][var], self.stats[1][var]
        return (tensor - mean) / std

    def _denormalize(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Reverse normalization transformation.

        Args:
            tensor: Normalized tensor.
            var: Variable name for retrieving statistics.

        Returns:
            Denormalized tensor in original scale.
        """
        mean, std = self.stats[0][var], self.stats[1][var]
        return tensor * std + mean

    def _apply_boxcox(self, tensor: torch.Tensor, var: str) -> torch.Tensor:
        """Apply Box-Cox transformation to a tensor.

        Transforms the data to improve normality and stabilize variance.
        The transformation requires positive values, so a shift is applied first.

        Args:
            tensor: Input tensor (field or scalar).
            var: Variable name for retrieving transformation parameters.

        Returns:
            Box-Cox transformed tensor.
        """
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
        """Apply inverse Box-Cox transformation to a tensor.

        Reverses the Box-Cox transformation to return data to original scale.

        Args:
            tensor: Box-Cox transformed tensor.
            var: Variable name for retrieving transformation parameters.

        Returns:
            Tensor in original scale before Box-Cox transformation.
        """
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
        """Return the total number of cases in the dataset.

        Returns:
            Number of cases available in the dataset.
        """
        return self.len

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[list[torch.Tensor], list[torch.Tensor]],
        tuple[list[torch.Tensor], list[torch.Tensor]],
    ]:
        """Retrieve a single item from the dataset.

        Loads and processes the case at the given index, separating field and scalar
        variables for both inputs and outputs. Optionally applies data augmentation.

        Args:
            idx: Index of the case to retrieve.

        Returns:
            Tuple containing ((input_fields, input_scalars), (output_fields, output_scalars)).
            Each element is a list of processed PyTorch tensors.
        """
        case = self._get_case(idx)

        input_fields = [
            self._process_variable(case, var, self.normalize_input)
            for var in self.input_vars
            if var not in self.SCALARS
        ]
        input_scalars = [
            self._process_variable(case, var, self.normalize_input)
            for var in self.input_vars
            if var in self.SCALARS
        ]

        output_fields = [
            self._process_variable(case, var, self.normalize_output)
            for var in self.output_vars
            if var not in self.SCALARS
        ]
        output_scalars = [
            self._process_variable(case, var, self.normalize_output)
            for var in self.output_vars
            if var in self.SCALARS
        ]

        # Apply data augmentation (horizontal flipping)
        if self.augment and torch.rand(1).item() > 0.5:
            input_fields = [torch.flip(f, [0]) for f in input_fields]
            output_fields = [torch.flip(f, [0]) for f in output_fields]

        return (input_fields, input_scalars), (output_fields, output_scalars)

    def __del__(self) -> None:
        """Clean up open HDF5 file handles when the dataset is destroyed."""
        if self.h5_file is not None:
            self.h5_file.close()
