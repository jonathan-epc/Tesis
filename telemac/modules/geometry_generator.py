import numpy as np
import xarray as xr
from typing import Tuple, List
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from loguru import logger
import math
import hashlib

class GeometryGenerator:
    """
    A class to generate geometry for a simulation.

    Methods
    -------
    generate_geometry(idx, SLOPE, flat_mesh, x, y, noise_grid_x, noise_grid_y, num_points_x, num_points_y, channel_length)
        Generates the geometry for the simulation and saves it to files.
    """

    @staticmethod
    def generate_geometry(
        idx: int,
        slope: float,
        bottom_type: str,
        flat_mesh: xr.Dataset,
        num_points_x: int,
        num_points_y: int,
        channel_length: float,
        channel_width: float,
        **kwargs
    ) -> Tuple[List[float], List[float]]:
        
        """
        Generate channel geometry based on specified parameters.

        Args:
            bottom_type (str): Type of channel bottom ('NOISE', 'HYBRID', 'BUMP2D', 'BUMP', 'STEP', 'SMOOTH_STEP', 'SINUSOIDAL_X', 'SINUSOIDAL_Y', 'BARS')
            slope (float): Channel slope
            channel_length (float): Length of the channel
            channel_width (float): Width of the channel
            num_points_x (int): Number of points along x-axis
            num_points_y (int): Number of points along y-axis
            **kwargs: Additional parameters for specific bottom types

        Returns:
            np.ndarray: 2D array representing the channel geometry
        """

        x = np.linspace(0, channel_length, num_points_x)
        y = np.linspace(0, channel_width, num_points_y)
        xv, yv = np.meshgrid(x, y, indexing="ij")

        z = slope * (channel_length - xv)

        variation_function = getattr(GeometryGenerator, f"_{bottom_type.lower()}_variation", None)
        if variation_function:
            variation = variation_function(xv, yv, channel_length, channel_width, idx=idx, **kwargs)
            z += GeometryGenerator._normalize_variation(variation, kwargs.get('variation_min', 0.0), kwargs.get('variation_max', 0.1))
        geometry_hash = hashlib.md5(z.tobytes()).hexdigest()
        logger.debug(f"Geometry hash for idx {idx}, bottom_type {bottom_type}: {geometry_hash}")
        # Save the generated geometry to file
        flat_mesh["B"].values = z.T.reshape(1, flat_mesh.y.shape[0])
        flat_mesh.selafin.write(f"geometry/3x3_{bottom_type}_{idx}.slf")

        z_left = z[0::num_points_x].max()
        z_right = z[num_points_x - 1 :: num_points_x].max()
        # Check for NaN values
        if any(math.isnan(x) for x in [z_left, z_right]):
            raise ValueError(f"NaN value encountered in elevation calculations. z={z_left, z_right}")
        return [z_left, z_right]

    @staticmethod
    def _normalize_variation(variation, min_var, max_var):
        """Normalize the variation between min_var and max_var."""
        return min_var + (variation - variation.min()) / (variation.ptp()) * (
            max_var - min_var
        )

    @staticmethod
    def _noise_variation(xv, yv, channel_length, channel_width, **kwargs):
        return gaussian_filter(
            np.random.rand(*xv.shape) * kwargs.get("noise_amplitude", 0.15),
            sigma=kwargs.get("noise_smoothness", 0.95),
        )

    @staticmethod
    def _hybrid_variation(xv, yv, channel_length, channel_width, **kwargs):
        noise = GeometryGenerator._noise_variation(
            xv, yv, channel_length, channel_width, **kwargs
        )
        transition_point = int(
            xv.shape[0] // 2 + np.random.uniform(-0.4, 0.4) * channel_length
        )
        noise[:transition_point] = 0
        return noise

    @staticmethod
    def _bump2d_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        bump_center_x = channel_length / 2 + (
            np.random.uniform(-0.3, 0.3) * channel_length if randomize else 0
        )
        bump_center_y = channel_width / 2 + (
            np.random.uniform(-0.05, 0.05) * channel_width if randomize else 0
        )
        bump_width = kwargs.get("bump_width_x", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )
        bump_amplitude = kwargs.get("bump_amplitude", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )

        return bump_amplitude * np.exp(
            -((xv - bump_center_x) ** 2 + (yv - bump_center_y) ** 2)
            / (2 * bump_width**2)
        )

    @staticmethod
    def _bump_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        bump_center = channel_length / 2 + (
            np.random.uniform(-0.3, 0.3) * channel_length if randomize else 0
        )
        bump_width = kwargs.get("bump_width_x", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )
        bump_amplitude = kwargs.get("bump_amplitude", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )

        return bump_amplitude * np.exp(-((xv - bump_center) ** 2) / (2 * bump_width**2))

    @staticmethod
    def _abrupt_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        step_position = kwargs.get("abrupt_position", 0.5) + (
            np.random.uniform(-0.05, 0.05) if randomize else 0
        )
        step_height = kwargs.get("abrupt_height", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )

        step_index = int(step_position * xv.shape[0])
        step = np.zeros_like(xv)
        step[step_index:] = step_height
        return step

    @staticmethod
    def _step_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        step_position = kwargs.get("step_position", 0.5) + (
            np.random.uniform(-0.05, 0.05) if randomize else 0
        )
        step_height = kwargs.get("h0", 0.05) * 0.5
        step_length = kwargs.get("h0", 0.01) * (
            np.random.uniform(1.5, 3.0) if randomize else 1
        )

        step_index = int(step_position * xv.shape[0])
        step_finish = step_index+int(np.ceil(step_length * xv.shape[0]/channel_length))
        step = np.zeros_like(xv)
        step[step_index:step_finish] = step[step_index]+step_height
        return step

    @staticmethod
    def _smooth_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        step_center = kwargs.get("smooth_position", 0.5) * channel_length + (
            np.random.uniform(-0.05, 0.05) * channel_length if randomize else 0
        )
        step_width = (
            0.01 * channel_length * (np.random.uniform(0.9, 1.1) if randomize else 1)
        )
        step_height = kwargs.get("smooth_height", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )

        return step_height / (1 + np.exp(-(xv - step_center) / step_width))

    @staticmethod
    def _sinusoidal_x_variation(xv, yv, channel_length, channel_width, **kwargs):
        return GeometryGenerator._sinusoidal_variation(
            xv, yv, channel_length, channel_width, "x", **kwargs
        )

    @staticmethod
    def _sinusoidal_y_variation(xv, yv, channel_length, channel_width, **kwargs):
        return GeometryGenerator._sinusoidal_variation(
            xv, yv, channel_length, channel_width, "y", **kwargs
        )

    @staticmethod
    def _bars_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        bars_frequency = kwargs.get(
            "bars_frequency", 2 * np.pi * channel_width / channel_length
        ) * (np.random.uniform(10, 15) if randomize else 1)
        bars_amplitude = kwargs.get("bars_amplitude", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )
        phase_shift = kwargs.get("phase_shift", 1) * (
            np.random.uniform(0, 2 * np.pi) if randomize else 1
        )

        x_sin = np.sin(bars_frequency * xv + phase_shift)

        transition_type = kwargs.get("transition_type", "sigmoid")
        transition = GeometryGenerator._get_transition(yv, transition_type)

        combined_sin = (
            x_sin * (1 - transition)
            + np.sin(bars_frequency * xv + phase_shift + np.pi) * transition
        )

        return bars_amplitude * combined_sin

    @staticmethod
    def _real_bars_variation(xv, yv, channel_length, channel_width, **kwargs):
        randomize = kwargs.get("randomize", True)
        bars_frequency = kwargs.get(
            "bars_frequency", 2 * np.pi * channel_width / channel_length
        ) * (np.random.uniform(10, 15) if randomize else 1)
        bars_amplitude = kwargs.get("bars_amplitude", 0.1) * (
            np.random.uniform(0.9, 1.1) if randomize else 1
        )
        phase_shift = kwargs.get("phase_shift", 1) * (
            np.random.uniform(0, 2 * np.pi) if randomize else 1
        )
        longitudinal_variation = kwargs.get("longitudinal_variation", 0.2)
        irregularity_chance = kwargs.get("irregularity_chance", 0.1)
        irregularity_magnitude = kwargs.get("irregularity_magnitude", 0.05)

        x_sin = np.sin(bars_frequency * xv + phase_shift)

        harmonics = kwargs.get("harmonics", [(2, 0.3)])
        for harmonic, amplitude in harmonics:
            x_sin += amplitude * np.sin(harmonic * bars_frequency * xv + phase_shift)

        transition_type = kwargs.get("transition_type", "sigmoid")
        transition = GeometryGenerator._get_transition(yv, transition_type)

        combined_sin = (
            x_sin * (1 - transition)
            + np.sin(bars_frequency * xv + phase_shift + np.pi) * transition
        )
        bars = bars_amplitude * combined_sin

        longitudinal_factor = 1 + longitudinal_variation * np.sin(
            2 * np.pi * xv / channel_length
        )
        bars *= longitudinal_factor

        irregularities = np.random.random(xv.shape) < irregularity_chance
        bars[irregularities] += np.random.uniform(
            -irregularity_magnitude, irregularity_magnitude, size=np.sum(irregularities)
        )

        return bars

    @staticmethod
    def _get_transition(yv, transition_type):
        normalized_y = (yv - yv.min()) / yv.ptp()
        if transition_type == "linear":
            return normalized_y
        elif transition_type == "sigmoid":
            return 1 / (1 + np.exp(-10 * (normalized_y - 0.5)))
        elif transition_type == "exponential":
            return np.exp(-normalized_y * 5)
        elif transition_type == "sinusoidal":
            return np.sin(-normalized_y * 5)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")