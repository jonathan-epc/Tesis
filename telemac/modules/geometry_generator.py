import numpy as np
import xarray as xr
from typing import Tuple, List
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

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
        SLOPE: float,
        BOTTOM_TYPE: str,
        flat_mesh: xr.Dataset,
        x: np.ndarray,
        y: np.ndarray,
        noise_grid_x: np.ndarray,
        noise_grid_y: np.ndarray,
        num_points_x: int,
        num_points_y: int,
        channel_length: float,
        bump_amplitude: float = 0.1,
        bump_width: float = 0.2,
        step_height: float = 0.1,
        step_position: float = 0.5,
        noise_amplitude: float = 0.15,
        noise_smoothness: float = 0.95,
        sinusoidal_amplitude: float = 0.05,
        sinusoidal_frequency: float = 3,
        random_seed: int = None
    ) -> Tuple[List[float], List[float]]:
        """
        Generates the geometry for the simulation and saves it to files.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Base slope calculation
        z_slope = SLOPE * (channel_length - flat_mesh["x"].values)

        # Initialize z with slope
        z = np.zeros_like(z_slope)
        # if BOTTOM_TYPE == 'SLOPE' or BOTTOM_TYPE == 'NOISE':
        z = z_slope

        # Add noise if required
        if BOTTOM_TYPE == 'NOISE':
            random_noise = np.random.rand(num_points_y, num_points_x)
            scaled_random_noise = noise_amplitude * random_noise
            smoothed_random_noise = gaussian_filter(scaled_random_noise, sigma=noise_smoothness)
            interpolator = RegularGridInterpolator(
                (noise_grid_y, noise_grid_x),
                smoothed_random_noise,
                bounds_error=False,
                fill_value=None,
            )
            z_noise = interpolator((y, x))
            z += z_noise

        # Add bump if required
        if BOTTOM_TYPE == 'BUMP':
            bump_center = channel_length / 2
            bump = bump_amplitude * np.exp(-((flat_mesh["x"].values - bump_center) ** 2) / (2 * bump_width ** 2))
            z += bump

        # Add step if required
        if BOTTOM_TYPE == 'STEP':
            step_index = int(step_position * num_points_x)
            z[:step_index] = 0
            z[step_index:] = step_height

        # Add sinusoidal variation
        if BOTTOM_TYPE == 'SINUSOIDAL':
            z += sinusoidal_amplitude * np.sin(2 * np.pi * sinusoidal_frequency * flat_mesh["x"].values / channel_length)

        # Hybrid approach: flat with slope on one side, noise on the other
        if BOTTOM_TYPE == 'HYBRID':
            transition_point = int(num_points_x * 0.5)  # 50% flat, 50% noise
            z[:transition_point] = z_slope[:transition_point]
            z[transition_point:] = z_slope[transition_point:] + z_noise[transition_point:]

        # Save the generated geometry to file
        flat_mesh["B"].values = z.reshape(1, flat_mesh.y.shape[0])
        flat_mesh.selafin.write(f"geometry/3x3_{BOTTOM_TYPE}_{idx}.slf")

        z_left = z[0::num_points_x].max()
        z_right = z[num_points_x - 1 :: num_points_x].max()
        return [z_left, z_right]