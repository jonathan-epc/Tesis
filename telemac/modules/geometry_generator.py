import numpy as np
import xarray as xr
from typing import Tuple, List
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class GeometryGenerator:
    @staticmethod
    def generate_geometry(
        idx: int,
        SLOPE: float,
        flat_mesh: xr.Dataset,
        x: np.ndarray,
        y: np.ndarray,
        noise_grid_x: np.ndarray,
        noise_grid_y: np.ndarray,
        num_points_x: int,
        num_points_y: int,
        channel_length: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Generate geometry with random noise and save it to a file.

        Parameters:
        ----------
        idx : int
            Index to identify the geometry file
        SLOPE : float
            Slope value for the geometry
        flat_mesh : xr.Dataset
            Dataset containing the geometry data
        x : np.ndarray
            X-coordinates for interpolation
        y : np.ndarray
            Y-coordinates for interpolation
        noise_grid_x : np.ndarray
            Grid X-coordinates for the random noise
        noise_grid_y : np.ndarray
            Grid Y-coordinates for the random noise
        num_points_x : int
            Number of points in the X direction for the random noise grid
        num_points_y : int
            Number of points in the Y direction for the random noise grid
        channel_length : float
            Length of the channel

        Returns:
        -------
        Tuple[List[float], List[float]]
            borders_flat and borders_noise values
        """
        min_value, max_value = 0, 0.15
        sigma = 0.95

        random_noise = np.random.rand(num_points_y, num_points_x)
        scaled_random_noise = min_value + (random_noise * (max_value - min_value))
        smoothed_random_noise = gaussian_filter(scaled_random_noise, sigma=sigma)

        interpolator = RegularGridInterpolator(
            (noise_grid_y, noise_grid_x),
            smoothed_random_noise,
            bounds_error=False,
            fill_value=None,
        )

        z_slope = SLOPE * (channel_length - flat_mesh["x"].values)
        z_left_slope = z_slope[0::num_points_x].max()
        z_right_slope = z_slope[num_points_x - 1 :: num_points_x].max()

        z_noise = interpolator((y, x))

        z = z_slope + z_noise

        z_left = z[0::num_points_x].max()
        z_right = z[num_points_x - 1 :: num_points_x].max()

        flat_mesh["B"].values = z_slope.reshape(1, flat_mesh.y.shape[0])
        flat_mesh.selafin.write(f"geometry/geometry_3x3_FLAT_{idx}.slf")

        flat_mesh["B"].values = z.reshape(1, flat_mesh.y.shape[0])
        flat_mesh.selafin.write(f"geometry/geometry_3x3_NOISE_{idx}.slf")

        borders_flat = [z_left_slope, z_right_slope]
        borders_noise = [z_left, z_right]

        return borders_flat, borders_noise