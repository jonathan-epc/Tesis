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
        BOTTOM: str,
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
        Generates the geometry for the simulation and saves it to files.

        Parameters
        ----------
        idx : int
            The index of the geometry file.
        SLOPE : float
            The slope of the channel.
        flat_mesh : xarray.Dataset
            The dataset containing the flat mesh geometry.
        x : numpy.ndarray
            The x-coordinates of the flat mesh.
        y : numpy.ndarray
            The y-coordinates of the flat mesh.
        noise_grid_x : numpy.ndarray
            The x-coordinates of the noise grid.
        noise_grid_y : numpy.ndarray
            The y-coordinates of the noise grid.
        num_points_x : int
            The number of points along the x-axis.
        num_points_y : int
            The number of points along the y-axis.
        channel_length : float
            The length of the channel.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing the flat and noisy border elevations.

        Examples
        --------
        >>> GeometryGenerator.generate_geometry(
        ...     1, 0.01, flat_mesh, x, y, noise_grid_x, noise_grid_y, 100, 100, 10.0
        ... )
        ([0.0, 0.1], [0.01, 0.11])
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

        if BOTTOM == 'FLAT':
            flat_mesh["B"].values = z_slope.reshape(1, flat_mesh.y.shape[0])
            flat_mesh.selafin.write(f"geometry/3x3_FLAT_{idx}.slf")

        elif BOTTOM == 'NOISE':
            flat_mesh["B"].values = z.reshape(1, flat_mesh.y.shape[0])
            flat_mesh.selafin.write(f"geometry/3x3_NOISE_{idx}.slf")

        borders_flat = [z_left_slope, z_right_slope]
        borders_noise = [z_left, z_right]

        return borders_flat, borders_noise
