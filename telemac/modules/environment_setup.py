import yaml
import xarray as xr
import numpy as np
from loguru import logger

class EnvironmentSetup:
    """
    A class to handle the setup of the environment for a simulation.

    Attributes
    ----------
    constants : dict
        A dictionary containing the constants loaded from the YAML file.
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

    Methods
    -------
    get_setup_data()
        Returns a dictionary containing the setup data.
    """

    def __init__(self, constants_file="constants.yml", flat_mesh_path="geometry/mesh_3x3.slf"):
        """
        Initializes the EnvironmentSetup instance.

        Parameters
        ----------
        constants_file : str, optional
            The path to the YAML file containing the constants. Default is "constants.yml".
        flat_mesh_path : str, optional
            The path to the flat mesh geometry file. Default is "geometry/mesh_3x3.slf".
        """
        self.constants = self._load_constants(constants_file)
        self._adjust_channel_dimensions()
        self.flat_mesh = self._load_geometry_dataset(flat_mesh_path)

    def _load_constants(self, constants_file):
        """
        Loads the constants from a YAML file.

        Parameters
        ----------
        constants_file : str
            The path to the YAML file containing the constants.

        Returns
        -------
        dict
            A dictionary containing the constants.
        """
        with open(constants_file) as f:
            constants = yaml.safe_load(f)
        logger.info(f"Loaded constants from {constants_file}")
        return constants

    def _adjust_channel_dimensions(self):
        """
        Adjusts the channel dimensions based on the wall thickness.
        """
        if self.constants["channel"]["wall_thickness"] > 0:
            self.constants["channel"]["width"] += 2 * self.constants["channel"]["wall_thickness"]
            self.constants["channel"]["length"] += 2 * self.constants["channel"]["wall_thickness"]
        logger.info("Adjusted channel dimensions for walls")

    def _load_geometry_dataset(self, flat_mesh_path):
        """
        Loads the geometry dataset from a file.

        Parameters
        ----------
        flat_mesh_path : str
            The path to the flat mesh geometry file.

        Returns
        -------
        tuple
            A tuple containing the flat mesh dataset, x-coordinates, and y-coordinates.
        """
        flat_mesh = xr.open_dataset(flat_mesh_path, engine="selafin")
        logger.info(f"Loaded geometry dataset from {flat_mesh_path}")
        return flat_mesh


    def get_setup_data(self):
        """
        Returns a dictionary containing the setup data.

        Returns
        -------
        dict
            A dictionary containing the constants, flat mesh, x and y coordinates,
            and noise grid coordinates.
        """
        return {
            "constants": self.constants,
            "flat_mesh": self.flat_mesh,
        }
