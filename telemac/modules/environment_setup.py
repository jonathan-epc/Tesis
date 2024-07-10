import yaml
import xarray as xr
import numpy as np
from loguru import logger

class EnvironmentSetup:
    def __init__(self, constants_file="constants.yml", flat_mesh_path="geometry/mesh_3x3.slf"):
        self.constants = self._load_constants(constants_file)
        self._adjust_channel_dimensions()
        self.flat_mesh, self.x, self.y = self._load_geometry_dataset(flat_mesh_path)
        self.noise_grid_x, self.noise_grid_y = self._create_noise_grids()

    def _load_constants(self, constants_file):
        with open(constants_file) as f:
            constants = yaml.safe_load(f)
        logger.info(f"Loaded constants from {constants_file}")
        return constants

    def _adjust_channel_dimensions(self):
        if self.constants["channel"]["wall_thickness"] > 0:
            self.constants["channel"]["width"] += 2 * self.constants["channel"]["wall_thickness"]
            self.constants["channel"]["length"] += 2 * self.constants["channel"]["wall_thickness"]
        logger.info("Adjusted channel dimensions for walls")

    def _load_geometry_dataset(self, flat_mesh_path):
        flat_mesh = xr.open_dataset(flat_mesh_path, engine="selafin")
        x = flat_mesh["x"].values
        y = flat_mesh["y"].values
        logger.info(f"Loaded geometry dataset from {flat_mesh_path}")
        return flat_mesh, x, y

    def _create_noise_grids(self):
        noise_grid_x = np.linspace(
            0, 
            self.constants["channel"]["length"], 
            self.constants["mesh"]["num_points_x"]
        )
        noise_grid_y = np.linspace(
            0, 
            self.constants["channel"]["width"], 
            self.constants["mesh"]["num_points_y"]
        )
        logger.info("Created noise grids")
        return noise_grid_x, noise_grid_y

    def get_setup_data(self):
        return {
            "constants": self.constants,
            "flat_mesh": self.flat_mesh,
            "x": self.x,
            "y": self.y,
            "noise_grid_x": self.noise_grid_x,
            "noise_grid_y": self.noise_grid_y
        }