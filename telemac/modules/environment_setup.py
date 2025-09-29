# telemac/modules/environment_setup.py

from pathlib import Path

import xarray as xr
from loguru import logger
from nconfig import Config  # Import the main Config object


class EnvironmentSetup:
    def __init__(self, config: Config):
        self.config = config
        self.flat_mesh = self._load_geometry_dataset()

    def _load_geometry_dataset(self) -> xr.Dataset:
        # Construct path relative to telemac_dir from config
        flat_mesh_path = Path(self.config.paths.telemac_dir) / "geometry/mesh_3x3.slf"
        flat_mesh = xr.open_dataset(flat_mesh_path, engine="selafin")
        logger.info(f"Loaded geometry dataset from {flat_mesh_path}")
        return flat_mesh

    def get_setup_data(self) -> dict:
        return {
            "config": self.config,
            "flat_mesh": self.flat_mesh,
        }
