# telemac/modules/telemac_case.py
"""Module containing TelemacCase dataclass for TELEMAC simulation management.

This module provides the TelemacCase dataclass that encapsulates all information
needed for a single TELEMAC simulation case, including parameter management,
file path generation, and automated file creation for geometry and steering files.

Typical usage example:
    case = TelemacCase(
        case_id=1,
        params={
            "SLOPE": 0.01,
            "BOTTOM": "smooth",
            "Q0": 1.0,
            "H0": 2.0,
            "L": 100.0,
            "W": 50.0,
            "n": 0.03,
            "direction": "x",
            "num_points_x": 11,
            "num_points_y": 11
        }
    )
    geometry = case.generate_geometry(flat_mesh)
    case.generate_steering_file(borders)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TelemacCase:
    """A data class to hold all information for a single TELEMAC simulation case.

    This class encapsulates case identification, simulation parameters, and manages
    file paths for geometry, steering, and results files. It provides methods to
    automatically generate geometry and steering files based on the configured
    parameters.

    Attributes:
        case_id: Unique identifier for the simulation case.
        params: Dictionary containing all simulation parameters including geometry,
            boundary conditions, and physical properties.
        steering_file_path: Path to the generated steering (.cas) file.
        geometry_file_path: Path to the generated geometry (.slf) file.
        results_file_path: Path where simulation results will be stored.
    """

    # Core identifying and physical parameters
    case_id: int | str
    params: dict[str, Any]

    # Paths to generated files (set automatically after initialization)
    steering_file_path: Path = field(init=False)
    geometry_file_path: Path = field(init=False)
    results_file_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Set up file paths after the object is created.

        Constructs standardized file paths based on case_id and bottom type.
        The paths follow the convention:
        - Steering files: telemac/steering/{case_id}.cas
        - Geometry files: telemac/geometry/3x3_{bottom_type}_{case_id}.slf
        - Results files: telemac/results/{case_id}.slf
        """
        bottom_type = self.params.get("BOTTOM", "UNKNOWN")

        # Using Path objects for robust path handling
        self.steering_file_path = Path(f"telemac/steering/{self.case_id}.cas")
        self.geometry_file_path = Path(
            f"telemac/geometry/3x3_{bottom_type}_{self.case_id}.slf"
        )
        self.results_file_path = Path(f"telemac/results/{self.case_id}.slf")

    def generate_geometry(self, flat_mesh: Any) -> list[float]:
        """Generate the geometry file for this simulation case.

        Creates geometry data by calling the GeometryGenerator with parameters
        specific to this case including slope, bottom type, channel dimensions,
        and mesh resolution.

        Args:
            flat_mesh: Mesh object containing the base geometric structure.
                The exact type depends on the mesh library being used.

        Returns:
            A list of float values representing the generated geometry data,
            typically containing node coordinates and connectivity information.

        Raises:
            ImportError: If the geometry_generator module cannot be imported.
            ValueError: If required parameters are missing from self.params.
        """
        from .geometry_generator import (
            GeometryGenerator,
        )

        return GeometryGenerator.generate_geometry(
            idx=self.case_id,
            slope=self.params["SLOPE"],
            bottom_type=self.params["BOTTOM"],
            flat_mesh=flat_mesh,
            num_points_x=self.params["num_points_x"],
            num_points_y=self.params["num_points_y"],
            channel_length=self.params["L"],
            channel_width=self.params["W"],
            h0=self.params["H0"],
            adimensional=self.params.get("adimensional", False),
        )

    def generate_steering_file(self, borders: list[float]) -> None:
        """Generate and write the steering file for this simulation case.

        Creates a TELEMAC steering file (.cas) containing all simulation
        parameters, boundary conditions, and file references. The file is
        automatically saved to the path specified in steering_file_path.

        Args:
            borders: List of boundary coordinates defining the computational
                domain boundaries. Values represent spatial coordinates in
                the same units as the geometry.

        Raises:
            ImportError: If boundary_conditions or steering_file_generator
                modules cannot be imported.
            OSError: If the steering file cannot be written to disk.
            ValueError: If required parameters are missing from self.params.
        """
        from .boundary_conditions import BoundaryConditions
        from .steering_file_generator import SteeringFileGenerator

        # Generate boundary conditions and prescribed elevations
        boundary_file, prescribed_elevations = (
            BoundaryConditions.get_boundary_and_elevations(
                direction=self.params["direction"],
                h0=self.params["H0"],
                bottom=self.params["BOTTOM"],
                borders=borders,
            )
        )

        # Generate steering file content
        steering_content = SteeringFileGenerator.generate_steering_file(
            geometry_file=str(self.geometry_file_path),
            boundary_file=boundary_file,
            results_file=str(self.results_file_path),
            title=f"Case {self.case_id}",
            duration=30,  # TODO: Make this configurable via params
            time_step=0.02,  # TODO: Make this configurable via params
            initial_depth=self.params["H0"],
            prescribed_flowrates=(0.0, self.params["Q0"]),
            prescribed_elevations=prescribed_elevations,
            friction_coefficient=self.params["n"],
            viscosity=self.params.get("nut", 1e-6) + 1e-6,
        )

        # Ensure the directory exists and write the file
        self.steering_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.steering_file_path.write_text(steering_content)
