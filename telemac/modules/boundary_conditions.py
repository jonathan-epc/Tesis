from typing import Tuple, Literal, List
import math

class BoundaryConditions:
    """
    A class to handle boundary conditions for a simulation.
    Methods
    -------
    get_boundary_and_elevations(direction, h0, bottom, borders)
        Determines the boundary file and elevations based on the given parameters.
    """
    @staticmethod
    def get_boundary_and_elevations(
        direction: Literal["Left to right", "Right to left"],
        h0: float,
        bottom: str,
        borders: List[float],
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Determines the boundary file and elevations based on the given parameters.
        Raises a ValueError if any NaN values are encountered.
        
        Parameters
        ----------
        direction : Literal["Left to right", "Right to left"]
            The direction of the flow.
        h0 : float
            The initial water height.
        bottom : str
            The type of bottom surface, either "FLAT" or another string.
        borders : List[float]
            A list containing the left and right elevations.
        
        Returns
        -------
        Tuple[str, Tuple[float, float]]
            A tuple containing the boundary file path and the elevations.
        
        Raises
        ------
        ValueError
            If any NaN values are encountered in the calculations.
        
        Examples
        --------
        >>> BoundaryConditions.get_boundary_and_elevations(
        ...     "Left to right", 1.0, "FLAT", [0.0, 0.0])
        ('boundary/3x3_tor.cli', (0.0, 1.0))
        >>> BoundaryConditions.get_boundary_and_elevations(
        ...     "Right to left", 1.0, "NOISY", [0.1, 0.1])
        ('boundary/3x3_riv.cli', (1.1, 0.0))
        """
        z_left, z_right = borders
        
        if direction == "Left to right":
            boundary_file = "boundary/3x3_tor.cli"
            elevations = (0.0, z_left + h0)
        else:
            boundary_file = "boundary/3x3_riv.cli"
            elevations = (z_right + h0, 0.0)
        
        # Check for NaN values
        if any(math.isnan(x) for x in elevations):
            raise ValueError(f"NaN value encountered in elevation calculations. z={z_left, z_right} h0={h0}")
        
        return boundary_file, elevations