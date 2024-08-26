from typing import Tuple, Literal, List

class BoundaryConditions:
    """
    A class to handle boundary conditions for a simulation.

    Methods
    -------
    get_boundary_and_elevations(direction, h0, bottom, borders_flat, borders_noise)
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

        Parameters
        ----------
        direction : Literal["Left to right", "Right to left"]
            The direction of the flow.
        h0 : float
            The initial water height.
        bottom : str
            The type of bottom surface, either "FLAT" or another string.
        borders_flat : List[float]
            A list containing the left and right elevations for a flat bottom.
        borders_noise : List[float]
            A list containing the left and right elevations for a noisy bottom.

        Returns
        -------
        Tuple[str, Tuple[float, float]]
            A tuple containing the boundary file path and the elevations.

        Examples
        --------
        >>> BoundaryConditions.get_boundary_and_elevations(
        ...     "Left to right", 1.0, "FLAT", [0.0, 0.0], [0.1, 0.1])
        ('boundary/3x3_tor.cli', (0.0, 1.0))

        >>> BoundaryConditions.get_boundary_and_elevations(
        ...     "Right to left", 1.0, "NOISY", [0.0, 0.0], [0.1, 0.1])
        ('boundary/3x3_riv.cli', (1.1, 0.0))
        """
        z_left, z_right = borders

        if direction == "Left to right":
            return "boundary/3x3_tor.cli", (0.0, z_left + h0)
        else:
            return "boundary/3x3_riv.cli", (z_right + h0, 0.0)
