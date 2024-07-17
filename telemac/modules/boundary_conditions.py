from typing import Tuple, Literal, List
class BoundaryConditions:
    @staticmethod
    def get_boundary_and_elevations(
        direction: Literal["Left to right", "Right to left"],
        h0: float,
        bottom: str,
        borders_flat: List[float],
        borders_noise: List[float],
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Determine the boundary file and prescribed elevations based on flow direction.

        Parameters:
        ----------
        direction : Literal["Left to right", "Right to left"]
            The direction of flow
        z_left : float
            Left elevation
        z_right : float
            Right elevation
        h0 : float
            Initial water depth

        Returns:
        -------
        Tuple[str, Tuple[float, float]]
            A tuple containing the boundary file path and prescribed elevations
        """
        if bottom == "FLAT":
            z_left, z_right = borders_flat
        else:
            z_left, z_right = borders_noise

        if direction == "Left to right":
            return "boundary/3x3_tor.cli", (0.0, z_left + h0)
        else:
            return "boundary/3x3_riv.cli", (z_right + h0, 0.0)