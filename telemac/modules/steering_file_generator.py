from typing import List

class SteeringFileGenerator:
    """
    A class used to generate steering files for hydraulic simulations.

    Methods
    -------
    generate_steering_file(geometry_file, boundary_file, results_file, title, duration, time_step, initial_depth, prescribed_flowrates=None, prescribed_elevations=None, friction_coefficient=0.0025)
        Generates a steering file based on the provided parameters.
    """

    @staticmethod
    def generate_steering_file(
        geometry_file: str,
        boundary_file: str,
        results_file: str,
        title: str,
        duration: float,
        time_step: float,
        initial_depth: float,
        prescribed_flowrates: List[float] = None,
        prescribed_elevations: List[float] = None,
        friction_coefficient: float = 0.0025,
    ) -> str:
        """
        Generates a steering file based on the provided parameters.

        Parameters
        ----------
        geometry_file : str
            Path to the geometry file.
        boundary_file : str
            Path to the boundary file.
        results_file : str
            Path to the results file.
        title : str
            Title of the simulation.
        duration : float
            Duration of the simulation in seconds.
        time_step : float
            Time step of the simulation in seconds.
        initial_depth : float
            Initial depth of the simulation.
        prescribed_flowrates : List[float], optional
            List of prescribed flow rates (default is [0.0, 0.0]).
        prescribed_elevations : List[float], optional
            List of prescribed elevations (default is [0.0, 0.0]).
        friction_coefficient : float, optional
            Friction coefficient (default is 0.0025).

        Returns
        -------
        str
            The generated steering file content as a string.

        Raises
        ------
        ValueError
            If prescribed_flowrates or prescribed_elevations have less than two elements.
            If duration, time_step, initial_depth, or friction_coefficient are negative.
        FileNotFoundError
            If the steering template file is not found.

        Notes
        -----
        The method reads a template file named "steering_template.txt" and replaces placeholders with the provided parameters.
        """
        if prescribed_flowrates is None:
            prescribed_flowrates = [0.0, 0.0]
        if prescribed_elevations is None:
            prescribed_elevations = [0.0, 0.0]

        if len(prescribed_flowrates) < 2 or len(prescribed_elevations) < 2:
            raise ValueError(
                "prescribed_flowrates and prescribed_elevations must have at least two elements"
            )

        if duration < 0 or time_step < 0 or initial_depth < 0 or friction_coefficient < 0:
            raise ValueError(
                "duration, time_step, initial_depth, and friction_coefficient must be non-negative"
            )

        try:
            with open("steering_template.txt", "r") as f:
                template_text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("steering_template.txt not found")

        steering_text = template_text.format(
            geometry_file=geometry_file,
            boundary_file=boundary_file,
            results_file=results_file,
            title=title,
            duration=duration,
            time_step=time_step,
            initial_depth=initial_depth,
            prescribed_flowrates=prescribed_flowrates,
            prescribed_elevations=prescribed_elevations,
            friction_coefficient=friction_coefficient,
        )

        return steering_text
