from typing import List
class SteeringFileGenerator:
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
        Generates a TELEMAC-2D steering file with user-provided parameters.
    
        Parameters:
        ----------
        geometry_file : str
            Path to the geometry file
        boundary_file : str
            Path to the boundary conditions file
        results_file : str
            Path to the results file
        title : str
            Title for the simulation
        duration : float
            Simulation duration in seconds
        time_step : float
            Time step for the simulation in seconds
        initial_depth : float
            Initial water depth for the simulation
        prescribed_flowrates : List[float], optional
            List of prescribed flow rates at boundaries (default: None)
        prescribed_elevations : List[float], optional
            List of prescribed water elevations at boundaries (default: None)
        friction_coefficient : float, optional
            Bottom friction coefficient (default: 0.0025)
    
        Returns:
        -------
        str
            The complete steering file content as a string
    
        Raises:
        ------
        ValueError
            If input parameters are invalid
        FileNotFoundError
            If template file is not found
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