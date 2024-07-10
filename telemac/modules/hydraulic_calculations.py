import numpy as np
from functools import lru_cache
from typing import Union
from constants import GRAVITY
class HydraulicCalculations:
    GRAVITY = GRAVITY
    @staticmethod
    @lru_cache(maxsize=128)
    def manning_equation(
        depth: Union[float, np.ndarray],
        flow_rate: Union[float, np.ndarray],
        bottom_width: Union[float, np.ndarray],
        slope: Union[float, np.ndarray],
        roughness_coefficient: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """
        Calculate the depth of flow using Manning's equation.

        Parameters:
        ----------
        depth : float or np.ndarray
            Depth of flow (meters)
        flow_rate : float or np.ndarray
            Flow rate (cubic meters per second)
        bottom_width : float or np.ndarray
            Bottom width of the channel (meters)
        slope : float or np.ndarray
            Slope of the channel bed
        roughness_coefficient : float or np.ndarray
            Manning's roughness coefficient

        Returns:
        -------
        float or np.ndarray
            Depth of flow (meters)

        Raises:
        ------
        ValueError
            If any input parameter is negative
        """
        if np.any(
            np.array([depth, flow_rate, bottom_width, slope, roughness_coefficient]) < 0
        ):
            raise ValueError("All input parameters must be non-negative")

        area = depth * bottom_width
        wetted_perimeter = bottom_width + 2 * depth * 0
        hydraulic_radius = area / wetted_perimeter
        return (
            flow_rate
            - area
            * hydraulic_radius ** (2 / 3)
            * slope ** (1 / 2)
            / roughness_coefficient
        )

    @staticmethod
    def froude_number(
        depth: float, flow_rate: float, top_width: float, epsilon: float = 1e-10
    ) -> float:
        """
        Calculate the Froude number.

        Parameters:
        ----------
        depth : float
            Depth of flow (meters)
        flow_rate : float
            Flow rate (cubic meters per second)
        top_width : float
            Top width of the channel (meters)
        epsilon : float, optional
            Small value to prevent division by zero (default: 1e-10)

        Returns:
        -------
        float
            Froude number minus 1
        """
        area = top_width * depth
        numerator = flow_rate**2 * top_width
        denominator = area**3 * GRAVITY + epsilon
        return (numerator / denominator) ** 0.5 - 1

    @np.vectorize
    def normal_depth(
        flow_rate: float,
        bottom_width: float,
        slope: float,
        roughness_coefficient: float,
    ) -> float:
        """
        Calculate the normal depth of flow using Manning's equation and numerical methods.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        bottom_width : float
            Bottom width of the channel (meters)
        slope : float
            Slope of the channel
        roughness_coefficient : float
            Manning's roughness coefficient

        Returns:
        -------
        float
            Normal depth of flow (meters)
        """
        solution = fsolve(
            manning_equation,
            x0=1,
            args=(flow_rate, bottom_width, slope, roughness_coefficient),
        )[0]
        return solution

    @np.vectorize
    def critical_depth(flow_rate: float, bottom_width: float) -> float:
        """
        Calculate the critical depth of flow using Froude's equation and numerical methods.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        bottom_width : float
            Bottom width of the channel (meters)

        Returns:
        -------
        float
            Critical depth of flow (meters)
        """
        solution = fsolve(froude_number, x0=0.001, args=(flow_rate, bottom_width))[0]
        return solution

    def critical_slope(
        flow_rate: float, bottom_width: float, roughness_coefficient: float
    ) -> float:
        """
        Calculate the critical slope of flow using numerical methods.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        bottom_width : float
            Bottom width of the channel (meters)
        roughness_coefficient : float
            Manning's roughness coefficient

        Returns:
        -------
        float
            Critical slope of flow
        """

        def depth_difference_equation(
            slope: float,
            flow_rate: float,
            bottom_width: float,
            roughness_coefficient: float,
        ) -> float:
            return normal_depth(
                flow_rate, bottom_width, slope, roughness_coefficient
            ) - critical_depth(flow_rate, bottom_width)

        solution = fsolve(
            depth_difference_equation,
            x0=1e-10,
            args=(flow_rate, bottom_width, roughness_coefficient),
        )
        return solution[0]

    def normal_depth_simple(
        flow_rate: float,
        bottom_width: float,
        slope: float,
        roughness_coefficient: float,
    ) -> float:
        """
        Calculate the normal depth of flow using a simplified form of Manning's equation.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        bottom_width : float
            Bottom width of the channel (meters)
        slope : float
            Slope of the channel
        roughness_coefficient : float
            Manning's roughness coefficient

        Returns:
        -------
        float
            Normal depth of flow (meters)
        """
        return (
            roughness_coefficient * flow_rate / bottom_width * slope ** (-1 / 2)
        ) ** (3 / 5)

    def critical_depth_simple(flow_rate: float, top_width: float) -> float:
        """
        Calculate the critical depth of flow using a simplified form of Froude's equation.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        top_width : float
            Top width of the channel (meters)

        Returns:
        -------
        float
            Critical depth of flow (meters)
        """
        return (flow_rate / top_width) ** (2 / 3) * GRAVITY ** (-1 / 3)

    def critical_slope_simple(
        flow_rate: float, bottom_width: float, roughness_coefficient: float
    ) -> float:
        """
        Calculate the critical slope of flow using a simplified form.

        Parameters:
        ----------
        flow_rate : float
            Flow rate (cubic meters per second)
        bottom_width : float
            Bottom width of the channel (meters)
        roughness_coefficient : float
            Manning's roughness coefficient

        Returns:
        -------
        float
            Critical slope of flow
        """
        return (
            (bottom_width / flow_rate) ** (2 / 9)
            * GRAVITY ** (10 / 9)
            * roughness_coefficient**2
        )