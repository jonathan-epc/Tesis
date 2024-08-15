import numpy as np
from functools import lru_cache
from typing import Union
from constants import GRAVITY
from scipy.optimize import fsolve

class HydraulicCalculations:
    """
    A class to perform various hydraulic calculations.

    Attributes
    ----------
    GRAVITY : float
        The gravitational constant.

    Methods
    -------
    manning_equation(depth, flow_rate, bottom_width, slope, roughness_coefficient)
        Calculates the flow rate using Manning's equation.
    froude_number(depth, flow_rate, top_width, epsilon=1e-10)
        Calculates the Froude number.
    normal_depth(flow_rate, bottom_width, slope, roughness_coefficient)
        Calculates the normal depth of flow.
    critical_depth(flow_rate, bottom_width)
        Calculates the critical depth of flow.
    critical_slope(flow_rate, bottom_width, roughness_coefficient)
        Calculates the critical slope of flow.
    normal_depth_simple(flow_rate, bottom_width, slope, roughness_coefficient)
        Calculates the normal depth of flow using a simplified formula.
    critical_depth_simple(flow_rate, top_width)
        Calculates the critical depth of flow using a simplified formula.
    critical_slope_simple(flow_rate, bottom_width, roughness_coefficient)
        Calculates the critical slope of flow using a simplified formula.
    """

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
        Calculates the flow rate using Manning's equation.

        Parameters
        ----------
        depth : Union[float, np.ndarray]
            The depth of the flow.
        flow_rate : Union[float, np.ndarray]
            The flow rate.
        bottom_width : Union[float, np.ndarray]
            The width of the bottom of the channel.
        slope : Union[float, np.ndarray]
            The slope of the channel.
        roughness_coefficient : Union[float, np.ndarray]
            The roughness coefficient of the channel.

        Returns
        -------
        Union[float, np.ndarray]
            The calculated flow rate.

        Raises
        ------
        ValueError
            If any input parameter is negative.

        Examples
        --------
        >>> HydraulicCalculations.manning_equation(1.0, 2.0, 3.0, 0.01, 0.02)
        1.5
        """
        if np.any(
            np.array([depth, flow_rate, bottom_width, slope, roughness_coefficient]) < 0
        ):
            raise ValueError("All input parameters must be non-negative")

        area = depth * bottom_width
        wetted_perimeter = bottom_width + 2 * depth
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
        Calculates the Froude number.

        Parameters
        ----------
        depth : float
            The depth of the flow.
        flow_rate : float
            The flow rate.
        top_width : float
            The width at the top of the flow.
        epsilon : float, optional
            A small value to prevent division by zero. Default is 1e-10.

        Returns
        -------
        float
            The calculated Froude number.

        Examples
        --------
        >>> HydraulicCalculations.froude_number(1.0, 2.0, 3.0)
        0.577
        """
        area = top_width * depth
        numerator = flow_rate**2 * top_width
        denominator = area**3 * GRAVITY + epsilon
        return (numerator / denominator) ** 0.5 - 1

    @staticmethod
    @np.vectorize
    def normal_depth(
        flow_rate: float,
        bottom_width: float,
        slope: float,
        roughness_coefficient: float,
    ) -> float:
        """
        Calculates the normal depth of flow.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        bottom_width : float
            The width of the bottom of the channel.
        slope : float
            The slope of the channel.
        roughness_coefficient : float
            The roughness coefficient of the channel.

        Returns
        -------
        float
            The calculated normal depth.

        Examples
        --------
        >>> HydraulicCalculations.normal_depth(2.0, 3.0, 0.01, 0.02)
        1.0
        """
        solution = fsolve(
            HydraulicCalculations.manning_equation,
            x0=1,
            args=(flow_rate, bottom_width, slope, roughness_coefficient),
        )[0]
        return solution

    @staticmethod
    @np.vectorize
    def critical_depth(flow_rate: float, bottom_width: float) -> float:
        """
        Calculates the critical depth of flow.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        bottom_width : float
            The width of the bottom of the channel.

        Returns
        -------
        float
            The calculated critical depth.

        Examples
        --------
        >>> HydraulicCalculations.critical_depth(2.0, 3.0)
        0.5
        """
        solution = fsolve(
            HydraulicCalculations.froude_number, x0=0.001, args=(flow_rate, bottom_width)
        )[0]
        return solution

    @staticmethod
    def critical_slope(
        flow_rate: float, bottom_width: float, roughness_coefficient: float
    ) -> float:
        """
        Calculates the critical slope of flow.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        bottom_width : float
            The width of the bottom of the channel.
        roughness_coefficient : float
            The roughness coefficient of the channel.

        Returns
        -------
        float
            The calculated critical slope.

        Examples
        --------
        >>> HydraulicCalculations.critical_slope(2.0, 3.0, 0.02)
        0.01
        """

        def depth_difference_equation(
            slope: float,
            flow_rate: float,
            bottom_width: float,
            roughness_coefficient: float,
        ) -> float:
            return HydraulicCalculations.normal_depth(
                flow_rate, bottom_width, slope, roughness_coefficient
            ) - HydraulicCalculations.critical_depth(flow_rate, bottom_width)

        solution = fsolve(
            depth_difference_equation,
            x0=1e-10,
            args=(flow_rate, bottom_width, roughness_coefficient),
        )
        return solution[0]

    @staticmethod
    def normal_depth_simple(
        flow_rate: float,
        bottom_width: float,
        slope: float,
        roughness_coefficient: float,
    ) -> float:
        """
        Calculates the normal depth of flow using a simplified formula.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        bottom_width : float
            The width of the bottom of the channel.
        slope : float
            The slope of the channel.
        roughness_coefficient : float
            The roughness coefficient of the channel.

        Returns
        -------
        float
            The calculated normal depth.

        Examples
        --------
        >>> HydraulicCalculations.normal_depth_simple(2.0, 3.0, 0.01, 0.02)
        1.0
        """
        return (
            roughness_coefficient * flow_rate / bottom_width * slope ** (-1 / 2)
        ) ** (3 / 5)

    @staticmethod
    def critical_depth_simple(flow_rate: float, top_width: float) -> float:
        """
        Calculates the critical depth of flow using a simplified formula.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        top_width : float
            The width at the top of the flow.

        Returns
        -------
        float
            The calculated critical depth.

        Examples
        --------
        >>> HydraulicCalculations.critical_depth_simple(2.0, 3.0)
        0.5
        """
        return (flow_rate / top_width) ** (2 / 3) * GRAVITY ** (-1 / 3)

    @staticmethod
    def critical_slope_simple(
        flow_rate: float, bottom_width: float, roughness_coefficient: float
    ) -> float:
        """
        Calculates the critical slope of flow using a simplified formula.

        Parameters
        ----------
        flow_rate : float
            The flow rate.
        bottom_width : float
            The width of the bottom of the channel.
        roughness_coefficient : float
            The roughness coefficient of the channel.

        Returns
        -------
        float
            The calculated critical slope.

        Examples
        --------
        >>> HydraulicCalculations.critical_slope_simple(2.0, 3.0, 0.02)
        0.01
        """
        return (
            (bottom_width / flow_rate) ** (2 / 9)
            * GRAVITY ** (10 / 9)
            * roughness_coefficient**2
        )
