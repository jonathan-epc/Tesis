# src/telemac/modules/parameter_manager.py

"""ParameterManager Module

This module provides the `ParameterManager` class for managing parameters used
in hydraulic calculations. The class handles loading existing parameters,
generating new parameters, and combining both.

Classes:
    ParameterManager

Dependencies:
    pandas, numpy, loguru, nconfig, modules.sample_generator,
    modules.hydraulic_calculations, modules.telemac_case
"""

import numpy as np
import pandas as pd
from loguru import logger

from nconfig import Config

from .hydraulic_calculations import HydraulicCalculations
from .sample_generator import SampleGenerator
from .telemac_case import TelemacCase


class ParameterManager:
    """Manages parameters for hydraulic calculations.

    Handles loading existing parameters, generating new parameters, and combining
    both based on the specified mode of operation.

    Attributes:
        config: Configuration object containing simulation parameters.
        sample_size: The size of the sample to generate.
        mode: The mode of operation ('new', 'read', or 'add').
        parameters_df: DataFrame containing the parameters.
    """

    def __init__(
        self, config: Config, sample_size: int = 113**2, mode: str = "add"
    ) -> None:
        """Initialize the ParameterManager.

        Args:
            config: Configuration object containing constants and parameters.
            sample_size: The size of the sample to generate.
            mode: The mode of operation. Must be 'new', 'read', or 'add'.

        Raises:
            ValueError: If mode is not one of 'new', 'read', or 'add'.
        """
        self.config = config
        self.sample_size = sample_size
        self.mode = mode
        self.parameters_df = self._load_or_generate_parameters()

    def _load_old_parameters(self) -> pd.DataFrame | None:
        """Load existing parameters from a CSV file.

        Returns:
            DataFrame containing the parameters if the file exists, otherwise None.

        Note:
            Logs appropriate messages for successful loading, file not found,
            or other errors.
        """
        try:
            df = pd.read_csv("parameters.csv")
            logger.info(
                f"Loaded existing parameters from parameters.csv. Shape: {df.shape}"
            )
            return df
        except FileNotFoundError:
            logger.info("No existing parameters.csv found.")
            return None
        except Exception as e:
            logger.error(f"Error loading parameters.csv: {str(e)}")
            return None

    def _generate_new_parameters(self) -> pd.DataFrame:
        """Generate new parameters based on configuration settings.

        Returns:
            DataFrame containing the newly generated parameters with all
            calculated properties.

        Raises:
            Exception: If parameter generation fails.

        Note:
            Generation method depends on config.simulation_params.adimensional_generation.
            If True, generates from adimensional numbers then inverts to physical parameters.
            If False, generates base parameters then calculates derived properties.
        """
        logger.info(f"Generating new parameters. Sample size: {self.sample_size}")
        try:
            if self.config.simulation_params.adimensional_generation:
                # Generate a sampling of the nondimensional numbers
                new_params = self._generate_adimensional_parameters()
                # Invert them to obtain the physical parameters, including nut
                new_params = self._calculate_from_adimensionals(new_params)
                # Calculate remaining hydraulic properties
                new_params = self._calculate_hydraulic_properties(new_params)
            else:
                # Generate base parameters
                new_params = self._generate_base_parameters()
                # Calculate hydraulic properties (yn, yc) first
                new_params = self._calculate_hydraulic_properties(new_params)
                # **FIX**: Estimate nut based on calculated hydraulic properties
                new_params = self._estimate_turbulent_viscosity(new_params)
                # Balance samples after all physical properties are known
                new_params = self._balance_samples(new_params)
                # Calculate adimensional numbers as the final step
                new_params = self._calculate_adimensionals(new_params)

            # Here, add a simple direction flag based on a comparison of two hydraulic depths.
            new_params["direction"] = np.where(
                new_params["yn"] > new_params["yc"], "Right to left", "Left to right"
            )
            new_params = self._add_bottom_values(new_params)
            return new_params
        except Exception as e:
            logger.error(f"Error generating new parameters: {str(e)}")
            raise

    def _generate_base_parameters(self) -> pd.DataFrame:
        """Generate base physical parameters using configured ranges.

        Returns:
            DataFrame containing base parameters (SLOPE, n, Q0, H0) sampled
            from their respective ranges.
        """
        param_ranges: dict[str, tuple[float, float]] = {
            "SLOPE": (
                self.config.simulation_params.parameter_ranges["slope_min"],
                self.config.simulation_params.parameter_ranges["slope_max"],
            ),
            "n": (
                self.config.simulation_params.parameter_ranges["n_min"],
                self.config.simulation_params.parameter_ranges["n_max"],
            ),
            "Q0": (
                self.config.simulation_params.parameter_ranges["q0_min"],
                self.config.simulation_params.parameter_ranges["q0_max"],
            ),
            "H0": (
                self.config.simulation_params.parameter_ranges["h0_min"],
                self.config.simulation_params.parameter_ranges["h0_max"],
            ),
        }
        # **FIX**: Add channel dimensions directly to the dataframe
        df = SampleGenerator.sample_combinations(self.sample_size, param_ranges)
        df["L"] = self.config.channel.length
        df["W"] = self.config.channel.width
        return df

    def _generate_adimensional_parameters(self) -> pd.DataFrame:
        """Generate adimensional parameters using configured ranges.

        Returns:
            DataFrame containing adimensional parameters (Ar, Hr, Fr, Re, M)
            sampled from their respective ranges.

        Note:
            The ranges are taken from configuration and should be adjusted
            based on physical constraints of the problem.
        """
        param_ranges: dict[str, tuple[float, float]] = {
            "Ar": (
                self.config.simulation_params.parameter_ranges["Ar_min"],
                self.config.simulation_params.parameter_ranges["Ar_max"],
            ),
            "Hr": (
                self.config.simulation_params.parameter_ranges["Hr_min"],
                self.config.simulation_params.parameter_ranges["Hr_max"],
            ),
            "Fr": (
                self.config.simulation_params.parameter_ranges["Fr_min"],
                self.config.simulation_params.parameter_ranges["Fr_max"],
            ),
            "Re": (
                self.config.simulation_params.parameter_ranges["Re_min"],
                self.config.simulation_params.parameter_ranges["Re_max"],
            ),
            "M": (
                self.config.simulation_params.parameter_ranges["M_min"],
                self.config.simulation_params.parameter_ranges["M_max"],
            ),
        }
        return SampleGenerator.sample_combinations(self.sample_size, param_ranges)

    def _calculate_adimensionals(self, params: pd.DataFrame) -> pd.DataFrame:
        """Calculate adimensional numbers from physical parameters.

        Args:
            params: DataFrame containing physical parameters.

        Returns:
            DataFrame with added adimensional numbers (Ar, Vr, Hr, Fr, Re, M).

        Note:
            Uses gravity constant from configuration. Calculates characteristic
            scales from geometric and flow parameters.
        """
        g = self.config.simulation_params.gravity
        xc = params["L"]
        yc = params["W"]
        bc = params["L"] * params["SLOPE"]
        hc = params["H0"]
        uc = params["Q0"] / (params["H0"] * params["W"])
        vc = params["Q0"] / (params["H0"] * params["L"])

        params["Ar"] = xc / yc
        params["Vr"] = uc / vc
        params["Hr"] = bc / hc
        params["Fr"] = uc / (np.sqrt(g * hc))
        # **FIX**: Ensure Re is calculated correctly for the dimensional case
        params["Re"] = (uc * xc) / (params["nut"])
        params["M"] = g * (params["n"] ** 2) * xc / (hc ** (4 / 3))
        return params

    def _calculate_from_adimensionals(self, params: pd.DataFrame) -> pd.DataFrame:
        """Calculate physical parameters from adimensional numbers.

        Args:
            params: DataFrame containing adimensional parameters.

        Returns:
            DataFrame with calculated physical parameters.

        Note:
            Uses configured characteristic scales to derive dimensional values.
            **FIX**: This now correctly calculates `nut` from the sampled `Re`.
        """
        g = self.config.simulation_params.gravity
        # Use characteristic scales from the config
        xc = self.config.channel.length
        yc = self.config.channel.width
        hc = self.config.channel.depth  # A reference scale

        # Calculate dimensional parameters from non-dimensional ones
        params["W"] = yc
        params["L"] = params["Ar"] * params["W"]
        params["H0"] = hc

        uc = params["Fr"] * np.sqrt(g * params["H0"])
        params["Q0"] = uc * params["H0"] * params["W"]
        params["SLOPE"] = (params["Hr"] * params["H0"]) / params["L"]
        params["n"] = np.sqrt(params["M"] * (hc ** (4 / 3)) / (g * xc))

        # **FIX**: Calculate nut from the sampled Reynolds number
        params["nut"] = (uc * xc) / params["Re"]

        return params

    def _calculate_hydraulic_properties(self, params: pd.DataFrame) -> pd.DataFrame:
        """Calculate hydraulic properties for the given parameters.

        Args:
            params: DataFrame containing base flow parameters.

        Returns:
            DataFrame with calculated hydraulic properties including normal depth (yn),
            critical depth (yc), and flow regime classification (subcritical).
            **FIX**: This method no longer calculates `nut`.
        """
        params["yn"] = HydraulicCalculations.normal_depth_simple(
            params["Q0"],
            params["W"],
            params["SLOPE"],
            params["n"],
        )
        params["yc"] = HydraulicCalculations.critical_depth_simple(
            params["Q0"], params["W"]
        )
        params["subcritical"] = params["yn"] > params["yc"]
        return params

    def _estimate_turbulent_viscosity(self, params: pd.DataFrame) -> pd.DataFrame:
        """
        **FIX (New Method)**: Estimate turbulent viscosity for the dimensional case.
        Uses a physics-based formula based on shear velocity and normal depth.
        """
        g = self.config.simulation_params.gravity
        kappa = 0.41  # Von Kármán constant

        # Approximate hydraulic radius R_h with normal depth yn (wide channel assumption)
        shear_velocity_sq = g * params["yn"] * params["SLOPE"]
        shear_velocity = np.sqrt(shear_velocity_sq.clip(min=0))  # Avoid negative sqrt

        # Estimate eddy viscosity
        params["nut"] = (kappa / 6) * shear_velocity * params["yn"]
        logger.info("Estimated turbulent viscosity for dimensional samples.")
        return params

    def _balance_samples(self, params: pd.DataFrame) -> pd.DataFrame:
        """Balance subcritical and supercritical flow samples.

        Args:
            params: DataFrame containing parameters with flow regime classification.

        Returns:
            DataFrame with equal numbers of subcritical and supercritical samples,
            randomly shuffled.

        Note:
            Uses random_state=0 for reproducible sampling. The final sample size
            will be 2 * min(subcritical_count, supercritical_count).
        """
        subcritical = params[params["subcritical"]]
        supercritical = params[~params["subcritical"]]

        if len(subcritical) == 0 or len(supercritical) == 0:
            logger.warning(
                "One flow regime has zero samples. Cannot balance. Returning original dataframe."
            )
            return params

        required_samples = min(len(subcritical), len(supercritical))

        logger.info(
            f"Balancing samples. Selected {required_samples} from each category."
        )

        selected_subcritical = subcritical.sample(n=required_samples, random_state=0)
        selected_supercritical = supercritical.sample(
            n=required_samples, random_state=0
        )

        return (
            pd.concat([selected_subcritical, selected_supercritical])
            .sample(frac=1, random_state=0)
            .reset_index(drop=True)
        )

    def _add_bottom_values(self, params: pd.DataFrame) -> pd.DataFrame:
        """Add bottom type values to the parameters.

        Args:
            params: DataFrame containing hydraulic parameters.

        Returns:
            DataFrame with bottom type assignments for each parameter set.

        Note:
            Creates copies of the parameter set for each bottom type specified
            in the configuration, effectively multiplying the dataset size.
        """
        bottom_values = self.config.simulation_params.bottom_types
        logger.info(f"Adding bottom values: {bottom_values}")
        return pd.concat(
            [params.assign(BOTTOM=bottom) for bottom in bottom_values]
        ).reset_index(drop=True)

    def _load_or_generate_parameters(self) -> pd.DataFrame:
        """Load or generate parameters based on the specified mode.

        Returns:
            DataFrame containing the final parameter set.

        Raises:
            FileNotFoundError: If mode is 'read' but no parameters.csv exists.
            ValueError: If mode is not one of 'new', 'read', or 'add'.

        Note:
            - 'new' mode: Generate completely new parameters and save to CSV
            - 'read' mode: Load existing parameters from CSV only
            - 'add' mode: Combine existing parameters with newly generated ones
        """
        if self.mode == "new":
            logger.info("Generating completely new parameter file.")
            new_params = self._generate_new_parameters()
            if self.config.simulation_params.adimensional_generation:
                new_params.index = [f"a{id}" for id in range(len(new_params))]
            new_params.index.name = "id"
            new_params.to_csv("parameters.csv", index=True)
            logger.info(
                f"Wrote new parameters to parameters.csv. Shape: {new_params.shape}"
            )
            return new_params

        elif self.mode == "read":
            old_params = self._load_old_parameters()
            if old_params is None:
                raise FileNotFoundError(
                    "No existing parameters.csv found and mode is set to 'read'."
                )
            return old_params

        elif self.mode == "add":
            old_params = self._load_old_parameters()
            new_params = self._generate_new_parameters()

            if old_params is not None:
                if "id" in old_params.columns:
                    old_params.set_index("id", inplace=True)
                if "id" in new_params.columns:
                    new_params.set_index("id", inplace=True)
                if self.config.simulation_params.adimensional_generation:
                    new_params.index = [f"a{id}" for id in range(len(new_params))]

                combined_params = pd.concat([old_params, new_params], ignore_index=True)
                combined_params = combined_params.drop_duplicates(keep="first")
                combined_params = combined_params.reset_index(drop=True)
                combined_params.index.name = "id"
                combined_params.to_csv("parameters.csv", index=True)
                logger.info(
                    f"Combined old and new parameters. New shape: {combined_params.shape}"
                )
                return combined_params
            else:
                if self.config.simulation_params.adimensional_generation:
                    new_params.index = [f"a{id}" for id in range(len(new_params))]
                else:
                    new_params.index.name = "id"
                new_params.to_csv("parameters.csv", index=True)
                logger.info(
                    f"Wrote new parameters to parameters.csv. Shape: {new_params.shape}"
                )
                return new_params
        else:
            raise ValueError("Invalid mode. Choose 'new', 'read', or 'add'.")

    def create_cases(self) -> list[TelemacCase]:
        """Create a list of TelemacCase objects from the parameters DataFrame.

        Returns:
            List of TelemacCase objects, one for each parameter set.

        Note:
            Adds mesh configuration constants to each parameter set before
            creating TelemacCase objects. The case_id is taken from the
            DataFrame index.
        """
        cases: list[TelemacCase] = []

        # Add constants needed by geometry generator to each row
        self.parameters_df["num_points_x"] = self.config.mesh.num_points_x
        self.parameters_df["num_points_y"] = self.config.mesh.num_points_y
        self.parameters_df["adimensional"] = (
            self.config.simulation_params.adimensional_generation
        )

        for case_id, params_series in self.parameters_df.iterrows():
            case = TelemacCase(case_id=case_id, params=params_series.to_dict())
            cases.append(case)

        logger.info(f"Created {len(cases)} TelemacCase objects.")
        return cases
