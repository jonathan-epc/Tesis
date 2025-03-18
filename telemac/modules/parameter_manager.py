"""
ParameterManager Module
========================

This module provides the `ParameterManager` class for managing parameters used in hydraulic calculations.
The class handles loading existing parameters, generating new parameters, and combining both.

Classes:
    ParameterManager

Dependencies:
    pandas, numpy, loguru, modules.sample_generator, modules.hydraulic_calculations
"""

import pandas as pd
import numpy as np
from loguru import logger
from modules.sample_generator import SampleGenerator
from modules.hydraulic_calculations import HydraulicCalculations


class ParameterManager:
    """
    ParameterManager Class
    ----------------------

    Manages parameters for hydraulic calculations, including loading existing parameters,
    generating new parameters, and combining both.

    Parameters
    ----------
    constants : dict
        A dictionary containing constants required for parameter generation.
    sample_size : int, optional
        The size of the sample to generate. Default is 113**2.
    mode : str, optional
        The mode of operation. Can be 'new', 'read', or 'add'. Default is 'add'.

    Attributes
    ----------
    constants : dict
        Constants required for parameter generation.
    sample_size : int
        The size of the sample to generate.
    mode : str
        The mode of operation.
    parameters_df : pandas.DataFrame
        DataFrame containing the parameters.

    Methods
    -------
    get_parameters()
        Returns the parameters DataFrame.
    validate_constants()
        Validates the constants dictionary.
    """

    def __init__(self, constants, sample_size=113**2, mode="add"):
        self.constants = constants
        self.sample_size = sample_size
        self.mode = mode
        self.parameters_df = self._load_or_generate_parameters()
    def _load_old_parameters(self):
        """
        Load existing parameters from a CSV file.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing the parameters if the file exists, otherwise None.
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

    def _generate_new_parameters(self):
        """
        Generate new parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the newly generated parameters.
        """
        logger.info(f"Generating new parameters. Sample size: {self.sample_size}")

        try:
            if self.constants["adimensional"]:
                # Generate a sampling of the nondimensional numbers
                new_params = self._generate_adimensional_parameters()
                # Invert them to obtain the physical parameters.
                new_params = self._calculate_from_adimensionals(new_params)
                new_params = self._calculate_hydraulic_properties(new_params)
            else:
                # Generate base parameters and then calculate hydraulic and adimensional properties.
                new_params = self._generate_base_parameters()
                new_params = self._calculate_hydraulic_properties(new_params)
                new_params = self._balance_samples(new_params)
                new_params = self._calculate_adimensionals(new_params)
            # Here, add a simple direction flag based on a comparison of two hydraulic depths.
            new_params["direction"] = np.where(new_params["yn"] > new_params["yc"], "Right to left", "Left to right")
            new_params = self._add_bottom_values(new_params)
            return new_params
        except Exception as e:
            logger.error(f"Error generating new parameters: {str(e)}")
            raise
    
    def _generate_base_parameters(self):
        """
        Generate base parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the base parameters.
        """
        param_ranges = {
            "SLOPE": (self.constants["parameters"]["slope_min"], self.constants["parameters"]["slope_max"]),
            "n": (self.constants["parameters"]["n_min"], self.constants["parameters"]["n_max"]),
            "Q0": (self.constants["parameters"]["q0_min"], self.constants["parameters"]["q0_max"]),
            "H0": (self.constants["parameters"]["h0_min"], self.constants["parameters"]["h0_max"]),
        }
        return SampleGenerator.sample_combinations(self.sample_size, param_ranges)
    def _generate_adimensional_parameters(self):
        # Sample 5 nondimensional numbers: Ar, Hr, Fr, Re, M.
        # Note: The ranges here are illustrative; adjust as needed.
        param_ranges = {
            "Ar": (self.constants["parameters"]["Ar_min"], self.constants["parameters"]["Ar_max"]),
            "Hr": (self.constants["parameters"]["Hr_min"], self.constants["parameters"]["Hr_max"]),
            "Fr": (self.constants["parameters"]["Fr_min"], self.constants["parameters"]["Fr_max"]),
            "Re": (self.constants["parameters"]["Re_min"], self.constants["parameters"]["Re_max"]),
            "M": (self.constants["parameters"]["M_min"], self.constants["parameters"]["M_max"]),
        }
        return SampleGenerator.sample_combinations(self.sample_size, param_ranges)

    def _calculate_adimensionals(self, params):
        # Calculate the nondimensional numbers from the physical parameters.
        g = self.constants["gravity"]
        xc = params["L"]
        yc = params["W"]
        bc = params["L"]*params["SLOPE"]
        hc = params["H0"]
        uc = params["Q0"]/(params["H0"]*params["W"])
        vc = params["Q0"]/(params["H0"]*params["L"])
        
        params["Ar"] = xc / yc
        params["Vr"] = uc / vc
        params["Hr"] = bc / hc
        params["Fr"] = uc / (np.sqrt(g * hc))
        params["Re"] = uc * xc / (params["nut"])
        params["M"] = g * (params["n"]**2) * xc / (hc**(4/3))
        return params
        
    def _calculate_from_adimensionals(self, params):
        g = self.constants["gravity"]
        params["W"] = 1.0
        params["H0"] = 0.1
        params["Vr"] = params["Ar"]
        params["L"] = params["Ar"]
        params["Q0"] = params["Fr"] * params["H0"] * params["W"] * np.sqrt(g * params["H0"])
        params["SLOPE"] = (params["Hr"] * params["H0"]) / (params["Ar"] * params["W"])
        params["n"] = (params["H0"]**(2/3) * np.sqrt(params["M"])) / (np.sqrt(params["Ar"]) * np.sqrt(g) * np.sqrt(params["W"]))
        return params
        
    def _calculate_hydraulic_properties(self, params):
        """
        Calculate hydraulic properties for the given parameters.

        Parameters
        ----------
        params : pandas.DataFrame
            DataFrame containing the parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame with calculated hydraulic properties.
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
        # params["nut"] = 0.41 * params["yn"] / 2 * np.sqrt(9.81 * params["SLOPE"] * params["yn"] * (params["W"])/(2*params["yn"] + params["W"]))
        uc = params["Q0"] / (params["H0"] * params["W"])
        params["nut"] = uc * params["L"] / params["Re"] #Correct nut calculation
        return params

    def _balance_samples(self, params):
        """
        Balance subcritical and supercritical samples.

        Parameters
        ----------
        params : pandas.DataFrame
            DataFrame containing the parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame with balanced samples.
        """
        subcritical = params[params["subcritical"]]
        supercritical = params[~params["subcritical"]]

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

    def _add_bottom_values(self, params):
        """
        Add bottom values to the parameters.

        Parameters
        ----------
        params : pandas.DataFrame
            DataFrame containing the parameters.

        Returns
        -------
        pandas.DataFrame
            DataFrame with added bottom values.
        """
        bottom_values = self.constants["parameters"]["bottom_values"]
        logger.info(f"Adding bottom values: {bottom_values}")

        return pd.concat(
            [params.assign(BOTTOM=bottom) for bottom in bottom_values]
        ).reset_index(drop=True)

    def _load_or_generate_parameters(self):
        """
        Load or generate parameters based on the mode.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parameters.
        """
        if self.mode == "new":
            logger.info("Generating completely new parameter file.")
            new_params = self._generate_new_parameters()
            
            if self.constants["adimensional"]:
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
                
                if self.constants["adimensional"]:
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
                if self.constants["adimensional"]:
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

            

    def get_parameters(self):
        """
        Get the parameters DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the parameters.
        """
        return self.parameters_df