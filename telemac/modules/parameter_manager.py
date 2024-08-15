import pandas as pd
import numpy as np
from loguru import logger
from modules.sample_generator import SampleGenerator
from modules.hydraulic_calculations import HydraulicCalculations

class ParameterManager:
    """
    A class used to manage parameters for hydraulic calculations.

    Attributes
    ----------
    constants : dict
        A dictionary containing constants and parameters for the calculations.
    sample_size : int, optional
        The size of the sample to generate (default is 113**2).
    parameters_df : pd.DataFrame
        A DataFrame containing the parameters.

    Methods
    -------
    _load_old_parameters():
        Loads existing parameters from a CSV file.
    _generate_new_parameters():
        Generates new parameters.
    _generate_base_parameters():
        Generates base parameters without bottom values.
    _calculate_hydraulic_properties(params):
        Calculates hydraulic properties for the given parameters.
    _balance_samples(params):
        Balances subcritical and supercritical samples.
    _add_bottom_values(params):
        Adds bottom values to the parameters.
    _load_or_generate_parameters():
        Loads or generates parameters and combines them if necessary.
    get_parameters():
        Returns the parameters DataFrame.
    validate_constants():
        Validates the constants dictionary.
    """

    def __init__(self, constants, sample_size=113**2):
        """
        Parameters
        ----------
        constants : dict
            A dictionary containing constants and parameters for the calculations.
        sample_size : int, optional
            The size of the sample to generate (default is 113**2).
        """
        self.constants = constants
        self.sample_size = sample_size
        self.parameters_df = self._load_or_generate_parameters()

    def _load_old_parameters(self):
        """
        Loads existing parameters from a CSV file.

        Returns
        -------
        pd.DataFrame or None
            The DataFrame containing the parameters, or None if the file is not found.
        """
        try:
            df = pd.read_csv("parameters.csv")
            logger.info(f"Loaded existing parameters from parameters.csv. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.info("No existing parameters.csv found. Will create new DataFrame.")
            return None
        except Exception as e:
            logger.error(f"Error loading parameters.csv: {str(e)}")
            return None

    def _generate_new_parameters(self):
        """
        Generates new parameters.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the new parameters.
        """
        logger.info(f"Generating new parameters. Sample size: {self.sample_size}")

        try:
            # Generate parameters without bottom
            new_params = self._generate_base_parameters()

            # Calculate hydraulic properties
            new_params = self._calculate_hydraulic_properties(new_params)

            # Balance subcritical and supercritical samples
            balanced_params = self._balance_samples(new_params)

            # Add bottom values
            final_params = self._add_bottom_values(balanced_params)

            # Calculate direction
            final_params["direction"] = np.where(final_params["H0"] > final_params["yc"],
                                                 "Right to left", "Left to right")

            return final_params

        except Exception as e:
            logger.error(f"Error generating new parameters: {str(e)}")
            raise

    def _generate_base_parameters(self):
        """
        Generates base parameters without bottom values.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the base parameters.
        """
        return SampleGenerator.sample_combinations(
            self.sample_size,
            self.constants["parameters"]["slope_min"],
            self.constants["parameters"]["slope_max"],
            self.constants["parameters"]["n_min"],
            self.constants["parameters"]["n_max"],
            self.constants["parameters"]["q0_min"],
            self.constants["parameters"]["q0_max"],
            self.constants["parameters"]["h0_min"],
            self.constants["parameters"]["h0_max"],
        )

    def _calculate_hydraulic_properties(self, params):
        """
        Calculates hydraulic properties for the given parameters.

        Parameters
        ----------
        params : pd.DataFrame
            The DataFrame containing the parameters.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added hydraulic properties.
        """
        params["yn"] = HydraulicCalculations.normal_depth_simple(
            params["Q0"],
            self.constants["channel"]["width"],
            params["SLOPE"],
            params["n"],
        )
        params["yc"] = HydraulicCalculations.critical_depth_simple(
            params["Q0"], self.constants["channel"]["width"]
        )
        params["subcritical"] = params["yn"] > params["yc"]
        return params

    def _balance_samples(self, params):
        """
        Balances subcritical and supercritical samples.

        Parameters
        ----------
        params : pd.DataFrame
            The DataFrame containing the parameters.

        Returns
        -------
        pd.DataFrame
            The DataFrame with balanced samples.
        """
        subcritical = params[params["subcritical"]]
        supercritical = params[~params["subcritical"]]

        required_samples = min(len(subcritical), len(supercritical))
        logger.info(f"Balancing samples. Selected {required_samples} from each category.")

        selected_subcritical = subcritical.sample(n=required_samples, random_state=0)
        selected_supercritical = supercritical.sample(n=required_samples, random_state=0)

        return pd.concat([selected_subcritical, selected_supercritical]).sample(frac=1, random_state=0).reset_index(drop=True)

    def _add_bottom_values(self, params):
        """
        Adds bottom values to the parameters.

        Parameters
        ----------
        params : pd.DataFrame
            The DataFrame containing the parameters.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added bottom values.
        """
        bottom_values = self.constants["parameters"]["bottom_values"]
        logger.info(f"Adding bottom values: {bottom_values}")

        return pd.concat([params.assign(BOTTOM=bottom) for bottom in bottom_values]).reset_index(drop=True)

    def _load_or_generate_parameters(self):
        """
        Loads or generates parameters and combines them if necessary.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the combined parameters.
        """
        old_params = self._load_old_parameters()
        new_params = self._generate_new_parameters()

        if old_params is not None:
            # Combine old and new parameters
            combined_params = pd.concat([old_params, new_params], ignore_index=True)
            # Remove duplicates based on all columns except the index
            combined_params = combined_params.drop_duplicates(subset=combined_params.columns.drop('id'), keep='first')
            # Reset the index to create a new continuous id column
            combined_params = combined_params.reset_index(drop=True)
            combined_params.index.name = 'id'
            combined_params.to_csv("parameters.csv", index=True)
            logger.info(f"Combined old and new parameters. New shape: {combined_params.shape}")
            return combined_params
        else:
            new_params.index.name = 'id'
            new_params.to_csv("parameters.csv", index=True)
            logger.info(f"Wrote new parameters to parameters.csv. Shape: {new_params.shape}")
            return new_params

    def get_parameters(self):
        """
        Returns the parameters DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the parameters.
        """
        return self.parameters_df

    def validate_constants(self):
        """
        Validates the constants dictionary.

        Raises
        ------
        ValueError
            If any required keys or parameters are missing.
        """
        required_keys = ["parameters", "channel"]
        required_params = ["slope_min", "slope_max", "n_min", "n_max", "q0_min", "q0_max", "h0_min", "h0_max", "bottom_values"]
        required_channel = ["width"]

        if not all(key in self.constants for key in required_keys):
            raise ValueError(f"Missing required keys in constants: {required_keys}")

        if not all(param in self.constants["parameters"] for param in required_params):
            raise ValueError(f"Missing required parameters: {required_params}")

        if not all(channel_param in self.constants["channel"] for channel_param in required_channel):
            raise ValueError(f"Missing required channel parameters: {required_channel}")

        logger.info("Constants validation passed.")
