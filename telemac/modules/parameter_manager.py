import pandas as pd
from loguru import logger
from modules.sample_generator import SampleGenerator
from modules.hydraulic_calculations import HydraulicCalculations

class ParameterManager:
    def __init__(self, constants):
        self.constants = constants
        self.parameters_df = self._load_or_generate_parameters()

    def _load_old_parameters(self):
        try:
            return pd.read_csv("parameters.csv")
        except FileNotFoundError:
            logger.info("No existing parameters.csv found. Creating new DataFrame.")
            return None

    def _generate_new_parameters(self):
        new_params = SampleGenerator.sample_combinations(
            79**2,
            self.constants["parameters"]["slope_min"],
            self.constants["parameters"]["slope_max"],
            self.constants["parameters"]["n_min"],
            self.constants["parameters"]["n_max"],
            self.constants["parameters"]["q0_min"],
            self.constants["parameters"]["q0_max"],
            self.constants["parameters"]["h0_min"],
            self.constants["parameters"]["h0_max"],
            self.constants["parameters"]["bottom_values"],
        )
        
        new_params["yn"] = HydraulicCalculations.normal_depth_simple(
            new_params["Q0"],
            self.constants["channel"]["width"],
            new_params["SLOPE"],
            new_params["n"],
        )
        new_params["yc"] = HydraulicCalculations.critical_depth_simple(
            new_params["Q0"], self.constants["channel"]["width"]
        )
        new_params["subcritical"] = new_params["yn"] > new_params["yc"]
        new_params["direction"] = new_params["H0"] > new_params["yc"]
        new_params["direction"] = new_params["direction"].apply(
            lambda x: "Right to left" if x else "Left to right"
        )
        
        new_params.to_csv("parameters.csv", index=True, index_label="id")
        logger.info("Wrote new parameters to parameters.csv")
        return new_params

    def _load_or_generate_parameters(self):
        old_params = self._load_old_parameters()
        if old_params is not None:
            return old_params
        else:
            return self._generate_new_parameters()

    def get_parameters(self):
        return self.parameters_df
