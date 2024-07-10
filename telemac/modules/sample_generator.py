import pandas as pd
from scipy.stats import qmc
from typing import List
class SampleGenerator:
    @staticmethod
    def sample_combinations(
        n: int,
        SLOPE_min: float,
        SLOPE_max: float,
        n_min: float,
        n_max: float,
        Q0_min: float,
        Q0_max: float,
        H0_min: float,
        H0_max: float,
        BOTTOM_values: List[str]
    ) -> pd.DataFrame:
        """
        Generate a DataFrame of sample combinations using Latin Hypercube Sampling (LHS).
    
        Parameters:
        ----------
        n : int
            Number of samples to generate
        SLOPE_min : float
            Minimum value for S
        SLOPE_max : float
            Maximum value for S
        n_min : float
            Minimum value for n
        n_max : float
            Maximum value for n
        Q0_min : float
            Minimum value for Q
        Q0_max : float
            Maximum value for Q
        H0_min : float
            Minimum value for H0
        H0_max : float
            Maximum value for H0
        BOTTOM_values : List[str]
            List of values for BOTTOM to be combined with each sample
    
        Returns:
        -------
        pd.DataFrame
            DataFrame containing the sample combinations with columns ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
        """
        sampler = qmc.LatinHypercube(d=4, strength=2, seed=1618)
        sample = sampler.random(n=n)
    
        lower_bounds = [SLOPE_min, n_min, Q0_min, H0_min]
        upper_bounds = [SLOPE_max, n_max, Q0_max, H0_max]
    
        sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    
        combinations = [
            combination.tolist() + [bottom]
            for combination in sample_scaled
            for bottom in BOTTOM_values
        ]
    
        column_names = ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
        return pd.DataFrame(combinations, columns=column_names)