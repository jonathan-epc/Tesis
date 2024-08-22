import pandas as pd
from scipy.stats import qmc
from typing import List

class SampleGenerator:
    """
    A class to generate samples of combinations for given parameter ranges.

    Methods
    -------
    sample_combinations(n, SLOPE_min, SLOPE_max, n_min, n_max, Q0_min, Q0_max, H0_min, H0_max)
        Generates a DataFrame of sampled combinations for the specified parameter ranges.
    """

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
    ) -> pd.DataFrame:
        """
        Generates a DataFrame of sampled combinations for the specified parameter ranges.

        Parameters
        ----------
        n : int
            The number of samples to generate.
        SLOPE_min : float
            The minimum value for the SLOPE parameter.
        SLOPE_max : float
            The maximum value for the SLOPE parameter.
        n_min : float
            The minimum value for the n parameter.
        n_max : float
            The maximum value for the n parameter.
        Q0_min : float
            The minimum value for the Q0 parameter.
        Q0_max : float
            The maximum value for the Q0 parameter.
        H0_min : float
            The minimum value for the H0 parameter.
        H0_max : float
            The maximum value for the H0 parameter.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sampled combinations of the parameters.

        Examples
        --------
        >>> generator = SampleGenerator()
        >>> df = generator.sample_combinations(10, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 5.0, 15.0)
        >>> print(df)
        """

        sampler = qmc.LatinHypercube(d=4, strength=2, seed=0)
        sample = sampler.random(n=n)

        lower_bounds = [SLOPE_min, n_min, Q0_min, H0_min]
        upper_bounds = [SLOPE_max, n_max, Q0_max, H0_max]

        sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

        column_names = ["SLOPE", "n", "Q0", "H0"]
        return pd.DataFrame(sample_scaled, columns=column_names)
