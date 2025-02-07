import pandas as pd
from scipy.stats import qmc
from typing import Dict

class SampleGenerator:
    """
    A class to generate samples of combinations for given parameter ranges.

    Methods
    -------
    sample_combinations(n, param_ranges)
        Generates a DataFrame of sampled combinations for the specified parameter ranges.
    """

    @staticmethod
    def sample_combinations(n: int, param_ranges: Dict[str, tuple]) -> pd.DataFrame:
        """
        Generates a DataFrame of sampled combinations for the specified parameter ranges.

        Parameters
        ----------
        n : int
            The number of samples to generate.
        param_ranges : Dict[str, tuple]
            A dictionary where keys are parameter names and values are tuples (min, max).

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sampled combinations of the parameters.

        Examples
        --------
        >>> param_ranges = {
        ...     "SLOPE": (0.1, 0.5),
        ...     "n": (1.0, 2.0),
        ...     "Q0": (10.0, 20.0),
        ...     "H0": (5.0, 15.0)
        ... }
        >>> generator = SampleGenerator()
        >>> df = generator.sample_combinations(10, param_ranges)
        >>> print(df)
        """
        param_names = list(param_ranges.keys())
        lower_bounds = [param_ranges[p][0] for p in param_names]
        upper_bounds = [param_ranges[p][1] for p in param_names]

        sampler = qmc.LatinHypercube(d=len(param_names), strength=2, seed=43)
        sample = sampler.random(n=n)
        sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

        return pd.DataFrame(sample_scaled, columns=param_names)
