from loguru import logger
import re

def check_flux_boundaries(log_file):
    """
    Checks the flux boundaries from the log file.

    Parameters
    ----------
    log_file : str
        The path to the log file.

    Returns
    -------
    tuple
        A tuple containing the flux boundary values (flux_boundary_1, flux_boundary_2).
        Returns (None, None) if the flux boundaries are not found or an error occurs.

    Examples
    --------
    >>> check_flux_boundaries('log.txt')
    (1.23, -1.23)
    """
    try:
        with open(log_file, 'r', encoding="latin1") as f:
            content = f.read()

        # Find the last occurrence of BALANCE OF WATER VOLUME
        last_balance = re.findall(r'BALANCE OF WATER VOLUME(.*?)(?=\n\n)', content, re.DOTALL)
        if not last_balance:
            return None, None

        last_balance = last_balance[-1]

        # Extract flux boundaries
        flux_boundary_1 = re.search(r'FLUX BOUNDARY\s+1:\s+([-\d.E]+)', last_balance)
        flux_boundary_2 = re.search(r'FLUX BOUNDARY\s+2:\s+([-\d.E]+)', last_balance)

        if flux_boundary_1 and flux_boundary_2:
            return float(flux_boundary_1.group(1)), float(flux_boundary_2.group(1))
        else:
            return None, None
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return None, None

def is_flux_balanced(flux_1, flux_2, threshold=1e-3):
    """
    Checks if the flux boundaries are balanced within a given threshold.

    Parameters
    ----------
    flux_1 : float
        The value of the first flux boundary.
    flux_2 : float
        The value of the second flux boundary.
    threshold : float, optional
        The threshold value for considering the fluxes balanced. Default is 1e-3.

    Returns
    -------
    bool
        True if the fluxes are balanced within the threshold, False otherwise.

    Examples
    --------
    >>> is_flux_balanced(1.23, -1.23)
    True
    """
    return abs(flux_1 + flux_2) < threshold
