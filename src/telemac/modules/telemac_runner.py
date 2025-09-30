import os
import platform
import subprocess

from loguru import logger


def run_telemac2d(filename: str, output_dir: str) -> None:
    """
    Runs a Telemac2D simulation and writes the output to a specified directory.

    Parameters
    ----------
    filename : str
        The name of the input file for the Telemac2D simulation.
    output_dir : str
        The directory where the output file will be saved.

    Raises
    ------
    subprocess.CalledProcessError
        If the Telemac2D simulation fails to run.

    Examples
    --------
    >>> run_telemac2d('input_file.txt', '/path/to/output/dir')
    """
    command = (
        ["telemac2d.py"]
        if platform.system() == "Linux"
        else ["python", "-m", "telemac2d"]
    )
    name, ext = os.path.splitext(filename)
    output_file = os.path.join(output_dir, f"{name}.txt")

    try:
        with open(output_file, "w") as output_fh:
            subprocess.run(
                command + ["--ncsize=8", filename],
                check=True,
                stdout=output_fh,
                stderr=subprocess.STDOUT,
            )
        logger.info(f"Completed Telemac2D simulation for {filename}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Telemac2D simulation for {filename}: {e}")
        raise
