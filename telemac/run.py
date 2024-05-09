import argparse
import glob
import multiprocessing
import os
import platform
import shlex
import signal
import subprocess
import sys

import pandas as pd
from loguru import logger
from tqdm.autonotebook import tqdm


def run_telemac2d(filename, output_dir="outputs"):
    """
    Run telemac2d simulation on a specific file.

    Args:
        filename (str): The name of the input file.
        output_dir (str, optional): The directory to store output files. Defaults to "outputs".
        linux (bool, optional): Indicates if running on Linux. Defaults to False.
    """
    if platform.system() == "Linux":
        command = f"telemac2d.py --ncsize=4 {filename} >> {os.path.join(output_dir, filename)}.txt"
    else:
        command = f"python -m telemac2d --ncsize=4 {filename} >> {os.path.join(output_dir, filename)}.txt"
    subprocess.run(shlex.split(command), check=True, capture_output=True)


def run_telemac2d_on_files(start, end, output_dir, parameters):
    """
    Run telemac2d simulation on a range of files.

    Args:
        start (int): The starting index of the files.
        end (int): The ending index of the files.
        output_dir (str): The directory to store output files.
        linux (bool, optional): Indicates if running on Linux. Defaults to False.
    """
    total_files = end - start + 1
    with tqdm(total=total_files) as pbar:
        for i in range(start, end + 1):
            filename = f"steering_{i}.cas"
            case_parameters = parameters.iloc[i]
            tqdm_desc = f"Running case {i}: S={case_parameters['S']} | n = {case_parameters['n']:.4f} | Q={case_parameters['Q']:.4f} | H0={case_parameters['H0']:.4f} | {'subcritical' if case_parameters['subcritical'] else 'supercritical'} | B: {case_parameters['BOTTOM']}"
            pbar.set_description(tqdm_desc)
            run_telemac2d(filename, output_dir)
            pbar.update(1)


if __name__ == "__main__":
    # Set up logger configuration
    logger.add("logfile.log", rotation="500 MB", level="INFO")  # Output log to file
    logger.add(sys.stderr, level="WARNING")  # Output log to console
    logger.info("Starting Telemac2D simulations...")

    # Read parameters DataFrame once
    parameters = pd.read_csv("parameters.csv", index_col="id")

    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument("start", type=int, help="Start file index (0 for all files)")
    parser.add_argument("end", type=int, help="End file index (0 for all files)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_dir",
        help="Output directory for the simulations (default: output_dir)",
    )
    parser.add_argument(
        "--linux",
        action="store_true",
        help="Flag to indicate running on Linux (default: False)",
    )
    args = parser.parse_args()

    # Process all .cas files in the folder if start_file and end_file are both 0
    if args.start == 0 and args.end == 0:
        cas_files = [file for file in os.listdir() if file.endswith(".cas")]
        start_index = 1
        end_index = len(cas_files)
    else:
        start_index = args.start
        end_index = args.end

    # Check if start index is less than or equal to end index
    if start_index > end_index:
        logger.error("Start index cannot be greater than end index.")
    else:
        # Example usage
        try:
            run_telemac2d_on_files(
                start_index,
                end_index,
                args.output_dir,
                parameters,
                linux=args.linux,
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
