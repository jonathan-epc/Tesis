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
    Run Telemac2D simulation for a given case file.

    Args:
        filename (str): Name of the case file.
        output_dir (str, optional): Directory to save the output. Defaults to "outputs".
    """
    # Determine the command based on the operating system
    if platform.system() == "Linux":
        command = ["telemac2d.py"]
    else:
        command = ["python", "-m", "telemac2d"]

    # Define the file path for the output
    output_file = os.path.join(output_dir, filename + ".txt")

    try:
        with open(output_file, "w") as output_fh:
            subprocess.run(
                command + ["--ncsize=8", filename],
                check=True,
                stdout=output_fh,
                stderr=subprocess.STDOUT
            )
    except subprocess.CalledProcessError as e:
        pass
        #logger.error(f"Error running Telemac2D simulation for {filename}: {e}")
    else:
        pass
        #logger.info(f"Completed Telemac2D simulation for {filename}")


def run_telemac2d_on_files(start, end, output_dir, parameters):
    """
    Run Telemac2D simulations for a range of case files with given parameters.

    Args:
        start (int): Starting index of the case files.
        end (int): Ending index of the case files.
        output_dir (str): Directory to save the outputs.
        parameters (pandas.DataFrame): DataFrame containing parameters for each case.
    """
    total_files = end - start
    print(f"Running {total_files} Telemac2D simulations from case {start} to case {end}")

    with tqdm(total=total_files, unit="case", dynamic_ncols=True) as pbar:
        for i in range(start, end):
            filename = f"steering_{i}.cas"
            case_parameters = parameters.loc[i]
            tqdm_desc = (
                f"Case {i}: S={case_parameters['SLOPE']} | "
                f"n={case_parameters['n']:.4f} | Q={case_parameters['Q0']:.4f} | "
                f"H0={case_parameters['H0']:.4f} | "
                f"{'subcritical' if case_parameters['subcritical'] else 'supercritical'} | "
                f"B: {case_parameters['BOTTOM']}"
            )
            pbar.set_description(tqdm_desc)
            run_telemac2d(filename, output_dir)
            pbar.update(1)

    print("All Telemac2D simulations completed")


if __name__ == "__main__":
    # Read parameters DataFrame once
    parameters = pd.read_csv("parameters.csv", index_col="id")

    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument(
        "--start", type=int, default=0, help="Start file index (default: 0 for all files)"
    )
    parser.add_argument(
        "--end", type=int, default=0, help="End file index (default: 0 for all files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for the simulations (default: outputs)",
    )
    args = parser.parse_args()

    # Process all .cas files in the folder if start_file and end_file are both 0
    if args.start == 0 and args.end == 0:
        cas_files = [file for file in os.listdir() if file.endswith(".cas")]
        start_index = 0
        end_index = len(cas_files)
    else:
        start_index = args.start
        end_index = args.end

    # Check if start index is less than or equal to end index
    if start_index > end_index:
        print("Start index cannot be greater than end index.")
    else:
        # Example usage
        try:
            run_telemac2d_on_files(start_index, end_index, args.output_dir, parameters)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
