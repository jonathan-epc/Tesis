"""
Simulation Data Processor

This module processes simulation results, calculates statistics, and saves the data
in both raw and normalized formats. It handles two types of data: NOISE and FLAT.

The main function, `process_simulation_results`, orchestrates the entire process.

Dependencies:
    - os
    - h5py
    - pandas
    - loguru
    - modules.file_processing
    - modules.statistics

Usage:
    Run this script directly to process simulation results.
    Ensure that the required input files and directories exist before running.

Author: [Your Name]
Date: [Current Date]
"""

import os
from typing import Dict, List

import h5py
import pandas as pd
from loguru import logger

from modules.file_processing import process_and_save, process_and_save_normalized
from modules.statistics import (
    calculate_statistics,
    combine_statistics,
    normalize_statistics,
)


def process_simulation_results(
    base_dir: str,
    output_file_noise: str,
    output_file_flat: str,
    normalized_output_file_noise: str,
    normalized_output_file_flat: str,
) -> None:
    """
    Process simulation results, calculate statistics, and save data.

    This function reads simulation parameters, processes result files,
    calculates statistics, and saves both raw and normalized data to HDF5 files.

    Parameters
    ----------
    base_dir : str
        Base directory containing simulation data.
    output_file_noise : str
        Path to output HDF5 file for NOISE data.
    output_file_flat : str
        Path to output HDF5 file for FLAT data.
    normalized_output_file_noise : str
        Path to output HDF5 file for normalized NOISE data.
    normalized_output_file_flat : str
        Path to output HDF5 file for normalized FLAT data.

    Returns
    -------
    None

    Notes
    -----
    This function processes both NOISE and FLAT data types.
    It assumes the existence of a 'parameters.csv' file in the base directory
    and '.slf' result files in the 'results' subdirectory.
    """
    logger.info("Processing start")

    # Read and process parameters
    parameters = pd.read_csv(os.path.join(base_dir, "parameters.csv"))
    parameter_names = ["H0", "Q0", "SLOPE", "n"]
    parameter_stats = {param: parameters[param].describe() for param in parameter_names}
    parameter_table = (
        pd.DataFrame(parameter_stats)
        .T.reset_index()
        .rename(columns={"index": "names", "std": "variance"})
    )
    parameter_table["variance"] = parameter_table["variance"] ** 2

    # Define variable names and find result files
    variable_names = ["F", "B", "H", "Q", "S", "U", "V"]
    result_files = [
        f for f in os.listdir(os.path.join(base_dir, "results")) if f.endswith(".slf")
    ]

    # Process and save raw data
    with h5py.File(output_file_noise, "w") as hdf5_file_noise, h5py.File(
        output_file_flat, "w"
    ) as hdf5_file_flat:
        table_noise = process_and_save(
            hdf5_file_noise,
            "NOISE",
            base_dir,
            parameters,
            parameter_table,
            parameter_names,
            variable_names,
            result_files,
        )
        table_flat = process_and_save(
            hdf5_file_flat,
            "FLAT",
            base_dir,
            parameters,
            parameter_table,
            parameter_names,
            variable_names,
            result_files,
        )

    # Process and save normalized data
    with h5py.File(
        normalized_output_file_noise, "w"
    ) as hdf5_file_noise_norm, h5py.File(
        normalized_output_file_flat, "w"
    ) as hdf5_file_flat_norm:
        process_and_save_normalized(
            hdf5_file_noise_norm,
            "NOISE",
            table_noise,
            base_dir,
            parameters,
            parameter_names,
            variable_names,
            result_files,
        )
        process_and_save_normalized(
            hdf5_file_flat_norm,
            "FLAT",
            table_flat,
            base_dir,
            parameters,
            parameter_names,
            variable_names,
            result_files,
        )

    logger.success("Script completed successfully")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "simulation_data_processor.log", format="{time} {level} {message}", level="INFO"
    )

    # Define input and output paths
    BASE_DIR = "telemac"
    OUTPUT_FILE_NOISE = "ML/simulation_data_noise.hdf5"
    OUTPUT_FILE_FLAT = "ML/simulation_data_flat.hdf5"
    NORMALIZED_OUTPUT_FILE_NOISE = "ML/simulation_data_normalized_noise.hdf5"
    NORMALIZED_OUTPUT_FILE_FLAT = "ML/simulation_data_normalized_flat.hdf5"

    # Run the main processing function
    process_simulation_results(
        BASE_DIR,
        OUTPUT_FILE_NOISE,
        OUTPUT_FILE_FLAT,
        NORMALIZED_OUTPUT_FILE_NOISE,
        NORMALIZED_OUTPUT_FILE_FLAT,
    )
