import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from logger_config import setup_logger
from run_configurations import OUTPUT_FOLDER, PARAMETERS_FILE, STEERING_FOLDER

from modules.file_handler import prepare_steering_file
from modules.flux_checker import check_flux_boundaries, is_flux_balanced
from modules.param_utils import load_parameters
from modules.simulation_runner import run_telemac2d_on_files

def main():
    args = parse_arguments()
    logger = setup_logging()

    try:
        # Load and validate parameters
        parameters = load_and_validate_parameters(PARAMETERS_FILE, logger)

        # Determine the range of files to process
        start_index, end_index = determine_file_range(args, STEERING_FOLDER)
        cas_files = get_cas_files(STEERING_FOLDER)
        selected_files = cas_files[start_index:end_index]

        # Apply bottom filter if specified
        if args.bottom is not None:
            # Filter parameters by the specified bottom value
            parameters = filter_parameters_by_bottom(parameters, args.bottom, logger)

            # Ensure selected files are filtered based on IDs in the filtered parameters
            parameter_ids = set(parameters.index.astype(str))
            selected_files = [
                file
                for file in selected_files
                if os.path.splitext(file)[0] in parameter_ids
            ]
            # Pre-check flux balance for the filtered range of files
        unbalanced_files = filter_unbalanced_files(
            selected_files, args.output_dir, logger
        )

        if args.dry_run:
            logger.info(f"Dry run: would process files {unbalanced_files}")
            logger.info(f"Number of unbalanced files: {len(unbalanced_files)}")
        else:
            run_telemac2d_on_files(
                unbalanced_files, parameters, args.output_dir, STEERING_FOLDER
            )
            logger.info("Telemac2D simulations completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)

def filter_unbalanced_files(files: List[str], output_dir: str, logger) -> List[str]:
    """
    Filter files to include only those with unbalanced flux boundaries.

    Parameters
    ----------
    files : List[str]
        List of steering file names.
    output_dir : str
        Directory where result files are stored.
    logger : Logger
        Logger for logging messages.

    Returns
    -------
    List[str]
        List of file names with unbalanced flux boundaries.
    """
    unbalanced_files = []
    logger.info(f"Checking flux boundaries of {len(files)} files . . .")
    for filename in files:
        result_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        if os.path.exists(result_file):
            flux_1, flux_2 = check_flux_boundaries(result_file)
            if flux_1 is not None and flux_2 is not None:
                if is_flux_balanced(flux_1, flux_2):
                    logger.info(f"Skipping {filename}: flux boundaries are balanced.")
                    continue
                else:
                    logger.info(f"Unbalanced flux for {filename}.")
            else:
                logger.info(f"Cannot determine flux for {filename}. Including in run.")
        else:
            logger.info(f"No result file found for {filename}. Including in run.")
        unbalanced_files.append(filename)
    return unbalanced_files

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument(
        "--start", type=int, default=0, help="Start index for simulation files"
    )
    parser.add_argument(
        "--end", type=int, default=0, help="End index for simulation files"
    )
    parser.add_argument(
        "--bottom",
        type=str,
        default="",
        choices=["SLOPE", "NOISE", "BUMP", "BARS", "STEP"],
        help="Select a specific bottom value for simulation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_FOLDER,
        help="Output directory for simulation results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actually running simulations",
    )
    return parser.parse_args()

def setup_logging():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()

    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)

    return logger

def load_and_validate_parameters(file_path: str, logger) -> pd.DataFrame:
    try:
        parameters = load_parameters(file_path)
        if parameters.empty:
            raise ValueError("Parameters file is empty")
        required_columns = ["SLOPE", "n", "Q0", "H0", "BOTTOM"]
        missing_columns = [
            col for col in required_columns if col not in parameters.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in parameters file: {', '.join(missing_columns)}"
            )
        logger.info(f"Loaded and validated parameters from {file_path}")
        return parameters
    except pd.errors.EmptyDataError:
        raise ValueError(f"Parameters file {file_path} is empty")
    except FileNotFoundError:
        raise FileNotFoundError(f"Parameters file not found: {file_path}")

def determine_file_range(
    args: argparse.Namespace, steering_folder: str
) -> Tuple[int, int]:
    if args.start == 0 and args.end == 0:
        cas_files = get_cas_files(steering_folder)
        return 0, len(cas_files)
    elif args.start < 0 or args.end < args.start:
        raise ValueError("Invalid start or end index")
    else:
        return args.start, args.end

def get_cas_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Steering folder not found: {folder}")
    cas_files = [f for f in os.listdir(folder) if f.endswith(".cas")]
    if not cas_files:
        raise ValueError(f"No .cas files found in {folder}")
    return cas_files

def filter_parameters_by_bottom(
    parameters: pd.DataFrame, selected_bottom: float, logger
) -> pd.DataFrame:
    if "BOTTOM" not in parameters.columns:
        logger.warning(
            "'BOTTOM' column not found in parameters. Running all simulations."
        )
        return parameters

    filtered_params = parameters[parameters["BOTTOM"] == selected_bottom]

    if filtered_params.empty:
        logger.warning(
            f"No parameters found for bottom value {selected_bottom}. Running all simulations."
        )
        return parameters

    logger.info(
        f"Filtered parameters to include only bottom value {selected_bottom}. "
        f"Running {len(filtered_params)} simulations."
    )
    return filtered_params

if __name__ == "__main__":
    main()
