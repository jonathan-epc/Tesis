import os
import sys

from run_configurations import OUTPUT_FOLDER, PARAMETERS_FILE, STEERING_FOLDER

from modules.cli import parse_arguments, setup_logging
from modules.file_handler import prepare_steering_file
from modules.flux_checker import check_flux_boundaries, is_flux_balanced
from modules.param_utils import load_parameters
from modules.simulation_runner import run_telemac2d_on_files


def main():
    args = parse_arguments()
    logger = setup_logging()

    try:
        parameters = load_parameters(PARAMETERS_FILE)
        logger.info(f"Loaded parameters from {PARAMETERS_FILE}")

        start_index, end_index = determine_file_range(args, STEERING_FOLDER)

        run_telemac2d_on_files(
            start_index, end_index, parameters, args.output_dir, STEERING_FOLDER
        )

        logger.info("Telemac2D simulations completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


def determine_file_range(args, steering_folder):
    if args.start == 0 and args.end == 0:
        cas_files = [f for f in os.listdir(steering_folder) if f.endswith(".cas")]
        return 0, len(cas_files)
    else:
        return args.start, args.end


if __name__ == "__main__":
    main()
