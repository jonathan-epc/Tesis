import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from config import OUTPUT_FOLDER, PARAMETERS_FILE, STEERING_FOLDER
from logger_config import setup_logger
from loguru import logger
from modules.file_utils import move_file, setup_output_dir
from modules.param_utils import load_parameters
from modules.telemac_runner import run_telemac2d
from tqdm import tqdm


def run_telemac2d_on_files(start, end, output_dir, parameters):
    setup_output_dir(output_dir)
    total_files = end - start

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

            try:
                src_file = os.path.join(STEERING_FOLDER, filename)
                dst_file = filename
                move_file(src_file, dst_file)

                run_telemac2d(filename, output_dir)

            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
            finally:
                move_file(dst_file, src_file)

            pbar.update(1)

    logger.info("All Telemac2D simulations completed")


if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    
    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)  # Use the log_name variable here

    parser = argparse.ArgumentParser(description="Run Telemac2D simulations.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start file index (default: 0 for all files)",
    )
    parser.add_argument(
        "--end", type=int, default=0, help="End file index (default: 0 for all files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_FOLDER,
        help=f"Output directory for the simulations (default: {OUTPUT_FOLDER})",
    )

    args = parser.parse_args()

    logger.info(f"Starting Telemac2D simulations with arguments: {args}")

    try:
        parameters = load_parameters(PARAMETERS_FILE)
        logger.info(f"Loaded parameters from {PARAMETERS_FILE}")

        if args.start == 0 and args.end == 0:
            cas_files = [f for f in os.listdir(STEERING_FOLDER) if f.endswith(".cas")]
            start_index = 0
            end_index = len(cas_files)
            logger.info(f"Processing all {end_index} .cas files")
        else:
            start_index = args.start
            end_index = args.end
            logger.info(f"Processing .cas files from index {start_index} to {end_index}")

        if start_index > end_index:
            raise ValueError("Start index cannot be greater than end index.")

        run_telemac2d_on_files(start_index, end_index, args.output_dir, parameters)
        
        logger.info("Telemac2D simulations completed successfully")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)
