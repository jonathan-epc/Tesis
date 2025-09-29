# telemac/input_generator.py

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from common.utils import setup_logger
from loguru import logger
from nconfig import get_config
from tqdm import tqdm

from modules.environment_setup import EnvironmentSetup
from modules.parameter_manager import ParameterManager


def process_case(case, flat_mesh, overwrite=False):
    """
    Processes a single TelemacCase by generating its geometry and steering file.
    """
    if case.steering_file_path.exists() and not overwrite:
        logger.info(f"Skipping case {case.case_id}: steering file exists.")
        return

    try:
        logger.debug(f"Processing case {case.case_id}...")
        # 1. Generate geometry and get the border elevations
        borders = case.generate_geometry(flat_mesh)

        # 2. Generate the steering file using the border elevations
        case.generate_steering_file(borders)

        logger.debug(f"Successfully generated files for case {case.case_id}.")
    except Exception as e:
        logger.exception(f"Error processing case {case.case_id}: {e}")


def main(param_mode="add", sample_size=100, overwrite=False):
    logger.info("Starting input generation process...")

    config = get_config("nconfig.yml")

    env_setup = EnvironmentSetup(config)
    setup_data = env_setup.get_setup_data()

    # Seed for reproducibility
    np.random.seed(config.seed)
    random.seed(config.seed)

    param_manager = ParameterManager(config, mode=param_mode, sample_size=sample_size)

    # ParameterManager now returns a list of TelemacCase objects
    cases_to_process = param_manager.create_cases()

    for case in tqdm(cases_to_process, desc="Generating simulation cases"):
        process_case(case, setup_data["flat_mesh"], overwrite)

    logger.info("Input generation process completed.")


if __name__ == "__main__":
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{script_name}_{timestamp}"
    logger = setup_logger(log_name)

    parser = argparse.ArgumentParser(
        description="Generate TELEMAC-2D simulation input files."
    )
    parser.add_argument(
        "--mode",
        choices=["new", "read", "add"],
        default="add",
        help="Parameter file handling mode.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of new samples to generate (if applicable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing geometry and steering files.",
    )
    args = parser.parse_args()

    main(param_mode=args.mode, sample_size=args.sample_size, overwrite=args.overwrite)
