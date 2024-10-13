import argparse
import os
import random
from datetime import datetime
from pathlib import Path
import numpy as np

from logger_config import setup_logger
from loguru import logger
from tqdm.autonotebook import tqdm

from modules.boundary_conditions import BoundaryConditions
from modules.environment_setup import EnvironmentSetup
from modules.geometry_generator import GeometryGenerator
from modules.parameter_manager import ParameterManager
from modules.steering_file_generator import SteeringFileGenerator

def process_case(index, case, setup_data):
    steering_file_path = Path(f"steering/{index}.cas")

    if steering_file_path.exists():
        logger.info(
            f"Steering file for case {index} already exists. Skipping processing."
        )
        return

    try:
        logger.debug(f"Processing case {index}")

        # Generate geometry
        logger.debug(f"Generating geometry")
        geometry_generator = GeometryGenerator()
        borders = geometry_generator.generate_geometry(
            index,
            case["SLOPE"],
            case["BOTTOM"],
            setup_data["flat_mesh"],
            setup_data["constants"]["mesh"]["num_points_x"],
            setup_data["constants"]["mesh"]["num_points_y"],
            setup_data["constants"]["channel"]["length"],
            setup_data["constants"]["channel"]["width"],
        )

        # Get boundary conditions
        logger.debug(f"Getting boundary conditions")
        boundary_file, prescribed_elevations = (
            BoundaryConditions.get_boundary_and_elevations(
                case["direction"], case["H0"], case["BOTTOM"], borders
            )
        )

        # Generate steering file
        logger.debug(f"Getting steering file")
        steering_generator = SteeringFileGenerator()
        steering_file_content = steering_generator.generate_steering_file(
            geometry_file=f"geometry/3x3_{case['BOTTOM']}_{index}.slf",
            boundary_file=boundary_file,
            results_file=f"results/{index}.slf",
            title=f"Case {index}",
            duration=120,
            time_step=0.02,
            initial_depth=case["H0"],
            prescribed_flowrates=(0.0, case["Q0"]),
            prescribed_elevations=prescribed_elevations,
            friction_coefficient=case["n"],
        )

        # Write the steering file content to a file
        steering_file_path.parent.mkdir(parents=True, exist_ok=True)
        steering_file_path.write_text(steering_file_content)
        logger.debug(f"Wrote steering file for case {index}")

    except Exception as e:
        logger.error(f"Error processing case {index}: {str(e)}")

def main(param_mode="add", sample_size=179**2) -> None:
    logger.info(f"Starting main process with parameter mode: {param_mode}")

    # Set up the environment
    env_setup = EnvironmentSetup()
    setup_data = env_setup.get_setup_data()
    seed = setup_data["constants"]["seed"]
    random.seed(seed)
    np.random.seed(seed)

    # Generate or load parameters
    param_manager = ParameterManager(
        setup_data["constants"], mode=param_mode, sample_size=sample_size
    )
    parameters_df = param_manager.get_parameters()
    subcritical_counts = parameters_df["subcritical"].value_counts()
    print(f"Subcritical cases: {subcritical_counts[True]}")
    print(f"Supercritical cases: {subcritical_counts[False]}")
    # Process cases
    for index, case in tqdm(
        parameters_df.iterrows(), total=len(parameters_df), desc="Processing cases"
    ):
        process_case(index, case, setup_data)

    logger.info("Main process completed")

if __name__ == "__main__":
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)

    parser = argparse.ArgumentParser(
        description="Run the main process with specified parameter mode."
    )
    parser.add_argument(
        "--mode",
        choices=["new", "read", "add"],
        default="add",
        help="Mode for parameter management: 'new' to generate new, 'read' to only read existing, 'add' to add to existing",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=179,
        help="Sample size to be used, will be adjusted to the nearest square of a prime if necessary.",
    )
    args = parser.parse_args()

    main(param_mode=args.mode, sample_size=args.sample_size**2)
