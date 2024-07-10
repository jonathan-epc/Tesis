import os
from datetime import datetime
from pathlib import Path

from logger_config import setup_logger
from loguru import logger
from modules.boundary_conditions import BoundaryConditions
from modules.environment_setup import EnvironmentSetup
from modules.geometry_generator import GeometryGenerator
from modules.parameter_manager import ParameterManager
from modules.steering_file_generator import SteeringFileGenerator
from tqdm.autonotebook import tqdm


def main() -> None:
    logger.info("Starting main process")

    # Set up the environment
    env_setup = EnvironmentSetup()
    setup_data = env_setup.get_setup_data()

    # Generate or load parameters
    param_manager = ParameterManager(setup_data["constants"])
    parameters_df = param_manager.get_parameters()

    for index, case in tqdm(parameters_df.iterrows(), total=len(parameters_df)):
        try:
            logger.debug(f"Processing case {index}")

            # Generate geometry
            geometry_generator = GeometryGenerator()
            borders_flat, borders_noise = geometry_generator.generate_geometry(
                index,
                case["SLOPE"],
                setup_data["flat_mesh"],
                setup_data["x"],
                setup_data["y"],
                setup_data["noise_grid_x"],
                setup_data["noise_grid_y"],
                setup_data["constants"]["mesh"]["num_points_x"],
                setup_data["constants"]["mesh"]["num_points_y"],
                setup_data["constants"]["channel"]["length"],
            )

            # Get boundary conditions
            boundary_file, prescribed_elevations = (
                BoundaryConditions.get_boundary_and_elevations(
                    case["direction"],
                    case["H0"],
                    case["BOTTOM"],
                    borders_flat,
                    borders_noise,
                )
            )

            # Generate steering file
            steering_generator = SteeringFileGenerator()
            steering_file_content = steering_generator.generate_steering_file(
                geometry_file=f"geometry/geometry_3x3_{case['BOTTOM']}_{index}.slf",
                boundary_file=boundary_file,
                results_file=f"results/results_{index}.slf",
                title=f"Caso {index}",
                duration=120,
                time_step=0.02,
                initial_depth=case["H0"],
                prescribed_flowrates=(0.0, case["Q0"]),
                prescribed_elevations=prescribed_elevations,
                friction_coefficient=case["n"],
            )
            # Write the steering file content to a file

            Path("steering").mkdir(parents=True, exist_ok=True)
            try:
                with open(f"steering/steering_{index}.cas", "w") as f:
                    f.write(steering_file_content)
                logger.debug(f"Wrote steering file for case {index}")
            except IOError as e:
                logger.error(f"Error writing steering file for case {index}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing case {index}: {str(e)}")
            continue
    logger.info("Main process completed")


if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[
        0
    ]  # Gets 'main' from 'main.py'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()

    log_name = f"{script_name}_{timestamp}_{process_id}"
    logger = setup_logger(log_name)
    main()
