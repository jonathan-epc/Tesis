import os

from loguru import logger
from tqdm import tqdm

from modules.file_handler import prepare_steering_file
from modules.file_utils import move_file, setup_output_dir
from modules.flux_checker import check_flux_boundaries, is_flux_balanced
from modules.telemac_runner import run_telemac2d


def run_single_simulation(i, filename, case_parameters, output_dir, steering_folder):
    result_file = os.path.join(output_dir, f"{filename}.txt")
    run_simulation = True

    if os.path.exists(result_file):
        flux_1, flux_2 = check_flux_boundaries(result_file)
        if flux_1 is not None and flux_2 is not None:
            if is_flux_balanced(flux_1, flux_2):
                logger.info(
                    f"Skipping simulation for {filename} as flux boundaries are balanced."
                )
                return False
            else:
                logger.info(
                    f"Re-running simulation for {filename} as flux boundaries are not balanced."
                )
        else:
            logger.info(
                f"Re-running simulation for {filename} as flux boundaries could not be determined."
            )

    tqdm_desc = (
        f"Case {i}: S={case_parameters['SLOPE']} | "
        f"n={case_parameters['n']:.4f} | Q={case_parameters['Q0']:.4f} | "
        f"H0={case_parameters['H0']:.4f} | "
        f"{'subcritical' if case_parameters['subcritical'] else 'supercritical'} | "
        f"B: {case_parameters['BOTTOM']}"
    )

    try:
        src_file = os.path.join(steering_folder, filename)
        dst_file = filename
        prepare_steering_file(src_file, dst_file, result_file, i)
        run_telemac2d(filename, output_dir)
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    finally:
        move_file(dst_file, src_file)

    return True


def run_telemac2d_on_files(start, end, parameters, output_dir, steering_folder):
    setup_output_dir(output_dir)
    total_files = end - start

    with tqdm(total=total_files, unit="case", dynamic_ncols=True) as pbar:
        for i in range(start, end):
            filename = f"steering_{i}.cas"
            case_parameters = parameters.loc[i]

            pbar.set_description(f"Processing {filename}")
            simulation_run = run_single_simulation(
                i, filename, case_parameters, output_dir, steering_folder
            )

            if simulation_run:
                pbar.update(1)
            else:
                pbar.total -= 1
                pbar.refresh()

    logger.info("All Telemac2D simulations completed")