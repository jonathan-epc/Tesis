import os

from loguru import logger
from tqdm import tqdm

from modules.file_handler import prepare_steering_file
from modules.file_utils import move_file, setup_output_dir
from modules.flux_checker import check_flux_boundaries, is_flux_balanced
from modules.telemac_runner import run_telemac2d

def run_single_simulation(i, filename, case_parameters, output_dir, steering_folder):
    """
    Run a single Telemac2D simulation.

    Parameters
    ----------
    i : int
        The index of the simulation case.
    filename : str
        The name of the steering file for the simulation.
    case_parameters : dict
        A dictionary containing the parameters for the simulation case.
    output_dir : str
        The directory where the output files will be saved.
    steering_folder : str
        The folder containing the steering files.

    Returns
    -------
    bool
        True if the simulation was run, False if it was skipped.
    """
    name, ext = os.path.splitext(filename)
    result_file = os.path.join(output_dir, f"{name}.txt")
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
        prepare_steering_file(src_file, dst_file, result_file, i, False)
        run_telemac2d(filename, output_dir)
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    finally:
        move_file(dst_file, src_file)

    return True

def run_telemac2d_on_files(start, end, parameters, output_dir, steering_folder):
    """
    Run Telemac2D simulations on a range of files.

    Parameters
    ----------
    start : int
        The starting index of the simulation cases.
    end : int
        The ending index of the simulation cases.
    parameters : pandas.DataFrame
        A DataFrame containing the parameters for each simulation case.
    output_dir : str
        The directory where the output files will be saved.
    steering_folder : str
        The folder containing the steering files.

    Returns
    -------
    None
    """
    setup_output_dir(output_dir)

    # Filter the parameters DataFrame to match the selected range.
    filtered_parameters = parameters.iloc[start:end]
    total_files = len(filtered_parameters)

    with tqdm(total=total_files, unit="case", dynamic_ncols=True) as pbar:
        for i, (index, case_parameters) in enumerate(filtered_parameters.iterrows()):
            filename = f"{index}.cas"  # Use the original index as part of the filename

            pbar.set_description(f"Processing {filename}")
            simulation_run = run_single_simulation(
                index, filename, case_parameters, output_dir, steering_folder
            )

            if simulation_run:
                pbar.update(1)
            else:
                pbar.total -= 1
                pbar.refresh()

    logger.info("All Telemac2D simulations completed")
