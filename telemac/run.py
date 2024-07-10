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
