import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import pandas as pd
from loguru import logger

from modules.file_processing import process_and_save, process_and_save_normalized
from modules.statistics import (
    calculate_statistics,
    combine_statistics,
    normalize_statistics,
)

@dataclass
class ProcessingConfig:
    base_dir: str
    generate_normalized: bool = False
    separate_critical_states: bool = False
    bottom_types: Optional[List[str]] = (
        None  # Specify which bottom types to process (None for all)
    )

def get_output_files(config: ProcessingConfig) -> Dict[str, Dict[str, str]]:
    base_path = os.path.join(config.base_dir, "ML")
    os.makedirs(base_path, exist_ok=True)

    bottom_types = config.bottom_types or ["NOISE", "SLOPE", "BUMP", "BARS"]

    files = {}
    for bottom_type in bottom_types:
        files[bottom_type] = {
            "main": os.path.join(base_path, f"simulation_data_{bottom_type}.hdf5"),
            "normalized": os.path.join(
                base_path, f"simulation_data_normalized_{bottom_type}.hdf5"
            ),
        }

        if config.separate_critical_states:
            for data_type in ["main", "normalized"]:
                base_name = files[bottom_type][data_type]
                files[bottom_type][f"{data_type}_subcritical"] = base_name.replace(
                    ".hdf5", "_subcritical.hdf5"
                )
                files[bottom_type][f"{data_type}_supercritical"] = base_name.replace(
                    ".hdf5", "_supercritical.hdf5"
                )

    return files

def load_and_validate_data(
    config: ProcessingConfig,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    parameters_file = os.path.join(config.base_dir, "parameters.csv")
    results_dir = os.path.join(config.base_dir, "results")

    if not os.path.exists(parameters_file):
        raise FileNotFoundError(f"Parameters file not found: {parameters_file}")
    if not os.path.isdir(results_dir):
        raise NotADirectoryError(f"Results directory not found: {results_dir}")

    parameters = pd.read_csv(parameters_file)
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".slf")]

    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    return parameters, result_files, ["F", "B", "H", "Q", "S", "U", "V"]

def prepare_parameter_table(
    parameters: pd.DataFrame, parameter_names: List[str]
) -> pd.DataFrame:
    parameter_stats = {param: parameters[param].describe() for param in parameter_names}
    parameter_table = (
        pd.DataFrame(parameter_stats)
        .T.reset_index()
        .rename(columns={"index": "names", "std": "variance"})
    )
    parameter_table["variance"] = parameter_table["variance"] ** 2
    return parameter_table

def process_data(
    config: ProcessingConfig,
    files: Dict[str, Dict[str, str]],
    parameters: pd.DataFrame,
    result_files: List[str],
    variable_names: List[str],
    parameter_names: List[str],
):
    parameter_table = prepare_parameter_table(parameters, parameter_names)
    bottom_types = config.bottom_types or ["NOISE", "SLOPE", "BUMP", "BARS", "STEP"]

    for bottom_type in bottom_types:
        if config.separate_critical_states:
            for critical_state in ["subcritical", "supercritical"]:
                mask = (
                    parameters["subcritical"]
                    if critical_state == "subcritical"
                    else ~parameters["subcritical"]
                )
                filtered_parameters = parameters[mask]
                filtered_result_files = [f for f, m in zip(result_files, mask) if m]

                with h5py.File(
                    files[bottom_type][f"main_{critical_state}"], "w"
                ) as hdf5_file:
                    process_and_save(
                        hdf5_file,
                        bottom_type.upper(),
                        config.base_dir,
                        filtered_parameters,
                        parameter_table,
                        parameter_names,
                        variable_names,
                        filtered_result_files,
                    )

                if config.generate_normalized:
                    with h5py.File(
                        files[bottom_type][f"normalized_{critical_state}"], "w"
                    ) as hdf5_file_norm:
                        process_and_save_normalized(
                            hdf5_file_norm,
                            bottom_type.upper(),
                            None,
                            config.base_dir,
                            filtered_parameters,
                            parameter_names,
                            variable_names,
                            filtered_result_files,
                        )
        else:
            with h5py.File(files[bottom_type]["main"], "w") as hdf5_file:
                table = process_and_save(
                    hdf5_file,
                    bottom_type.upper(),
                    config.base_dir,
                    parameters,
                    parameter_table,
                    parameter_names,
                    variable_names,
                    result_files,
                )

            if config.generate_normalized:
                with h5py.File(files[bottom_type]["normalized"], "w") as hdf5_file_norm:
                    process_and_save_normalized(
                        hdf5_file_norm,
                        bottom_type.upper(),
                        table,
                        config.base_dir,
                        parameters,
                        parameter_names,
                        variable_names,
                        result_files,
                    )

def process_simulation_results(config: ProcessingConfig) -> None:
    logger.info("Processing started")

    try:
        parameters, result_files, variable_names = load_and_validate_data(config)
        parameter_names = ["H0", "Q0", "SLOPE", "n"]
        files = get_output_files(config)

        process_data(
            config, files, parameters, result_files, variable_names, parameter_names
        )

        logger.success("Processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")

def parse_args() -> ProcessingConfig:
    parser = argparse.ArgumentParser(description="Process simulation data.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default= "telemac",
        help="Base directory for the simulation data",
    )
    parser.add_argument(
        "--generate_normalized",
        default=False,
        action="store_true",
        help="Flag to generate normalized data",
    )
    parser.add_argument(
        "--separate_critical_states",
        default=False,
        action="store_true",
        help="Flag to separate subcritical and supercritical states",
    )
    parser.add_argument(
        "--bottom_types",
        type=str,
        nargs="+",
        choices=["NOISE", "SLOPE", "BUMP", "BARS"],
        help="Bottom types to process (e.g., --bottom_types noise slope). Default is all.",
    )

    args = parser.parse_args()

    return ProcessingConfig(
        base_dir=args.base_dir,
        generate_normalized=args.generate_normalized,
        separate_critical_states=args.separate_critical_states,
        bottom_types=args.bottom_types,
    )

if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log", format="{time} {level} {message}", level="INFO"
    )

    config = parse_args()
    process_simulation_results(config)
