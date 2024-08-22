import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

def get_output_files(config: ProcessingConfig) -> Dict[str, Dict[str, str]]:
    base_path = os.path.join(config.base_dir, "ML")
    os.makedirs(base_path, exist_ok=True)

    files = {
        "noise": {
            "main": os.path.join(base_path, "simulation_data_noise.hdf5"),
            "normalized": os.path.join(
                base_path, "simulation_data_normalized_noise.hdf5"
            ),
        },
        "flat": {
            "main": os.path.join(base_path, "simulation_data_flat.hdf5"),
            "normalized": os.path.join(
                base_path, "simulation_data_normalized_flat.hdf5"
            ),
        },
    }

    if config.separate_critical_states:
        for flow_type in ["noise", "flat"]:
            for data_type in ["main", "normalized"]:
                base_name = files[flow_type][data_type]
                files[flow_type][f"{data_type}_subcritical"] = base_name.replace(
                    ".hdf5", "_subcritical.hdf5"
                )
                files[flow_type][f"{data_type}_supercritical"] = base_name.replace(
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

    for flow_type in ["noise", "flat"]:
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
                    files[flow_type][f"main_{critical_state}"], "w"
                ) as hdf5_file:
                    process_and_save(
                        hdf5_file,
                        flow_type.upper(),
                        config.base_dir,
                        filtered_parameters,
                        parameter_table,
                        parameter_names,
                        variable_names,
                        filtered_result_files,
                    )

                if config.generate_normalized:
                    with h5py.File(
                        files[flow_type][f"normalized_{critical_state}"], "w"
                    ) as hdf5_file_norm:
                        process_and_save_normalized(
                            hdf5_file_norm,
                            flow_type.upper(),
                            None,
                            config.base_dir,
                            filtered_parameters,
                            parameter_names,
                            variable_names,
                            filtered_result_files,
                        )
        else:
            with h5py.File(files[flow_type]["main"], "w") as hdf5_file:
                table = process_and_save(
                    hdf5_file,
                    flow_type.upper(),
                    config.base_dir,
                    parameters,
                    parameter_table,
                    parameter_names,
                    variable_names,
                    result_files,
                )

            if config.generate_normalized:
                with h5py.File(files[flow_type]["normalized"], "w") as hdf5_file_norm:
                    process_and_save_normalized(
                        hdf5_file_norm,
                        flow_type.upper(),
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

if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log", format="{time} {level} {message}", level="INFO"
    )

    config = ProcessingConfig(
        base_dir="telemac", generate_normalized=False, separate_critical_states=True
    )

    process_simulation_results(config)
