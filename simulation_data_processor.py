import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

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
    channel_length: float = 12
    channel_width: float = 0.3
    bottom_types: Optional[List[str]] = None
    parameter_names: Optional[List[str]] = None
    variable_names: Optional[List[str]] = None

def load_yaml_config(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)

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

    # Compute adimensional numbers
    def compute_adimensional_numbers(row):
        g = 9.81
        xc, yc = config.channel_length, config.channel_width
        bc, hc = row["SLOPE"] * xc, row["H0"]
        uc = row["Q0"] / (hc * yc)
        nut = row["nut"]

        return {
            "Ar": xc / yc,
            "Vr": 1,
            "Fr": uc / (g * hc) ** 0.5,
            "Hr": bc / hc,
            "Re": (uc * xc) / nut,
            "M": g * row["n"]**2 * xc / (hc ** (4 / 3)),
        }

    adim_params = parameters.apply(compute_adimensional_numbers, axis=1, result_type="expand")
    parameters = pd.concat([parameters, adim_params], axis=1)
    return (
        parameters,
        result_files,
        config.variable_names or ["H", "U", "V"],
    )

def prepare_parameter_table(parameters: pd.DataFrame, parameter_names: List[str], config: ProcessingConfig) -> pd.DataFrame:
    # Compute stats for all parameters (including adimensional)
    parameter_stats = {param: parameters[param].describe() for param in parameters.columns if param in parameter_names}
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
    parameter_table = prepare_parameter_table(parameters, parameter_names, config)
    bottom_types = config.bottom_types or ["NOISE", "SLOPE", "BUMP", "BARS"]

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
                        config.channel_width,
                        config.channel_length,
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
                            config.channel_width,
                            config.channel_length,
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
                    config.channel_width,
                    config.channel_length,
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
                        config.channel_width,
                        config.channel_length,
                        parameters,
                        parameter_names,
                        variable_names,
                        result_files,
                    )

def process_simulation_results(config: ProcessingConfig, yaml_config: Dict) -> None:
    logger.info("Processing started")

    try:
        config.bottom_types = yaml_config.get(
            "bottom_types", ["NOISE", "SLOPE", "BUMP", "BARS"]
        )
        config.parameter_names = yaml_config.get(
            "parameter_names", ["H0", "Q0", "SLOPE", "n"]
        )
        config.variable_names = yaml_config.get("variable_names", ["H", "U", "V"])
        
        config.channel_length = yaml_config.get("channel_length", 12)

        config.channel_width = yaml_config.get("channel_width", 12)

        parameters, result_files, variable_names = load_and_validate_data(config)
        files = get_output_files(config)

        process_data(
            config,
            files,
            parameters,
            result_files,
            variable_names,
            config.parameter_names,
        )

        logger.success("Processing completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")

def parse_args() -> Tuple[ProcessingConfig, str]:
    parser = argparse.ArgumentParser(description="Process simulation data.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="telemac",
        help="Base directory for the simulation data",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file",
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

    args = parser.parse_args()

    return (
        ProcessingConfig(
            base_dir=args.base_dir,
            generate_normalized=args.generate_normalized,
            separate_critical_states=args.separate_critical_states,
        ),
        args.config_file,
    )

if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {function} | {line} | {message}",
        level="DEBUG",
        rotation="00:00",  # Rotate the log file at midnight
        retention= 1,  # Keep only the most recent log file
    )

    config, config_file = parse_args()
    yaml_config = load_yaml_config(config_file)
    process_simulation_results(config, yaml_config)
