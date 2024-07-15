import os
import pandas as pd
from loguru import logger
import h5py
from modules.file_processing import process_and_save, process_and_save_normalized
from modules.statistics import calculate_statistics, combine_statistics, normalize_statistics


def process_simulation_results(
    base_dir: str, 
    output_file_noise: str, 
    output_file_flat: str, 
    normalized_output_file_noise: str, 
    normalized_output_file_flat: str
):
    logger.info("Processing start")

    parameters = pd.read_csv(os.path.join(base_dir, "parameters.csv"))
    parameter_names = ["H0", "Q0", "SLOPE", "n"]
    parameter_stats = {param: parameters[param].describe() for param in parameter_names}
    parameter_table = (
        pd.DataFrame(parameter_stats)
        .T.reset_index()
        .rename(columns={"index": "names", "std": "variance"})
    )
    parameter_table["variance"] = parameter_table["variance"] ** 2

    variable_names = ["F", "B", "H", "Q", "S", "U", "V"]
    result_files = [
        f
        for f in os.listdir(os.path.join(base_dir, "results"))
        if f.startswith("results_") and f.endswith(".slf")
    ]

    with h5py.File(output_file_noise, "w") as hdf5_file_noise, \
         h5py.File(output_file_flat, "w") as hdf5_file_flat:
        table_noise = process_and_save(hdf5_file_noise, "NOISE", base_dir, parameters, parameter_table, parameter_names, variable_names, result_files)
        table_flat = process_and_save(hdf5_file_flat, "FLAT", base_dir, parameters, parameter_table, parameter_names, variable_names, result_files)

    with h5py.File(normalized_output_file_noise, "w") as hdf5_file_noise_norm, \
         h5py.File(normalized_output_file_flat, "w") as hdf5_file_flat_norm:
        process_and_save_normalized(hdf5_file_noise_norm, "NOISE", table_noise, base_dir, parameters, parameter_names, variable_names, result_files)
        process_and_save_normalized(hdf5_file_flat_norm, "FLAT", table_flat, base_dir, parameters, parameter_names, variable_names, result_files)

    logger.success("Script completed successfully")


if __name__ == "__main__":
    logger.remove()
    logger.add(
        "simulation_data_processor.log", format="{time} {level} {message}", level="INFO"
    )

    BASE_DIR = "telemac"
    OUTPUT_FILE_NOISE = "ML/simulation_data_noise.hdf5"
    OUTPUT_FILE_FLAT = "ML/simulation_data_flat.hdf5"
    NORMALIZED_OUTPUT_FILE_NOISE = "ML/simulation_data_normalized_noise.hdf5"
    NORMALIZED_OUTPUT_FILE_FLAT = "ML/simulation_data_normalized_flat.hdf5"

    process_simulation_results(BASE_DIR, OUTPUT_FILE_NOISE, OUTPUT_FILE_FLAT, NORMALIZED_OUTPUT_FILE_NOISE, NORMALIZED_OUTPUT_FILE_FLAT)
