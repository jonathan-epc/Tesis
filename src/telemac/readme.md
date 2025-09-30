# Telemac Simulation Automation and Analysis

## Overview

This project provides a pipeline for automating the setup, execution, and basic analysis of Telemac2D hydraulic simulations. It allows users to:

1.  Generate a large set of simulation input parameters using sampling techniques (Latin Hypercube).
2.  Automatically create the necessary input files for Telemac2D based on these parameters, including geometry files (`.slf`) with various bottom configurations and steering files (`.cas`).
3.  Run the Telemac2D simulations in batch, with options for filtering based on parameters or previous run status (flux balance).
4.  Provide utilities for reading and visualizing simulation results.

The system is designed to handle variations in parameters like slope, roughness, flow rate, initial water depth, and bottom geometry types (e.g., FLAT, SLOPE, NOISE, BUMP, BARS, STEP). It also supports handling parameters in both dimensional and adimensional forms.

## Features

*   **Parameter Generation:** Creates `parameters.csv` with combinations of input parameters (Slope, Manning's n, Flow Rate Q0, Initial Depth H0) using Latin Hypercube sampling. Supports dimensional and adimensional parameter workflows.
*   **Input File Generation:** Automatically generates:
    *   Geometry files (`.slf`) based on a base mesh and specified bottom type (SLOPE, NOISE, BUMP, BARS, STEP, etc.) using `GeometryGenerator`.
    *   Steering files (`.cas`) using a template and generated parameters via `SteeringFileGenerator`.
    *   Boundary condition files (`.cli`) selection based on flow direction (`BoundaryConditions`).
*   **Simulation Execution:** Runs Telemac2D simulations sequentially using `run_telemac_simulations.py`.
    *   Command-line interface to specify simulation ranges, filter by bottom type, set logging levels, and perform dry runs.
    *   Optionally checks for flux balance in previous simulation logs (`.txt` files in `telemac_logs/`) and skips or re-runs simulations accordingly (`FluxChecker`).
    *   Handles simulation continuation or duration updates based on previous results.
*   **Result Analysis:** Includes scripts (`results_reading.py`, `exploring.py`) for:
    *   Reading Telemac result files (`.slf`) using `xarray` and potentially a custom `TelemacFile` class.
    *   Plotting 1D water depth profiles along polylines.
    *   Generating 2D contour plots of variables like bottom elevation, velocity components.
*   **Configuration:** Uses `constants.yml` for primary configuration (parameter ranges, channel dimensions, mesh details, seed) and `run_configurations.py` for specific paths.
*   **Logging:** Comprehensive logging using Loguru via `logger_config.py`.

## Installation & Setup

1.  **Prerequisites:**
    *   Python 3.x
    *   Telemac-Mascaret suite installed and configured in your environment (i.e., `telemac2d.py` command should be accessible).
    *   Required Python libraries (see Dependencies).

2.  **Get the Code:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

3.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt # TODO: Create requirements.txt
    ```
    *See Dependencies section below for a list of required packages.*

4.  **Configuration Files:**
    *   Ensure the main configuration file `constants.yml` exists and is populated with desired settings (parameter ranges, channel details, mesh paths, etc.).
    *   Ensure the base mesh file (e.g., `geometry/mesh_3x3.slf`) specified in `constants.yml` exists.
    *   Ensure the steering file template `steering_template.txt` exists in the root directory.
    *   Ensure necessary boundary condition files (e.g., `boundary/3x3_tor.cli`, `boundary/3x3_riv.cli`) exist.

5.  **Directories:** Create the necessary input/output directories if they don't exist (though the scripts might create some):
    ```bash
    mkdir geometry steering boundary results telemac_logs logs img
    ```
    *   Place your base mesh (`mesh_3x3.slf`) in `geometry/`.
    *   Place your boundary files (`.cli`) in `boundary/`.

## Usage

### 1. Generating Input Files

Use `input_generator.py` to create the `parameters.csv` file and the corresponding geometry (`.slf`) and steering (`.cas`) files.

```bash
python input_generator.py [OPTIONS]
```

**Options:**

*   `--mode [new|read|add]`:
    *   `new`: Generate a completely new `parameters.csv` and associated files. Overwrites existing `parameters.csv`.
    *   `read`: Only read the existing `parameters.csv` (useful for debugging setup). Fails if `parameters.csv` doesn't exist.
    *   `add` (default): Load existing `parameters.csv` (if any), generate new parameters, combine them (removing duplicates), and save. Generates files for the new parameters.
*   `--sample_size <N>`: The base number of samples for parameter generation (default: 179). The actual number of parameter sets generated before adding bottom types might be N*N.
*   `--overwrite`: If set, existing geometry and steering files will be overwritten during generation. Otherwise, existing files are skipped.

**Example:** Generate inputs, adding to any existing parameters, using a base sample size of 50, and overwrite existing files.

```bash
python input_generator.py --mode add --sample_size 50 --overwrite
```

### 2. Running Telemac Simulations

Use `run_telemac_simulations.py` to execute the simulations defined by the `.cas` files in the `steering/` directory.

```bash
python run_telemac_simulations.py [OPTIONS]
```

**Options:**

*   `--start <index>`: Starting index (row number in `parameters.csv`, 0-based) of simulations to run (default: 0).
*   `--end <index>`: Ending index (exclusive) of simulations to run (default: run all from start).
*   `--bottom [SLOPE|NOISE|BUMP|BARS|STEP]`: Run only simulations corresponding to a specific `BOTTOM` type defined in `parameters.csv`.
*   `--output-dir <path>`: Directory to store Telemac log files (`.txt`) (default: `telemac_logs`). Telemac result files (`.slf`) are typically saved in `results/` as defined in the `.cas` files.
*   `--log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]`: Set the logging level for the script (default: INFO).
*   `--dry-run`: Print the files that would be processed without actually running Telemac.
*   `--adimensional`: Filter `.cas` files to only include those starting with 'a' (intended for adimensional runs).

**Example:** Run simulations for cases 50 to 99 (exclusive) that use the 'BUMP' bottom type.

```bash
python run_telemac_simulations.py --start 50 --end 100 --bottom BUMP
```

**Example:** Perform a dry run for all adimensional simulations.

```bash
python run_telemac_simulations.py --adimensional --dry-run
```

### 3. Analyzing Results

The scripts `results_reading.py` and `exploring.py` contain functions for plotting. They appear designed for interactive use (e.g., in a Jupyter Notebook or IPython session).

*   **`results_reading.py`:**
    *   `read_parameters()`: Loads `parameters.csv`.
    *   `plot_results(indices, parameters_df)`: Plots 1D water depth profiles for specified case indices, comparing with normal and critical depths.
    *   `plot_results_2d(index)`: Creates 2D contour plots for a given case index (Note: multiple definitions exist in the provided code).
*   **`exploring.py`:** Contains various functions for parameter exploration, sensitivity analysis (e.g., critical slope plotting), and preliminary result plotting similar to `results_reading.py`.

**Example (Conceptual - Run in Python/IPython/Jupyter):**

```python
import pandas as pd
from results_reading import plot_results, plot_results_2d # Assuming functions are importable

parameters_df = pd.read_csv("parameters.csv", index_col="id")

# Plot water depth for cases 0, 10, 20
plot_results([0, 10, 20], parameters_df)

# Plot 2D results for case 5
plot_results_2d(5)
```

## Project Structure

```
.
├── boundary/                 # Boundary condition files (.cli)
│   ├── 3x3_riv.cli
│   └── 3x3_tor.cli
├── geometry/                 # Geometry files (.slf)
│   └── mesh_3x3.slf          # Base mesh
├── img/                      # Saved output plots
├── logs/                     # Log files from these Python scripts
├── modules/                  # Reusable Python modules
│   ├── boundary_conditions.py
│   ├── environment_setup.py
│   ├── file_handler.py
│   ├── file_utils.py
│   ├── flux_checker.py
│   ├── geometry_generator.py
│   ├── hydraulic_calculations.py
│   ├── param_utils.py
│   ├── parameter_manager.py
│   ├── sample_generator.py
│   ├── simulation_runner.py
│   ├── steering_file_generator.py
│   └── telemac_runner.py
├── results/                  # Telemac simulation results (.slf) - OUTPUT
├── steering/                 # Generated steering files (.cas) - OUTPUT
├── telemac_logs/             # Telemac execution logs (.txt) - OUTPUT
├── constants.py              # Physical constants (e.g., GRAVITY)
├── constants.yml             # Main configuration file
├── exploring.py              # Script for exploration and sensitivity analysis
├── input_generator.py        # Script to generate inputs (parameters.csv, .slf, .cas)
├── logger_config.py          # Logging setup using Loguru
├── parameters.csv            # Generated simulation parameters - OUTPUT
├── README.md                 # This documentation file
├── results_reading.py        # Script for reading/plotting results
├── run_configurations.py     # Configuration for the simulation runner script
├── run_telemac_simulations.py # Main script to run simulations
└── steering_template.txt     # Template for generating .cas files```

## Dependencies

*   Python 3.x
*   numpy
*   pandas
*   scipy
*   matplotlib
*   xarray (likely requires `netcdf4` and `h5netcdf` as well)
*   PyYAML
*   Loguru
*   tqdm
*   IPython (for `results_reading.py` display functions)
*   **Telemac-Mascaret Suite** (External dependency)

*TODO: Create a `requirements.txt` file.*
```bash
pip install numpy pandas scipy matplotlib xarray PyYAML loguru tqdm ipython netcdf4 h5netcdf
```

## Suggestions & TODOs

*   **Refactoring:**
    *   Consolidate the multiple `plot_results_2d` function definitions in `results_reading.py`.
    *   Review `exploring.py`: Much of its functionality (hydraulic calculations, parameter generation, steering file logic) seems duplicated or superseded by dedicated modules (`hydraulic_calculations.py`, `parameter_manager.py`, `steering_file_generator.py`). Clean up `exploring.py` to remove redundancy, perhaps keeping only the unique exploration/plotting parts.
    *   Consolidate flux checking logic: `filter_unbalanced_files` in `run_telemac_simulations.py` seems redundant with checks inside `run_single_simulation` (in `simulation_runner.py`). Streamline this.
*   **Configuration:** Centralize configuration. Move paths defined in `run_configurations.py` (like `OUTPUT_FOLDER`, `STEERING_FOLDER`) into `constants.yml` for a single config file.
*   **Requirements File:** Create a `requirements.txt` file for easier dependency management.
*   **Error Handling:** Improve robustness. Add checks for the existence of required files like `steering_template.txt`, base mesh, boundary files before starting generation or simulation runs. Provide more informative error messages if Telemac fails.
*   **Documentation:**
    *   Add detailed docstrings to complex classes and functions (e.g., `ParameterManager`, `GeometryGenerator`, dimensional/adimensional logic).
    *   Clarify the role and source/definition of the `TelemacFile` class used in `results_reading.py` and `exploring.py`. If it's custom code, include it; if it's from a library, specify it.
*   **Testing:** Implement unit tests, especially for critical modules like `hydraulic_calculations.py`, `parameter_manager.py`, and `geometry_generator.py`, to ensure correctness.
*   **Parallelism:** The `run_telemac_simulations.py` script runs simulations sequentially. For large batches, this can be slow. Explore parallel execution of independent Telemac runs using Python's `multiprocessing` module or tools like `joblib`.
*   **Result Analysis:** Make the plotting functionalities in `results_reading.py` more robust and potentially scriptable (e.g., add command-line arguments to generate specific plots).
*   **Adimensional Workflow:** The handling of adimensional parameters adds complexity. Ensure this workflow is clearly documented, robustly implemented, and indexing (`a{id}` vs numeric) is consistent across scripts.
*   **Code Style:** Ensure consistent code style (e.g., using a linter like Flake8 or Black).

## License

*TODO: Add a license file (e.g., MIT, Apache 2.0) and reference it here.*
