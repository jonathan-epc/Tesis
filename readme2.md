# Telemac-ML Hydraulic Simulation Surrogate Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges as needed: build status, code coverage, etc. -->
<!-- [![Build Status]([Link to Build Status])]() -->

A project coupling Telemac2D hydraulic simulations with Machine Learning (specifically Fourier Neural Operators) to create surrogate models for predicting hydrodynamic variables under various conditions.

## Table of Contents

*   [Features](#features)
*   [Project Structure](#project-structure)
*   [Installation](#installation)
    *   [Dependencies](#dependencies)
    *   [Environment Setup](#environment-setup)
*   [Configuration](#configuration)
*   [Usage](#usage)
    *   [1. Generating Simulation Cases](#1-generating-simulation-cases)
    *   [2. Running Telemac Simulations](#2-running-telemac-simulations)
    *   [3. Processing Simulation Data](#3-processing-simulation-data)
    *   [4. Training the ML Model](#4-training-the-ml-model)
    *   [5. Evaluating the ML Model](#5-evaluating-the-ml-model)
*   [Modules Overview](#modules-overview)
*   [Known Issues & TODO](#known-issues--todo)
*   [Contributing](#contributing)
*   [License](#license)
*   [Contact](#contact)

## Features

*   **Hydraulic Simulation:** Leverages Telemac2D for simulating 2D free-surface flows.
*   **Parametric Study:** Generates diverse simulation scenarios by varying parameters (e.g., flow rate, roughness, slope, bottom geometry).
*   **Data Processing:** Efficiently processes raw Telemac `.slf` output files into ML-ready HDF5 datasets.
*   **Surrogate Modeling:** Employs Fourier Neural Operators (FNO) using PyTorch and `neuralop` to learn the mapping from input parameters/fields to output hydrodynamic fields.
*   **Physics-Informed ML:** Incorporates physical constraints (Saint-Venant equations) into the loss function for potentially improved accuracy and generalization.
*   **Advanced Training:** Features a robust training pipeline including:
    *   K-Fold Cross-Validation
    *   Hyperparameter Optimization (Optuna)
    *   Experiment Tracking (WandB)
    *   Mixed-Precision Training
    *   Early Stopping
    *   Dynamic Loss Weighting (ReLoBRaLo)
*   **Data Handling:** Supports normalization, Box-Cox transformation, preloading, and chunking for large datasets.

## Project Structure

```
.
├── ML/                     # Machine Learning components
│   ├── modules/            # ML specific modules (data, models, loss, training, etc.)
│   ├── *.py                # ML scripts (main training, evaluation, etc.)
│   ├── *.ipynb             # Jupyter notebooks for ML exploration/visualization
│   └── config.yaml         # ML specific configuration (-> To be consolidated)
├── telemac/                # Telemac simulation components
│   ├── modules/            # Simulation specific modules (geometry, parameters, etc.)
│   ├── boundary/           # Boundary condition files (*.cli)
│   ├── geometry/           # Geometry/mesh files (*.slf)
│   ├── steering/           # Generated steering files (*.cas)
│   ├── logs/               # Simulation log files
│   ├── *.py                # Simulation scripts (input generation, run script)
│   └── *.ipynb             # Jupyter notebooks for simulation setup/analysis
├── modules/                # Top-level modules (data processing utilities) -> Potential Refactor
├── data/                   # Directory for storing processed HDF5 data (Example)
├── savepoints/             # Directory for saving trained model checkpoints
├── plots/                  # Directory for saving generated plots
├── studies/                # Directory for Optuna study databases
├── .gitignore              # Specifies intentionally untracked files
├── config.yml              # Main configuration file (-> To be consolidated)
├── environment.yml         # Conda environment definition
├── LICENSE                 # Project license file (e.g., MIT)
├── README.md               # This file
├── requirements.txt        # Pip requirements file
└── simulation_data_processor.py # Script to process simulation output for ML
```

## Installation

### Dependencies

*   **Telemac-Mascaret:** Requires a working installation of the Telemac-Mascaret suite. Please refer to the official [Telemac installation guide](http://www.opentelemac.org/). Ensure the necessary environment variables (`PYTHONPATH`, `SYSTELCFG`) are set correctly.
*   **Python:** Python 3.8+ is recommended.
*   **Other Libraries:** Primarily managed via Conda and Pip. Key libraries include PyTorch, `neuralop`, `h5py`, `xarray`, `selafin_tools` (or equivalent for reading `.slf`), `numpy`, `pandas`, `scipy`, `loguru`, `pyyaml`, `tqdm`, `optuna`, `wandb`, `matplotlib`, `seaborn`, `pydantic`.

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[Your Name/Org]/[Your Repo Name].git
    cd [Your Repo Name]
    ```

2.  **Create and activate Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate <your_env_name> # Replace <your_env_name> with the name defined in environment.yml
    ```
    *Note: Ensure your Telemac installation is compatible with this environment or modify `environment.yml` accordingly.*

3.  **(Optional) Install remaining packages with Pip:** If some packages are not handled by Conda or you prefer Pip for certain packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **WandB Login (Optional):** If using Weights & Biases for logging:
    ```bash
    wandb login
    ```
    You might need to set the `WANDB_API_KEY` environment variable or place it in the config.

## Configuration

*(Note: Configuration files currently need consolidation. The primary configuration for ML is intended to be `ML/config.yaml` loaded via `ML/config.py`. Simulation parameters are partly in `config.yml` and `telemac/constants.yml`.)*

Key configuration files:

*   `ML/config.yaml` (and loaded by `ML/config.py`): Defines ML model architecture, training hyperparameters, data paths, input/output variables, Optuna settings, logging preferences.
*   `config.yml`: Contains channel dimensions, data processing parameters (bottom types, variable names).
*   `telemac/constants.yml`: Contains simulation-specific constants (mesh details, parameter ranges for generation).

**TODO:** Consolidate these into a single, validated configuration system (likely using the Pydantic models in `ML/config.py`).

## Usage

The project involves several steps:

### 1. Generating Simulation Cases

This step creates the necessary input files (geometry, steering files) for various simulation scenarios based on parameter sampling.

```bash
cd telemac
python input_generator.py --mode add --sample_size 100 # Example: add 100^2 new samples
# Use --mode new to overwrite, --mode read to use existing parameters.csv
# Use --overwrite to force regeneration of existing files
cd ..
```
This script uses `telemac/constants.yml` for parameter ranges and generates `parameters.csv` along with files in `telemac/geometry/` and `telemac/steering/`.

### 2. Running Telemac Simulations

This script executes the Telemac2D simulations based on the generated steering files.

```bash
cd telemac
python run_telemac_simulations.py --output-dir telemac_logs
# Optional arguments:
# --start <N> --end <M>: Run cases from index N to M-1
# --bottom <TYPE>: Run only cases with a specific BOTTOM value (e.g., --bottom BARS)
# --dry-run: Show which files would be run without executing
# --adimensional: Filter for files starting with 'a' (if using adimensional IDs)
cd ..
```
This script reads `parameters.csv`, runs `telemac2d` for files in `telemac/steering/`, checks for balanced fluxes in existing log files (in `--output-dir`), and saves simulation results (`.slf`) and logs (`.txt`).

### 3. Processing Simulation Data

This step converts the raw simulation `.slf` output into structured HDF5 files suitable for the ML pipeline.

```bash
python simulation_data_processor.py --base_dir telemac --config_file config.yml --generate_normalized
# Optional arguments:
# --separate_critical_states: Create separate HDF5 files for sub/supercritical states```
This script reads `telemac/parameters.csv` and simulation results from `telemac/results/`, processes the data based on `config.yml`, calculates statistics, handles normalization, and saves output to HDF5 files in the `ML/` directory (e.g., `ML/simulation_data_BARS.hdf5`, `ML/simulation_data_normalized_BARS.hdf5`).

### 4. Training the ML Model

The main script for training, hyperparameter optimization, or repeating specific trials is `ML/main.py`.

*   **Hyperparameter Optimization (Optuna):**
    ```bash
    cd ML
    python main.py --mode hypertuning --config config.yaml
    ```
    This will run an Optuna study defined in `config.yaml`, performing K-fold cross-validation for each trial and saving results to the study database (`studies/`). The best model across all folds of the best trial is saved in `savepoints/`.

*   **Single Training Run (using defaults from config):**
    ```bash
    cd ML
    python main.py --mode training --config config.yaml
    ```
    This performs a single K-fold cross-validation run using the hyperparameters defined directly in `config.yaml` and saves the best model in `savepoints/`.

*   **Repeat a Specific Optuna Trial:**
    ```bash
    cd ML
    python main.py --mode repeat --trial_id <TRIAL_NUMBER> --config config.yaml
    ```
    This re-runs the training using the hyperparameters from the specified Optuna trial number.

### 5. Evaluating the ML Model

The script `ML/see_results.py` can be used to load a trained model and evaluate its predictions on a test set, generating plots and metrics.

```bash
cd ML
# Ensure the study_name and trial_number in see_results.py (or make them args)
# correspond to the model you want to evaluate.
python see_results.py --study <STUDY_TYPE> # e.g., --study ia
cd ..
```
This script loads the best model checkpoint, loads the test dataset, performs inference, calculates metrics (saved to `case_metrics.csv`), and generates various plots (scatter, field comparisons, analysis) saved in `plots/<model_name>/`.

## Modules Overview

*   **`telemac/modules`**: Contains helper modules for simulation setup:
    *   `parameter_manager.py`: Handles generation and loading of simulation parameters.
    *   `geometry_generator.py`: Creates different channel bottom geometries.
    *   `hydraulic_calculations.py`: Performs calculations like normal/critical depth.
    *   `steering_file_generator.py`: Creates Telemac steering files (`.cas`).
    *   `simulation_runner.py`: Logic for running individual simulations, checking status.
    *   `telemac_runner.py`: Interface to execute the `telemac2d` command.
    *   ... and others for boundaries, file handling, etc.
*   **`ML/modules`**: Contains core ML components:
    *   `data.py`: `HDF5Dataset` class for loading data.
    *   `models.py`: FNO model definition (`FNOnet`).
    *   `loss.py`: `PhysicsInformedLoss` including data and physics terms.
    *   `training.py`: `Trainer` class, cross-validation logic, Optuna integration.
    *   `plots.py`: Functions for generating evaluation plots.
    *   `utils.py`: Helper functions (seeding, logging, metrics, early stopping).
    *   `config.py`: Pydantic models for configuration management.
*   **`modules` (Root)**: General utilities used by `simulation_data_processor.py`.
    *   `file_processing.py`: Handles reading `.slf` and writing to HDF5.
    *   `statistics.py`: Statistical calculations.

## Known Issues & TODO

*   **Configuration Consolidation:** Multiple configuration files exist across the project. Need to unify into a single system, likely based on `ML/config.py` (Pydantic).
*   **Redundant Calculations:** Adimensional numbers and potentially other hydraulic properties are calculated multiple times (parameter generation and data processing). This should be done only once.
*   **Lack of Automated Tests:** No unit or integration tests are currently implemented. Tests are needed for core physics calculations, data processing logic, and ML components.
*   **Notebook/Script Synchronization:** Decide on the role of Jupyter notebooks versus Python scripts and ensure consistency or `.gitignore` notebooks if purely exploratory.
*   **Refactor Root `modules`:** Clarify the scope of the top-level `modules` directory or integrate its contents where appropriate.

## Contributing

Contributions are welcome! Please follow standard practices like Forking the repository, creating a new branch for your feature or bug fix, and submitting a Pull Request. [Add more specific contribution guidelines if applicable].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Your Name/Email/Lab] - [Project Link/Paper Citation if available]
