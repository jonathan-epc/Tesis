# Surrogate Modeling of Channel Flow using Fourier Neural Operators

This repository contains the complete codebase for the Master's thesis and scientific article titled "[**Your Paper Title Here**]". The project leverages a Fourier Neural Operator (FNO) to create a surrogate model for TELEMAC-2D simulations of shallow water channel flow. It addresses both direct problems (predicting flow fields) and inverse problems (inferring channel geometry and hydraulic parameters).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Project Goal](#project-goal)
- [Methodology Overview](#methodology-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Workflow and Usage](#workflow-and-usage)
  - [1. Data Generation (TELEMAC-2D)](#1-data-generation-telemac-2d)
  - [2. Data Processing](#2-data-processing)
  - [3. Model Training & Optimization](#3-model-training--optimization)
- [How to Cite](#how-to-cite)
- [License](#license)

## Project Goal

Traditional Computational Fluid Dynamics (CFD) solvers for hydrodynamic problems are computationally expensive. This research explores the use of data-driven surrogate models, specifically Fourier Neural Operators (FNOs), to accelerate simulations and solve complex inverse problems in channel flow dynamics.

The key objectives are:
- To build a robust FNO-based model capable of predicting steady-state flow fields (water depth, velocity) under varying hydraulic and geometric conditions.
- To solve the inverse problem: inferring channel bed geometry and hydraulic parameters (e.g., Manning's coefficient) from observable flow velocity fields.
- To systematically compare purely data-driven models against physics-informed approaches and evaluate the impact of non-dimensionalization on model performance.
- To analyze the model's generalizability across different channel bed complexities (Slope, Noise, Bars).

## Methodology Overview

The workflow begins by generating a comprehensive dataset using the **TELEMAC-2D** solver. A wide parameter space, including channel slope, inflow, water height, and Manning's roughness, is explored using Latin Hypercube Sampling.

The generated data is then used to train a **Fourier Neural Operator (FNO)**, a deep learning architecture particularly effective for learning resolution-invariant solutions to PDEs. The training process is managed with **PyTorch** and automated hyperparameter optimization is performed using **Optuna**.

Key methodological features include:
- **Physics-Informed Learning:** A custom loss function incorporating the shallow water equation residuals was implemented and evaluated.
- **Dynamic Loss Weighting:** The ReLoBRaLo algorithm was used to dynamically balance the data-driven and physics-based components of the loss function.
- **Transfer Learning:** Experiments were conducted to assess knowledge transfer between models trained on different bed geometries.

For a complete description of the methodology, please refer to the scientific paper.

## Repository Structure

The project is organized into distinct, modular components:

Tesis/
├── common/              # Shared utility functions (logger, seeder)
├── data/                # Processed HDF5 datasets for training
├── logs/                # Log files from script executions
├── ML/                  # Machine Learning source code
│   ├── core/            # Core classes for training, optimization, etc.
│   ├── modules/         # Low-level modules (data loader, models, loss)
│   └── scripts/         # High-level executable scripts
├── plots/               # Saved plots and figures
├── savepoints/          # Saved model checkpoints (.pth files)
├── studies/             # Optuna study databases (.db files)
├── telemac/             # TELEMAC-2D data generation source code
│   ├── modules/         # Modules for case generation, file writing, etc.
│   └── geometry/        # Base mesh file
├── .gitignore
├── environment.yml      # The definitive Conda environment file
├── nconfig.yml          # The SINGLE configuration file for the entire project
└── README.md            # This file

## Installation

The project uses Conda for environment management. To set up the environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url.git
    cd Tesis
    ```

2.  **Create the Conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate ML-Tesis
    ```

4.  **Perform an "Editable Install":** This crucial step makes your project's source code in the `src/` directory importable by Python. **You must run this command from the project root.**
    ```bash
    pip install -e .
    ```

5.  **(Optional) Set up WandB:** If you wish to use Weights & Biases for logging, set your API key as an environment variable.
    ```bash
    export WANDB_API_KEY="your_key_here"
    ```

## Workflow and Usage

The project workflow is divided into three main stages. All experiments are controlled by editing the central **`nconfig.yml`** file.

### 1. Data Generation (TELEMAC-2D)

This step generates the steering and geometry files required to run the TELEMAC-2D simulations.

- **Configure:** Edit `nconfig.yml` to set the desired `simulation_params` (e.g., `bottom_types`, parameter ranges).
- **Run:** Execute the input generator script from the project root.
  ```bash
  python telemac/input_generator.py --mode new --sample_size 1000 --overwrite
  ```
- **Execute Simulations:** This step requires a local installation of TELEMAC-2D. Run the simulations using the generated files.
  ```bash
  python telemac/run_telemac_simulations.py
  ```

### 2. Data Processing

This step processes the raw `.slf` output from TELEMAC-2D into a single, analysis-ready HDF5 file for each geometry type.

- **Run:**
  ```bash
  python simulation_data_processor.py
  ```
- **Output:** This will create the `.hdf5` files inside the `data/` directory.

### 3. Model Training & Optimization

All machine learning tasks are managed via the scripts in `ML/scripts/`. **Before running any script, configure `nconfig.yml`** to define the specific experiment (e.g., set the correct `data.file_path`, `data.inputs`, `data.outputs`, and `optuna.study_name`).

- **To run a single training session:**
  ```bash
  python ML/scripts/train_model.py
  ```

- **To run a full hyperparameter search with Optuna:**
  ```bash
  python ML/scripts/run_hyperparameter_search.py
  ```

- **To re-run a specific trial from a completed study:**
  ```bash
  # First, configure nconfig.yml to point to the correct study and dataset
  python ML/scripts/run_trial_repeat.py <TRIAL_ID>
  ```

- **To run all transfer learning experiments:**
  ```bash
  python ML/scripts/run_transfer_learning.py
  ```

## How to Cite

If you use this work in your research, please cite the following paper:

```bibtex
@article{your_citation_key_here,
  title   = {{Your Paper Title Here}},
  author  = {{Your Name(s) Here}},
  journal = {{Journal/Conference Name Here}},
  year    = {2025},
  ...
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
