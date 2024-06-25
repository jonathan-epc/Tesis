# Fluid Flow Machine Learning Project

This project aims to train a machine learning model using simulations of fluid flow and evaluate its ability to reproduce the results and predict input variables from the output.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Key Aspects](#key-aspects)
3. [Usage](#usage)
4. [Results](#results)
5. [License](#license)

## Project Structure
- `Avances/`: Contains project progress reports and presentations.
- `ML/`: Contains machine learning model training scripts and data.
- `simulation_data_processor.py`: Script for processing and analyzing simulation results.
- `telemac/`: Contains TELEMAC-2D simulation files, input generators, and result processors.
- `LICENSE`: Project license.
- `README.md`: This file.
- `Vault`: Empty directory for future use.

## Key Aspects
1. **Input Generation**: Create input files for TELEMAC-2D simulations with various hydraulic parameters, geometries, and boundary conditions.
2. **Simulation Execution**: Automate the process of running multiple TELEMAC-2D simulations and save the output.
3. **Result Processing and Analysis**: Analyze simulation results, calculate descriptive statistics, and store the data in HDF5 format.
4. **Model Training and Evaluation**: Train a simple neural network model using the generated data and evaluate its performance.

## Usage
1. Run the `input_generator.py` script in the `telemac/` directory to generate input files for TELEMAC-2D simulations.
2. Execute the `run.py` script in the `telemac/` directory to run the simulations and save the output.
3. Process and analyze the simulation results using the `simulation_data_processor.py` script.
4. Train a machine learning model using the `training.py` script in the `ML/` directory and evaluate its performance.

## Results
The results of the fluid flow simulations, machine learning model performance, and data analysis can be found in the `telemac/results/` and `ML/` directories.

## License
This project is licensed under the [MIT License](LICENSE).
