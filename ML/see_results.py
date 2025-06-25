import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmap import Colormap
from config import get_config
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.amp import autocast
from torch.utils.data import DataLoader, random_split
from torcheval.metrics.functional import r2_score

from modules.data import HDF5Dataset
from modules.models import *
from modules.utils import denormalize_outputs_and_targets, get_hparams, set_seed


from modules.plots import (
    PlotManager,
    plot_field_analysis,
    plot_field_comparisons,
    plot_field_fourier,
    plot_scatter,
)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Calculate regression metrics between prediction and target tensors."""
    # Flatten tensors to handle both 2D and 4D inputs
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)

    # Basic metrics
    mse = F.mse_loss(pred_flat, target_flat).item()
    mae = F.l1_loss(pred_flat, target_flat).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()

    # R-squared calculation
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    r2 = (1 - ss_res / ss_tot).item()

    # Additional metrics
    diff = torch.abs(pred_flat - target_flat)

    # Mean Absolute Percentage Error (MAPE)
    mape = (
        torch.mean(torch.abs(diff / target_flat)) * 100
        if torch.all(target_flat != 0)
        else float("nan")
    )

    # Normalized Root Mean Square Error (NRMSE)
    target_range = target_flat.max() - target_flat.min()
    nrmse = rmse / target_range.item() if target_range != 0 else float("nan")

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = (
        torch.mean(
            2 * torch.abs(diff) / (torch.abs(target_flat) + torch.abs(pred_flat))
        )
        * 100
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "nrmse": nrmse,
        "smape": smape,
    }


def evaluate_predictions(
    field_predictions: Optional[torch.Tensor],
    field_targets: Optional[torch.Tensor],
    scalar_predictions: Optional[torch.Tensor],
    scalar_targets: Optional[torch.Tensor],
    config,
    csv_output_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Calculate combined metrics for both field and scalar predictions.
    Args:
        field_predictions: Field predictions [batch_size, num_fields, height, width], or None
        field_targets: Field targets [batch_size, num_fields, height, width], or None
        scalar_predictions: Scalar predictions [batch_size, num_scalars], or None
        scalar_targets: Scalar targets [batch_size, num_scalars], or None
        config: Config object containing data.outputs, data.non_scalars, data.scalars
        csv_output_path: Optional path to save the per-case metrics DataFrame as CSV
    Returns:
        Tuple containing:
        - Dictionary with overall and per-variable metrics
        - DataFrame with per-case metrics for each variable
    """
    metrics = {}
    per_case_metrics = []

    # Get variable names
    field_names = (
        [var for var in config.data.outputs if var in config.data.non_scalars]
        if field_predictions is not None and field_targets is not None
        else []
    )
    scalar_names = (
        [var for var in config.data.outputs if var in config.data.scalars]
        if scalar_predictions is not None and scalar_targets is not None
        else []
    )

    # Initialize flattened predictions and targets
    all_pred = []
    all_targ = []

    # Process field predictions
    if field_predictions is not None and field_targets is not None:
        field_pred_flat = field_predictions.reshape(field_predictions.shape[0], -1)
        field_targ_flat = field_targets.reshape(field_targets.shape[0], -1)
        all_pred.append(field_pred_flat)
        all_targ.append(field_targ_flat)

        for idx, name in enumerate(field_names):
            pred = field_predictions[:, idx]
            targ = field_targets[:, idx]
            metrics[name] = calculate_metrics(pred, targ)

            # Calculate per-case metrics
            for case_idx in range(pred.shape[0]):
                case_metrics = calculate_metrics(
                    pred[case_idx].unsqueeze(0), targ[case_idx].unsqueeze(0)
                )
                per_case_metrics.append(
                    {
                        "case": case_idx,
                        "variable": name,
                        "type": "field",
                        **case_metrics,
                    }
                )

    # Process scalar predictions
    if scalar_predictions is not None and scalar_targets is not None:
        scalar_pred_flat = scalar_predictions.reshape(scalar_predictions.shape[0], -1)
        scalar_targ_flat = scalar_targets.reshape(scalar_targets.shape[0], -1)
        all_pred.append(scalar_pred_flat)
        all_targ.append(scalar_targ_flat)

        for idx, name in enumerate(scalar_names):
            pred = scalar_predictions[:, idx]
            targ = scalar_targets[:, idx]
            metrics[name] = calculate_metrics(pred, targ)

            # Calculate per-case metrics
            for case_idx in range(pred.shape[0]):
                case_metrics = calculate_metrics(
                    pred[case_idx].unsqueeze(0), targ[case_idx].unsqueeze(0)
                )
                per_case_metrics.append(
                    {
                        "case": case_idx,
                        "variable": name,
                        "type": "scalar",
                        **case_metrics,
                    }
                )

    # Calculate overall metrics if there are valid predictions and targets
    if all_pred and all_targ:
        all_pred = torch.cat(all_pred, dim=1)
        all_targ = torch.cat(all_targ, dim=1)
        metrics["overall"] = calculate_metrics(all_pred, all_targ)

        # Calculate per-case overall metrics
        for case_idx in range(all_pred.shape[0]):
            case_metrics = calculate_metrics(
                all_pred[case_idx].unsqueeze(0), all_targ[case_idx].unsqueeze(0)
            )
            per_case_metrics.append(
                {
                    "case": case_idx,
                    "variable": "overall",
                    "type": "combined",
                    **case_metrics,
                }
            )
    else:
        metrics["overall"] = {}

    # Convert per-case metrics to DataFrame
    per_case_df = pd.DataFrame(per_case_metrics)

    # Save DataFrame to CSV if path is provided
    if csv_output_path is not None:
        per_case_df.to_csv(csv_output_path, index=False)

    return metrics, per_case_df


def evaluate_model(model, test_dataloader, config, dataset):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        all_field_outputs, all_scalar_outputs = [], []
        all_field_targets, all_scalar_targets = [], []
        all_field_inputs, all_scalar_inputs = [], []

        for batch in test_dataloader:
            inputs, targets = batch
            input_fields, input_scalars = inputs
            target_fields, target_scalars = targets

            # Move inputs and targets to device
            input_fields = [field.to(config.device) for field in input_fields]
            input_scalars = (
                [scalar.to(config.device) for scalar in input_scalars]
                if input_scalars
                else []
            )
            target_fields = [field.to(config.device) for field in target_fields]
            target_scalars = (
                [scalar.to(config.device) for scalar in target_scalars]
                if target_scalars
                else []
            )

            # Reconstruct the tuples
            inputs = (input_fields, input_scalars)
            targets = (target_fields, target_scalars)

            # Forward pass
            outputs = model(inputs)
            outputs, targets = denormalize_outputs_and_targets(
                outputs,
                targets,
                dataset,
                config,
                config.data.normalize_output 
            )
            target_fields, target_scalars = targets
            # Collect field and scalar outputs and targets
            field_outputs, scalar_outputs = outputs

            if field_outputs is not None:
                all_field_outputs.append(field_outputs)
                if target_fields:
                    all_field_targets.append(torch.stack(target_fields, dim=1))

            # Only collect scalar outputs and targets if they exist
            if scalar_outputs is not None:
                all_scalar_outputs.append(scalar_outputs)
                if target_scalars:
                    all_scalar_targets.append(
                        torch.stack(target_scalars, dim=1)
                    )  # Stack to preserve batch structure

        # After processing all batches, concatenate outputs and targets along batch dimension
        all_field_outputs = (
            torch.cat(all_field_outputs, dim=0) if all_field_outputs else None
        )
        all_field_targets = (
            torch.cat(all_field_targets, dim=0) if all_field_targets else None
        )

        # Only concatenate scalar outputs if they exist
        all_scalar_outputs = (
            torch.cat(all_scalar_outputs, dim=0) if all_scalar_outputs else None
        )
        all_scalar_targets = (
            torch.cat(all_scalar_targets, dim=0) if all_scalar_targets else None
        )

        return (
            all_field_outputs,
            all_field_targets,
            all_scalar_outputs,
            all_scalar_targets,
        )

def get_input_output_counts(config: Any) -> Dict[str, int]:
    """Calculate input and output dimensions based on configuration."""
    return {
        "field_inputs": len(
            [p for p in config.data.inputs if p in config.data.non_scalars]
        ),
        "scalar_inputs": len(
            [p for p in config.data.inputs if p in config.data.scalars]
        ),
        "field_outputs": len(
            [p for p in config.data.outputs if p in config.data.non_scalars]
        ),
        "scalar_outputs": len(
            [p for p in config.data.outputs if p in config.data.scalars]
        ),
    }


def get_output_names(config: Any) -> Dict[str, List[str]]:
    """Get lists of field and scalar output variable names."""
    return {
        "field": [var for var in config.data.outputs if var in config.data.non_scalars],
        "scalar": [var for var in config.data.outputs if var in config.data.scalars],
    }


def setup_datasets(
    config: Any, hparams: Dict
) -> Tuple[DataLoader, List[str], List[str]]:
    """Initialize and split datasets, create data loader."""
    full_dataset = HDF5Dataset(
        file_path=config.data.file_name,
        input_vars=config.data.inputs,
        output_vars=config.data.outputs,
        numpoints_x=config.data.numpoints_x,
        numpoints_y=config.data.numpoints_y,
        channel_length=config.channel.length,
        channel_width=config.channel.width,
        normalize_input=config.data.normalize_input,
        normalize_output=config.data.normalize_output,
        device=config.device,
        preload=False,
    )

    # Split dataset
    test_size = int(config.training.test_frac * len(full_dataset))
    train_val_size = len(full_dataset) - test_size
    _, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False
    )

    return test_dataloader, full_dataset


def load_model(
    config: Any, hparams: Dict, io_counts: Dict[str, int], model_name: str
) -> torch.nn.Module:
    """Initialize and load the model."""
    # Ensure the model class (FNOnet) is available in the scope.
    # If FNOnet is defined in modules.models, you'd typically have:
    # from modules.models import FNOnet (or *)
    # Then config.model.class_name ("FNOnet") can be resolved.
    try:
        model_class = globals()[config.model.class_name]
    except KeyError:
        # Fallback if not in globals, try importing from modules.models dynamically
        # This is a bit more robust if the script structure changes.
        import importlib

        models_module = importlib.import_module("modules.models")
        model_class = getattr(models_module, config.model.class_name)

    model = model_class(
        field_inputs_n=io_counts["field_inputs"],
        scalar_inputs_n=io_counts["scalar_inputs"],
        field_outputs_n=io_counts["field_outputs"],
        scalar_outputs_n=io_counts["scalar_outputs"],
        **hparams,
    ).to(config.device)

    model_path = Path(f"savepoints/{model_name}_best_model.pth")
    print(f"INFO: Attempting to load model from: {model_path.resolve()}")

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path.resolve()}")
        raise FileNotFoundError(f"Model checkpoint not found at {model_path.resolve()}")

    checkpoint = None
    try:
        # Try loading with weights_only=True first for security
        checkpoint = torch.load(
            model_path, map_location=config.device, weights_only=True
        )
        print("INFO: Successfully loaded checkpoint with weights_only=True.")
    except Exception as e_true:
        print(f"WARNING: Loading model with weights_only=True failed: {e_true}.")
        print("INFO: Attempting to load with weights_only=False as a fallback.")
        try:
            # Fallback to weights_only=False
            checkpoint = torch.load(
                model_path, map_location=config.device, weights_only=False
            )
            print("INFO: Successfully loaded checkpoint with weights_only=False.")
        except Exception as e_false:
            print(
                f"ERROR: Fallback loading (weights_only=False) also failed: {e_false}"
            )
            raise  # Re-raise the exception if both loading attempts fail

    # Clean the checkpoint if it's a dictionary and contains '_metadata'
    # This is crucial because your debug script shows 'checkpoint' IS the state_dict (an OrderedDict)
    if isinstance(checkpoint, dict):
        if "_metadata" in checkpoint:
            print("INFO: Removing '_metadata' key from loaded checkpoint dictionary.")
            checkpoint.pop("_metadata", None)  # Use .pop() for safety

        # Now, load the (potentially cleaned) state_dict
        model.load_state_dict(checkpoint)
        print("INFO: Model state_dict loaded successfully.")
    else:
        # This case should ideally not happen if you saved model.state_dict()
        # If the entire model object was saved (torch.save(model, ...)), this would be different
        error_msg = (
            f"ERROR: Loaded checkpoint is not a dictionary (type: {type(checkpoint)}). "
            "Cannot load state_dict. Ensure the model was saved using `torch.save(model.state_dict(), ...)`."
        )
        print(error_msg)
        raise TypeError(error_msg)

    return model


def configure_study(config, study_type: str):
    """
    Configure study-specific parameters based on study_type.

    Args:
        config: The configuration object.
        study_type (str): One of {"dd", "id", "da", "ia"}.

    Returns:
        trial_number (int), study_name (str), inputs (list), outputs (list)
    """
    mapping = {
        "ddb": {
            "trial_number": 161,
            "study_name": "study28ddb",
            "inputs": ["H0", "Q0", "n", "nut", "B"],
            "outputs": ["H", "U", "V"],
        },
        "idb": {
            "trial_number": 57,
            "study_name": "study25idb",
            "inputs": ["U", "V"],
            "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
        },
        "dab": {
            "trial_number": 35,
            "study_name": "study26dab",
            "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
            "outputs": ["H*", "U*", "V*"],
        },
        "iab": {
            "trial_number": 157,
            "study_name": "study36iab",
            "inputs": ["U*", "V*"],
            "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
        },
        "dds": {
            "trial_number": 93,
            "study_name": "study30dds",
            "inputs": ["H0", "Q0", "n", "nut", "B"],
            "outputs": ["H", "U", "V"],
        },
        "ids": {
            "trial_number": 63,
            "study_name": "study34ids",
            "inputs": ["U", "V"],
            "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
        },
        "das": {
            "trial_number": 26,
            "study_name": "study31das",
            "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
            "outputs": ["H*", "U*", "V*"],
        },
        "ias": {
            "trial_number": 36,
            "study_name": "study35ias",
            "inputs": ["U*", "V*"],
            "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
        },
        "ddn": {
            "trial_number": 194,
            "study_name": "study29ddb",
            "inputs": ["H0", "Q0", "n", "nut", "B"],
            "outputs": ["H", "U", "V"],
        },
        "idn": {
            "trial_number": 94,
            "study_name": "study33ids",
            "inputs": ["U", "V"],
            "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
        },
        "dan": {
            "trial_number": 195,
            "study_name": "study32dan",
            "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
            "outputs": ["H*", "U*", "V*"],
        },
        "ian": {
            "trial_number": 88,
            "study_name": "study37ian",
            "inputs": ["U*", "V*"],
            "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
        },
    }
    if study_type not in mapping:
        raise ValueError(
            f"Unknown study type '{study_type}'. Choose from {list(mapping.keys())}."
        )
    params = mapping[study_type]
    return (
        params["trial_number"],
        params["study_name"],
        params["inputs"],
        params["outputs"],
    )


def main(study_type: str = "ddb", bottom_type: str = "barsa") -> None:
    """Main function to run the evaluation pipeline with study-specific configuration."""
    # Initial setup
    config = get_config()
    set_seed(config.seed)

    # Define and create a custom colormap
    colors = ["white", "#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_turbo", colors, N=256)

    # Set study-specific configuration
    trial_number, study_name, inputs, outputs = configure_study(config, study_type)
    config.optuna.study_name = study_name
    config.data.inputs = inputs
    config.data.outputs = outputs

    # Create or load the Optuna study
    study = optuna.create_study(
        study_name=config.optuna.study_name,
        load_if_exists=True,
        storage=config.optuna.storage,
    )
    specific_trial_params = study.trials[trial_number].params
    print(f"DEBUG: Optuna parameters for {study_name}, trial {trial_number}: {specific_trial_params}")
    # Get model I/O counts and names
    io_counts = get_input_output_counts(config)
    output_names = get_output_names(config)
    model_name = f"{study_name}_{config.model.architecture}_trial_{trial_number}"

    # Load hyperparameters and update config
    hparams = study.trials[trial_number].params
    config.training.use_physics_loss = hparams["use_physics_loss"]
    # config.data.normalize_output = hparams["normalize_output"]
    config.data.normalize_output = True
    config.data.file_name = "data/"+bottom_type+".hdf5"

    # Setup model and data
    model = load_model(config, hparams, io_counts, model_name)
    test_dataloader, full_dataset = setup_datasets(config, hparams)

    # Evaluate model predictions
    all_field_outputs, all_field_targets, all_scalar_outputs, all_scalar_targets = (
        evaluate_model(model, test_dataloader, config, full_dataset)
    )
    metrics, per_case_df = evaluate_predictions(
        all_field_outputs,
        all_field_targets,
        all_scalar_outputs,
        all_scalar_targets,
        config,
        csv_output_path= "metrics/" + study_type + "_" + bottom_type + "_" + "case_metrics.csv",
    )

    # Define plotting parameters
    units = {
        "H": "m",
        "U": "m/s",
        "V": "m/s",
        "B": "m",
        "H*": "-",
        "U*": "-",
        "V*": "-",
        "H0": "m",
        "Q0": "m³/s",
        "n": "s/m^(1/3)",
        "nut": "m²/s",
        "Hr": "-",
        "Fr": "-",
        "M": "-",
        "Re": "-",
        "B*": "-",
        "Ar": "-",
        "Vr": "-",
    }
    long_names = {
        "H": {"en": "Depth", "es": "Profundidad"},
        "U": {"en": "Longitudinal velocity", "es": "Velocidad longitudinal"},
        "V": {"en": "Transversal velocity", "es": "Velocidad transversal"},
        "B": {"en": "Bottom height", "es": "Profundidad del fondo"},
        "H*": {"en": "Adimensional depth", "es": "Profundidad adimensional"},
        "U*": {
            "en": "Adimensional transversal velocity",
            "es": "Velocidad transversal adimensional",
        },
        "V*": {
            "en": "Adimensional longitudinal velocity",
            "es": "Velocidad longitudinal adimensional",
        },
        "H0": {"en": "Initial depth", "es": "Profundidad inicial"},
        "Q0": {"en": "Initial flow rate", "es": "Caudal inicial"},
        "n": {
            "en": "Manning's roughness coefficient",
            "es": "Coeficiente de rugosidad de Manning",
        },
        "nut": {"en": "Kinematic viscosity", "es": "Viscosidad cinemática"},
        "Hr": {"en": "Height ratio", "es": "Relación de altura"},
        "Fr": {"en": "Froude number", "es": "Número de Froude"},
        "M": {
            "en": "Friction adimensional number",
            "es": "Número adimensional friccional",
        },
        "Re": {"en": "Reynolds number", "es": "Número de Reynolds"},
        "B*": {
            "en": "Adimensional bottom height",
            "es": "Profundidad del fondo adimensional",
        },
        "Ar": {"en": "Aspect ratio", "es": "Relación de aspecto"},
        "Vr": {"en": "Velocity ratio", "es": "Relación de velocidad"},
    }
    language = "en"
    detailed = True
    data_percentile = 99
    random_idx = random.randint(0, len(all_field_outputs) - 1)
    plot_manager = PlotManager(base_dir=f"plots/{study_type}_{bottom_type}")

    # Generate visualizations
    plot_scatter(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        scalar_outputs=all_scalar_outputs,
        scalar_targets=all_scalar_targets,
        scalar_names=output_names["scalar"],
        units=units,
        long_names=long_names,
        metrics=metrics,
        custom_cmap=custom_cmap,
        plot_manager=plot_manager,
        language=language,
        detailed=detailed,
        data_percentile=data_percentile,
    )
    plot_field_comparisons(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        case_idx=random_idx,
        custom_cmap=custom_cmap,
        plot_manager=plot_manager,
        units=units,
        long_names=long_names,
        language=language,
        detailed=detailed,
    )
    plot_field_analysis(
        all_field_outputs,
        all_field_targets,
        output_names["field"],
        custom_cmap=custom_cmap,
        plot_manager=plot_manager,
        units=units,
        language=language,
        detailed=detailed,
    )


def run_all_evaluations():
    """
    Runs the evaluation and plotting pipeline for all study types
    defined in the configure_study mapping.
    """
    all_study_types = ["ddb", "idb", "dab", "iab", "dds", "ids", "das", "ias", "ddn", "idn", "dan", "ian"]
    all_bottom_types = ["barsa", "slopea", "noisea"]
    for current_study_type in all_study_types:
        for current_bottom_type in all_bottom_types:
            print(f"\n\n{'='*30} PROCESSING STUDY TYPE: {current_study_type.upper()} {'='*30}\n")
            try:
                # Call your existing main function with the current study type
                main(study_type=current_study_type, bottom_type=current_bottom_type)
                print(f"\n{'='*30} SUCCESSFULLY COMPLETED STUDY TYPE: {current_study_type.upper()} in {current_bottom_type.upper()} {'='*30}\n")
            except Exception as e:
                print(f"\n!!!!!! ERROR PROCESSING STUDY TYPE: {current_study_type.upper()} in {current_bottom_type.upper()} !!!!!!")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc() # Print the full traceback for debugging
                print(f"!!!!!! SKIPPING TO NEXT STUDY TYPE (IF ANY) !!!!!!\n")


if __name__ == "__main__":
    # main("ddb", "barsa") # Call for a single study type

    # Call the new function to process all studies
    run_all_evaluations()
