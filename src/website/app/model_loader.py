# src/website/app/model_loader.py
import json
from pathlib import Path

import torch

# This is where your good project structure pays off!
# We can directly import the FNOnet class from your main ML code.
from ML.modules.models import FNOnet

# --- Configuration ---
DEVICE = torch.device("cpu")  # We know we are on a CPU
MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store" / "direct_problem"
MODEL_PATH = MODEL_DIR / "model.pth"
CONFIG_PATH = MODEL_DIR / "model_config.json"

# --- Helper Functions ---


def load_model_and_config():
    """Loads the model configuration and instantiates the FNOnet model."""
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Model or config not found. Make sure model.pth and model_config.json are in src/website/models_store/direct_problem/"
        )

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # Extract the model's architectural hyperparameters from the config
    model_hparams = config.get("model_hparams", {})
    print(f"Instantiating FNOnet with hyperparameters: {model_hparams}")

    model = FNOnet(
        field_inputs_n=len(config["input_fields"]),
        scalar_inputs_n=len(config["input_scalars"]),
        field_outputs_n=len(config["output_fields"]),
        scalar_outputs_n=len(config["output_scalars"]),
        **model_hparams,  # Pass the loaded hyperparameters to the model constructor
    )

    # --- Robust State Dict Loading ---
    # Load the state dictionary, making sure it's mapped to the CPU
    # Use weights_only=True for security, with a fallback
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except Exception:
        print("Warning: weights_only=True failed. Falling back to weights_only=False.")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    # The error mentioned an unexpected '_metadata' key. This is a common
    # issue with saving/loading. We can safely remove it.
    if "_metadata" in state_dict:
        state_dict.pop("_metadata")

    model.load_state_dict(state_dict)  # Now it should match!
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode

    print("--- Model and configuration loaded successfully for website. ---")
    return model, config


def preprocess_input(api_input: dict):
    """Converts the flat JSON input from the API into tensors for the FNO model."""
    config = loaded_config  # Use the globally loaded config

    # 1. Process Scalars
    scalar_values = api_input.get("scalar_features", [])
    scalar_tensors = [
        torch.tensor([v], dtype=torch.float32, device=DEVICE) for v in scalar_values
    ]

    # 2. Process Field Data
    field_flat = api_input.get("field_data_flat", [])
    dims = config["grid_dims"]
    expected_size = dims["height"] * dims["width"]

    if len(field_flat) != expected_size:
        raise ValueError(
            f"Field data has {len(field_flat)} elements, but expected {expected_size} (for {dims['height']}x{dims['width']} grid)."
        )

    field_tensor = torch.tensor(field_flat, dtype=torch.float32, device=DEVICE).view(
        dims["height"], dims["width"]
    )

    # 3. Add batch dimension (batch size = 1) and package for FNOnet
    field_tensors = [field_tensor.unsqueeze(0)]  # Shape: [1, H, W]
    scalar_tensors_batched = [
        s.unsqueeze(0) for s in scalar_tensors
    ]  # Shape: [1, 1] but FNOnet handles it.

    # FNOnet expects (list_of_field_tensors, list_of_scalar_tensors)
    return (field_tensors, scalar_tensors_batched)


def postprocess_output(raw_predictions):
    """Converts the model's output tensors into a JSON-serializable dictionary."""
    field_preds, scalar_preds = raw_predictions

    results = {"error": None}

    if field_preds is not None:
        # --- START: REPLACE THE OLD 'field_preds' BLOCK WITH THIS ---
        field_preds_squeezed = field_preds.squeeze(0).cpu().numpy()

        output_field_names = loaded_config["output_fields"]  # e.g., ["H", "U", "V"]

        results["field_predictions_shape"] = list(field_preds_squeezed.shape)
        results["field_predictions_info"] = (
            f"Returning {len(output_field_names)} field(s). Each has shape {results['field_predictions_shape'][1:]}."
        )

        # Create the dictionary for the flat data
        flat_data_dict = {}
        for i, name in enumerate(output_field_names):
            # Flatten the 2D array for each field and convert to a list
            flat_data_dict[name] = field_preds_squeezed[i].flatten().tolist()

        results["field_predictions_flat"] = flat_data_dict
        # --- END: REPLACEMENT BLOCK ---

    if scalar_preds is not None:
        results["scalar_predictions"] = scalar_preds.squeeze(0).cpu().numpy().tolist()

    return results


# --- Load Model on Startup ---
# This code runs once when the application starts
model, loaded_config = load_model_and_config()
