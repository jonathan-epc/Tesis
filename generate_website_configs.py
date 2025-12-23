# generate_website_configs.py
import json
from pathlib import Path

# --- Data from your paper and nconfig.yml ---

# Hyperparameters for each model architecture (from Table 4)
HYPERPARAMS = {
    "ddb": {
        "batch_size": 32,
        "lr": 2.84e-4,
        "wd": 2.67e-3,
        "layers": 15,
        "modes_x": 350,
        "modes_y": 4,
        "hidden": 102,
        "lift": 209,
        "proj": 211,
    },
    "idb": {
        "batch_size": 8,
        "lr": 7.38e-4,
        "wd": 0.34e-3,
        "layers": 1,
        "modes_x": 384,
        "modes_y": 2,
        "hidden": 111,
        "lift": 133,
        "proj": 154,
    },
    "dab": {
        "batch_size": 64,
        "lr": 0.51e-4,
        "wd": 3.54e-3,
        "layers": 11,
        "modes_x": 180,
        "modes_y": 4,
        "hidden": 20,
        "lift": 57,
        "proj": 123,
    },
    "iab": {
        "batch_size": 16,
        "lr": 5.01e-4,
        "wd": 1.30e-3,
        "layers": 15,
        "modes_x": 212,
        "modes_y": 4,
        "hidden": 103,
        "lift": 26,
        "proj": 152,
    },
    "dds": {
        "batch_size": 8,
        "lr": 2.37e-4,
        "wd": 0.83e-3,
        "layers": 9,
        "modes_x": 382,
        "modes_y": 2,
        "hidden": 57,
        "lift": 166,
        "proj": 63,
    },
    "ids": {
        "batch_size": 16,
        "lr": 7.33e-4,
        "wd": 1.97e-3,
        "layers": 9,
        "modes_x": 384,
        "modes_y": 8,
        "hidden": 31,
        "lift": 153,
        "proj": 222,
    },
    "das": {
        "batch_size": 8,
        "lr": 2.16e-4,
        "wd": 3.12e-3,
        "layers": 4,
        "modes_x": 274,
        "modes_y": 4,
        "hidden": 89,
        "lift": 242,
        "proj": 219,
    },
    "ias": {
        "batch_size": 16,
        "lr": 4.99e-4,
        "wd": 4.50e-3,
        "layers": 5,
        "modes_x": 372,
        "modes_y": 2,
        "hidden": 149,
        "lift": 3,
        "proj": 106,
    },
    "ddn": {
        "batch_size": 8,
        "lr": 8.14e-4,
        "wd": 0.16e-3,
        "layers": 16,
        "modes_x": 38,
        "modes_y": 4,
        "hidden": 170,
        "lift": 195,
        "proj": 7,
    },
    "idn": {
        "batch_size": 8,
        "lr": 4.23e-4,
        "wd": 0.37e-3,
        "layers": 2,
        "modes_x": 2,
        "modes_y": 10,
        "hidden": 149,
        "lift": 148,
        "proj": 196,
    },
    "dan": {
        "batch_size": 16,
        "lr": 0.39e-4,
        "wd": 3.19e-3,
        "layers": 4,
        "modes_x": 224,
        "modes_y": 8,
        "hidden": 92,
        "lift": 15,
        "proj": 27,
    },
    "ian": {
        "batch_size": 16,
        "lr": 2.21e-4,
        "wd": 5.00e-3,
        "layers": 8,
        "modes_x": 156,
        "modes_y": 6,
        "hidden": 154,
        "lift": 193,
        "proj": 205,
    },
}

# Input/Output variables for each model
IO_CONFIGS = {
    "ddb": {"inputs": ["H0", "Q0", "n", "nut", "B"], "outputs": ["H", "U", "V"]},
    "idb": {"inputs": ["U", "V"], "outputs": ["H0", "Q0", "n", "nut", "B", "H"]},
    "dab": {
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "iab": {
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "dds": {"inputs": ["H0", "Q0", "n", "nut", "B"], "outputs": ["H", "U", "V"]},
    "ids": {"inputs": ["U", "V"], "outputs": ["H0", "Q0", "n", "nut", "B", "H"]},
    "das": {
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "ias": {
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "ddn": {"inputs": ["H0", "Q0", "n", "nut", "B"], "outputs": ["H", "U", "V"]},
    "idn": {"inputs": ["U", "V"], "outputs": ["H0", "Q0", "n", "nut", "B", "H"]},
    "dan": {
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "ian": {
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
}

# Physical ranges for the sliders (from nconfig.yml)
PARAMETER_RANGES = {
    "H0": {"min": 0.01, "max": 0.25, "step": 0.001, "default": 0.15},
    "Q0": {"min": 0.005, "max": 0.02, "step": 0.001, "default": 0.01},
    "n": {"min": 0.01, "max": 0.2, "step": 0.001, "default": 0.025},
    "nut": {
        "min": 1e-6,
        "max": 1e-3,
        "step": 1e-6,
        "default": 1e-5,
    },  # Using an estimate for nut
    "Ar": {"min": 1, "max": 40, "step": 1, "default": 40},
    "Hr": {"min": 0.001, "max": 30, "step": 0.1, "default": 1},
    "Fr": {"min": 0.001, "max": 10, "step": 0.1, "default": 1.5},
    "M": {"min": 0.001, "max": 500, "step": 1, "default": 10},
    "Re": {"min": 100, "max": 75000, "step": 100, "default": 20000},
    "Vr": {"min": 1, "max": 40, "step": 1, "default": 40},  # Tied to Ar
}

ALL_FIELD_VARS = ["B", "F", "H", "Q", "S", "U", "V", "D", "B*", "H*", "U*", "V*"]
ALL_SCALAR_VARS = ["H0", "Q0", "SLOPE", "n", "nut", "Vr", "Fr", "Re", "Ar", "Hr", "M"]

# Descriptions for the dropdown
DESCRIPTIONS = {
    "ddb": "Direct Dimensional (BARS)",
    "idb": "Inverse Dimensional (BARS)",
    "dab": "Direct Adimensional (BARS)",
    "iab": "Inverse Adimensional (BARS)",
    "dds": "Direct Dimensional (SLOPE)",
    "ids": "Inverse Dimensional (SLOPE)",
    "das": "Direct Adimensional (SLOPE)",
    "ias": "Inverse Adimensional (SLOPE)",
    "ddn": "Direct Dimensional (NOISE)",
    "idn": "Inverse Dimensional (NOISE)",
    "dan": "Direct Adimensional (NOISE)",
    "ian": "Inverse Adimensional (NOISE)",
}


def main():
    models_store_dir = Path("src/website/models_store")
    for model_key in IO_CONFIGS:
        model_dir = models_store_dir / model_key
        if not model_dir.exists():
            continue

        print(f"Generating config for '{model_key}'...")

        hparams = HYPERPARAMS[model_key]
        io = IO_CONFIGS[model_key]

        input_fields = [v for v in io["inputs"] if v in ALL_FIELD_VARS]
        input_scalars_names = [v for v in io["inputs"] if v in ALL_SCALAR_VARS]

        # Create a list of objects for scalars, including their ranges
        input_scalars_with_ranges = []
        for name in input_scalars_names:
            if name in PARAMETER_RANGES:
                input_scalars_with_ranges.append(
                    {"name": name, **PARAMETER_RANGES[name]}
                )
            else:
                input_scalars_with_ranges.append(
                    {"name": name}
                )  # Fallback if range not defined

        output_fields = [v for v in io["outputs"] if v in ALL_FIELD_VARS]
        output_scalars = [v for v in io["outputs"] if v in ALL_SCALAR_VARS]

        config_data = {
            "model_key": model_key,
            "description": DESCRIPTIONS.get(model_key, f"Model for {model_key}"),
            "input_fields": input_fields,
            "input_scalars": input_scalars_with_ranges,  # Use the new list of objects
            "output_fields": output_fields,
            "output_scalars": output_scalars,
            "grid_dims": {"height": 11, "width": 401},
            "model_hparams": {
                "n_layers": hparams["layers"],
                "n_modes_x": hparams["modes_x"],
                "n_modes_y": hparams["modes_y"],
                "hidden_channels": hparams["hidden"],
                "lifting_channels": hparams["lift"],
                "projection_channels": hparams["proj"],
            },
        }

        with open(model_dir / "model_config.json", "w") as f:
            json.dump(config_data, f, indent=2)

    print("\nAll model configuration files generated successfully!")


if __name__ == "__main__":
    main()
