# src/website/app/model_loader.py
import json
from pathlib import Path

import torch

from ML.modules.models import FNOnet

DEVICE = torch.device("cpu")


class ModelHandler:
    def __init__(self, model_store_path: Path):
        self._models = {}
        self._configs = {}
        self.load_all_models(model_store_path)

    def load_all_models(self, model_store_path: Path):
        print("--- Scanning for models... ---")
        if not model_store_path.is_dir():
            raise NotADirectoryError(f"Model store path not found: {model_store_path}")

        for model_dir in model_store_path.iterdir():
            if model_dir.is_dir():
                model_key = model_dir.name
                try:
                    model, config = self._load_single_model(model_dir)
                    self._models[model_key] = model
                    self._configs[model_key] = config
                    print(f"  [SUCCESS] Loaded model '{model_key}'")
                except Exception as e:
                    print(f"  [FAILURE] Could not load model '{model_key}': {e}")
        print(f"--- Finished loading {len(self._models)} models. ---")

    def _load_single_model(self, model_dir: Path):
        model_path = model_dir / "model.pth"
        config_path = model_dir / "model_config.json"

        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"model.pth or model_config.json missing in {model_dir}"
            )

        with open(config_path) as f:
            config = json.load(f)

        model_hparams = config.get("model_hparams", {})
        model = FNOnet(
            field_inputs_n=len(config["input_fields"]),
            scalar_inputs_n=len(config["input_scalars"]),
            field_outputs_n=len(config["output_fields"]),
            scalar_outputs_n=len(config["output_scalars"]),
            **model_hparams,
        )

        try:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        except Exception:
            state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if "_metadata" in state_dict:
            state_dict.pop("_metadata")

        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, config

    def get_available_models(self):
        return self._configs

    def predict(self, model_key: str, api_input: dict):
        if model_key not in self._models:
            raise KeyError(f"Model '{model_key}' not loaded.")

        model = self._models[model_key]
        config = self._configs[model_key]

        # 1. Preprocess
        model_inputs = self._preprocess_input(api_input, config)

        # 2. Infer
        with torch.no_grad():
            raw_predictions = model(model_inputs)

        # 3. Postprocess
        return self._postprocess_output(raw_predictions, config)

    def _preprocess_input(self, api_input: dict, config: dict):
        scalar_values = api_input.get("scalar_features", [])
        if len(scalar_values) != len(config["input_scalars"]):
            raise ValueError(
                f"Expected {len(config['input_scalars'])} scalar features, but got {len(scalar_values)}."
            )

        scalar_tensors = [
            torch.tensor([v], dtype=torch.float32, device=DEVICE) for v in scalar_values
        ]

        field_flat = api_input.get("field_data_flat", [])
        if config["input_fields"]:  # Only check if fields are expected
            dims = config["grid_dims"]
            expected_size = dims["height"] * dims["width"] * len(config["input_fields"])
            if len(field_flat) != expected_size:
                raise ValueError(
                    f"Expected {len(config['input_fields'])} field(s) with {expected_size} total elements, but got {len(field_flat)}."
                )

            # Reshape and split into list of fields
            all_fields_tensor = torch.tensor(
                field_flat, dtype=torch.float32, device=DEVICE
            ).view(len(config["input_fields"]), dims["height"], dims["width"])
            field_tensors = [
                f.unsqueeze(0) for f in all_fields_tensor
            ]  # Add batch dim to each
        else:
            field_tensors = []

        scalar_tensors_batched = [s.unsqueeze(0) for s in scalar_tensors]
        return (field_tensors, scalar_tensors_batched)

    def _postprocess_output(self, raw_predictions, config: dict):
        field_preds, scalar_preds = raw_predictions
        results = {"error": None}

        if field_preds is not None:
            field_preds_squeezed = field_preds.squeeze(0).cpu().numpy()
            output_field_names = config["output_fields"]
            results["field_predictions_shape"] = list(field_preds_squeezed.shape)
            results["field_predictions_info"] = (
                f"Returning {len(output_field_names)} field(s)."
            )

            flat_data_dict = {
                name: field_preds_squeezed[i].flatten().tolist()
                for i, name in enumerate(output_field_names)
            }
            results["field_predictions_flat"] = flat_data_dict

        if scalar_preds is not None:
            results["scalar_predictions"] = (
                scalar_preds.squeeze(0).cpu().numpy().tolist()
            )

        return results
