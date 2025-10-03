# src/ML/core/results.py

from pathlib import Path
from typing import Any

import optuna
import torch
from torch.utils.data import DataLoader, random_split

from common.utils import setup_logger
from ML.modules.data import HDF5Dataset
from ML.modules.models import FNOnet
from ML.modules.utils import seed_worker
from nconfig import Config, get_config

# --- Authoritative Information for Pre-trained Models and Datasets ---

TRAINED_MODELS_INFO = {
    "ddb": {
        "trial_number": 161,
        "study_name": "study28ddb",
        "source_geom": "b",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idb": {
        "trial_number": 57,
        "study_name": "study25idb",
        "source_geom": "b",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "dab": {
        "trial_number": 35,
        "study_name": "study26dab",
        "source_geom": "b",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "iab": {
        "trial_number": 157,
        "study_name": "study36iab",
        "source_geom": "b",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "dds": {
        "trial_number": 93,
        "study_name": "study30dds",
        "source_geom": "s",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "ids": {
        "trial_number": 63,
        "study_name": "study34ids",
        "source_geom": "s",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "das": {
        "trial_number": 26,
        "study_name": "study31das",
        "source_geom": "s",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "ias": {
        "trial_number": 36,
        "study_name": "study35ias",
        "source_geom": "s",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
    "ddn": {
        "trial_number": 194,
        "study_name": "study29ddn",
        "source_geom": "n",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idn": {
        "trial_number": 94,
        "study_name": "study33idn",
        "source_geom": "n",
        "inputs": ["U", "V"],
        "outputs": ["H0", "Q0", "n", "nut", "B", "H"],
    },
    "dan": {
        "trial_number": 195,
        "study_name": "study32dan",
        "source_geom": "n",
        "inputs": ["Hr", "Fr", "M", "Re", "B*", "Ar", "Vr"],
        "outputs": ["H*", "U*", "V*"],
    },
    "ian": {
        "trial_number": 88,
        "study_name": "study37ian",
        "source_geom": "n",
        "inputs": ["U*", "V*"],
        "outputs": ["Hr", "Fr", "M", "Re", "B*", "H*", "Ar", "Vr"],
    },
}

GEOMETRY_FILES = {
    "b": "BARSa.hdf5",
    "s": "SLOPEa.hdf5",
    "n": "NOISEa.hdf5",
}
GEOM_NAMES = {"b": "BARSa", "s": "SLOPEa", "n": "NOISEa"}


logger = setup_logger()


class ResultsLoader:
    """
    Loads all necessary artifacts from a completed experiment for analysis.

    This class takes a study name and an optional trial number, and handles
    loading the configuration, hyperparameters, model weights, and dataset
    to reproduce predictions on the test set.

    Args:
        study_name (str): The name of the Optuna study.
        trial_number (Optional[int]): The specific trial number to load.
            If None, the best trial from the study will be used.
        config_path (str): Path to the main nconfig.yml file.
    """

    def __init__(
        self,
        study_name: str,
        trial_number: int | None = None,
        config_path: str = "nconfig.yml",
    ):
        self.study_name = study_name
        self.base_config = get_config(config_path)
        self.trial_number = trial_number

        self.study: optuna.Study = self._load_study()
        self.trial: optuna.Trial = self._load_trial()
        self.hparams: dict[str, Any] = self.trial.params

        self.config: Config = self._reconstruct_config()

        self.model: torch.nn.Module = self._load_model()
        self.predictions: tuple[torch.Tensor, torch.Tensor]
        self.targets: tuple[torch.Tensor, torch.Tensor]

    def _load_study(self) -> optuna.Study:
        """Loads the Optuna study from the database."""
        storage_url = (
            f"sqlite:///{self.base_config.paths.studies_dir}/{self.study_name}.db"
        )
        logger.info(f"Loading study '{self.study_name}' from {storage_url}")
        return optuna.load_study(study_name=self.study_name, storage=storage_url)

    def _load_trial(self) -> optuna.Trial:
        """Loads the specific trial, defaulting to the best one if not specified."""
        if self.trial_number is None:
            logger.info("Trial number not specified, loading the best trial.")
            trial = self.study.best_trial
            self.trial_number = trial.number
        else:
            trial = self.study.trials[self.trial_number]

        logger.info(f"Loaded trial #{self.trial_number} (State: {trial.state})")
        return trial

    def _reconstruct_config(self) -> Config:
        """
        Reconstructs the precise configuration used for this specific trial
        by layering the base config with the study-level user attributes.
        """
        # Start with a copy of the base config data
        config_data = self.base_config.model_dump()

        # The user_attrs from the Optuna study contain the exact `data` section
        # (inputs, outputs, file_path) used for that study.
        study_config = self.study.user_attrs.get("config", {})

        if "data" in study_config:
            # --- THIS IS THE FIX ---
            # Instead of replacing, we UPDATE the data dictionary.
            # This preserves fields like 'all_field_vars' from the base config
            # while overriding the experiment-specific fields.
            logger.info("Updating data config from study's user attributes...")
            config_data["data"].update(study_config["data"])

            # The old config saved 'file_name', but the new one uses 'file_path'.
            # Let's handle this for backward compatibility.
            if (
                "file_name" in config_data["data"]
                and "file_path" not in study_config["data"]
            ):
                config_data["data"]["file_path"] = config_data["data"]["file_name"]

        else:
            logger.warning(
                "Could not find 'config' in study user_attrs. Using base config as-is."
            )

        # Create a new Config object from the merged data
        reconstructed_config = Config(**config_data)
        logger.info(
            f"Reconstructed data config: file_path='{reconstructed_config.data.file_path}', inputs={reconstructed_config.data.inputs}"
        )

        return reconstructed_config

    def _load_model(self) -> torch.nn.Module:
        """Instantiates the model and loads the saved weights for the trial."""
        model_name_stem = f"{self.study_name}_{self.config.model.class_name}_trial_{self.trial_number}"
        model_path = (
            Path(self.config.paths.savepoints_dir) / f"{model_name_stem}_best_model.pth"
        )

        logger.info(f"Loading model weights from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        model = FNOnet(
            field_inputs_n=len(self.config.data.input_fields),
            scalar_inputs_n=len(self.config.data.input_scalars),
            field_outputs_n=len(self.config.data.output_fields),
            scalar_outputs_n=len(self.config.data.output_scalars),
            **self.hparams,
        ).to(self.config.device)

        try:
            checkpoint = torch.load(
                model_path, map_location=self.config.device, weights_only=True
            )
        except Exception:
            logger.warning(
                "weights_only=True failed. Retrying with weights_only=False."
            )
            checkpoint = torch.load(
                model_path, map_location=self.config.device, weights_only=False
            )

        if isinstance(checkpoint, dict):
            checkpoint.pop("_metadata", None)
        model.load_state_dict(checkpoint)
        model.eval()

        logger.info("Model loaded successfully.")
        return model

    def _run_inference(self):
        """Runs the model on the test set to get predictions and targets."""
        logger.info("Running inference on the test set...")
        full_dataset = HDF5Dataset.from_config(
            self.config, file_path=self.config.data.file_path
        )
        test_size = int(self.config.training.test_frac * len(full_dataset))
        train_val_size = len(full_dataset) - test_size
        g_split = torch.Generator().manual_seed(self.config.seed)
        _, test_dataset = random_split(
            full_dataset, [train_val_size, test_size], generator=g_split
        )

        g_test = torch.Generator().manual_seed(self.config.seed + 1)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.hparams.get("batch_size", 32),
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_test,
        )

        all_field_preds, all_scalar_preds = [], []
        all_field_targs, all_scalar_targs = [], []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = (
                    [t.to(self.config.device) for t in inputs[0]],
                    [t.to(self.config.device) for t in inputs[1]],
                )

                preds = self.model(inputs)

                # Store raw predictions and targets for this batch
                field_p, scalar_p = preds
                field_t, scalar_t = targets

                if field_p is not None:
                    all_field_preds.append(field_p.cpu())
                    all_field_targs.append(torch.stack(field_t, dim=1).cpu())

                if scalar_p is not None:
                    all_scalar_preds.append(scalar_p.cpu())
                    all_scalar_targs.append(torch.stack(scalar_t, dim=1).cpu())

        # Concatenate all batches
        final_field_preds = torch.cat(all_field_preds) if all_field_preds else None
        final_scalar_preds = torch.cat(all_scalar_preds) if all_scalar_preds else None
        final_field_targs = torch.cat(all_field_targs) if all_field_targs else None
        final_scalar_targs = torch.cat(all_scalar_targs) if all_scalar_targs else None

        self.predictions = (final_field_preds, final_scalar_preds)
        self.targets = (final_field_targs, final_scalar_targs)

        logger.info("Inference complete.")

    def run_inference_on_dataset(self, eval_config: Config) -> tuple[tuple, tuple]:
        """Runs the loaded model on a specified dataset configuration."""
        full_dataset = HDF5Dataset.from_config(
            eval_config, file_path=eval_config.data.file_path
        )
        # Use the entire dataset for evaluation, not just a test split
        loader = DataLoader(
            full_dataset, batch_size=self.hparams.get("batch_size", 32), shuffle=False
        )

        all_preds, all_targs = [], []
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                inputs_on_device = (
                    [t.to(self.config.device) for t in inputs[0]],
                    [t.to(self.config.device) for t in inputs[1]],
                )
                preds = self.model(inputs_on_device)

                # Denormalize based on the hyperparameter of the LOADED model
                if self.hparams.get("normalize_output", False):
                    from ML.modules.utils import denormalize_outputs_and_targets

                    preds, targets = denormalize_outputs_and_targets(
                        preds, targets, full_dataset, eval_config, True
                    )

                all_preds.append(
                    (preds[0].cpu(), preds[1].cpu() if preds[1] is not None else None)
                )
                all_targs.append(
                    (
                        torch.stack(targets[0], dim=1).cpu(),
                        torch.stack(targets[1], dim=1).cpu() if targets[1] else None,
                    )
                )

        # Concatenate batches
        field_p = (
            torch.cat([p[0] for p in all_preds])
            if all_preds[0][0] is not None
            else None
        )
        scalar_p = (
            torch.cat([p[1] for p in all_preds])
            if all_preds[0][1] is not None
            else None
        )
        field_t = (
            torch.cat([t[0] for t in all_targs])
            if all_targs[0][0] is not None
            else None
        )
        scalar_t = (
            torch.cat([t[1] for t in all_targs])
            if all_targs[0][1] is not None
            else None
        )

        return (field_p, scalar_p), (field_t, scalar_t)
