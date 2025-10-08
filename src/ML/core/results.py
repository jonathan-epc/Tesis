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
        "study_name": "study29ddb",  # instead of ddn, the file is misnamed
        "source_geom": "n",
        "inputs": ["H0", "Q0", "n", "nut", "B"],
        "outputs": ["H", "U", "V"],
    },
    "idn": {
        "trial_number": 94,
        "study_name": "study33ids",  # instead of idn, the file is misnamed
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
    Loads artifacts from a completed experiment and provides access to the
    trained model and the test set DataLoader.
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

        logger.info(f"Loading artifacts for study '{self.study_name}'...")
        self.study: optuna.Study = self._load_study()
        self.trial: optuna.Trial = self._load_trial()
        self.hparams: dict[str, Any] = self.trial.params
        self.config: Config = self._reconstruct_config()
        self.model: torch.nn.Module = self._load_model()

        self.full_dataset: HDF5Dataset
        self.test_loader: DataLoader
        self._setup_datasets()

        logger.info(
            f"Successfully loaded model and test data loader for trial #{self.trial_number}."
        )

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
        config_data = self.base_config.model_dump()
        study_config = self.study.user_attrs.get("config", {})

        if "data" in study_config:
            logger.info("Updating data config from study's user attributes...")
            config_data["data"].update(study_config["data"])
            if (
                "file_name" in config_data["data"]
                and "file_path" not in study_config["data"]
            ):
                config_data["data"]["file_path"] = config_data["data"]["file_name"]
        else:
            logger.warning(
                "Could not find 'config' in study user_attrs. Using base config as-is."
            )

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

        # --- FIX: Instantiate model on CPU first ---
        model = FNOnet(
            field_inputs_n=len(self.config.data.input_fields),
            scalar_inputs_n=len(self.config.data.input_scalars),
            field_outputs_n=len(self.config.data.output_fields),
            scalar_outputs_n=len(self.config.data.output_scalars),
            **self.hparams,
        )

        try:
            # --- FIX: Load state dict to CPU to avoid VRAM fragmentation ---
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "weights_only=True failed. Retrying with weights_only=False."
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        # --- END FIX ---

        if isinstance(checkpoint, dict):
            checkpoint.pop("_metadata", None)

        model.load_state_dict(checkpoint)

        # --- FIX: Move the fully loaded model to the target device ---
        model.to(self.config.device)
        model.eval()
        # --- END FIX ---

        logger.info("Model loaded successfully.")
        return model

    def _setup_datasets(self):
        """
        Initializes the full dataset and creates a DataLoader for the test split.
        """
        logger.info(
            f"Setting up dataset and test loader for: {self.config.data.file_path}"
        )
        self.full_dataset = HDF5Dataset.from_config(
            self.config, file_path=self.config.data.file_path
        )

        test_size = int(self.config.training.test_frac * len(self.full_dataset))
        train_val_size = len(self.full_dataset) - test_size

        if test_size == 0 or train_val_size == 0:
            raise ValueError("Dataset is too small for the specified train/test split.")

        g_split = torch.Generator().manual_seed(self.config.seed)
        _, test_dataset = random_split(
            self.full_dataset, [train_val_size, test_size], generator=g_split
        )

        g_test = torch.Generator().manual_seed(self.config.seed + 1)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hparams.get("batch_size", 32),
            shuffle=False,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g_test,
        )
        logger.info(f"Test set contains {len(test_dataset)} samples.")
