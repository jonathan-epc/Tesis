# nconfig.py
import os
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# --- Sub-Models for Configuration Sections ---


class ApiKeys(BaseModel):
    wandb: str | None = None


class ChannelConfig(BaseModel):
    width: float = Field(0.3, gt=0)
    length: float = Field(12, gt=0)
    depth: float = Field(0.3, gt=0)
    wall_thickness: float = Field(0, ge=0)


class MeshConfig(BaseModel):
    num_points_x: int = Field(401, gt=0)
    num_points_y: int = Field(11, gt=0)


class SimulationParamsConfig(BaseModel):
    adimensional_generation: bool = False
    gravity: float = Field(9.81, gt=0)
    parameter_ranges: dict[str, float]
    bottom_types: list[str]


class PathsConfig(BaseModel):
    telemac_dir: str = "telemac"
    ml_dir: str = "ML"
    data_dir: str = "data"
    log_dir: str = "logs"
    savepoints_dir: str = "savepoints"
    plot_dir: str = "plots"
    studies_dir: str = "studies"


class DataConfig(BaseModel):
    file_path: str
    normalize_input: bool = True
    normalize_output: bool = False
    boxcox_transform: bool = False
    preload_hdf5: bool = True
    chunk_size_hdf5: int | None = None
    inputs: list[str]
    outputs: list[str]
    all_field_vars: list[str]
    all_numeric_scalar_vars: list[str]
    all_non_numeric_scalar_vars: list[str]

    @computed_field
    @property
    def all_scalar_vars(self) -> list[str]:
        return self.all_numeric_scalar_vars + self.all_non_numeric_scalar_vars

    @computed_field
    @property
    def input_scalars(self) -> list[str]:
        return [v for v in self.inputs if v in self.all_scalar_vars]

    @computed_field
    @property
    def input_fields(self) -> list[str]:
        return [v for v in self.inputs if v in self.all_field_vars]

    @computed_field
    @property
    def output_scalars(self) -> list[str]:
        return [v for v in self.outputs if v in self.all_scalar_vars]

    @computed_field
    @property
    def output_fields(self) -> list[str]:
        return [v for v in self.outputs if v in self.all_field_vars]

    @computed_field
    @property
    def is_adimensional(self) -> bool:
        """
        Determines if the current experiment is adimensional by checking if any
        input or output variable names contain an asterisk '*'.
        """
        all_vars = self.inputs + self.outputs
        return any("*" in var for var in all_vars)


class ModelConfig(BaseModel):
    name: str
    architecture: str
    class_name: str
    n_layers: int = Field(4, gt=0)
    n_modes_x: int = Field(72, gt=0)
    n_modes_y: int = Field(11, gt=0)
    hidden_channels: int = Field(64, gt=0)
    lifting_channels: int = Field(32, gt=0)
    projection_channels: int = Field(32, gt=0)


class TrainingConfig(BaseModel):
    kfolds: int = Field(1, ge=1)
    batch_size: int = Field(64, gt=0)
    num_epochs: int = Field(512, gt=0)
    num_workers: int = Field(0, ge=0)
    accumulation_steps: int = Field(1, gt=0)
    learning_rate: float = Field(1e-4, gt=0)
    test_frac: float = Field(0.2, ge=0, le=1)
    validation_frac: float = Field(0.2, ge=0, le=1)
    early_stopping_patience: int = Field(64, ge=0)
    early_stopping_delta: float = Field(0.0, ge=0)
    clip_grad_value: float = Field(1.0, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    pretrained_model_name: str | None = None
    use_physics_loss: bool = False


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    wandb_project: str = "Tesis"
    plot_enabled: bool = False


class OptunaConfig(BaseModel):
    study_name: str
    study_notes: str
    n_trials: int = Field(100, gt=0)
    time_limit_per_trial: float | None = Field(86400, gt=0)
    hyperparameter_space: dict[str, dict[str, Any]]


# --- Main Configuration Model ---
class Config(BaseModel):
    project_name: str = "FNO-TELEMAC-Thesis"
    seed: int = 43
    device: str = "cpu"  # Default to CPU, will be updated if CUDA is available
    api_keys: ApiKeys
    channel: ChannelConfig
    mesh: MeshConfig
    simulation_params: SimulationParamsConfig
    paths: PathsConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    optuna: OptunaConfig

    @field_validator("device", mode="before")
    @classmethod
    def set_device(cls, v: str) -> str:
        if v == "auto" or v == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

    @model_validator(mode="before")
    @classmethod
    def convert_hyperparameter_values(cls, data: Any) -> Any:
        """
        Recursively find 'low' and 'high' in the hyperparameter space
        and ensure they are parsed as floats, not strings.
        """
        if (
            isinstance(data, dict)
            and "optuna" in data
            and "hyperparameter_space" in data["optuna"]
        ):
            space = data["optuna"]["hyperparameter_space"]
            for _, settings in space.items():
                if "low" in settings and isinstance(settings["low"], str):
                    with suppress(ValueError):
                        settings["low"] = float(settings["low"])
                if "high" in settings and isinstance(settings["high"], str):
                    with suppress(ValueError):
                        settings["high"] = float(settings["high"])
        return data

    @model_validator(mode="after")
    def adjust_optuna_mode_bounds(self) -> "Config":
        space = self.optuna.hyperparameter_space
        if "n_modes_x" in space:
            max_modes_x = self.mesh.num_points_x // 2 + 1
            space["n_modes_x"]["high"] = min(space["n_modes_x"]["high"], max_modes_x)
        if "n_modes_y" in space:
            max_modes_y = self.mesh.num_points_y // 2 + 1
            space["n_modes_y"]["high"] = min(space["n_modes_y"]["high"], max_modes_y)
        return self

    @model_validator(mode="after")
    def add_timestamp_to_model_name(self) -> "Config":
        if "{timestamp}" in self.model.name:
            date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.model.name = self.model.name.replace("{timestamp}", date_str)
        return self

    @computed_field
    @property
    def optuna_storage_url(self) -> str:
        return f"sqlite:///{self.paths.studies_dir}/{self.optuna.study_name}.db"

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Get the directory where this nconfig.py file is located. This is our project root.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "nconfig.yml"

_config: Config | None = None


def load_config(config_path: Path) -> Config:
    """Loads configuration from YAML, validates with Pydantic, and handles secrets."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    try:
        with open(config_path) as file:
            yaml_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}") from exc

    # Handle API keys from environment variables
    if "api_keys" in yaml_config and "wandb" in yaml_config["api_keys"]:
        wandb_key = yaml_config["api_keys"]["wandb"]
        if isinstance(wandb_key, str) and wandb_key.startswith("${"):
            env_var = wandb_key.strip("${}")
            yaml_config["api_keys"]["wandb"] = os.environ.get(env_var)

    return Config(**yaml_config)


def get_config(config_path: str | None = None) -> Config:
    """
    Gets the singleton Config instance, loading if necessary.
    If config_path is None, it uses the default nconfig.yml in the project root.
    """
    global _config
    if _config is None:
        # If no path is provided, use the robust default path.
        # Otherwise, use the path the user provided.
        path_to_load = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        _config = load_config(path_to_load)
    return _config
