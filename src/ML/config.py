import os
from datetime import datetime
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, root_validator, validator


class TrainingConfig(BaseModel):
    kfolds: int = Field(1, ge=1)
    batch_size: int = Field(64, gt=0)
    num_epochs: int = Field(1024, gt=0)
    num_workers: int = Field(0, ge=0)
    accumulation_steps: int = Field(5, gt=0)
    learning_rate: float = Field(1e-4, gt=0)
    test_frac: float = Field(0.2, ge=0, le=1)
    validation_frac: float = Field(0.2, ge=0, le=1)
    early_stopping_patience: int = Field(64, ge=0)
    early_stopping_delta: int = Field(0, ge=0)
    clip_grad_value: float = Field(1.0, gt=0)
    weight_decay: float = Field(0.01, gt=0)
    pretrained_model_name: str | None = None
    time_limit: float = Field(86400, gt=0)
    use_physics_loss: bool = False
    lambda_physics: float = Field(0.5, ge=0)


class ModelConfig(BaseModel):
    name: str
    architecture: str
    class_name: str
    n_layers: int = Field(4, gt=0)
    n_modes_x: int = Field(72, gt=0)
    n_modes_y: int = Field(199, gt=0)
    hidden_channels: int = Field(6, gt=0)
    lifting_channels: int = Field(62, gt=0)
    projection_channels: int = Field(51, gt=0)


class ChannelConfig(BaseModel):
    width: float = Field(0.3, gt=0)
    length: float = Field(12, gt=0)
    depth: float = Field(0.3, gt=0)
    wall_thickness: float = Field(0, ge=0)


class DataConfig(BaseModel):
    file_name: str
    normalize_input: bool = True
    normalize_output: bool = False
    adimensional: bool = False
    boxcox_transform: bool = False
    numpoints_x: int = Field(401, gt=0)
    numpoints_y: int = Field(11, gt=0)
    preload: bool = True
    inputs: list[str]
    outputs: list[str]
    scalars: list[str]
    non_scalars: list[str]
    adimensionals: list[str]

    @root_validator(pre=True)
    def validate_adimensional_flag(cls, values):
        inputs = values.get("inputs", [])
        outputs = values.get("outputs", [])
        adimensionals = values.get("adimensionals", [])
        if any(item in inputs or item in outputs for item in adimensionals):
            values["adimensional"] = True
        else:
            values["adimensional"] = False
        return values


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    wandb_project: str = "Tesis"
    plot_enabled: bool = False
    save_dir: str = "plots"


class OptunaConfig(BaseModel):
    study_name: str
    study_notes: str
    n_trials: int = Field(100, gt=0)
    hyperparameter_space: dict[str, dict[str, Any]]
    base_storage_path: str = "sqlite:///studies/"

    @property
    def storage(self) -> str:
        return f"{self.base_storage_path}{self.study_name}.db"


class ApiKeys(BaseModel):
    wandb: str | None = None


class Config(BaseModel):
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    channel: ChannelConfig
    seed: int = 43
    logging: LoggingConfig
    optuna: OptunaConfig
    api_keys: ApiKeys

    @validator("api_keys", pre=True, always=True)
    def set_wandb_api_key(cls, v):
        if v is None:
            v = {}
        elif not isinstance(v, dict):
            raise ValueError("api_keys must be a dictionary")
        v["wandb"] = os.environ.get("WANDB_API_KEY", v.get("wandb"))
        return v

    @root_validator(pre=True)
    def validate_optuna_and_data(cls, values):
        optuna_config = values.get("optuna")
        data_config = values.get("data")

        if optuna_config and data_config:
            numpoints_x = data_config.get("numpoints_x", 77)
            numpoints_y = data_config.get("numpoints_y", 401)

            hyperparameter_space = optuna_config.get("hyperparameter_space", {})
            for mode, numpoints in zip(
                ["n_modes_x", "n_modes_y"], [numpoints_x, numpoints_y], strict=False
            ):
                if mode in hyperparameter_space:
                    hyperparameter_space[mode]["low"] = 2
                    hyperparameter_space[mode]["high"] = min(
                        numpoints, numpoints // 2 * 2
                    )
                    hyperparameter_space[mode]["step"] = 2

            optuna_config["hyperparameter_space"] = hyperparameter_space
            values["optuna"] = optuna_config

        return values

    def to_dict(self) -> dict[str, Any]:
        return self.dict()


def add_date_to_name(name: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{date_str}"


def load_config(config_path: str = "config.yaml") -> Config:
    try:
        with open(config_path) as file:
            yaml_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}") from exc

    if "model" in yaml_config and "class" in yaml_config["model"]:
        yaml_config["model"]["class_name"] = yaml_config["model"].pop("class")

    yaml_config["model"]["name"] = add_date_to_name(yaml_config["model"]["name"])

    return Config(**yaml_config)


_config: Config | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        try:
            _config = load_config(config_path)
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}") from e
    return _config


if __name__ == "__main__":
    config = get_config("nconfig.yml")
    print(config.json(indent=2))
