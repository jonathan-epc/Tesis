import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

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
    early_stopping_patience: int = Field(64, ge=0)
    clip_grad_value: float = Field(1.0, gt=0)
    weight_decay: float = Field(0.01, gt=0)
    pretrained_model_name: Union[str, None] = None
    time_limit: float = Field(86400, gt=0)
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


class DataConfig(BaseModel):
    file_name: str
    normalize_input: bool = True
    normalize_output: bool = False
    numpoints_x: int = Field(401, gt=0)
    numpoints_y: int = Field(11, gt=0)
    preload: bool = True
    inputs: List[str]
    outputs: List[str]
    scalars: List[str]
    non_scalars: List[str]


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    plot_enabled: bool = False
    save_dir: str = "plots"


class OptunaConfig(BaseModel):
    study_name: str
    n_trials: int = Field(100, gt=0)
    hyperparameter_space: Dict[str, Dict[str, Any]]
    base_storage_path: str = "sqlite:///studies/"

    @property
    def storage(self) -> str:
        return f"{self.base_storage_path}{self.study_name}.db"


class ApiKeys(BaseModel):
    wandb: Optional[str] = None


class Config(BaseModel):
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
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

    # Using a root validator to validate both 'optuna' and 'data'
    @root_validator(pre=True)
    def validate_optuna_and_data(cls, values):
        optuna_config = values.get("optuna")
        data_config = values.get("data")
        print(data_config)

        if optuna_config and data_config:
            # Access numpoints_x and numpoints_y from data_config
            numpoints_x = data_config.get("numpoints_x", 77)
            numpoints_y = data_config.get("numpoints_y", 401)

            hyperparameter_space = optuna_config.get("hyperparameter_space", {})

            # Ensure n_modes_x and n_modes_y are within valid bounds
            for mode, numpoints in zip(
                ["n_modes_x", "n_modes_y"], [numpoints_x, numpoints_y]
            ):
                if mode in hyperparameter_space:
                    hyperparameter_space[mode]["low"] = 2
                    hyperparameter_space[mode]["high"] = min(
                        numpoints, numpoints // 2 * 2
                    )
                    hyperparameter_space[mode]["step"] = (
                        2  # Ensure only even numbers are suggested
                    )

            # Update the optuna config in values
            optuna_config["hyperparameter_space"] = hyperparameter_space
            values["optuna"] = optuna_config

        return values


def add_date_to_name(name: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{date_str}"


def load_config(config_path: str = "config.yaml") -> Config:
    try:
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")

    # Handle the 'class' key in model config
    if "model" in yaml_config and "class" in yaml_config["model"]:
        yaml_config["model"]["class_name"] = yaml_config["model"].pop("class")

    # Add date to model name
    yaml_config["model"]["name"] = add_date_to_name(yaml_config["model"]["name"])

    return Config(**yaml_config)


# Global variable to store config, initialized to None
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        try:
            _config = load_config()
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")
    return _config


if __name__ == "__main__":
    config = get_config()  # Load the config when needed
    print(config.json(indent=2))
    print(f"Model name: {config.model.name}")
    print(f"Optuna study name: {config.optuna.study_name}")
    print(f"Optuna storage path: {config.optuna.storage}")