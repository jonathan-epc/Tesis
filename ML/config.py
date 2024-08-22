import os
from typing import Any, Dict, List, Optional

import torch
import yaml
from pydantic import BaseModel, Field, validator


class TrainingConfig(BaseModel):
    kfolds: int = 1
    batch_size: int = 64
    num_epochs: int = 1024
    num_workers: int = 0
    accumulation_steps: int = 5
    learning_rate: float = 2.847859594257385e-5
    test_frac: float = 0.2
    early_stopping_patience: int = 64


class ModelConfig(BaseModel):
    name: str = "FNOjustUV"
    architecture: str = "FNO"
    class_name: str = "FNOnet"
    n_layers: int = 4
    n_modes_x: int = 72
    n_modes_y: int = 199
    hidden_channels: int = 6
    lifting_channels: int = 62
    projection_channels: int = 51


class DataConfig(BaseModel):
    file_name: str = "simulation_data_noise.hdf5"
    normalize: List[bool] = [True, False]
    numpoints_x: int = 401
    numpoints_y: int = 11
    variables: List[str] = ["F", "H", "Q", "S", "U", "V"]
    variable_units: List[str] = ["dimensionless", "H", "Q", "S", "U", "V"]
    parameters: List[str] = ["H0", "Q0", "SLOPE", "n"]


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    plot_enabled: bool = False


class HyperparameterConfig(BaseModel):
    type: str
    low: float
    high: float
    log: Optional[bool] = None


class OptunaConfig(BaseModel):
    storage: str = "sqlite:///studies/study-justUV.db"
    study_name: str = "study-justUV"
    n_trials: int = 100
    hyperparameter_space: Dict[str, HyperparameterConfig]


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
        v = v or {}
        v["wandb"] = os.environ.get("WANDB_API_KEY")
        return v


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    # Convert 'class' key to 'class_name' in model config
    if "model" in yaml_config and "class" in yaml_config["model"]:
        yaml_config["model"]["class_name"] = yaml_config["model"].pop("class")

    return Config(**yaml_config)


# Create a global instance of the config
config = load_config()


def get_config() -> Config:
    return config