import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, validator


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
    normalize: List[bool] = [True, False]
    numpoints_x: int = Field(401, gt=0)
    numpoints_y: int = Field(11, gt=0)
    variables: List[str]
    variable_units: List[str]
    parameters: List[str]


class LoggingConfig(BaseModel):
    use_wandb: bool = True
    plot_enabled: bool = False


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
        v["wandb"] = os.environ.get("WANDB_API_KEY", v.get("wandb"))
        return v


def add_date_to_name(name: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{date_str}"


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    # Handle the 'class' key in model config
    if "model" in yaml_config and "class" in yaml_config["model"]:
        yaml_config["model"]["class_name"] = yaml_config["model"].pop("class")

    # Add date to model name
    yaml_config["model"]["name"] = add_date_to_name(yaml_config["model"]["name"])

    return Config(**yaml_config)


# Global config instance
config: Config = load_config()


def get_config() -> Config:
    return config


if __name__ == "__main__":
    print(config.json(indent=2))
    print(f"Model name: {config.model.name}")
    print(f"Optuna study name: {config.optuna.study_name}")
    print(f"Optuna storage path: {config.optuna.storage}")