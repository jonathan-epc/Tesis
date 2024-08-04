# config.py

import os
import yaml
import torch
from typing import Any, Dict

def load_config() -> Dict[str, Any]:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    config['api_keys']['wandb'] = os.environ.get('WANDB_API_KEY')
    
    return config

CONFIG = load_config()