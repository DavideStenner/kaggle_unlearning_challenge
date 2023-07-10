import gc
import json
import torch
import warnings

from typing import Dict

def get_config()-> dict:
    with open('config.json', 'r') as file:
        config = json.load(file)
    config['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"
    return config

def get_config_model()-> dict:
    with open('config_model.json', 'r') as file:
        config_model = json.load(file)

    return config_model

def get_all_config() -> Dict[str, dict]:
    config = get_config()
    config_model = get_config_model()
    return {'config': config, 'config_model': config_model}

def free_memory(*args) -> None:

    if args is not None:
        for variable in args:
            del variable
    
    gc.collect()
    torch.cuda.empty_cache()

def ignore_warning():
    warnings.filterwarnings("ignore", message=".*No audio backend is available.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")
