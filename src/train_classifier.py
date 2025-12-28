import os
import sys
import torch
import pandas as pd 
import numpy as np

import yaml
from pathlib import Path


def load_training_config(
    training_cfg_path="config/training.yaml",
    model_cfg_path="config/model.yaml"
):
    """
    Load training and model configuration files
    and merge them into a single config dictionary.
    """

    training_cfg_path = Path(training_cfg_path)
    model_cfg_path = Path(model_cfg_path)

    if not training_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {training_cfg_path}")

    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_cfg_path}")

    # Load YAML files
    with open(training_cfg_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Merge configs
    config = {
        **training_cfg,
        **model_cfg
    }

    return config
