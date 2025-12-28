import os
import sys
import torch
import pandas as pd 
import numpy as np
import random
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

def setup_environment(config):
    """
    Setup training environment:
    - device (CPU or GPU)
    - reproducibility
    - output directories
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Reproducibility
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")

    # Output directories
    output_root = Path(config.get("output_dir", "results"))
    checkpoints_dir = output_root / "checkpoints"
    logs_dir = output_root / "training_logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Checkpoints directory: {checkpoints_dir}")
    print(f"[INFO] Logs directory: {logs_dir}")

    # Save paths in config for later use
    config["checkpoints_dir"] = checkpoints_dir
    config["logs_dir"] = logs_dir

    return device