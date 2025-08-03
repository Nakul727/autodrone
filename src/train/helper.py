import os
import sys
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

def create_env(gui=False, **kwargs):
    """Create AutoDroneAviary environment with default parameters."""
    return AutoDroneAviary(
        gui=gui,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.05,
        episode_len_sec=12,
        **kwargs
    )


def path_from_dir(filepath: str, directory_name: str) -> Path:
    """
    Trims the path to start from the 'models' directory.
    If 'models' is not in the path, returns the full resolved path.
    
    Args:
        filepath (str): The path to trim.
        directory_name (str): The directory trim to
    
    Returns:
        Path: The trimmed path starting from directory_name, or the full path.
    """
    filepath = Path(filepath).resolve()
    parts = filepath.parts

    if directory_name in parts:
        models_index = parts.index(directory_name)
        return Path(*parts[models_index:])
    else:
        return filepath
