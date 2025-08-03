import os
import sys
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
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
