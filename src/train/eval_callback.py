import os
import sys
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

class EvalCallback(BaseCallback):
    """
    Simple callback for displaying training progress with minimal output.
    """
    def __init__(self, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        """Initialize progress bar at training start."""
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            ncols=80
        )

    def _on_step(self) -> bool:
        """Update progress bar on each step."""
        if self.pbar is not None:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        """Close progress bar at training end."""
        if self.pbar is not None:
            self.pbar.close()
