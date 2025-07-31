"""
train.py
This file will train the model using RL algorithm like PPO
based on the AutoDroneAviary env. 
"""

import os
import sys
import numpy as np
import time
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

def create_train_env():
    """
    Create AutoDroneAviary environment
    """
    return AutoDroneAviary(
        gui=False,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.05,
        episode_len_sec=12
    )

def train_model(total_timesteps=500000):
    """
    Train the PPO model
    """
    print(f"Starting training for {total_timesteps:,} timesteps...")
    start_time = time.time()
    
    env = create_train_env()
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    model.save("autodrone_model")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/60:.1f} minutes")
    
    return model

if __name__ == "__main__":
    model = train_model()
    print("Model saved as 'autodrone_model.zip'")