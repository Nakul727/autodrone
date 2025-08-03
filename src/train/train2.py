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
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    BaseCallback
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import signal

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

class PerformanceCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(PerformanceCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.step_count = 0
    
    def _on_step(self) -> bool:
        # just need to track step count for now
        self.step_count += 1
        return True 


class ProgressCallback(BaseCallback):
    def __init__(self, progress_bar):
        super(ProgressCallback, self).__init__()
        self.progress_bar = progress_bar
        self.last_update = 0
    
    def _on_step(self) -> bool:
        # Update progress bar every 1000 steps
        if self.n_calls - self.last_update >= 1000:
            self.progress_bar.update(1000)
            self.last_update = self.n_calls
        return True
    


def setup_checkpoint_callback(save_freq=10000, save_path="models/"):
    """
    Create callback that saves model checkpoints every save_freq steps
    """
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,  # Save every 10,000 steps
        save_path=save_path,  # Directory to save checkpoints
        name_prefix="autodrone_model"  # Prefix for checkpoint files
    )
    return checkpoint_callback



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

# def train_model(total_timesteps=500000):   fallback simple implementation
#     """
#     Train the PPO model
#     """
#     print(f"Starting training for {total_timesteps:,} timesteps...")
#     start_time = time.time()
    
#     env = create_train_env()
#     model = PPO("MlpPolicy", env, verbose=0)
#     model.learn(total_timesteps=total_timesteps)
#     model.save("autodrone_model")
    
#     elapsed_time = time.time() - start_time
#     print(f"Training completed in {elapsed_time/60:.1f} minutes")
    
#     return model

def train_model(total_timesteps=500000, save_freq=10000, log_dir="logs/",   # save_freq is the num of steps for checkpoints, log_dir is dir for Tensorboard logs 
                show_progress=True, handle_interrupts=True):                # show_progress shows progress bar, handle_interrupt controls whether to handle Ctrl+C gracefully
    """
    Enhanced training function with all features

    total timesteps: self explanatory
    save_freq: number of steps after which mdodel is saved (checkpoints)
    log_dir: directory for  TensorBoard logs
    show_progess: whether to show progress bar
    handle_interrupt: controls whether to handle Ctrl+C gracefully ~ warm shutdown
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models/", exist_ok=True)
    
    configure(log_dir, ["stdout", "csv", "tensorboard"])  # uses SB3s configure function, log output to console, csv file and tensorboard format (log_dir)
    
    env = Monitor(create_train_env())      # monitors (SB3) the env created ~ AutoDroneAviary, auto tracks episode rewards, lengths and success rates, used by SB3 logger to actually write the logs
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)   # actual PPO model, with the monitored environment, and tensorboard logging enabled

    
    callbacks = []
    performance_callback = PerformanceCallback()  # called every traiing step, if reward exists, adds it to the list ~ tracks training performance
    callbacks.append(performance_callback)
    
    checkpoint_callback = CheckpointCallback(    # iteratively saves model at checkpoints
        save_freq=save_freq,
        save_path="models/",
        name_prefix="autodrone_model"
    )
    callbacks.append(checkpoint_callback)

    if show_progress:
        progress_bar = tqdm(total=total_timesteps, desc="Training Progress")  # uses tqdm to show progress bar
        progress_callback = ProgressCallback(progress_bar)                    # updates progress bar every 1000 steps, showing competion %
        callbacks.append(progress_callback)
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    if show_progress:
        progress_bar.close()
    
    model.save("models/autodrone_model_final")     # saves fully trained model, can be loaded later with PPO.load
    
    return model

if __name__ == "__main__":
    model = train_model()  # default call for testing
    
    print(f"\nTraining completed successfully!")
    print(f"Final model saved as: models/autodrone_model_final.zip")
    print(f"Checkpoints saved in: models/")
    print(f"Logs available at: logs/")
    print(f"View training logs with: tensorboard --logdir logs/")