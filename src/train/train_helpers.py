"""
setup_training.py
Helper functions to create and setup directories, training and train eval environments, 
callbacks and PPO model architecture
"""

import os
import sys
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary
from custom_callbacks import TrainingProgressCallback, EnhancedEvalCallback, EpisodeLoggingCallback 

def create_env():
    """
    Create AutoDroneAviary environment (no GUI)
    """
    return AutoDroneAviary(
        gui=False,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.15,
        episode_len_sec=12,
        random_xyz=True
    )

def setup_directories(model_name: str):
    """
    Create directories for models, logs, and graphs
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = os.path.join(script_dir, "models", f"{model_name}_{timestamp}")
    log_dir = os.path.join(script_dir, "logs", f"{model_name}_{timestamp}")
    graphs_dir = os.path.join(script_dir, "graphs", f"{model_name}_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    return model_dir, log_dir, graphs_dir

def setup_environments(log_dir: str):
    """
    Create training and evaluation environments
    """
    train_env = Monitor(create_env(), log_dir)
    eval_env = Monitor(create_env(), os.path.join(log_dir, "eval"))
    return train_env, eval_env

def setup_callbacks(total_timesteps, model_dir, log_dir, eval_env, model_name):
    """
    Setup training callbacks
    """ 
    # Enhanced evaluation callback 
    eval_callback = EnhancedEvalCallback(
        eval_env=eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=model_name
    )
    
    # Progress callback
    progress_callback = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        verbose=1
    )
    
    # Episode logging callback
    episode_logger = EpisodeLoggingCallback(
        log_dir=log_dir,
        verbose=1
    )
    
    return [eval_callback, checkpoint_callback, progress_callback, episode_logger]

def setup_ppo_model(train_env, log_dir: str):
    """
    Create PPO model with optimized hyperparameters
    """
    return PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="auto"
    )