import os
import sys
import numpy as np
import time
import eval_callback as ecb
import enhanced_eval_callback as eecb
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from helper import create_env


# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary


def train_model(total_timesteps=100000, model_name="autodrone_ppo"):
    """Train the PPO model with minimal output."""
    
    print("=" * 50)
    print("AUTODRONE TRAINING")
    print("=" * 50)
    
    # Get the directory where this script is located (src/train/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories for saving models and logs in src/train/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(script_dir, "models", f"{model_name}_{timestamp}")
    log_dir = os.path.join(script_dir, "logs", f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Model save path: {save_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    
    # Create environments
    env = create_env(gui=False)
    env = Monitor(env, log_dir)
    eval_env = create_env(gui=False)
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
    # Setup callbacks
    simple_eval_callback = ecb.EvalCallback(total_timesteps)
    
    enhanced_eval_callback = eecb.EnhancedEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10000 steps
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # Save checkpoint every 20000 steps
        save_path=save_dir,
        name_prefix=model_name,
        verbose=0
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
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
        device="auto"
    )

    # Start training
    print(f"\nStarting training...")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[simple_eval_callback, enhanced_eval_callback, checkpoint_callback],
            tb_log_name="PPO"
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.1f} minutes")
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model.zip")
        model.save(final_model_path)

        print(f"Model saved: {final_model_path}")
        
        return model, save_dir, log_dir
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted!")
        interrupted_model_path = os.path.join(save_dir, "interrupted_model.zip")
        model.save(interrupted_model_path)
        print(f"Model saved: {interrupted_model_path}")
        return model, save_dir, log_dir

def main():
    """Main training pipeline."""
    
    print("AutoDrone Training Pipeline")
    
    # Get training parameters
    timesteps = int(input("Timesteps (default 100000): ") or "100000")
    model_name = input("Model name (default 'autodrone_ppo'): ").strip() or "autodrone_ppo"
    
    # Train model
    model, save_dir, log_dir = train_model(total_timesteps=timesteps, model_name=model_name)
    
    print(f"\nTraining complete!")
    print(f"Model directory: {save_dir}")
    print(f"Log directory: {log_dir}")

if __name__ == "__main__":
    main()
