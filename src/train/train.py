"""
train.py
Main file for training AutoDroneAviary agent
"""

import os
import time
from datetime import datetime

from setup_training import (
    setup_directories, 
    setup_environments, 
    setup_callbacks, 
    setup_ppo_model
)
from post_train_graphs import generate_all_graphs

def train_model(total_timesteps=500000, model_name="autodrone_ppo"):
    """
    Main training function
    """
    print("=" * 50)
    print("AUTODRONE TRAINING")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Setup
    model_dir, log_dir, graphs_dir = setup_directories(model_name)
    train_env, eval_env = setup_environments(log_dir)
    model = setup_ppo_model(train_env, log_dir)
    callbacks = setup_callbacks(total_timesteps, model_dir, log_dir, eval_env, model_name)
    
    start_time = time.time()
    
    try:
        # Training
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="PPO"
        )
        
        training_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        print(f"Duration: {training_time/60:.1f} minutes")

        final_model_path = os.path.join(model_dir, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}\n")
        
        episode_log_dir = os.path.join(log_dir, "episode_log.csv")
        if os.path.exists(episode_log_dir):
            print(f"Episode log saved to: {episode_log_dir}")
        
        final_graphs_dir = generate_all_graphs(log_dir, graphs_dir, model_name)
        print(f"Graphs saved to: {final_graphs_dir}")
        return
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user\n")

        interrupted_path = os.path.join(model_dir, "interrupted_model.zip")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}")
        
        episode_log_dir = os.path.join(log_dir, "episode_log.csv")
        if os.path.exists(episode_log_dir):
            print(f"Episode log saved to: {episode_log_dir}")
        
        try:
            final_graphs_dir = generate_all_graphs(log_dir, graphs_dir, model_name)
            print(f"Graphs saved to: {final_graphs_dir}")
            return
        except:
            return
    
    finally:
        train_env.close()
        eval_env.close()

def main():
    timesteps_input = input("\nTotal timesteps (default 500000): ").strip()
    timesteps = int(timesteps_input) if timesteps_input else 500000
    
    model_name_input = input("Model name (default 'autodrone_ppo'): ").strip()
    model_name = model_name_input if model_name_input else "autodrone_ppo"
    
    print(f"\nTraining: {model_name} for {timesteps:,} timesteps")
    confirm = input("Continue? (y/N): ").strip().lower()

    if confirm in ['y', 'yes']:
        train_model(total_timesteps=timesteps, model_name=model_name)
    else:
        print("Training cancelled")

if __name__ == "__main__":
    main()