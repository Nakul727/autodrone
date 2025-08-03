import os
import sys
import numpy as np
import time
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

def create_evaluation_env(gui=False):
    """
    Create AutoDroneAviary environment
    """
    return AutoDroneAviary(
        gui=gui,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.15,
        episode_len_sec=12
    )

def evaluate_model(model_path, n_episodes=5):
    """
    Evaluate trained model performance with GUI and n_episodes
    """
    model = PPO.load(model_path)
    env = create_evaluation_env(gui=True)
    
    # Evaluation metrics
    episode_rewards = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward if reward is not None else 0.0
            done = terminated or truncated
            
            time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
    
    env.close()
    
    # Calculate statistics
    success_rate = success_count / n_episodes
    avg_reward = np.mean(episode_rewards)
    
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    return success_rate, avg_reward

if __name__ == "__main__":
    model_path = input("Enter model path: ").strip()
    #model_path = "src/train/models/autodrone_model_final"   # since training script stores model at (fixed) predefined location, makes sense to hardcode
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure training has completed successfully.")
        exit(1)

    evaluate_model(model_path)