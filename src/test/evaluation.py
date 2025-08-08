import os
import sys
import numpy as np
import time
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary
from src.envs.AutoDroneAviaryGui import AutoDroneAviaryGui

def create_evaluation_env(gui=False):
    """
    Create AutoDroneAviary environment with or without GUI
    """
    if gui:
        # Use GUI version with visual target markers
        return AutoDroneAviaryGui(
            gui=True,
            target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
            success_threshold=0.15,
            episode_len_sec=12,
            random_xyz=False
        )
    else:
        # Use headless version for fast evaluation
        return AutoDroneAviary(
            gui=False,
            target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
            success_threshold=0.15,
            episode_len_sec=12,
            random_xyz=False
        )

def evaluate_model(model_path, n_episodes=5, gui=True):
    """
    Evaluate trained model performance with optional GUI and n_episodes
    """
    model = PPO.load(model_path)
    env = create_evaluation_env(gui=gui)
    
    # Evaluation metrics
    episode_rewards = []
    success_count = 0
    failure_reasons = {} 
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"Target: {info['target_position']}")
        print(f"Start: {info['start_position']}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward if reward is not None else 0.0
            done = terminated or truncated
            
            if gui:
                time.sleep(0.02)
        
        episode_rewards.append(episode_reward)
        
        # Use info to determine success (distance-based, not termination-based)
        final_distance = info.get('distance_to_target', float('inf'))
        is_success = final_distance < 0.15  # Use your success threshold
        
        # Get episode ending reason
        termination_reason = info.get('termination_reason', None)
        truncation_reason = info.get('truncation_reason', None)
        
        # Determine success based on final distance, not termination type
        if is_success:
            success_count += 1
            if truncation_reason == "time_limit":
                ending_reason = f"SUCCESS: reached target (time_limit)"
            elif terminated and termination_reason:
                ending_reason = f"SUCCESS: {termination_reason}"
            else:
                ending_reason = f"SUCCESS: reached target"
        else:
            # Only count as failure if didn't reach target
            if truncation_reason and truncation_reason != "time_limit":
                ending_reason = f"FAILURE: {truncation_reason}"
                failure_reasons[truncation_reason] = failure_reasons.get(truncation_reason, 0) + 1
            else:
                ending_reason = f"FAILURE: did not reach target"
                failure_reasons["target_not_reached"] = failure_reasons.get("target_not_reached", 0) + 1
        
        print(f"Episode {episode + 1} - {ending_reason}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Final distance: {final_distance:.3f}m")
        print(f"Episode length: {info.get('episode_step', 0)} steps")
        print(f"Time elapsed: {info.get('time_elapsed', 0):.2f}s")
        print("-" * 50)
    
    env.close()
    
    # Calculate statistics
    success_rate = success_count / n_episodes
    avg_reward = np.mean(episode_rewards)
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Reward Range: {min(episode_rewards):.2f} to {max(episode_rewards):.2f}")
    
    # Show failure breakdown
    if failure_reasons:
        print(f"\n=== FAILURE BREAKDOWN ===")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / n_episodes) * 100
            print(f"{reason}: {count} episodes ({percentage:.1f}%)")
    
    return success_rate, avg_reward

def evaluate_headless(model_path, n_episodes=100):
    """
    Fast headless evaluation for large-scale testing
    """
    print(f"Running headless evaluation with {n_episodes} episodes...")
    return evaluate_model(model_path, n_episodes=n_episodes, gui=False)

if __name__ == "__main__":
    model_path = input("\nEnter model path: ").strip()
    
    # Ask user for evaluation mode
    mode = input("Choose mode: (1) GUI visualization (2) Headless fast evaluation: ").strip()
    
    if mode == "2":
        n_episodes = int(input("Number of episodes (default 100): ") or "100")
        print('\n')
        evaluate_headless(model_path, n_episodes)
    else:
        n_episodes = int(input("Number of episodes (default 5): ") or "5")
        print('\n')
        evaluate_model(model_path, n_episodes, gui=True)