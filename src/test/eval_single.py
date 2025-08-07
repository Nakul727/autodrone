import numpy as np
import time
from stable_baselines3 import PPO
import helper as hp
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary
from src.envs.AutoDroneAviaryGui import AutoDroneAviaryGui

def evaluate_single(model, env, episodes, speed):
    """
    Evaluates a model over a number of individual episodes.
    
    Args:
        model: Trained PPO model.
        env: Drone environment.
        episodes: Number of episodes to run.
        speed: Delay between steps (in seconds).
        start_pos: Optional starting position.
    """
    success_count = 0
    rewards = []

    for ep in range(episodes):

        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward or 0.0
            time.sleep(speed)

        rewards.append(total_reward)
        if info.get('is_success', False):
            success_count += 1

        print(f"[Episode {ep+1}] Reward: {total_reward:.2f}, Success: {info.get('is_success', False)}, Time: {info.get('time_elapsed', 0):.2f}s")

    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {success_count}/{episodes} = {success_count / episodes:.2%}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print("\n--------------------------")

def main():
    """
    Main entry point for evaluation.
    Determines whether to use CLI args or prompt-based input,
    loads the trained model, and runs evaluation based on the task.
    """
    args = hp.parse_args() if len(sys.argv) > 1 else hp.prompt_args("eval_single")

    model = PPO.load(args.model_path)

    env = hp.create_env(gui=args.gui)

    evaluate_single(model, env, args.episodes, args.speed)

    env.close()

if __name__ == "__main__":
    main()

