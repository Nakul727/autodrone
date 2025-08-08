"""
eval_single.py
Evaluate a trained AutoDroneAviary agent to simulate single complete episode
"""

import sys
import numpy as np
import time
from stable_baselines3 import PPO
from eval_helpers import *

def evaluate_single(model, env, n_episodes, speed, gui):
    """
    Evaluates a model over a number of individual episodes
    """
    success_count = 0
    episode_rewards = []
    failure_reasons = {} 

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"Target: {info['target_position']}")
        print(f"Start: {info['start_position']}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward if reward is not None else 0.0
            
            if gui:
                time.sleep(speed)
        
        episode_rewards.append(episode_reward)

        at_target = info.get('at_target', False)
        termination_reason = info.get('termination_reason', None)
        truncation_reason = info.get('truncation_reason', None)
        
        if at_target:
            success_count += 1
            if truncation_reason == "time_limit":
                ending_reason = f"SUCCESS: reached target (time_limit)"
            elif terminated and termination_reason:
                ending_reason = f"SUCCESS: {termination_reason}"
            else:
                ending_reason = f"SUCCESS: reached target"
        else:
            if truncation_reason and truncation_reason != "time_limit":
                ending_reason = f"FAILURE: {truncation_reason}"
                failure_reasons[truncation_reason] = failure_reasons.get(truncation_reason, 0) + 1
            else:
                ending_reason = f"FAILURE: did not reach target"
                failure_reasons["target_not_reached"] = failure_reasons.get("target_not_reached", 0) + 1
        
        # Summarize episode
        print(f"Episode {episode + 1} - {ending_reason}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Episode length: {info.get('episode_step', 0)} steps")
        print(f"Time elapsed: {info.get('time_elapsed', 0):.2f}s")
        print("-" * 50)

    # Summarize evaluation results
    success_rate = success_count / n_episodes
    avg_reward = np.mean(episode_rewards)

    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Reward Range: {min(episode_rewards):.2f} to {max(episode_rewards):.2f}")

    if failure_reasons:
        print("\n--- Failure Breakdown ---")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / n_episodes) * 100
            print(f"{reason}: {count} episodes ({percentage:.1f}%)")

def main():
    """
    Main entry point for evaluation.
    """
    args = parse_args() if len(sys.argv) > 1 else prompt_args("eval_single")
    model = PPO.load(args.model_path)
    env = create_env(gui=args.gui)
    evaluate_single(model, env, args.n_episodes, args.speed, args.gui)
    env.close()

if __name__ == "__main__":
    main()