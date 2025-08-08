"""
eval_seq.py
Evaluate a trained AutoDroneAviary agent to simulate sequential target navigation
"""

import sys
import numpy as np
import time
from stable_baselines3 import PPO
from eval_helpers import *

def evaluate_sequence(model, env, n_targets, speed, gui):
    """
    Evaluates a model in a multi-target sequence navigation task
    """
    success_count = 0
    total_reward = 0
    target_rewards = []
    failure_reasons = {}

    # Generate random 3D target positions
    waypoints = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.3], size=(n_targets, 3))
    
    '''
    # Generate set waypoints
    waypoints = np.array([
        [0.5, 0.5, 1.5],
        [0.0, 1.2, 1.0],
        [0.75, 1.3, 0.95],
        [0.3, 0.3, 1.5],
        [0.95, -0.5, 1.05]
    ])
    ''' 

    print(f"Generated waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"Target {i+1}: [{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]")
    print("-" * 50)

    obs, info = env.reset()
    print(f"Start position: {info.get('start_position', 'Unknown')}")
    print("-" * 50)

    # Since we are running a single episode for multiple target
    # Increase episode length so it doesn't timeout
    env.EPISODE_LEN_SEC = 500000

    for target_idx, waypoint in enumerate(waypoints):
        env.set_target(waypoint)
        target_reward = 0
        done = False
        
        print(f"Target {target_idx + 1}/{n_targets}")
        print(f"Navigating to: [{waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}]")
        
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            target_reward += reward if reward is not None else 0.0
            step_count += 1
            
            if gui:
                time.sleep(speed)
            
            # Check if target reached
            if info.get('at_target', False):
                done = True
        
        target_rewards.append(target_reward)
        total_reward += target_reward

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
            
            # Stop episode on failure
            print(f"Target {target_idx + 1} - {ending_reason}")
            print(f"Reward: {target_reward:.2f}")
            print(f"Steps: {step_count}")
            print(f"Sequence stopped at target {target_idx + 1}")
            print("-" * 50)
            break
                    
        # Summarize episode
        print(f"Target {target_idx + 1} - {ending_reason}")
        print(f"Reward: {target_reward:.2f}")
        print(f"Steps: {step_count}")
        print("-" * 50)

        if hasattr(env, 'success_marker_placed'):
            env.success_marker_placed = False
            
    # Summarize sequence results
    success_rate = success_count / n_targets
    avg_target_reward = np.mean(target_rewards) if target_rewards else 0.0

    print("\n--- Sequence Navigation Summary ---")
    print(f"Targets Reached: {success_count}/{n_targets} ({success_rate:.1%})")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Target Reward: {avg_target_reward:.2f}")
    if target_rewards:
        print(f"Target Reward Range: {min(target_rewards):.2f} to {max(target_rewards):.2f}")

    if failure_reasons:
        print("\n--- Failure Breakdown ---")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / n_targets) * 100
            print(f"{reason}: {count} targets ({percentage:.1f}%)")

def main():
    """
    Main entry point for sequence evaluation.
    """
    args = parse_args() if len(sys.argv) > 1 else prompt_args("evaluate_sequence")
    model = PPO.load(args.model_path)
    env = create_env(gui=args.gui)
    evaluate_sequence(model, env, args.n_targets, args.speed, args.gui)
    env.close()

if __name__ == "__main__":
    main()