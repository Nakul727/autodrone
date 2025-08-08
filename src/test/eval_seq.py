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

def evaluate_sequence(model, env, num_targets=5, speed=0.01, start_pos=None):
    """
    Evaluates a model in a multi-target sequence navigation task.
    
    Args:
        model: Trained PPO model.
        env: Drone environment.
        num_targets: Number of sequential targets to hit.
        speed: Delay between steps (in seconds).
        start_pos: Optional starting position.
    """

    obs, info = env.reset()
    total_reward = 0
    success_count = 0

    # Generate random 3D target positions
    waypoints = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.3], size=(num_targets, 3))

    '''
    # Generate set waypoints
    waypoints = np.array([
        [0.5, 0.5, 1.5],       # A: Start
        [0.0, 1.2, 1.0],       # B: Move forward (base of triangle)
        [0.75, 1.3, 0.95],     # C: Top corner
        [0.3, 0.3, 1.5],       # A: Start
        [0.95, -0.5, 1.05]      # D: Continue forward along triangle direction
    ])
    '''




    print(f"Generated waypoints:\n{waypoints}")

    for i, wp in enumerate(waypoints):
        env.set_target(wp)
        done = False
        print(f"\n[Target {i+1}] Navigating to {wp}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or info.get('at_target', False)
            total_reward += reward or 0.0
            time.sleep(speed)
                
        if info.get('is_success', False):
            print(f"Reached Target {i+1} :D")
            success_count += 1
            print("\n--- Sequence Navigation {i+1} ---")
            print(f"Reached {success_count}/{num_targets} targets")
            print(f"Total Reward: {reward:.2f}")
            print("\n--------------------------")
        else:
            print(f"Failed to reach Target {i+1} :(")
            print("\n--- Sequence Navigation {i+1} ---")
            print(f"Reached {success_count}/{num_targets} targets")
            print(f"Total Reward: {reward:.2f}")
            print("\n--------------------------")
            break

    print("\n--- Sequence Navigation Summary ---")
    print(f"Reached {success_count}/{num_targets} targets")
    print(f"Total Reward: {total_reward:.2f}")
    print("\n--------------------------")
    input("Press Enter to exit and close the GUI...")

def main():
    """
    Main entry point for evaluation.
    Determines whether to use CLI args or prompt-based input,
    loads the trained model, and runs evaluation based on the task.
    """
    args = hp.parse_args() if len(sys.argv) > 1 else hp.prompt_args("evaluate_sequence")

    model = PPO.load(args.model_path)

    env = hp.create_env(gui=args.gui)

    evaluate_sequence(model, env, num_targets=args.targets, speed=args.speed)

    env.close()

if __name__ == "__main__":
    main()

