"""
helpers.py

"""

import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary
from src.envs.AutoDroneAviaryGui import AutoDroneAviaryGui

def create_env(gui=False):
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

def parse_args():
    """
    Parses command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained drone policies.")
    parser.add_argument("model_path", nargs='?', help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--targets", type=int, default=6, help="Number of targets")
    parser.add_argument("--gui", action="store_true", help="Choose mode: (1) GUI visualization (2) Headless fast evaluation:")
    parser.add_argument("--speed", type=float, default=0.01, help="Delay per step (in seconds)")
    
    args = parser.parse_args()
    return args

def prompt_args(task):
    """
    Prompts the user interactively for evaluation configuration.
    """
    print("\n")
    print("=" * 50)
    print("AUTODRONE EVALUATION")
    print("=" * 50)

    model_path = input("Enter model path (e.g. ./models/autodrone.zip): ").strip()

    n_episodes = 5
    n_targets = 6
    if task == "evaluate_sequence":
        try:
            n_targets = int(input("Number of targets (default: 6): ").strip() or 6)
        except:
            n_targets = 6
    else:
        try:
            n_episodes = int(input("Number of episodes (default: 5): ").strip() or 5)
        except:
            n_episodes = 5

    gui = input("Choose mode: (1) GUI visualization (2) Headless fast evaluation: ").strip().lower() == "1"
    try:
        speed = float(input("Step speed in seconds (default: 0.01): ").strip() or 0.01)
    except:
        speed = 0.01

    return argparse.Namespace(
        model_path=model_path,
        n_episodes=n_episodes,
        n_targets=n_targets,
        gui=gui,
        speed=speed,
    )