import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

def create_env(gui=False, record=False, start=None):
    """
    Creates the drone environment with standard settings.

    Args:
        gui: Enable GUI display.
        record: Enable recording (if implemented).
        start: Optional start position override.

    Returns:
        Configured AutoDroneAviary environment.
    """
    return AutoDroneAviary(
        gui=gui,
        record=record,
        initial_xyzs=start,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.15,
        episode_len_sec=12
    )

def init_pos(pos):
    """
    Converts a user-supplied tuple into a 1x3 array.
    If (999, 999, 999) is given, generate a random position.

    Args:
        pos: Tuple or array of (x, y, z)

    Returns:
        Numpy array of shape (1, 3)
    """
    if np.array_equal(pos, np.array([999, 999, 999])):
        return np.array([[np.random.uniform(-1.0, 1.0),
                          np.random.uniform(-1.0, 1.0),
                          np.random.uniform(0.3, 2.0)]])
    else:
        return np.array([pos])

def parse_args():
    """
    Parses command-line arguments for evaluation.

    Returns:
        argparse.Namespace with parsed values.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained drone policies.")
    parser.add_argument("model_path", nargs='?', help="Path to trained model")
    parser.add_argument("--task", choices=["evaluate_single", "evaluate_sequence"], default="evaluate_single", help="Task type")
    parser.add_argument("--start_pos", type=lambda s: np.array(eval(s)), default=None,
        help="""Starting position as a tuple, e.g. "(1.0, 1.0, 1.2)". Use "(999, 999, 999)" for random.""")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--targets", type=int, default=6, help="Number of targets")
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    parser.add_argument("--speed", type=float, default=0.01, help="Delay per step (in seconds)")
    parser.add_argument("--record", action="store_true", help="Enable recording (if implemented)")
    return parser.parse_args()

def prompt_args():
    """
    Prompts the user interactively for evaluation configuration.

    Returns:
        argparse.Namespace with collected values.
    """
    print("==== AutoDrone Evaluation Interface ====")

    model_path = input("Enter model path (e.g. ./models/autodrone.zip): ").strip()

    task = input("Choose task [evaluate_single / evaluate_sequence] (default: evaluate_single): ").strip()
    if task not in ["evaluate_single", "evaluate_sequence"]:
        task = "evaluate_single"

    start_pos_input = input('Enter start position as tuple (e.g. "1.0, 1.0, 1.2") or "random": ').strip()
    if start_pos_input.lower() == "random":
        start_pos = np.array([999, 999, 999])
    elif start_pos_input:
        try:
            start_pos = np.array([float(x) for x in start_pos_input.split(",")])
        except:
            print("Invalid format. Falling back to random start.")
            start_pos = np.array([999, 999, 999])
    else:
        start_pos = None

    episodes, targets = 5, 6
    if task == "evaluate_sequence":
        try:
            targets = int(input("Number of targets (default: 6): ").strip() or 6)
        except:
            targets = 6
    else:
        try:
            episodes = int(input("Number of episodes (default: 5): ").strip() or 5)
        except:
            episodes = 5

    gui = input("Enable GUI? [y/N]: ").strip().lower() == "y"

    try:
        speed = float(input("Step speed in seconds (default: 0.01): ").strip() or 0.01)
    except:
        speed = 0.01

    record = input("Enable recording? [y/N]: ").strip().lower() == "y"

    return argparse.Namespace(
        model_path=model_path,
        task=task,
        start_pos=start_pos,
        episodes=episodes,
        targets=targets,
        gui=gui,
        speed=speed,
        record=record
    )

