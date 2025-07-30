import os
import sys
import time
import argparse
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.AutoDroneAviary import AutoDroneAviary

def create_env(gui=False, record=False, start=None):
    return AutoDroneAviary(
        gui=gui,
        record=record,
        initial_xyzs=start,
        target_bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.5, 1.5]]),
        success_threshold=0.15,
        episode_len_sec=12
    )

def evaluate_single(model, env, episodes, speed, start_pos=None):
    success_count = 0
    rewards = []

    for ep in range(episodes):
        if start_pos is not None:
            env.INIT_XYZS = init_pos(start_pos)
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

def evaluate_sequence(model, env, num_targets=5, speed=0.01, start_pos=None):
    if start_pos is not None:
        env.INIT_XYZS = init_pos(start_pos)
    obs, info = env.reset()
    total_reward = 0
    success_count = 0
    waypoints = np.random.uniform(low=[-1, -1, 0.5], high=[1, 1, 1.5], size=(num_targets, 3))

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

        if info.get('at_target', False):
            print(f"Reached Target {i+1} :D")
            success_count += 1
        else:
            print(f"Failed to reach Target {i+1} :(")
            break  # Stop sequence if failed

    print("\n--- Sequence Navigation Summary ---")
    print(f"Reached {success_count}/{num_targets} targets")
    print(f"Total Reward: {total_reward:.2f}")
    print("\n--------------------------")


def init_pos(pos):
    '''
    Processes tuple start position.
    (999, 999, 999) results in random position.
    Returns np.array of shape (1, 3)
    '''
    if np.array_equal(pos, np.array([999, 999, 999])):
        xyz = np.array([[np.random.uniform(-1.0, 1.0),
                         np.random.uniform(-1.0, 1.0),
                         np.random.uniform(0.3, 2.0)]])
    else:
        xyz = np.array([pos])
    return xyz


def parse_args():
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
    targets = 6
    episodes = 5

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
    
    if task == "evaluate_sequence":
        try:
            targets = int(input("Number of targets (default: 6): ").strip() or 6)
        except:
            targets = 6

    else:
        try:
            episodes = int(input("Number of episodes (default: 5): ").strip() or 5)
        except:
            episodes = 6


    gui_input = input("Enable GUI? [y/N]: ").strip().lower()
    gui = gui_input == "y"

    try:
        speed = float(input("Step speed in seconds (default: 0.01): ").strip() or 0.01)
    except:
        speed = 0.01

    record_input = input("Enable recording? [y/N]: ").strip().lower()
    record = record_input == "y"

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

def main():
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = prompt_args()

    model = PPO.load(args.model_path)
    env = create_env(gui=args.gui, record=args.record)

    if args.task == "evaluate_single":
        evaluate_single(model, env, args.episodes, args.speed, start_pos=args.start_pos)
    else:
        evaluate_sequence(model, env, num_targets=args.targets, speed=args.speed, start_pos=args.start_pos)

    env.close()

if __name__ == "__main__":
    main()
