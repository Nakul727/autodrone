import numpy as np
import time
from helper import init_pos

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
    if start_pos is not None:
        env.INIT_XYZS = init_pos(start_pos)

    obs, info = env.reset()
    total_reward = 0
    success_count = 0

    # Generate random 3D target positions
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

        if info.get('is_success', False):
            print(f"Reached Target {i+1} :D")
            success_count += 1
        else:
            print(f"Failed to reach Target {i+1} :(")
            break

    print("\n--- Sequence Navigation Summary ---")
    print(f"Reached {success_count}/{num_targets} targets")
    print(f"Total Reward: {total_reward:.2f}")
    print("\n--------------------------")

