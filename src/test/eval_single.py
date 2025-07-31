import time
import numpy as np
from helper import init_pos

def evaluate_single(model, env, episodes, speed, start_pos=None):
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

