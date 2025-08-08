"""
custom_callbacks.py
Custom callbacks for drone training
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

class TrainingProgressCallback(BaseCallback):
    """
    Progress bar callback for training
    """
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        
    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            ncols=100
        )
        
    def _on_step(self) -> bool:
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return True
    
    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()

class EnhancedEvalCallback(EvalCallback):
    """
    Enhanced evaluation callback with logging
    """
    def __init__(self, *args, verbose: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_reward = -float('inf')
        self.best_success_rate = 0.0
        self.eval_count = 0
        self.verbose = verbose
        self.eval_successes = []
        
    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        evaluated = (
            self.eval_freq > 0 and
            self.n_calls % self.eval_freq == 0 and
            len(self.evaluations_results) > 0
        )

        if evaluated:
            self.eval_count += 1
            latest_rewards = self.evaluations_results[-1]
            avg_reward = np.mean(latest_rewards)
            std_reward = np.std(latest_rewards)
            
            # Get success count from at_target info
            success_count = len(self.eval_successes)
            success_rate = success_count / len(latest_rewards)
            
            print(f"\n{'='*60}")
            print(f"Evaluation #{self.eval_count} at timestep {self.num_timesteps:,}")
            print(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
            print(f"Success Rate: {success_rate:.1%} ({success_count}/{len(latest_rewards)})")
            print(f"Min Reward: {np.min(latest_rewards):.3f}")
            print(f"Max Reward: {np.max(latest_rewards):.3f}")
            
            # Update and log best reward
            if avg_reward > self.best_reward:
                improvement = avg_reward - self.best_reward
                self.best_reward = avg_reward
                print(f"NEW BEST REWARD! Improvement: +{improvement:.3f}")
            else:
                deficit = self.best_reward - avg_reward
                print(f"Best reward so far: {self.best_reward:.3f} (current is -{deficit:.3f})")
            
            # Clear successes for next evaluation
            self.eval_successes = []
            
            print(f"{'='*60}")
        
        return continue_training
    
    def _on_eval_episode_end(self, episode_info):
        if episode_info.get('at_target', False):
            self.eval_successes.append(True)

class EpisodeLoggingCallback(BaseCallback):
    """
    Callback to log detailed episode information
    """
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_data = []
        self.current_episode = 0
        self.episode_step = 0
        
        # Reward tracking for current episode
        self.current_episode_reward = 0.0
        
        # Metrics tracking
        self.metrics_window = 100
        self.metrics_log = []
        
    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get('infos', [{}])
        info = infos[0] if infos and len(infos) > 0 else {}
        
        # Get episode end flags
        dones = self.locals.get('dones', [False])
        episode_ended = dones[0] if dones else False
        
        # Get current step reward and accumulate
        rewards = self.locals.get('rewards', [0])
        current_reward = rewards[0] if rewards else 0
        self.current_episode_reward += float(current_reward)
        
        # If episode ended, log episode summary
        if episode_ended:
            episode_summary = {
                'episode': self.current_episode,
                'final_timestep': self.num_timesteps,
                'episode_length': self.episode_step + 1,
                
                # Target and position data
                'target_x': info.get('target_position', [0, 0, 0])[0],
                'target_y': info.get('target_position', [0, 0, 0])[1], 
                'target_z': info.get('target_position', [0, 0, 0])[2],
                'final_x': info.get('current_position', [0, 0, 0])[0],
                'final_y': info.get('current_position', [0, 0, 0])[1],
                'final_z': info.get('current_position', [0, 0, 0])[2],
                
                # Distance metrics
                'final_distance': info.get('distance_to_target', 0.0),
                'best_distance': info.get('best_distance', 0.0),
                'initial_distance': info.get('initial_distance', 0.0),
                'progress_ratio': info.get('progress_ratio', 0.0),
                
                # Performance metrics
                'final_speed': info.get('current_speed', 0.0),
                'velocity_x': info.get('velocity', [0, 0, 0])[0],
                'velocity_y': info.get('velocity', [0, 0, 0])[1],
                'velocity_z': info.get('velocity', [0, 0, 0])[2],
                'attitude_roll': info.get('attitude', [0, 0, 0])[0],
                'attitude_pitch': info.get('attitude', [0, 0, 0])[1],
                'attitude_yaw': info.get('attitude', [0, 0, 0])[2],
                'time_elapsed': info.get('time_elapsed', 0.0),
                'time_remaining': info.get('time_remaining', 0.0),
                'total_reward': self.current_episode_reward,
                
                # Success metrics
                'at_target': info.get('at_target', False),
                'hover_quality': info.get('hover_quality', 0.0),
                'hover_steps': info.get('hover_steps', 0),
                'required_hover_steps': info.get('required_hover_steps', 0),
                'hover_progress': info.get('hover_progress', 0.0),
                
                # Episode/Env config
                'episode_step': info.get('episode_step', 0),
                'success_threshold': info.get('success_threshold', 0.0),
                'random_start_enabled': info.get('random_start_enabled', False),
                
                # Episode ending information
                'termination_reason': info.get('termination_reason', None),
                'truncation_reason': info.get('truncation_reason', None),
            }
            
            self.episode_data.append(episode_summary)

            if len(self.episode_data) > 10:
                window_size = min(self.metrics_window, len(self.episode_data))
                recent_episodes = self.episode_data[-window_size:]
                
                success_rates = [ep['at_target'] for ep in recent_episodes]
                final_distances = [ep['final_distance'] for ep in recent_episodes]
                progress_ratios = [ep['progress_ratio'] for ep in recent_episodes]
                total_rewards = [ep['total_reward'] for ep in recent_episodes]
                episode_lengths = [ep['episode_length'] for ep in recent_episodes]

                metrics = {
                    'episode': self.current_episode,
                    'window_size': window_size,
                    'success_rate_mean': np.mean(success_rates),
                    'success_rate_std': np.std(success_rates),
                    'final_distance_mean': np.mean(final_distances),
                    'final_distance_std': np.std(final_distances),
                    'progress_ratio_mean': np.mean(progress_ratios),
                    'progress_ratio_std': np.std(progress_ratios),
                    'total_reward_mean': np.mean(total_rewards),
                    'total_reward_std': np.std(total_rewards),
                    'episode_length_mean': np.mean(episode_lengths),
                    'episode_length_std': np.std(episode_lengths)
                }
                
                self.metrics_log.append(metrics)

            # Save logs every 100 episodes
            if (self.current_episode + 1) % 100 == 0:
                self._save_logs()
            
            # Reset for next episode
            self.current_episode += 1
            self.episode_step = 0
            self.current_episode_reward = 0.0 
        else:
            self.episode_step += 1
        
        return True
    
    def _on_training_end(self) -> None:
        self._save_logs()
    
    def _save_logs(self):
        if self.episode_data:
            episode_log_path = os.path.join(self.log_dir, "episode_log.csv")
            episode_df = pd.DataFrame(self.episode_data)
            episode_df.to_csv(episode_log_path, index=False)
        
        if self.metrics_log:
            metrics_log_path = os.path.join(self.log_dir, "metrics_log.csv")
            metrics_df = pd.DataFrame(self.metrics_log)
            metrics_df = pd.DataFrame(self.metrics_log)
            metrics_df.to_csv(metrics_log_path, index=False)