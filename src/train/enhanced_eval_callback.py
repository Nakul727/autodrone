import os
import sys
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


class EnhancedEvalCallback(EvalCallback):
    """Enhanced EvalCallback with periodic best reward reporting."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -float('inf')
    
    def _on_step(self) -> bool:
        # Call parent evaluation logic
        continue_training = super()._on_step()
        
        # Print evaluation results if evaluation was performed
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_reward = np.mean(self.evaluations_results[-1])
            std_reward = np.std(self.evaluations_results[-1])
            
            print(f"\nEval at step {self.num_timesteps:,}: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Check if this is the best model so far
            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                print(f"New best model! Improvement: +{improvement:.2f}")
            else:
                print(f"Best so far: {self.best_mean_reward:.2f}")
        
        return continue_training
