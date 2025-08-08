"""
AutoDroneAviary.py
Custom drone environment inheriting from BaseRLAviary.
Provides point-to-point navigation task for reinforcement learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class AutoDroneAviary(BaseRLAviary):
    """
    Custom drone environment for point-to-point navigation.
    """ 
    def __init__(
        self,
        drone_model: DroneModel=DroneModel.CF2X,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        physics: Physics=Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.RPM,
        target_bounds: Optional[np.ndarray] = None,
        success_threshold: float = 0.1,
        episode_len_sec: int = 15,
        random_xyz: bool = True,
        start_bounds: Optional[np.ndarray] = None,
    ) -> None: 
        """
        Initialize AutoDroneAviary environment.
        """
        # Episode configuration
        self.SUCCESS_THRESHOLD = success_threshold
        self.EPISODE_LEN_SEC = episode_len_sec
        self.RANDOM_XYZ = random_xyz
        
        # Spatial bounds
        self.START_BOUNDS = start_bounds if start_bounds is not None else \
            np.array([[-1.5, 1.5], [-1.5, 1.5], [0.3, 1.0]])
        self.TARGET_BOUNDS = target_bounds if target_bounds is not None else \
            np.array([[-2.0, 2.0], [-2.0, 2.0], [0.2, 2.0]])
        self.MAX_XY_DISTANCE = 3.0
        self.MAX_Z_HEIGHT = 3.0
        self.MIN_Z_HEIGHT = 0.05
        self.DISTANCE_FAILURE_LIMIT = 6.0
        self.MAX_RELATIVE_DISTANCE = 5.0
        
        # Motion limits
        self.MAX_ROLL_PITCH = 0.4
        self.MAX_SPEED = 10.0
        self.HOVER_SPEED_THRESHOLD = 0.2
        
        # Reward parameters
        self.HOVER_REWARD_BASE = 10.0
        self.VELOCITY_BONUS_MAX = 20.0
        self.VELOCITY_BONUS_DECAY = 5.0
        self.PERFECT_HOVER_BONUS = 5.0
        self.PERFECT_HOVER_DIST_RATIO = 0.5
        self.PERFECT_HOVER_SPEED = 0.1
        self.PROGRESS_REWARD_BASE = 2.0
        self.VELOCITY_PENALTY_FACTOR = 0.1
        self.STEP_PENALTY = 0.01
        
        # Success parameters
        self.HOVER_DURATION_SEC = 2.0
        
        # Episode state variables
        self.current_target = None
        self.initial_distance = None
        self.best_distance = float('inf')
        self.previous_distance = None
        self.hover_steps = 0
        self.termination_reason = None
        self.truncation_reason = None
        
        # Backup configuration
        self.BASE_INITIAL_XYZS = initial_xyzs
        
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and generate new target and optional random start position.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random initial position if enabled
        if self.RANDOM_XYZ:
            random_start = np.array([
                np.random.uniform(self.START_BOUNDS[0][0], self.START_BOUNDS[0][1]),
                np.random.uniform(self.START_BOUNDS[1][0], self.START_BOUNDS[1][1]),
                np.random.uniform(self.START_BOUNDS[2][0], self.START_BOUNDS[2][1])
            ]).reshape(1, 3)
            
            # Temporarily override the initial position
            original_init_xyzs = self.INIT_XYZS.copy()
            self.INIT_XYZS = random_start
        
        # Generate random target
        self.current_target = np.array([
            np.random.uniform(self.TARGET_BOUNDS[0][0], self.TARGET_BOUNDS[0][1]),
            np.random.uniform(self.TARGET_BOUNDS[1][0], self.TARGET_BOUNDS[1][1]),
            np.random.uniform(self.TARGET_BOUNDS[2][0], self.TARGET_BOUNDS[2][1])
        ])
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Restore original initial position config
        if self.RANDOM_XYZ:
            self.INIT_XYZS = original_init_xyzs
        
        # Reset episode distance tracking variables
        drone_pos = self._getDroneStateVector(0)[0:3]
        self.initial_distance = np.linalg.norm(self.current_target - drone_pos)
        self.best_distance = self.initial_distance
        self.previous_distance = self.initial_distance
        
        # Reset episode state variables
        self.hover_steps = 0
        self.termination_reason = None
        self.truncation_reason = None
        
        # Initialize episode variables to track
        info.update({
            'target_position': self.current_target.copy(),
            'start_position': drone_pos.copy(),
            'initial_distance': self.initial_distance,
            'success_threshold': self.SUCCESS_THRESHOLD,
            'random_start_enabled': self.RANDOM_XYZ
        })
        
        return obs, info
    
    def _observationSpace(self) -> spaces.Box:
        """
        Define observation space including target position.
        """
        base_obs_space = super()._observationSpace()

        low = np.concatenate([
            base_obs_space.low,
            np.array([[-self.MAX_RELATIVE_DISTANCE, -self.MAX_RELATIVE_DISTANCE, -self.MAX_RELATIVE_DISTANCE]])
        ], axis=1)
        
        high = np.concatenate([
            base_obs_space.high,
            np.array([[self.MAX_RELATIVE_DISTANCE, self.MAX_RELATIVE_DISTANCE, self.MAX_RELATIVE_DISTANCE]])
        ], axis=1)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        """
        Compute observation including drone state and relative target position.
        """
        drone_obs = super()._computeObs()

        drone_pos = self._getDroneStateVector(0)[0:3]
        relative_target = self.current_target - drone_pos
        relative_target_2d = relative_target.reshape(1, 3)
        final_obs = np.concatenate([drone_obs, relative_target_2d], axis=1).astype(np.float32)

        return final_obs
    
    def _computeReward(self):
        """
        Calculate the reward signal for the current state and action.
        """
        if self.current_target is None:
            return 0.0
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        speed = np.linalg.norm(vel)
        distance_to_target = np.linalg.norm(self.current_target - pos)

        if distance_to_target < self.best_distance:
            self.best_distance = distance_to_target

        total_reward = 0.0

        if distance_to_target < self.SUCCESS_THRESHOLD:
            # Hovering reward
            normalized_dist = distance_to_target / self.SUCCESS_THRESHOLD
            hover_reward = self.HOVER_REWARD_BASE * (1.0 - normalized_dist)
            velocity_bonus = self.VELOCITY_BONUS_MAX * np.exp(-speed * self.VELOCITY_BONUS_DECAY)
            perfect_hover_bonus = self.PERFECT_HOVER_BONUS if (
                normalized_dist < self.PERFECT_HOVER_DIST_RATIO and 
                speed < self.PERFECT_HOVER_SPEED
            ) else 0.0
            total_reward = hover_reward + velocity_bonus + perfect_hover_bonus
        else:
            # Progress reward
            progress_reward = max(0, self.PROGRESS_REWARD_BASE - distance_to_target)
            velocity_penalty = speed * self.VELOCITY_PENALTY_FACTOR
            total_reward = progress_reward - velocity_penalty - self.STEP_PENALTY

        return total_reward

    def _computeTerminated(self) -> bool:
        """
        Check if episode should be terminated (success).
        GUI version never terminates to allow visual hover inspection.
        """
        if self.GUI:
            self.termination_reason = None
            return False
        
        if self.current_target is None:
            self.termination_reason = None
            return False
        
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        distance = np.linalg.norm(self.current_target - pos)
        speed = np.linalg.norm(vel)
        
        # Check if hovering at target
        if distance < self.SUCCESS_THRESHOLD and speed < self.HOVER_SPEED_THRESHOLD:
            self.hover_steps += 1
            required_hover_steps = int(self.HOVER_DURATION_SEC * self.CTRL_FREQ)
            if self.hover_steps >= required_hover_steps:
                self.termination_reason = "success_hover"
                return True
        else:
            self.hover_steps = 0
        
        self.termination_reason = None
        return False

    def _computeTruncated(self) -> bool:
        """
        Check if episode should be truncated (failure conditions or time limit).
        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]

        # Boundary violations
        if (abs(current_pos[0]) > self.MAX_XY_DISTANCE or
            abs(current_pos[1]) > self.MAX_XY_DISTANCE or
            current_pos[2] > self.MAX_Z_HEIGHT or
            current_pos[2] < self.MIN_Z_HEIGHT):
            self.truncation_reason = "boundary_violation"
            return True

        # Attitude failures
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_ROLL_PITCH or abs(pitch) > self.MAX_ROLL_PITCH:
            self.truncation_reason = "attitude_failure"
            return True

        # Time limit exceeded
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            self.truncation_reason = "time_limit"
            return True
        
        # Distance failure (too far from target)
        distance = np.linalg.norm(self.current_target - current_pos)
        if distance > self.DISTANCE_FAILURE_LIMIT:
            self.truncation_reason = "distance_failure"
            return True

        # Speed failure (moving too fast)
        velocity = state[10:13]
        speed = np.linalg.norm(velocity)
        if speed > self.MAX_SPEED:
            self.truncation_reason = "speed_failure"
            return True

        self.truncation_reason = None
        return False

    def _computeInfo(self) -> Dict[str, Any]:
        """
        Compute additional information for logging and debugging.
        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        velocity = state[10:13]
        attitude = state[7:10]
        
        # Core calculations
        current_distance = np.linalg.norm(self.current_target - current_pos) if self.current_target is not None else 0.0
        speed = np.linalg.norm(velocity)
        
        # Progress calculation
        progress_ratio = 0.0
        if self.initial_distance is not None and self.initial_distance > 0:
            progress_ratio = (self.initial_distance - current_distance) / self.initial_distance
        
        # Success detection
        at_target = current_distance < self.SUCCESS_THRESHOLD if self.current_target is not None else False
        
        # Hover quality (0 to 1, where 1 is perfect)
        hover_quality = max(0, 1.0 - speed) if at_target else 0.0
        
        # Required hover steps for success
        required_hover_steps = int(self.HOVER_DURATION_SEC * self.CTRL_FREQ)
        hover_progress = 0.0
        if required_hover_steps > 0:
            hover_progress = float(self.hover_steps / required_hover_steps)
    

        return {
            # Positions and targets
            'target_position': self.current_target.copy() if self.current_target is not None else np.zeros(3),
            'current_position': current_pos.copy(),
            
            # Distance metrics
            'distance_to_target': float(current_distance),
            'best_distance': float(self.best_distance) if self.best_distance != float('inf') else 0.0,
            'initial_distance': float(self.initial_distance) if self.initial_distance is not None else 0.0,
            'progress_ratio': float(progress_ratio),
            
            # Motion metrics
            'current_speed': float(speed),
            'velocity': velocity.copy(),
            'attitude': attitude.copy(),
            
            # Success metrics
            'at_target': at_target,
            'hover_quality': float(hover_quality),
            'hover_steps': self.hover_steps,
            'required_hover_steps': required_hover_steps,
            'hover_progress': hover_progress,
            
            # Episode metrics
            'episode_step': self.step_counter,
            'time_elapsed': float(self.step_counter / self.PYB_FREQ),
            'time_remaining': float(self.EPISODE_LEN_SEC - (self.step_counter / self.PYB_FREQ)),
            
            # Configuration
            'success_threshold': float(self.SUCCESS_THRESHOLD),
            'random_start_enabled': self.RANDOM_XYZ,
            
            # Episode ending
            'termination_reason': getattr(self, 'termination_reason', None),
            'truncation_reason': getattr(self, 'truncation_reason', None)
        }

    def set_target(self, target_position: np.ndarray) -> None:
        """
        Manually set target position.
        """
        self.current_target = np.array(target_position)

    def get_target(self) -> np.ndarray:
        """
        Get current target position.
        """
        return self.current_target.copy() if self.current_target is not None else None
