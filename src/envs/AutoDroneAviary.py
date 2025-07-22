"""
AutoDroneAviary.py
Custom drone environment inheriting from BaseRLAviary.
Provides point-to-point navigation task for reinforcement learning.
"""

import numpy as np
import pybullet as p
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
    ) -> None: 
        """
        Initialize AutoDroneAviary environment.
        """

        # Target generation bounds
        self.TARGET_BOUNDS = target_bounds if target_bounds is not None else \
        np.array([[-2.0, 2.0], [-2.0, 2.0], [0.2, 2.0]])
    
        # Target state
        self.current_target = None

        # Episode constants
        self.SUCCESS_THRESHOLD = success_threshold
        self.EPISODE_LEN_SEC = episode_len_sec

        # Target GUI element id's
        self.target_marker_id = None
        self.success_sphere_id = None

        # Add distance tracking
        self.initial_distance = None
        self.best_distance = float('inf')

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
        Reset environment and generate new target.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random target
        self.current_target = np.array([
            np.random.uniform(self.TARGET_BOUNDS[0][0], self.TARGET_BOUNDS[0][1]),
            np.random.uniform(self.TARGET_BOUNDS[1][0], self.TARGET_BOUNDS[1][1]),
            np.random.uniform(self.TARGET_BOUNDS[2][0], self.TARGET_BOUNDS[2][1])
        ])
        
        obs, info = super().reset(seed=seed, options=options)
        
        if self.GUI:
            self._add_target_markers()
        
        # Initialize distance tracking
        drone_pos = self._getDroneStateVector(0)[0:3]
        self.initial_distance = np.linalg.norm(self.current_target - drone_pos)
        self.best_distance = self.initial_distance
        self.last_distance = self.initial_distance
        
        info.update({
            'target_position': self.current_target.copy(),
            'initial_distance': self.initial_distance,
            'success_threshold': self.SUCCESS_THRESHOLD
        })
        
        return obs, info

    def _observationSpace(self) -> spaces.Box:
        """
        Define observation space including target position.
        """
        base_obs_space = super()._observationSpace()

        # Maximum relative target distance 
        max_relative_dist = 5.0
        
        low = np.concatenate([
            base_obs_space.low,
            np.array([[-max_relative_dist, -max_relative_dist, -max_relative_dist]])
        ], axis=1)
        
        high = np.concatenate([
            base_obs_space.high,
            np.array([[max_relative_dist, max_relative_dist, max_relative_dist]])
        ], axis=1)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def _computeObs(self) -> np.ndarray:
        """
        Compute observation including drone state and relative target position.
        """
        drone_obs = super()._computeObs()
        
        # Find relative target position for current state
        drone_pos = self._getDroneStateVector(0)[0:3]
        relative_target = self.current_target - drone_pos
        relative_target_2d = relative_target.reshape(1, 3)

        return np.concatenate([drone_obs, relative_target_2d], axis=1).astype(np.float32)

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
        distance = np.linalg.norm(self.current_target - pos)
        
        if distance < self.SUCCESS_THRESHOLD:
            # Hovering reward
            normalized_dist = distance / self.SUCCESS_THRESHOLD
            hover_reward = 10.0 * (1.0 - normalized_dist)
            velocity_bonus = 20.0 * np.exp(-speed * 5.0)
            return hover_reward + velocity_bonus
        else:
            # Progress reward
            progress_reward = max(0, 2.0 - distance)
            velocity_penalty = speed * 0.1
            return progress_reward - velocity_penalty - 0.01

    def _computeTerminated(self) -> bool:
        """
        Check if episode should be terminated (success).
        Always returning Flase, allows the drone to continue hovering at 
        the target until episode end or truncation (failure)
        """
        return False

    def _computeTruncated(self) -> bool:
        """
        Check if episode should be truncated (failure conditions or time limit).
        """
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]

        # Boundary violations
        # Check if drone has moved outside the allowed flight area
        # Manually set as using TARGET_BOUNDS requires reading variable muliple times
        if (abs(current_pos[0]) > 3.0 or
            abs(current_pos[1]) > 3.0 or
            current_pos[2] > 3.0 or
            current_pos[2] < 0.05):
            return True
            
        # Time limit exceeded
        # Convert step counter to elapsed time and check against episode limit
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
            
        # Distance failure (too far from target)
        # Max allowed distance = 6 meters (manually set)
        distance = np.linalg.norm(self.current_target - current_pos)
        if distance > 6.0:
            return True
            
        # Speed failure (moving too fast)
        # Max velocity = 10 ms/s
        velocity = state[10:13]
        speed = np.linalg.norm(velocity)
        if speed > 10.0:
            return True

        return False

    def _computeInfo(self) -> Dict[str, Any]:
        """
        Compute additional information for logging and debugging.
        """
        drone_pos = self._getDroneStateVector(0)[0:3]
        distance = np.linalg.norm(self.current_target - drone_pos) if self.current_target is not None else 0.0
        
        return {
            'step': self.step_counter,
            'target_position': self.current_target.copy() if self.current_target is not None else np.zeros(3),
            'current_position': drone_pos.copy(),
            'distance_to_target': distance
        }
    
    def set_target(self, target_position: np.ndarray) -> None:
        """
        Manually set target position.
        """
        self.current_target = np.array(target_position)
        if self.GUI:
            self._add_target_markers()
   
    def get_target(self) -> np.ndarray:
        """
        Get current target position.
        """
        return self.current_target.copy() if self.current_target is not None else None
    
    def _add_target_markers(self):
        """Add visual markers for target position and success threshold."""
        if not self.GUI or self.current_target is None:
            return
            
        # Remove existing markers
        if self.target_marker_id is not None:
            p.removeBody(self.target_marker_id, physicsClientId=self.CLIENT)
        if self.success_sphere_id is not None:
            p.removeBody(self.success_sphere_id, physicsClientId=self.CLIENT)
        
        # Create target marker (red dot)
        target_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=self.CLIENT
        )
        
        self.target_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.current_target,
            physicsClientId=self.CLIENT
        )
        
        # Create success sphere (transparent red)
        success_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.SUCCESS_THRESHOLD,
            rgbaColor=[1, 0, 0, 0.2],
            physicsClientId=self.CLIENT
        )
        
        self.success_sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=success_visual,
            basePosition=self.current_target,
            physicsClientId=self.CLIENT
        )