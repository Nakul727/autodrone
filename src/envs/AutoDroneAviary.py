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
        target_bounds: Optional[np.ndarray] = None
    ) -> None: 
        """
        Initialize AutoDroneAviary environment.
        """

        # Target generation bounds
        self.TARGET_BOUNDS = target_bounds if target_bounds is not None else \
        np.array([[-2.0, 2.0], [-2.0, 2.0], [0.2, 2.0]])
    
        # Target state
        self.current_target = None

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
        info['target_position'] = self.current_target.copy()
        return obs, info

    def _observationSpace(self) -> spaces.Box:
        """
        Define observation space including target position.
        """
        base_obs_space = super()._observationSpace()
        return base_obs_space

    def _computeObs(self) -> np.ndarray:
        """
        Compute observation including drone state. 
        """
        drone_obs = super()._computeObs()
        return drone_obs

    def _computeReward(self):
        """
        Calculate the reward signal for the current state and action.
        """
        state = self._getDroneStateVector(0)
        total_reward = 1
        return total_reward

    def _computeTerminated(self) -> bool:
        """
        Check if episode should be terminated (success).
        """
        return True

    def _computeTruncated(self) -> bool:
        """
        Check if episode should be truncated (failure conditions or time limit).
        """
        return True

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