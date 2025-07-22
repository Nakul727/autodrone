"""
AutoDroneAviary.py
Custom drone environment inheriting from BaseRLAviary.
Provides point-to-point navigation task for reinforcement learning.
"""

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional
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
        act: ActionType=ActionType.RPM
    ) -> None: 
        """
        Initialize AutoDroneAviary environment.
        """

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

    def _observationSpace(self) -> spaces.Box:
        """
        Define observation space.
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

        # Reward will be based on the state
        # Temporary: returns 1
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

        debug_info = "debug"
        return {'debug': debug_info}