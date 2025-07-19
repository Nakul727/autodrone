"""
AutodroneAviary.py
This class implements our custom drone agent
It inherits the BaseRLAvairy and BaseAviary classes that provide useful 
functions that handle physics and simulation in Pybullet environment.
"""

import numpy as np
import gymnasium as gym
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class AutoDroneAviary(BaseRLAviary):

    def __init__(
        self,
        drone_model: DroneModel=DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics=Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui=False,
        record=False,
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.RPM
    ):
        
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

    def _observationSpace(self):
        """
        Define the observation space dimensions and bounds for the environment.
        Should return a gymnasium.spaces.Box defining the shape and limits
        of the observation vector that the agent will receive.
        """
        pass

    def _computeObs(self):
        """
        Compute and return the current observation vector for the drone.
        This should include drone state information like position, velocity,
        orientation, and any task-specific observations.
        """
        pass

    def _computeReward(self):
        """
        Calculate the reward signal for the current state and action.
        Should return a scalar reward value that guides the learning process
        based on task objectives like reaching targets, avoiding crashes, etc.
        """
        pass

    def _computeTerminated(self):
        """
        Determine if the episode should terminate due to success or failure.
        Should return True if the drone has crashed, completed the task,
        or violated safety constraints.
        """
        pass

    def _computeTruncated(self):
        """
        Determine if the episode should be truncated due to time limits.
        Should return True if maximum episode length is reached or
        other truncation conditions are met.
        """
        pass

    def _computeInfo(self):
        """
        Compute additional information dictionary for debugging and monitoring.
        Should return a dictionary containing useful metrics like distance
        to target, energy consumption, or other diagnostic information.
        """
        pass