"""
AutoDroneAviaryGui.py
GUI wrapper for AutoDroneAviary that handles all visual elements.
Provides target markers and visual feedback for the drone environment.
"""

import numpy as np
import pybullet as p
from typing import Dict, Any, Optional, Tuple
from .AutoDroneAviary import AutoDroneAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class AutoDroneAviaryGui(AutoDroneAviary):
    """
    GUI-enabled version of AutoDroneAviary with visual target markers.
    """

    def __init__(
        self,
        drone_model: DroneModel=DroneModel.CF2X,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        physics: Physics=Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = True,  # Default to True for GUI class
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
        Initialize AutoDroneAviaryGui environment.
        """
        # Target GUI elements
        self.target_marker_id = None
        self.success_sphere_id = None

        # Force GUI = True
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=True,
            record=record,
            obs=obs,
            act=act,
            target_bounds=target_bounds,
            success_threshold=success_threshold,
            episode_len_sec=episode_len_sec,
            random_xyz=random_xyz,
            start_bounds=start_bounds
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and add visual target markers.
        """
        obs, info = super().reset(seed=seed, options=options)
        
        # Add target markers in GUI mode
        if self.GUI:
            self._add_target_markers()
        
        return obs, info

    def set_target(self, target_position: np.ndarray) -> None:
        """
        Manually set target position and update visual markers.
        """
        super().set_target(target_position)
        if self.GUI:
            self._add_target_markers()

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

    def close(self):
        """Clean up GUI elements before closing."""
        if self.GUI:
            # Remove target markers
            if self.target_marker_id is not None:
                try:
                    p.removeBody(self.target_marker_id, physicsClientId=self.CLIENT)
                except:
                    pass
            if self.success_sphere_id is not None:
                try:
                    p.removeBody(self.success_sphere_id, physicsClientId=self.CLIENT)
                except:
                    pass
        
        super().close()