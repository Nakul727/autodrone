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
    GUI-enabled version of AutoDroneAviary with visual target markers and flight path.
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
        show_flight_path: bool = True,
        path_length: int = 100,
    ) -> None:
        """
        Initialize AutoDroneAviaryGui environment.
        """
        # Target GUI elements
        self.target_marker_id = None
        self.success_sphere_id = None
        
        # Flight path visualization
        self.show_flight_path = show_flight_path
        self.path_length = path_length
        self.flight_path = []
        self.path_line_ids = []
        self.last_position = None

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
        
        # Configure clean GUI after initialization
        self._configure_clean_gui()

    def _configure_clean_gui(self):
        """Configure PyBullet GUI visualization."""
        if not self.GUI:
            return
            
        # Remove all side panels and debug info 
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=self.CLIENT)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.CLIENT)
        
        # Set camera for optimal square view
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5],
            physicsClientId=self.CLIENT
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and add visual target markers.
        """
        obs, info = super().reset(seed=seed, options=options)
        
        # Clear flight path
        self._clear_flight_path()
        
        # Add target markers
        if self.GUI:
            self._add_target_markers()
        
        return obs, info

    def step(self, action):
        """
        Step the environment and update flight path visualization.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update flight path
        if self.GUI and self.show_flight_path:
            self._update_flight_path()
        
        return obs, reward, terminated, truncated, info

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
        if hasattr(self, 'target_marker_id') and self.target_marker_id is not None:
            try:
                body_info = p.getBodyInfo(self.target_marker_id, physicsClientId=self.CLIENT)
                if body_info:
                    p.removeBody(self.target_marker_id, physicsClientId=self.CLIENT)
            except:
                pass
            self.target_marker_id = None
            
        if hasattr(self, 'success_sphere_id') and self.success_sphere_id is not None:
            try:
                body_info = p.getBodyInfo(self.success_sphere_id, physicsClientId=self.CLIENT)
                if body_info:
                    p.removeBody(self.success_sphere_id, physicsClientId=self.CLIENT)
            except:
                pass
            self.success_sphere_id = None
        
        # Create target marker (red dot)
        target_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.03,
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
            rgbaColor=[1, 0, 0, 0.15],
            physicsClientId=self.CLIENT
        )
        
        self.success_sphere_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=success_visual,
            basePosition=self.current_target,
            physicsClientId=self.CLIENT
        )

    def _update_flight_path(self):
        """Update the flight path visualization with current drone position."""
        if not self.GUI or not self.show_flight_path:
            return

        state = self._getDroneStateVector(0)
        current_pos = state[0:3].copy()
        
        # Only add new line segment if we have a previous position
        if self.last_position is not None:
            # Draw line segment from last position to current position
            line_id = p.addUserDebugLine(
                lineFromXYZ=self.last_position,
                lineToXYZ=current_pos,
                lineColorRGB=[0.8, 0.2, 0.2],
                lineWidth=8.0,
                lifeTime=0,
                physicsClientId=self.CLIENT
            )
            self.path_line_ids.append(line_id)
        
        # Update last position for next segment
        self.last_position = current_pos.copy()
        
        # Optional: Limit total number of line segments to prevent memory issues
        if len(self.path_line_ids) > self.path_length:
            # Remove oldest line segment
            try:
                p.removeUserDebugItem(self.path_line_ids[0], physicsClientId=self.CLIENT)
            except:
                pass
            self.path_line_ids.pop(0)

    def _clear_flight_path(self):
        """Clear the flight path visualization."""
        if not self.GUI:
            return
        
        # Remove all existing path lines
        for line_id in self.path_line_ids:
            try:
                p.removeUserDebugItem(line_id, physicsClientId=self.CLIENT)
            except:
                pass
        
        # Reset path data
        self.path_line_ids.clear()
        self.last_position = None