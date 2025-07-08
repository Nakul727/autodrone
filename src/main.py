import time
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

def main():
    """Simple drone rendering program using gym-pybullet-drones"""
    
    # Configuration
    DRONE_MODEL = DroneModel.CF2X
    NUM_DRONES = 1
    PHYSICS = Physics.PYB
    SIMULATION_FREQ_HZ = 240
    CONTROL_FREQ_HZ = 48
    DURATION_SEC = 10
    GUI = True
    
    # Initial position and orientation
    INIT_XYZ = np.array([[0, 0, 1]]) 
    INIT_RPY = np.array([[0, 0, 0]])
    
    # Create the environment
    env = CtrlAviary(
        drone_model=DRONE_MODEL,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=PHYSICS,
        pyb_freq=SIMULATION_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=GUI,
        record=False,
        obstacles=False,
        user_debug_gui=True
    )
    
    print("Starting drone simulation...")
    print(f"Drone model: {DRONE_MODEL}")
    print(f"Initial position: {INIT_XYZ[0]}")
    print(f"Duration: {DURATION_SEC} seconds")
    print("Press Ctrl+C to stop early")
    
    # Initialize action (RPMs for 4 motors)
    action = np.zeros((NUM_DRONES, 4))
    hover_rpm = np.sqrt(env.M * env.G / (4 * env.KF))
    action[0, :] = hover_rpm 
    
    # Run the simulation
    START = time.time()
    try:
        for i in range(0, int(DURATION_SEC * env.CTRL_FREQ)):
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print drone state every second
            if i % env.CTRL_FREQ == 0:
                pos = obs[0][0:3]
                print(f"Time: {i/env.CTRL_FREQ:.1f}s, Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Render the environment
            env.render()
            
            if GUI:
                sync(i, START, env.CTRL_TIMESTEP)
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Close the environment
    env.close()
    print("Simulation ended")

if __name__ == "__main__":
    main()