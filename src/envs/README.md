## Inherited methods from gym-pybllet-drones via BaseRLAviary and BaseAviary

## Public Methods

| Method Name | Inheritance | Description |
|-------------|-------------|-------------|
| `__init__()` | Overridden in BaseRLAviary | Environment initialization |
| `reset()` | Inherited from BaseAviary | Episode reset |
| `step()` | Inherited from BaseAviary | Environment step |
| `render()` | Inherited from BaseAviary | Visual rendering |
| `close()` | Inherited from BaseAviary | Environment cleanup |
| `getPyBulletClient()` | Inherited from BaseAviary | Get physics client |
| `getDroneIds()` | Inherited from BaseAviary | Get drone IDs |

## Private Methods

| Method Name | Inheritance | Description |
|-------------|-------------|-------------|
| `_housekeeping()` | Inherited from BaseAviary | Internal state management |
| `_updateAndStoreKinematicInformation()` | Inherited from BaseAviary | Update drone kinematics |
| `_startVideoRecording()` | Inherited from BaseAviary | Begin video capture |
| `_getDroneStateVector()` | Inherited from BaseAviary | Extract drone state |
| `_getDroneImages()` | Inherited from BaseAviary | Capture drone cameras |
| `_exportImage()` | Inherited from BaseAviary | Save image files |
| `_getAdjacencyMatrix()` | Inherited from BaseAviary | Compute drone neighbors |
| `_physics()` | Inherited from BaseAviary | Apply physics forces |
| `_groundEffect()` | Inherited from BaseAviary | Ground effect forces |
| `_drag()` | Inherited from BaseAviary | Air drag forces |
| `_downwash()` | Inherited from BaseAviary | Propeller downwash effects |
| `_dynamics()` | Inherited from BaseAviary | Drone dynamics integration |
| `_integrateQ()` | Inherited from BaseAviary | Quaternion integration |
| `_normalizedActionToRPM()` | Inherited from BaseAviary | Convert actions to RPM |
| `_showDroneLocalAxes()` | Inherited from BaseAviary | Visualize drone axes |
| `_calculateNextStep()` | Inherited from BaseAviary | Path planning step |
| `_parseURDFParameters()` | Inherited from BaseAviary | Load drone parameters |
| `_addObstacles()` | Overridden in BaseRLAviary | Add environment obstacles |
| `_actionSpace()` | Implemented in BaseRLAviary | Define action bounds |
| `_preprocessAction()` | Implemented in BaseRLAviary | Process input actions |
| `_observationSpace()` | Implemented in BaseRLAviary | Define observation bounds |
| `_computeObs()` | Implemented in BaseRLAviary | Generate current observations |

## Abstract Methods

| Method Name | Inheritance | Description |
|-------------|-------------|-------------|
| `_computeReward()` | **Must be implemented** | Calculate episode rewards |
| `_computeTerminated()` | **Must be implemented** | Check episode termination |
| `_computeTruncated()` | **Must be implemented** | Check episode truncation |
| `_computeInfo()` | **Must be implemented** | Provide debug information |


## Public Methods and Their Internal Function Calls

### `__init__()`
1. Calls `_parseURDFParameters()` - Loads drone parameters from URDF
2. Calls parent `__init__()` which triggers:
   - `_actionSpace()` - Defines action space bounds
   - `_observationSpace()` - Defines observation space bounds
   - `_housekeeping()` - Initialize variables and PyBullet setup
   - `_updateAndStoreKinematicInformation()` - Store initial drone states
   - `_startVideoRecording()` - Setup recording if enabled
   - `_addObstacles()` - Add obstacles to environment

### `reset()`
1. `_housekeeping()` - Reset all variables and reinitialize PyBullet
2. `_updateAndStoreKinematicInformation()` - Update drone kinematic data
3. `_startVideoRecording()` - Restart video recording
4. `_computeObs()` - Generate initial observations
5. `_computeInfo()` - Generate initial info dict

### `step()`
**Action Processing:**
1. `_preprocessAction()` - Convert actions to RPM values

**Physics Loop (repeated PYB_STEPS_PER_CTRL times):**

2. `_updateAndStoreKinematicInformation()` (if multi-step physics)
3. **Physics application (one of):**
   - `_physics()` - Basic PyBullet physics
   - `_dynamics()` - Custom dynamics implementation
4. **Optional effects (depending on physics mode):**
   - `_groundEffect()` - Ground effect forces
   - `_drag()` - Air drag simulation
   - `_downwash()` - Multi-drone downwash effects

**Post-Physics:**

5. `_updateAndStoreKinematicInformation()` - Update final states
6. `_computeObs()` - Generate new observations
7. `_computeReward()` - Calculate rewards
8. `_computeTerminated()` - Check termination
9. `_computeTruncated()` - Check truncation
10. `_computeInfo()` - Generate info dict

### Additional Internal Calls in `step()`

**Video Recording (if enabled):**
- `_getDroneImages()` - Capture drone camera views
- `_exportImage()` - Save images to files

**Multi-drone scenarios:**
- `_getAdjacencyMatrix()` - Compute neighbor relationships (in `_computeObs()`)

**Action preprocessing calls:**
- `_getDroneStateVector()` - Get current drone state (in PID/VEL modes)
- `_calculateNextStep()` - Calculate waypoints (in PID mode)