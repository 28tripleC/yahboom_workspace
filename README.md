# Yahboom Warehouse Vision Workspace

This ROS 2 workspace contains the Yahboom robot packages used for warehouse navigation, camera streaming, ArUco marker detection, and inventory checking.

The main directory for this project is:

```text
src/warehouse_vision
```

`warehouse_vision` is the package that handles waypoint recording, camera calibration, ArUco detection, shelf scanning, patrol navigation, and inventory report generation.

## Main Workspace Layout

```text
yahboom_workspace/
├── start_warehouse.sh              # Starts the full warehouse system in tmux
├── src/
│   ├── warehouse_vision/           # Main project package
│   ├── yahboomcar_bringup/         # Robot bringup
│   ├── yahboom_esp32_camera/       # ESP32 camera launch package
│   └── yahboomcar_nav/             # Navigation, map, AMCL, Nav2, RViz
└── README.md
```

## Requirements

- ROS 2 Humble
- `colcon`
- `tmux`
- Yahboom robot bringup packages in this workspace
- ESP32 camera publishing images on `/esp32_img`
- Nav2, AMCL, map, and TF running correctly
- Python dependencies used by the package, including OpenCV, NumPy, PyYAML, and `cv_bridge`

## Build

From the workspace root:

```bash
cd /path/to/yahboom_workspace
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Replace `/path/to/yahboom_workspace` with your actual workspace path. Also update the `WS` variable in `start_warehouse.sh` if it does not match your workspace path.

## Prepare Before Running

### 1. Calibrate the Camera

Run this once before normal use:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run warehouse_vision calibrate_camera
```

The calibration files are saved under:

```text
~/camera_calibration/
```

### 2. Record Patrol Waypoints

Start robot bringup, camera, navigation, and RViz first. Set the robot initial pose in RViz, then run:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run warehouse_vision waypoint_recorder
```

Move the robot to each shelf position and enter the shelf ArUco marker ID when prompted. The waypoints are saved to:

```text
~/waypoints.yaml
```

## Run the Full Program

The easiest way is to use the startup script from the workspace root:

```bash
cd /path/to/yahboom_workspace
chmod +x start_warehouse.sh
./start_warehouse.sh
```

This starts a `tmux` session named `warehouse` with these windows:

- `bringup`: robot bringup
- `camera`: ESP32 camera
- `nav`: Yahboom navigation
- `aruco`: warehouse ArUco detector
- `patrol`: warehouse patrol and shelf scanning
- `rviz2`: RViz display

To run only the robot, camera, navigation, ArUco detector, and RViz without patrol:

```bash
./start_warehouse.sh test
```

To leave the tmux session without stopping the program, press:

```text
Ctrl+B, then D
```

To stop the whole session:

```bash
tmux kill-session -t warehouse
```

## Manual Run Order

If you do not use `start_warehouse.sh`, start the system in this order:

```bash
ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py
ros2 launch yahboom_esp32_camera yahboom_esp32_camera_launch.py
ros2 launch yahboomcar_nav navigation_dwb_launch.py
ros2 run warehouse_vision aruco_detector
ros2 launch warehouse_vision patrol_launch.py
ros2 launch yahboomcar_nav display_launch.py
```

Use a separate terminal for each command, and source ROS plus the workspace setup file in each terminal.

## Output Files

The program creates these files during use:

```text
~/camera_calibration/               # Camera calibration files
~/waypoints.yaml                    # Saved patrol waypoints
~/warehouse_log/baseline.json       # First inventory registration
~/warehouse_log/inventory_*.json    # Later inspection reports
```

If you want to register a new baseline inventory, delete:

```bash
rm ~/warehouse_log/baseline.json
```

## Useful Checks

Before running patrol, check that the required topics are publishing:

```bash
ros2 topic echo /esp32_img --once
ros2 topic echo /amcl_pose --once
ros2 topic echo /odom --once
```

Common issues:

- If patrol cannot start scanning, make sure `aruco_detector` is already running.
- If waypoints cannot load, check that `~/waypoints.yaml` exists.
- If marker distance is inaccurate, run camera calibration again.
- If navigation fails, set the initial pose in RViz and confirm AMCL is publishing.
