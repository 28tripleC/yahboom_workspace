# warehouse_vision

`warehouse_vision` is a ROS 2 Python package for a Yahboom-style warehouse robot that patrols shelf waypoints, uses an ESP32 camera stream to detect ArUco markers, and records inventory status in map coordinates.

The package combines:

- Camera calibration from `/esp32_img` using a chessboard target.
- Waypoint recording from AMCL poses into `~/waypoints.yaml`.
- ArUco marker detection with pose estimation, map-frame transforms, shelf scanning, and RViz marker publishing.
- Nav2 patrol between saved shelf waypoints with optional visual alignment to shelf markers.
- Baseline inventory registration on the first scan run and later inspection reports that classify items as `Normal`, `Misplaced`, `Missing`, `New item`, or `Out of range`.

## Repository Layout

```text
warehouse_vision/
├── launch/
│   └── patrol_launch.py          # Launches patrol_node with package params
├── params/
│   └── patrol_params.yaml        # Default patrol/navigation parameters
├── test/                         # Pytest/ament test files
├── warehouse_vision/
│   ├── aruco_detector.py         # Main ArUco inventory detector
│   ├── aruco_distance_test.py    # Standalone marker distance test tool
│   ├── calibrate_camera.py       # Camera calibration node
│   ├── patrol_node.py            # Nav2 patrol and shelf scan coordinator
│   └── waypoint_recorder.py      # Saves AMCL waypoints to YAML
├── package.xml
└── setup.py
```

Older experiment files such as `aruco_v1.py`, `patrol_v1.py`, and `waypoint_v1.py` are kept in the package directory, but the installed console scripts use the current files listed above.

## Requirements

- ROS 2 with `rclpy` and `ament_python`.
- Nav2 running and localized with AMCL.
- A robot base accepting `/cmd_vel`.
- An ESP32 camera image publisher on `/esp32_img` as `sensor_msgs/Image`.
- Camera servo controller subscribed to `servo_s2`.
- Python/OpenCV with ArUco support, NumPy, PyYAML, and `cv_bridge`.
- A map, AMCL, TF tree, and Nav2 stack that provide `map`, `base_link`, `/amcl_pose`, and `/odom`.

ROS package dependencies declared in `package.xml`:

- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `nav_msgs`
- `visualization_msgs`
- `cv_bridge`
- `tf2_ros`
- `nav2_simple_commander`

## Build

Run these commands from the ROS workspace root. In this project, that is typically:

```bash
cd ~/Downloads/project/yahboom_workspace
source /opt/ros/<ros-distro>/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select warehouse_vision
source install/setup.bash
```

Replace `<ros-distro>` with your ROS 2 distribution, for example `humble`, `iron`, or `jazzy`.

## Runtime Data Files

The nodes create and use these files in your home directory:

```text
~/camera_calibration/calibration_data.npz   # Camera matrix and distortion coefficients
~/camera_calibration/calibration_data.json  # Human-readable calibration result
~/waypoints.yaml                            # Patrol waypoints and optional shelf IDs
~/warehouse_log/baseline.json               # First-run inventory baseline
~/warehouse_log/inventory_*.json            # Inventory reports from each detector shutdown
```

Delete `~/warehouse_log/baseline.json` before a new registration run if you want to rebuild the inventory baseline.

## Main Topics and Services

Subscriptions:

- `/esp32_img` (`sensor_msgs/Image`): camera input for calibration and ArUco detection.
- `/amcl_pose` (`geometry_msgs/PoseWithCovarianceStamped`): robot pose for waypoint recording and patrol.
- `/odom` (`nav_msgs/Odometry`): yaw feedback for patrol rotations.
- `/aruco_visible_markers` (`std_msgs/String`): detector output consumed by patrol alignment.
- `/camera_angle` (`std_msgs/Int32`): manual or patrol camera angle command consumed by the detector.
- `/current_shelf_id` (`std_msgs/Int32`): shelf context for inventory logging.

Publications:

- `/aruco_detected_img` (`sensor_msgs/Image`): annotated camera stream.
- `/inventory_markers` (`visualization_msgs/MarkerArray`): inventory markers for RViz.
- `/aruco_visible_markers` (`std_msgs/String`): visible shelf marker angles and distances.
- `/cmd_vel` (`geometry_msgs/Twist`): patrol movement commands.
- `/camera_angle` (`std_msgs/Int32`): camera tilt commands from patrol.
- `servo_s2` (`std_msgs/Int32`): servo command sent by the detector.

Service:

- `scan_shelf` (`std_srvs/Trigger`): makes `aruco_detector` scan all configured camera rows.

## Step 1: Start Robot Bringup

Start the robot, ESP32 camera publisher, TF, map server, AMCL, Nav2, and RViz using your robot's normal launch files.

Before running this package, verify that these commands show data:

```bash
ros2 topic echo /esp32_img --once
ros2 topic echo /amcl_pose --once
ros2 topic echo /odom --once
ros2 topic list | grep cmd_vel
```

In RViz, set the initial pose with `2D Pose Estimate` so AMCL publishes a stable `/amcl_pose`.

## Step 2: Calibrate the Camera

Use a 9 by 7 inner-corner chessboard with 0.025 m squares, matching the values in `calibrate_camera.py`.

```bash
cd ~/Downloads/project/yahboom_workspace
source install/setup.bash
ros2 run warehouse_vision calibrate_camera
```

Show the chessboard to the camera at different positions and angles. The node captures automatically when the chessboard is detected, up to at least 25 captures. Press `q` in the OpenCV window or press `Ctrl+C` in the terminal to finish and save:

```text
~/camera_calibration/calibration_data.npz
~/camera_calibration/calibration_data.json
```

The ArUco detector can run without this file, but calibrated camera parameters give better marker distance and map-position estimates.

## Step 3: Record Patrol Waypoints

Start Nav2 and AMCL first, then run:

```bash
cd ~/Downloads/project/yahboom_workspace
source install/setup.bash
ros2 run warehouse_vision waypoint_recorder
```

For each shelf:

1. Drive the robot to the desired stopping pose.
2. Make sure AMCL pose is stable in RViz.
3. Enter the shelf marker ID at the `Shelf ID:` prompt and press Enter.
4. Leave the prompt blank for a waypoint that has no shelf marker.

The default command starts a new session and deletes old `waypoints*.yaml` files from the waypoint directory before saving. To append to an existing file:

```bash
ros2 run warehouse_vision waypoint_recorder --ros-args -p append:=true
```

To use a custom waypoint file:

```bash
ros2 run warehouse_vision waypoint_recorder --ros-args -p waypoints_file:=/path/to/waypoints.yaml
```

Expected waypoint format:

```yaml
waypoints:
- x: 1.0
  y: 2.0
  oz: 0.0
  ow: 1.0
  shelf_id: 10
```

## Step 4: Test ArUco Distance Detection

This optional tool is useful before running full inventory scans:

```bash
cd ~/Downloads/project/yahboom_workspace
source install/setup.bash
ros2 run warehouse_vision aruco_distance_test
```

It subscribes to `/esp32_img`, detects `DICT_4X4_50` markers, draws axes in an OpenCV window, and logs marker distance. The default marker size is `0.06` m:

```bash
ros2 run warehouse_vision aruco_distance_test --ros-args -p marker_size:=0.06
```

## Step 5: Start the ArUco Inventory Detector

Run the detector in a separate terminal:

```bash
cd ~/Downloads/project/yahboom_workspace
source install/setup.bash
ros2 run warehouse_vision aruco_detector
```

Useful parameters:

```bash
ros2 run warehouse_vision aruco_detector --ros-args \
  -p waypoints_file:=~/waypoints.yaml \
  -p row_angles:="[0, 15, 25]" \
  -p scan_duration_per_row:=4.0 \
  -p roi_band_frac:=0.55
```

Behavior:

- If `~/warehouse_log/baseline.json` does not exist, the detector starts in registration mode and saves a baseline when it exits.
- If `baseline.json` exists, the detector starts in inspection mode and compares detected marker map positions against the baseline.
- Shelf IDs from the waypoint file are treated as fixed shelf markers for patrol alignment and drift compensation.

## Step 6: Run Patrol and Shelf Scans

Keep `aruco_detector` running, then start patrol in another terminal:

```bash
cd ~/Downloads/project/yahboom_workspace
source install/setup.bash
ros2 launch warehouse_vision patrol_launch.py
```

Or run the node directly:

```bash
ros2 run warehouse_vision patrol_node --ros-args --params-file src/warehouse_vision/params/patrol_params.yaml
```

The patrol node will:

1. Wait for Nav2 to become active.
2. Load waypoints from `~/waypoints.yaml`.
3. Move to each waypoint with Nav2.
4. Rotate toward the waypoint yaw.
5. Use the shelf ArUco marker for visual yaw alignment.
6. Adjust distance to the shelf marker.
7. Trigger the detector's `scan_shelf` service.
8. Continue until all waypoints are scanned.

To change log verbosity:

```bash
ros2 launch warehouse_vision patrol_launch.py log_level:=debug
```

## Step 7: View Results

In RViz:

- Add the annotated image topic `/aruco_detected_img`.
- Add a `MarkerArray` display for `/inventory_markers`.
- Use `map` as the fixed frame.

After stopping `aruco_detector`, check the generated report:

```bash
ls -lt ~/warehouse_log/inventory_*.json
cat ~/warehouse_log/baseline.json
```

Inventory statuses:

- `Normal`: marker matches the registered baseline after drift compensation.
- `Misplaced`: marker position differs from the baseline beyond configured thresholds.
- `Missing`: baseline marker was not seen during the inspection run.
- `New item`: marker was detected but does not exist in the baseline.
- `Out of range`: marker pose was too far away or did not have usable map coordinates.

## Running Tests

From the workspace root:

```bash
cd ~/Downloads/project/yahboom_workspace
source /opt/ros/<ros-distro>/setup.bash
colcon test --packages-select warehouse_vision
colcon test-result --verbose
```

You can also run the Python tests directly from the package root:

```bash
cd ~/Downloads/project/yahboom_workspace/src/warehouse_vision
python3 -m pytest test
```

## Troubleshooting

- `Waypoints file not found`: record waypoints first or pass `-p waypoints_file:=...`.
- `scan_shelf service unavailable`: start `aruco_detector` before `patrol_node`.
- `TF transform failed`: verify the TF tree connects `map`, `base_link`, and `camera_frame`.
- No marker alignment: confirm the waypoint `shelf_id` matches the physical shelf ArUco marker ID.
- Poor distance estimates: rerun camera calibration and verify the printed marker size is correct.
- OpenCV window does not appear: run from a desktop session with display access, not a headless SSH session unless X forwarding is configured.

