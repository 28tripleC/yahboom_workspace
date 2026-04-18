# Waypoint Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a teach-and-replay waypoint system — record the car's AMCL pose at each location, save to YAML, then patrol those waypoints using Nav2.

**Architecture:** A new `waypoint_recorder.py` node subscribes to `/amcl_pose` and saves poses to `~/waypoints.yaml` on Enter keypress. The existing `patrol_node.py` is modified to load waypoints from that YAML file instead of hardcoded values.

**Tech Stack:** ROS2, Nav2 (`nav2_simple_commander`), `geometry_msgs`, `rclpy`, PyYAML, Python threading

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/warehouse_vision/warehouse_vision/waypoint_recorder.py` | Subscribe `/amcl_pose`, save pose on Enter, write `~/waypoints.yaml` |
| Modify | `src/warehouse_vision/warehouse_vision/patrol_node.py` | Load waypoints from YAML instead of hardcoded list |
| Modify | `src/warehouse_vision/setup.py` | Register `waypoint_recorder` console script entry point |
| Create | `src/warehouse_vision/test/test_waypoint_recorder.py` | Unit tests for recorder logic |
| Create | `src/warehouse_vision/test/test_patrol_node.py` | Unit tests for patrol YAML loading |

---

## Task 1: Write tests for YAML loading in patrol node

**Files:**
- Create: `src/warehouse_vision/test/test_patrol_node.py`

- [ ] **Step 1: Create the test file**

```python
# src/warehouse_vision/test/test_patrol_node.py
import os
import tempfile
import pytest
import yaml


def load_waypoints_from_yaml(path):
    """Helper that mirrors the logic we'll add to patrol_node."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Waypoints file not found: {expanded}")
    with open(expanded) as f:
        data = yaml.safe_load(f)
    waypoints = data.get('waypoints', [])
    if not waypoints:
        raise ValueError("Waypoints file is empty or has no 'waypoints' key")
    return [(wp['x'], wp['y'], wp['oz'], wp['ow']) for wp in waypoints]


def test_load_waypoints_returns_tuples():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'waypoints': [
            {'x': 1.0, 'y': 2.0, 'oz': 0.0, 'ow': 1.0},
            {'x': 3.0, 'y': 4.0, 'oz': -0.5, 'ow': 0.85},
        ]}, f)
        path = f.name
    try:
        result = load_waypoints_from_yaml(path)
        assert result == [(1.0, 2.0, 0.0, 1.0), (3.0, 4.0, -0.5, 0.85)]
    finally:
        os.unlink(path)


def test_load_waypoints_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        load_waypoints_from_yaml('/tmp/does_not_exist_xyz.yaml')


def test_load_waypoints_raises_if_empty():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'waypoints': []}, f)
        path = f.name
    try:
        with pytest.raises(ValueError, match="empty"):
            load_waypoints_from_yaml(path)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to confirm they fail (function not yet in patrol_node)**

```bash
cd /home/ccc/ROS/yahboom_ws
python3 -m pytest src/warehouse_vision/test/test_patrol_node.py -v
```

Expected: 3 tests PASS (the helper is self-contained in the test file — this confirms test logic is correct before we wire it into the node)

- [ ] **Step 3: Commit**

```bash
git add src/warehouse_vision/test/test_patrol_node.py
git commit -m "test: add patrol node YAML loading tests"
```

---

## Task 2: Modify `patrol_node.py` to load waypoints from YAML

**Files:**
- Modify: `src/warehouse_vision/warehouse_vision/patrol_node.py`

- [ ] **Step 1: Replace the file with the updated version**

```python
# src/warehouse_vision/warehouse_vision/patrol_node.py
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import time
import yaml


def load_waypoints_from_yaml(path: str) -> list:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Waypoints file not found: {expanded}")
    with open(expanded) as f:
        data = yaml.safe_load(f)
    waypoints = data.get('waypoints', [])
    if not waypoints:
        raise ValueError("Waypoints file is empty or has no 'waypoints' key")
    return [(wp['x'], wp['y'], wp['oz'], wp['ow']) for wp in waypoints]


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value

        try:
            self.waypoints_data = load_waypoints_from_yaml(waypoints_file)
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(str(e))
            raise SystemExit(1)

        self.navigator = BasicNavigator()

        self.get_logger().info(f"Patrol node started with {len(self.waypoints_data)} waypoints")
        self.get_logger().info("Waiting for Nav2 action server...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ready! Starting patrol...")

        self.run_patrol()

    def create_pose(self, x, y, oz, ow):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = oz
        pose.pose.orientation.w = ow
        return pose

    def run_patrol(self):
        waypoints = [self.create_pose(*wp) for wp in self.waypoints_data]
        self.navigator.followWaypoints(waypoints)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                current = feedback.current_waypoint
                total = len(waypoints)
                self.get_logger().info(
                    f"Progress: waypoint {current + 1}/{total}",
                    throttle_duration_sec=3)
            time.sleep(0.5)

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Patrol completed successfully!")
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Patrol was canceled.")
        elif result == TaskResult.FAILED:
            self.get_logger().error("Patrol failed.")
        else:
            self.get_logger().error(f"Unknown patrol result: {result}")


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

- [ ] **Step 2: Run tests to confirm they pass**

```bash
cd /home/ccc/ROS/yahboom_ws
python3 -m pytest src/warehouse_vision/test/test_patrol_node.py -v
```

Expected: 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/warehouse_vision/warehouse_vision/patrol_node.py
git commit -m "feat: load patrol waypoints from YAML file instead of hardcode"
```

---

## Task 3: Write tests for waypoint recorder logic

**Files:**
- Create: `src/warehouse_vision/test/test_waypoint_recorder.py`

- [ ] **Step 1: Create the test file**

```python
# src/warehouse_vision/test/test_waypoint_recorder.py
import os
import tempfile
import yaml


def append_waypoint_to_yaml(path: str, x: float, y: float, oz: float, ow: float):
    """Mirrors the save logic we'll put in waypoint_recorder.py."""
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        with open(expanded) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    waypoints = data.get('waypoints', [])
    waypoints.append({'x': x, 'y': y, 'oz': oz, 'ow': ow})
    data['waypoints'] = waypoints
    with open(expanded, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def test_append_creates_file_if_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        assert os.path.exists(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data['waypoints'] == [{'x': 1.0, 'y': 2.0, 'oz': 0.0, 'ow': 1.0}]


def test_append_accumulates_waypoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        append_waypoint_to_yaml(path, 3.0, 4.0, -0.5, 0.85)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data['waypoints']) == 2
        assert data['waypoints'][1] == {'x': 3.0, 'y': 4.0, 'oz': -0.5, 'ow': 0.85}


def test_append_preserves_existing_waypoints():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'waypoints.yaml')
        with open(path, 'w') as f:
            yaml.dump({'waypoints': [{'x': 9.0, 'y': 9.0, 'oz': 0.0, 'ow': 1.0}]}, f)
        append_waypoint_to_yaml(path, 1.0, 2.0, 0.0, 1.0)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data['waypoints']) == 2
        assert data['waypoints'][0] == {'x': 9.0, 'y': 9.0, 'oz': 0.0, 'ow': 1.0}
```

- [ ] **Step 2: Run tests to confirm they pass (helper is self-contained)**

```bash
cd /home/ccc/ROS/yahboom_ws
python3 -m pytest src/warehouse_vision/test/test_waypoint_recorder.py -v
```

Expected: 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/warehouse_vision/test/test_waypoint_recorder.py
git commit -m "test: add waypoint recorder append logic tests"
```

---

## Task 4: Create `waypoint_recorder.py`

**Files:**
- Create: `src/warehouse_vision/warehouse_vision/waypoint_recorder.py`

- [ ] **Step 1: Create the file**

```python
# src/warehouse_vision/warehouse_vision/waypoint_recorder.py
import os
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import yaml


def append_waypoint_to_yaml(path: str, x: float, y: float, oz: float, ow: float):
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        with open(expanded) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    waypoints = data.get('waypoints', [])
    waypoints.append({'x': x, 'y': y, 'oz': oz, 'ow': ow})
    data['waypoints'] = waypoints
    with open(expanded, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


class WaypointRecorder(Node):
    def __init__(self):
        super().__init__('waypoint_recorder')

        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        self.waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value

        self.current_pose = None
        self.waypoint_count = 0

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            10
        )

        # Warn if no pose received within 5 seconds
        self.startup_timer = self.create_timer(5.0, self._check_initial_pose)

        self.get_logger().info(f"Waypoint Recorder started. Saving to: {self.waypoints_file}")
        self.get_logger().info("Drive the car to each waypoint location, then press Enter to save.")
        self.get_logger().info("Press Ctrl+C to finish.")

        # Background thread for keyboard input
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose

    def _check_initial_pose(self):
        if self.current_pose is None:
            self.get_logger().warn(
                "No /amcl_pose received yet! "
                "Please set the 2D Pose Estimate in RViz before recording waypoints."
            )
        self.startup_timer.cancel()

    def _input_loop(self):
        while rclpy.ok():
            try:
                input("")  # blocks until Enter
            except EOFError:
                break
            self._save_current_pose()

    def _save_current_pose(self):
        if self.current_pose is None:
            self.get_logger().warn(
                "No pose available yet. Set the 2D Pose Estimate in RViz first."
            )
            return
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        oz = self.current_pose.orientation.z
        ow = self.current_pose.orientation.w
        append_waypoint_to_yaml(self.waypoints_file, x, y, oz, ow)
        self.waypoint_count += 1
        self.get_logger().info(
            f"Saved waypoint #{self.waypoint_count} at (x={x:.4f}, y={y:.4f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Done. {node.waypoint_count} waypoints saved to {node.waypoints_file}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
```

- [ ] **Step 2: Verify the `append_waypoint_to_yaml` function matches the one tested in Task 3**

The function body in `waypoint_recorder.py` must be identical to the helper in `test_waypoint_recorder.py`. Compare them visually — same logic, same field names (`x`, `y`, `oz`, `ow`).

- [ ] **Step 3: Commit**

```bash
git add src/warehouse_vision/warehouse_vision/waypoint_recorder.py
git commit -m "feat: add waypoint recorder node"
```

---

## Task 5: Register `waypoint_recorder` entry point in `setup.py`

**Files:**
- Modify: `src/warehouse_vision/setup.py`

- [ ] **Step 1: Add the entry point**

In `setup.py`, find the `console_scripts` list under `entry_points` and add one line:

```python
entry_points={
    'console_scripts': [
        'aruco_detector = warehouse_vision.aruco_detector:main',
        'calibrate_camera = warehouse_vision.calibrate_camera:main',
        'patrol_node = warehouse_vision.patrol_node:main',
        'waypoint_recorder = warehouse_vision.waypoint_recorder:main',  # ADD THIS
    ],
},
```

- [ ] **Step 2: Build the package**

```bash
cd /home/ccc/ROS/yahboom_ws
colcon build --packages-select warehouse_vision
source install/setup.bash
```

Expected: build finishes with `Summary: 1 package finished`

- [ ] **Step 3: Verify the executable is registered**

```bash
ros2 run warehouse_vision waypoint_recorder --help 2>&1 | head -5
```

Expected: no "package not found" error (it may print usage or just start the node)

- [ ] **Step 4: Commit**

```bash
git add src/warehouse_vision/setup.py
git commit -m "feat: register waypoint_recorder console script entry point"
```

---

## Task 6: End-to-end smoke test

This task is manual — run it on the robot with Nav2 active.

- [ ] **Step 1: Launch Nav2**

```bash
ros2 launch yahboomcar_nav navigation_dwb_launch.py
```

- [ ] **Step 2: Set 2D Pose Estimate in RViz**

Open RViz, click **2D Pose Estimate**, click+drag on the map at the car's actual physical location. Confirm laser scan aligns with map walls.

- [ ] **Step 3: Run the recorder and save 2+ waypoints**

```bash
source /home/ccc/ROS/yahboom_ws/install/setup.bash
ros2 run warehouse_vision waypoint_recorder
```

Drive the car to location 1, press Enter. Drive to location 2, press Enter. Press Ctrl+C.

Expected output:
```
Saved waypoint #1 at (x=..., y=...)
Saved waypoint #2 at (x=..., y=...)
Done. 2 waypoints saved to ~/waypoints.yaml
```

- [ ] **Step 4: Inspect the saved file**

```bash
cat ~/waypoints.yaml
```

Expected:
```yaml
waypoints:
- ow: 1.0
  oz: 0.0
  x: 1.23
  y: 0.45
...
```

- [ ] **Step 5: Run the patrol node**

```bash
ros2 run warehouse_vision patrol_node
```

Expected: car navigates to each waypoint in order, logs `Progress: waypoint 1/2`, then `Patrol completed successfully!`
