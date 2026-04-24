import math
import os
import signal
import time
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from std_srvs.srv import Trigger


def load_waypoints_from_yaml(path: str) -> list:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Waypoints file not found: {expanded}")
    with open(expanded) as f:
        data = yaml.safe_load(f) or {}
    waypoints = data.get('waypoints', [])
    if not waypoints:
        raise ValueError("No waypoints found in file")
    return [(wp['x'], wp['y'], wp['oz'], wp['ow']) for wp in waypoints]


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        # Parameters
        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        self.declare_parameter('scan_duration', 5.0)
        self.declare_parameter('rotation_speed', 0.4)
        self.declare_parameter('yaw_tolerance', 0.15)

        waypoints_file = self.get_parameter('waypoints_file').value
        self.scan_duration = self.get_parameter('scan_duration').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value

        # Load waypoints
        try:
            self.waypoints_data = load_waypoints_from_yaml(waypoints_file)
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(str(e))
            raise SystemExit(1)

        # Nav2
        self.navigator = BasicNavigator()

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self._pose_callback, 10)

        # Current pose from AMCL
        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        # State
        self.is_running = True

        # Scan service client
        self.scan_client = self.create_client(Trigger, 'scan_shelf')

    def start(self):
        self.get_logger().info(
            f"Patrol node: {len(self.waypoints_data)} waypoints loaded")
        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ready!")

        # Calibrate AMCL before patrol
        self.calibrate_amcl()

        # Run patrol
        self.run_patrol()

    # ========== Callbacks ==========

    def _pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w
        self.current_yaw = 2.0 * math.atan2(oz, ow)

    # ========== Robot Control ==========

    def create_pose(self, x, y, oz, ow):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = oz
        pose.pose.orientation.w = ow
        return pose

    def stop_robot(self, duration=1.5):
        """Publish zero velocity for duration seconds to override Nav2 commands."""
        stop = Twist()
        end = time.time() + duration
        while time.time() < end:
            try:
                self.cmd_vel_pub.publish(stop)
            except Exception:
                break
            time.sleep(0.05)

    def move(self, linear, angular, duration):
        """Move robot with given velocities for given duration."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        end_time = time.time() + duration
        while time.time() < end_time and self.is_running:
            rclpy.spin_once(self, timeout_sec=0)
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self.stop_robot()

    def wait_for_amcl(self, seconds=1.5):
        """Wait for AMCL to settle, with tiny nudge to trigger update."""
        nudge = Twist()
        nudge.angular.z = 0.05
        for _ in range(3):
            self.cmd_vel_pub.publish(nudge)
            time.sleep(0.1)
        nudge.angular.z = -0.05
        for _ in range(3):
            self.cmd_vel_pub.publish(nudge)
            time.sleep(0.1)
        self.stop_robot()

        end = time.time() + seconds
        while time.time() < end and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.1)

    # ========== AMCL Calibration ==========

    def calibrate_amcl(self):
        """Move robot slightly to help AMCL converge."""
        self.get_logger().info("Calibrating AMCL pose...")
        self.move(0.08, 0.0, 1.0)
        self.move(-0.08, 0.0, 1.0)
        self.move(0.0, 0.3, 2.0)
        self.move(0.0, -0.3, 2.0)
        time.sleep(0.5)
        self.get_logger().info("AMCL calibration done")

    # ========== Yaw Alignment ==========

    def rotate_to_yaw(self, target_oz, target_ow):
        """Small step rotation with AMCL correction after each step."""
        # Calculate target yaw snapped to nearest 90 degrees
        raw_yaw = 2.0 * math.atan2(target_oz, target_ow)
        target_yaw = round(raw_yaw / (math.pi / 2)) * (math.pi / 2)
        target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))

        # Log target direction
        deg = round(math.degrees(target_yaw))
        direction_map = {0: 'East', 90: 'North', -90: 'South', 180: 'West', -180: 'West'}
        direction = direction_map.get(deg, f'{deg}°')
        self.get_logger().info(f"Rotate target: {direction} ({deg}°)")

        # Iterative small-step rotation
        max_attempts = 4

        for attempt in range(max_attempts):
            if not self.is_running:
                break

            # Stop and wait for AMCL to settle
            self.stop_robot()
            self.wait_for_amcl(1.5)

            if self.current_yaw is None:
                self.get_logger().warn("No AMCL pose available, skipping")
                return

            # Calculate remaining angle
            diff = math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw))

            self.get_logger().info(
                f"  [{attempt + 1}/{max_attempts}] "
                f"current={math.degrees(self.current_yaw):.1f}° "
                f"target={math.degrees(target_yaw):.1f}° "
                f"diff={math.degrees(diff):.1f}°")

            # Check if aligned
            if abs(diff) < self.yaw_tolerance:
                self.get_logger().info("  Yaw aligned!")
                return

            overshoot = 1.2 if abs(diff) > math.radians(30) else 1.1
            duration = abs(diff) * overshoot / self.rotation_speed

            twist = Twist()
            twist.angular.z = self.rotation_speed if diff > 0 else -self.rotation_speed
            end_time = time.time() + duration
            while time.time() < end_time and self.is_running:
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)

            self.stop_robot()

        if self.current_yaw is not None:
            final_diff = abs(math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw)))
            self.get_logger().info(
                f"  Yaw done: {math.degrees(self.current_yaw):.1f}° "
                f"(target {deg}°, error {math.degrees(final_diff):.1f}°)")

    # ========== Patrol Logic ==========

    def run_patrol(self):
        """Navigate through all waypoints sequentially."""
        total = len(self.waypoints_data)

        for i, (x, y, oz, ow) in enumerate(self.waypoints_data):
            if not self.is_running:
                break

            wp_num = i + 1
            self.get_logger().info(
                f"=== Waypoint {wp_num}/{total}: ({x:.2f}, {y:.2f}) ===")

            # Send goal without orientation
            goal_pose = self.create_pose(x, y, 0.0, 1.0)
            self.navigator.goToPose(goal_pose)
            time.sleep(1.0)

            # Wait for Nav2 to complete
            while not self.navigator.isTaskComplete():
                if not self.is_running:
                    self.navigator.cancelTask()
                    return

                rclpy.spin_once(self, timeout_sec=0)

                feedback = self.navigator.getFeedback()
                if feedback:
                    self.get_logger().info(
                        f"  Distance remaining: "
                        f"{feedback.distance_remaining:.2f}m",
                        throttle_duration_sec=3)
                time.sleep(0.5)

            # Check navigation result
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(f"  Arrived at waypoint {wp_num}")
            elif result == TaskResult.CANCELED:
                self.get_logger().warn(f"  Waypoint {wp_num} canceled")
                continue
            else:
                self.get_logger().error(
                    f"  Failed to reach waypoint {wp_num}")
                continue

            # Rotate to face target direction
            self.rotate_to_yaw(oz, ow)

            # Trigger shelf scan (aruco_detector cycles through all rows)
            self.get_logger().info("  Triggering shelf scan...")
            if self.scan_client.wait_for_service(timeout_sec=2.0):
                future = self.scan_client.call_async(Trigger.Request())
                while not future.done():
                    rclpy.spin_once(self, timeout_sec=0.1)
                self.get_logger().info("  Shelf scan complete")
            else:
                self.get_logger().warn(
                    "  scan_shelf service unavailable, waiting fallback...")
                time.sleep(self.scan_duration)

        if self.is_running:
            self.get_logger().info("=============================")
            self.get_logger().info("===== PATROL COMPLETE =====")
            self.get_logger().info("=============================")

    # ========== Shutdown ==========

    def shutdown(self):
        """Clean shutdown."""
        self.is_running = False
        try:
            self.navigator.cancelTask()
        except Exception:
            pass
        # Wait for Nav2 controller to process the cancel before publishing stops
        time.sleep(0.5)
        self.stop_robot(duration=1.5)
        try:
            self.get_logger().info("Patrol node stopped")
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()

    # Handle SIGTERM (VSCode task kill) the same as Ctrl+C
    signal.signal(signal.SIGTERM, lambda *_: node.shutdown())

    try:
        node.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass