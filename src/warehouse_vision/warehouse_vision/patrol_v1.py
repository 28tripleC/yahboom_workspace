import math
import os
import signal
import time
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
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

        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        self.declare_parameter('scan_duration', 5.0)
        self.declare_parameter('rotation_speed', 0.3)
        self.declare_parameter('rotation_min_speed', 0.08)
        self.declare_parameter('rotation_kp', 1.2)
        self.declare_parameter('rotation_timeout', 8.0)
        self.declare_parameter('odom_angular_scale_correction', 1.0)
        self.declare_parameter('yaw_tolerance', 0.08)

        waypoints_file = self.get_parameter('waypoints_file').value
        self.scan_duration = self.get_parameter('scan_duration').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.rotation_min_speed = self.get_parameter('rotation_min_speed').value
        self.rotation_kp = self.get_parameter('rotation_kp').value
        self.rotation_timeout = self.get_parameter('rotation_timeout').value
        self.odom_angular_scale_correction = (
            self.get_parameter('odom_angular_scale_correction').value)
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value

        try:
            self.waypoints_data = load_waypoints_from_yaml(waypoints_file)
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(str(e))
            raise SystemExit(1)

        self.navigator = BasicNavigator()

        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.current_odom_yaw = None

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self._pose_callback, 10)
        self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 20)

        self.is_running = True
        self.scan_client = self.create_client(Trigger, 'scan_shelf')

    def start(self):
        self.get_logger().info(
            f"Patrol node: {len(self.waypoints_data)} waypoints loaded")
        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ready!")

        self.calibrate_amcl()
        self.run_patrol()

    # ========== Callbacks ==========

    @staticmethod
    def quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def _odom_callback(self, msg):
        self.current_odom_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

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
        stop = Twist()
        end = time.time() + duration
        while time.time() < end:
            try:
                self.cmd_vel_pub.publish(stop)
            except Exception:
                break
            time.sleep(0.05)

    def move(self, linear, angular, duration):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        end_time = time.time() + duration
        while time.time() < end_time and self.is_running:
            rclpy.spin_once(self, timeout_sec=0)
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self.stop_robot()

    def wait_for_pose_updates(self, seconds=1.0):
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

    # ========== Calibration & Alignment ==========

    def calibrate_amcl(self):
        self.get_logger().info("Calibrating AMCL pose...")
        self.move(0.08, 0.0, 1.0)
        self.move(-0.08, 0.0, 1.0)
        self.move(0.0, 0.3, 2.0)
        self.move(0.0, -0.3, 2.0)
        time.sleep(0.5)
        self.get_logger().info("AMCL calibration done")


    def rotate_to_yaw(self, target_oz, target_ow):
        target_yaw = self.normalize_angle(2.0 * math.atan2(target_oz, target_ow))
        self.get_logger().debug(
            f"Rotate target: {math.degrees(target_yaw):.1f} deg (map frame)")

        self.stop_robot()
        self.wait_for_pose_updates(1.0)

        if self.current_yaw is None:
            self.get_logger().warn("No AMCL pose available, skipping rotation")
            return
        if self.current_odom_yaw is None:
            self.get_logger().warn("No odom pose available, skipping rotation")
            return

        map_diff = self.normalize_angle(target_yaw - self.current_yaw)
        last_odom_yaw = self.current_odom_yaw
        corrected_turn = 0.0
        self.get_logger().debug(
            f"Map current={math.degrees(self.current_yaw):.1f} deg, "
            f"target={math.degrees(target_yaw):.1f} deg, "
            f"turn={math.degrees(map_diff):.1f} deg")

        if abs(map_diff) < self.yaw_tolerance:
            self.get_logger().debug("Yaw already aligned")
            return

        twist = Twist()
        deadline = time.time() + self.rotation_timeout
        last_error = map_diff

        while time.time() < deadline and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.02)
            odom_delta = self.normalize_angle(self.current_odom_yaw - last_odom_yaw)
            corrected_turn += odom_delta * self.odom_angular_scale_correction
            last_odom_yaw = self.current_odom_yaw

            error = map_diff - corrected_turn
            last_error = error

            if abs(error) < self.yaw_tolerance:
                self.get_logger().debug("Yaw aligned with odom feedback")
                break

            angular = max(
                self.rotation_min_speed,
                min(self.rotation_speed, abs(error) * self.rotation_kp))
            twist.angular.z = angular if error > 0 else -angular
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.02)

        self.stop_robot()

        if abs(last_error) >= self.yaw_tolerance:
            self.get_logger().warn(
                f"Rotation timeout/error: {math.degrees(last_error):.1f}°")
        else:
            self.get_logger().debug(
                f"Yaw done, corrected odom error={math.degrees(abs(last_error)):.1f} deg")

    # ========== Patrol Logic ==========

    def run_patrol(self):
        total = len(self.waypoints_data)

        for i, (x, y, oz, ow) in enumerate(self.waypoints_data):
            if not self.is_running:
                break

            wp_num = i + 1
            self.get_logger().info(
                f"=== Waypoint {wp_num}/{total}: ({x:.2f}, {y:.2f}) ===")

            goal_pose = self.create_pose(x, y, 0.0, 1.0)
            self.navigator.goToPose(goal_pose)
            time.sleep(1.0)

            while not self.navigator.isTaskComplete():
                if not self.is_running:
                    self.navigator.cancelTask()
                    return

                rclpy.spin_once(self, timeout_sec=0)
                feedback = self.navigator.getFeedback()
                if feedback:
                    self.get_logger().debug(
                        f"Distance remaining: {feedback.distance_remaining:.2f}m",
                        throttle_duration_sec=3)

                time.sleep(0.5)

            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(f"Arrived at waypoint {wp_num}")
            elif result == TaskResult.CANCELED:
                self.get_logger().warn(f"Waypoint {wp_num} canceled")
                continue
            else:
                self.get_logger().error(
                    f"Failed to reach waypoint {wp_num}")
                continue

            # Rotate to face target direction
            self.rotate_to_yaw(oz, ow)

            self.get_logger().debug("Triggering shelf scan")
            if self.scan_client.wait_for_service(timeout_sec=2.0):
                future = self.scan_client.call_async(Trigger.Request())
                while not future.done():
                    rclpy.spin_once(self, timeout_sec=0.1)
                self.get_logger().debug("Shelf scan complete")
            else:
                self.get_logger().warn(
                    "scan_shelf service unavailable, waiting fallback...")
                time.sleep(self.scan_duration)

        if self.is_running:
            self.get_logger().info("Patrol complete")

    # ========== Shutdown ==========

    def shutdown(self):
        self.is_running = False
        try:
            self.navigator.cancelTask()
        except Exception:
            pass
        time.sleep(0.5)
        self.stop_robot(duration=1.5)
        try:
            self.get_logger().info("Patrol node stopped")
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()

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
