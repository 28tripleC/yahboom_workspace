import json
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
from std_msgs.msg import Int32, String
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
    return [(wp['x'], wp['y'], wp['oz'], wp['ow'], wp.get('shelf_id'))
            for wp in waypoints]

class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        self.declare_parameter('scan_duration', 5.0)
        self.declare_parameter('rotation_speed', 0.3)
        self.declare_parameter('rotation_min_speed', 0.028)
        self.declare_parameter('rotation_start_speed', 0.08)
        self.declare_parameter('rotation_start_duration', 0.2)
        self.declare_parameter('rotation_kp', 1.2)
        self.declare_parameter('rotation_coast_time', 0.15)
        self.declare_parameter('rotation_timeout', 8.0)
        self.declare_parameter('odom_angular_scale_correction', 1.0)
        self.declare_parameter('yaw_tolerance', 0.08)
        self.declare_parameter('yaw_rotation_speed', 0.12)
        self.declare_parameter('yaw_rotation_kp', 0.8)
        self.declare_parameter('aruco_align_enabled', True)
        self.declare_parameter('aruco_align_tolerance', 0.035)
        self.declare_parameter('aruco_align_max_iters', 3)
        self.declare_parameter('aruco_settle_time', 0.4)
        self.declare_parameter('aruco_stale_threshold', 0.3)
        self.declare_parameter('aruco_sweep_angle', 0.26)
        self.declare_parameter('aruco_read_timeout', 1.0)
        self.declare_parameter('aruco_rotation_speed', 0.08)
        self.declare_parameter('aruco_rotation_kp', 0.6)
        self.declare_parameter('camera_nav_tilt', -30)
        self.declare_parameter('camera_align_tilt', 0)
        self.declare_parameter('camera_settle_time', 0.5)

        waypoints_file = self.get_parameter('waypoints_file').value
        self.scan_duration = self.get_parameter('scan_duration').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.rotation_min_speed = self.get_parameter('rotation_min_speed').value
        self.rotation_start_speed = self.get_parameter('rotation_start_speed').value
        self.rotation_start_duration = (
            self.get_parameter('rotation_start_duration').value)
        self.rotation_kp = self.get_parameter('rotation_kp').value
        self.rotation_coast_time = self.get_parameter('rotation_coast_time').value
        self.rotation_timeout = self.get_parameter('rotation_timeout').value
        self.odom_angular_scale_correction = (
            self.get_parameter('odom_angular_scale_correction').value)
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.yaw_rotation_speed = self.get_parameter('yaw_rotation_speed').value
        self.yaw_rotation_kp = self.get_parameter('yaw_rotation_kp').value
        self.aruco_align_enabled = self.get_parameter('aruco_align_enabled').value
        self.aruco_tol = self.get_parameter('aruco_align_tolerance').value
        self.aruco_max_iters = self.get_parameter('aruco_align_max_iters').value
        self.aruco_settle = self.get_parameter('aruco_settle_time').value
        self.aruco_stale = self.get_parameter('aruco_stale_threshold').value
        self.aruco_sweep = self.get_parameter('aruco_sweep_angle').value
        self.aruco_read_timeout = self.get_parameter('aruco_read_timeout').value
        self.aruco_rotation_speed = self.get_parameter('aruco_rotation_speed').value
        self.aruco_rotation_kp = self.get_parameter('aruco_rotation_kp').value
        self.camera_nav_tilt = self.get_parameter('camera_nav_tilt').value
        self.camera_align_tilt = self.get_parameter('camera_align_tilt').value
        self.camera_settle_time = self.get_parameter('camera_settle_time').value

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
        self.camera_angle_pub = self.create_publisher(Int32, '/camera_angle', 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self._pose_callback, 10)
        self.create_subscription(
            Odometry, '/odom',
            self._odom_callback, 20)

        self.latest_visible = {}
        self.latest_visible_stamp = 0.0
        self.create_subscription(
            String, '/aruco_visible_markers',
            self._visible_callback, 1)

        self.is_running = True
        self.scan_client = self.create_client(Trigger, 'scan_shelf')

    def start(self):
        self.get_logger().info(
            f"Patrol node: {len(self.waypoints_data)} waypoints loaded")
        self.set_camera_navigation_pose()
        self.get_logger().info("Waiting for Nav2...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ready!")

        self.calibrate_amcl()
        self.run_patrol()

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

    def set_camera_tilt(self, servo_angle_deg, label):
        msg = Int32()
        msg.data = int(servo_angle_deg)
        self.get_logger().debug(
            f"Setting camera {label} tilt: {msg.data} deg")

        wait_until = time.time() + 2.0
        while (self.camera_angle_pub.get_subscription_count() == 0 and
               time.time() < wait_until and self.is_running):
            rclpy.spin_once(self, timeout_sec=0.05)

        if self.camera_angle_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                "No subscriber on /camera_angle; aruco_detector may not receive camera command")

        for _ in range(5):
            self.camera_angle_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.05)
        time.sleep(self.camera_settle_time)

    def set_camera_navigation_pose(self):
        self.set_camera_tilt(self.camera_nav_tilt, 'navigation')

    def set_camera_alignment_pose(self):
        self.set_camera_tilt(self.camera_align_tilt, 'alignment')

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

        if self.current_odom_yaw is None:
            self.get_logger().warn("No odom pose available, skipping rotation")
            return

        self.stop_robot()
        self.wait_for_pose_updates(1.0)

        if self.current_yaw is None:
            self.get_logger().warn("No AMCL pose available, skipping rotation")
            return

        map_diff = self.normalize_angle(target_yaw - self.current_yaw)
        final_error = map_diff
        self.get_logger().debug(
            f"Map current={math.degrees(self.current_yaw):.1f} deg, "
            f"target={math.degrees(target_yaw):.1f} deg, "
            f"turn={math.degrees(map_diff):.1f} deg")

        if abs(map_diff) < self.yaw_tolerance:
            self.get_logger().debug("Yaw already aligned")
            return

        last_odom_yaw = self.current_odom_yaw
        corrected_turn = 0.0
        twist = Twist()
        deadline = time.time() + self.rotation_timeout
        last_error = map_diff

        while time.time() < deadline and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.02)
            odom_delta = self.normalize_angle(
                self.current_odom_yaw - last_odom_yaw)
            corrected_turn += odom_delta * self.odom_angular_scale_correction
            last_odom_yaw = self.current_odom_yaw

            error = map_diff - corrected_turn
            final_error = error

            angular = max(
                self.rotation_min_speed,
                min(self.yaw_rotation_speed, abs(error) * self.yaw_rotation_kp))
            predicted_stop = angular * self.rotation_coast_time

            if abs(error) < self.yaw_tolerance + predicted_stop:
                self.get_logger().debug(
                    f"Yaw aligned by predicted stop, residual={math.degrees(error):.2f} deg")
                break
            if error * last_error < 0.0:
                self.get_logger().debug("Yaw target crossed, stopping turn")
                break

            twist.angular.z = angular if error > 0 else -angular
            self.cmd_vel_pub.publish(twist)
            last_error = error
            time.sleep(0.02)

        self.stop_robot()
        self.wait_for_pose_updates(1.0)
        if self.current_yaw is not None:
            final_error = self.normalize_angle(target_yaw - self.current_yaw)

        if final_error is not None and abs(final_error) >= self.yaw_tolerance:
            self.get_logger().warn(
                f"Rotation timeout/error: {math.degrees(final_error):.1f}°")
            return
        self.get_logger().debug(
            f"Yaw done, corrected error={math.degrees(abs(final_error or 0.0)):.1f} deg")

    def _visible_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.latest_visible = {
                int(k): float(v) for k, v in data['markers'].items()}
            self.latest_visible_stamp = float(data['stamp'])
        except (ValueError, KeyError, TypeError):
            pass

    def rotate_by_odom(self, delta_yaw, timeout=6.0, tolerance=None, speed_limit=None, kp=None):
        tolerance = self.yaw_tolerance if tolerance is None else tolerance
        speed_limit = self.rotation_speed if speed_limit is None else speed_limit
        kp = self.rotation_kp if kp is None else kp

        if self.current_odom_yaw is None or abs(delta_yaw) < tolerance:
            return

        last = self.current_odom_yaw
        turned = 0.0
        twist = Twist()
        start_time = time.time()
        last_error = delta_yaw
        deadline = time.time() + timeout

        while time.time() < deadline and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.02)
            odom_delta = self.normalize_angle(self.current_odom_yaw - last)
            turned += odom_delta * self.odom_angular_scale_correction
            last = self.current_odom_yaw
            error = delta_yaw - turned

            angular = max(self.rotation_min_speed, min(speed_limit, abs(error) * kp))
            if (time.time() - start_time < self.rotation_start_duration and abs(turned) < tolerance):
                angular = max(angular, min(speed_limit, self.rotation_start_speed))
            predicted_stop = angular * self.rotation_coast_time

            if abs(error) < tolerance + predicted_stop:
                break
            if error * last_error < 0.0:
                break

            twist.angular.z = angular if error > 0 else -angular
            self.cmd_vel_pub.publish(twist)
            last_error = error
            time.sleep(0.02)

        self.stop_robot()

    def _wait_for_aruco_settle(self):
        """Let camera/marker callbacks update after motion has stopped."""
        deadline = time.time() + self.aruco_settle
        while time.time() < deadline and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _read_marker_angle(self, shelf_id):
        """Wait for a fresh post-stop reading; return angle (rad) or None."""
        settled_at = time.time()
        end = settled_at + self.aruco_read_timeout
        frames_seen = 0
        last_stamp = self.latest_visible_stamp
        while time.time() < end and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest_visible_stamp != last_stamp:
                frames_seen += 1
                last_stamp = self.latest_visible_stamp
            fresh = (self.latest_visible_stamp >= settled_at and
                     time.time() - self.latest_visible_stamp < self.aruco_stale)
            if fresh and shelf_id in self.latest_visible:
                return self.latest_visible[shelf_id]
        self.get_logger().warn(
            f"Read failed: {frames_seen} fresh frames in "
            f"{self.aruco_read_timeout:.1f}s, "
            f"last visible={list(self.latest_visible.keys())}")
        return None

    def align_to_marker(self, shelf_id):
        if not self.aruco_align_enabled:
            self.get_logger().info("ArUco align disabled, skipping")
            return
        if shelf_id is None:
            self.get_logger().info(
                "No shelf_id in waypoint, skipping ArUco align")
            return
        self.get_logger().info(f"Aligning to shelf marker {shelf_id}")

        self._wait_for_aruco_settle()
        angle = self._read_marker_angle(shelf_id)

        if angle is None:
            self.get_logger().warn(
                f"Marker {shelf_id} not visible, sweeping...")
            for delta in [self.aruco_sweep,
                          -2.0 * self.aruco_sweep,
                          self.aruco_sweep]:
                self.rotate_by_odom(delta)
                self._wait_for_aruco_settle()
                angle = self._read_marker_angle(shelf_id)
                if angle is not None:
                    break
            if angle is None:
                self.get_logger().warn(
                    f"Marker {shelf_id} not found, skipping align")
                return

        for i in range(self.aruco_max_iters):
            if abs(angle) < self.aruco_tol:
                self.get_logger().info(
                    f"Aligned: {math.degrees(angle):.2f}° "
                    f"(iter {i})")
                return

            self.get_logger().debug(
                f"Align iter {i}: marker at {math.degrees(angle):.2f} deg, rotating")

            self.rotate_by_odom(
                angle,
                tolerance=self.aruco_tol,
                speed_limit=self.aruco_rotation_speed,
                kp=self.aruco_rotation_kp)

            self._wait_for_aruco_settle()
            new_angle = self._read_marker_angle(shelf_id)
            if new_angle is None:
                self.get_logger().warn(
                    "Lost marker after rotation, stopping align")
                return
            angle = new_angle
        if abs(angle) < self.aruco_tol:
            self.get_logger().info(
                f"Aligned: {math.degrees(angle):.2f}° "
                f"(after {self.aruco_max_iters} iters)")
            return
        self.get_logger().warn(
            f"Max iters reached, residual {math.degrees(angle):.2f}°")

    def run_patrol(self):
        total = len(self.waypoints_data)

        for i, (x, y, oz, ow, shelf_id) in enumerate(self.waypoints_data):
            if not self.is_running:
                break

            wp_num = i + 1
            self.get_logger().info(
                f"=== Waypoint {wp_num}/{total}: ({x:.2f}, {y:.2f}) ===")

            self.set_camera_navigation_pose()
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

            self.stop_robot(duration=0.5)
            self.set_camera_alignment_pose()

            # Rotate to face target direction(roughly towards shelf)
            self.rotate_to_yaw(oz, ow)

            # Fine alignment using ArUco marker on shelf
            self.align_to_marker(shelf_id)

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

            self.set_camera_navigation_pose()

        if self.is_running:
            self.get_logger().info("Patrol complete")

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
