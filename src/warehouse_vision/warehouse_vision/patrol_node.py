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

# Cruise speed during marker-search rotations. Kept below the motors' max
# so the commanded rate matches actual odom rate (no command/actual gap)
# and odom feedback stays smooth.
ARUCO_SEARCH_CRUISE = 0.18
# Sweep / search speed when rotating to find a marker — slow enough that
# the camera frames stay sharp for ArUco detection.
ARUCO_SEARCH_SPEED = 0.12
# Number of consecutive fresh frames the marker must appear in before the
# rotation early-stops. Filters out single-frame false positives.
ARUCO_SEARCH_CONSECUTIVE = 2
# When searching for a marker, cruise at the normal rotation speed until the
# remaining error is within this angle (rad), then decelerate to the slower
# search speed so frames stay sharp near where the marker is expected.
ARUCO_SEARCH_DECEL_ANGLE = math.radians(45.0)


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
        self.declare_parameter('rotation_kp', 1.2)
        self.declare_parameter('rotation_timeout', 25.0)
        self.declare_parameter('angular_accel_limit', 2.0)
        self.declare_parameter('rotation_min_omega', 0.18)
        self.declare_parameter('odom_angular_scale_correction', 1.0)
        self.declare_parameter('yaw_tolerance', 0.08)
        self.declare_parameter('aruco_align_enabled', True)
        self.declare_parameter('aruco_align_tolerance', 0.035)
        self.declare_parameter('aruco_align_max_iters', 3)
        self.declare_parameter('aruco_rotation_speed', 0.12)
        self.declare_parameter('aruco_rotation_kp', 1.2)
        self.declare_parameter('aruco_pulse_kick_speed', 0.24)
        self.declare_parameter('aruco_pulse_kick_duration', 0.7)
        self.declare_parameter('aruco_pulse_startup_time', 1.0)
        self.declare_parameter('aruco_pulse_min_duration', 0.8)
        self.declare_parameter('aruco_pulse_max_duration', 2.2)
        self.declare_parameter('aruco_settle_time', 0.4)
        self.declare_parameter('aruco_stale_threshold', 0.3)
        self.declare_parameter('aruco_sweep_angle', 0.26)
        self.declare_parameter('aruco_read_timeout', 1.0)
        # Distance servoing: target marker forward-z (m), tolerance, per-iter
        # max step (clamps a single move so we don't lurch), and the linear
        # speed used while closing the distance gap.
        self.declare_parameter('aruco_target_z', 1.15)
        self.declare_parameter('aruco_distance_tolerance', 0.03)
        self.declare_parameter('aruco_distance_max_step', 0.20)
        self.declare_parameter('aruco_distance_speed', 0.07)
        self.declare_parameter('camera_nav_tilt', -30)
        self.declare_parameter('camera_align_tilt', 0)
        self.declare_parameter('camera_settle_time', 0.5)

        waypoints_file = self.get_parameter('waypoints_file').value
        self.scan_duration = self.get_parameter('scan_duration').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.rotation_kp = self.get_parameter('rotation_kp').value
        self.rotation_timeout = self.get_parameter('rotation_timeout').value
        self.angular_accel_limit = self.get_parameter('angular_accel_limit').value
        self.rotation_min_omega = self.get_parameter('rotation_min_omega').value
        self.odom_angular_scale_correction = (
            self.get_parameter('odom_angular_scale_correction').value)
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').value
        self.aruco_align_enabled = self.get_parameter('aruco_align_enabled').value
        self.aruco_tol = self.get_parameter('aruco_align_tolerance').value
        self.aruco_max_iters = self.get_parameter('aruco_align_max_iters').value
        self.aruco_rotation_speed = self.get_parameter(
            'aruco_rotation_speed').value
        self.aruco_rotation_kp = self.get_parameter(
            'aruco_rotation_kp').value
        self.aruco_pulse_kick_speed = self.get_parameter(
            'aruco_pulse_kick_speed').value
        self.aruco_pulse_kick_duration = self.get_parameter(
            'aruco_pulse_kick_duration').value
        self.aruco_pulse_startup_time = self.get_parameter(
            'aruco_pulse_startup_time').value
        self.aruco_pulse_min_duration = self.get_parameter(
            'aruco_pulse_min_duration').value
        self.aruco_pulse_max_duration = self.get_parameter(
            'aruco_pulse_max_duration').value
        self.aruco_settle = self.get_parameter('aruco_settle_time').value
        self.aruco_stale = self.get_parameter('aruco_stale_threshold').value
        self.aruco_sweep = self.get_parameter('aruco_sweep_angle').value
        self.aruco_read_timeout = self.get_parameter('aruco_read_timeout').value
        self.aruco_target_z = self.get_parameter('aruco_target_z').value
        self.aruco_distance_tolerance = self.get_parameter(
            'aruco_distance_tolerance').value
        self.aruco_distance_max_step = self.get_parameter(
            'aruco_distance_max_step').value
        self.aruco_distance_speed = self.get_parameter(
            'aruco_distance_speed').value
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
        # Tells aruco_detector which shelf the upcoming scan belongs to, so
        # detections can be bucketed for per-shelf AMCL drift compensation.
        self.pub_current_shelf = self.create_publisher(
            Int32, 'current_shelf_id', 10)

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
        self.get_logger().info(
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

    def settle_pose_updates(self, seconds=1.0):
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


    def rotate_to_yaw(self, target_oz, target_ow, stop_on_marker_id=None):
        target_yaw = self.normalize_angle(2.0 * math.atan2(target_oz, target_ow))

        self.get_logger().info(
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
        if abs(map_diff) < self.yaw_tolerance:
            return

        result = self.rotate_by_odom(
            map_diff,
            timeout=self.rotation_timeout,
            stop_on_marker_id=stop_on_marker_id)

        self.settle_pose_updates(1.0)
        if result and result.get('reason') == 'marker':
            self.get_logger().info(
                f"Coarse rotation stopped on marker {stop_on_marker_id}; "
                "skipping map-yaw residual because marker alignment is authoritative")
            return
        if self.current_yaw is not None:
            final_error = self.normalize_angle(target_yaw - self.current_yaw)
            if abs(final_error) >= self.yaw_tolerance:
                reason = result.get('reason', 'unknown') if result else 'none'
                elapsed = result.get('elapsed', 0.0) if result else 0.0
                timeout = result.get('timeout', self.rotation_timeout) if result else self.rotation_timeout
                turned = result.get('turned', 0.0) if result else 0.0
                self.get_logger().warn(
                    f"Rotation residual: {math.degrees(final_error):.1f}° "
                    f"(rotate_reason={reason}, "
                    f"elapsed={elapsed:.1f}/{timeout:.1f}s, "
                    f"odom_turned={math.degrees(turned):.1f}°)")

    def _visible_callback(self, msg):
        try:
            data = json.loads(msg.data)
            markers = {}

            for k, v in data.get('markers', {}).items():
                marker_id = int(k)

                if isinstance(v, dict):
                    markers[marker_id] = {
                        'angle': float(v.get('angle', 0.0)),
                        'z': float(v['z']) if v.get('z') is not None else None,
                        'distance': (
                            float(v['distance'])
                            if v.get('distance') is not None else None
                        ),
                    }
                else:
                    markers[marker_id] = {
                        'angle': float(v),
                        'z': None,
                        'distance': None,
                    }

            self.latest_visible = markers
            self.latest_visible_stamp = float(data['stamp'])

        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().warn(
                f"Failed to parse visible marker message: {e}",
                throttle_duration_sec=5
            )

    def rotate_by_odom(self, delta_yaw, timeout=15.0, tolerance=None,
                       speed_limit=None, kp=None, min_omega=None,
                       stop_on_marker_id=None):
        tolerance = self.yaw_tolerance if tolerance is None else tolerance
        if speed_limit is None:
            speed_limit = self.rotation_speed
        kp = self.rotation_kp if kp is None else kp
        min_omega = self.rotation_min_omega if min_omega is None else min_omega
        # Cap the floor so it can't exceed the speed limit (e.g. fine-align
        # passes a very low speed_limit; the floor must respect it).
        min_omega = min(min_omega, speed_limit)
        # Marker-search rotations use a steadier cruise cap (motors track it
        # cleanly so odom feedback matches commanded rate), then drop to the
        # slower search speed inside the decel zone where the marker is
        # likely to appear in FOV.
        search_cruise_cap = min(ARUCO_SEARCH_CRUISE, speed_limit)
        search_decel_cap = min(ARUCO_SEARCH_SPEED, speed_limit)
        search_min_omega = min(min_omega, search_decel_cap)

        if self.current_odom_yaw is None or abs(delta_yaw) < tolerance:
            return {'reason': 'skipped', 'turned': 0.0, 'error': delta_yaw}

        # Stop early to leave room for braking inertia inside the tolerance.
        # Conservative estimate: time to ramp from min_omega to 0 with the
        # configured accel limit, traveling ~0.5 * min_omega * t_brake.
        t_brake = min_omega / max(self.angular_accel_limit, 0.1)
        brake_margin = 0.5 * min_omega * t_brake
        effective_tol = max(tolerance - brake_margin, math.radians(0.5))

        last = self.current_odom_yaw
        turned = 0.0
        twist = Twist()
        last_error = delta_yaw
        start_time = time.time()
        deadline = start_time + timeout

        dt = 0.02
        max_delta = self.angular_accel_limit * dt
        current_cmd = 0.0
        marker_streak = 0
        last_seen_stamp = self.latest_visible_stamp
        reason = 'timeout'
        error = delta_yaw

        while time.time() < deadline and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.02)
            odom_delta = self.normalize_angle(self.current_odom_yaw - last)
            turned += odom_delta * self.odom_angular_scale_correction
            last = self.current_odom_yaw
            error = delta_yaw - turned

            if abs(error) < effective_tol:
                reason = 'tolerance'
                break
            if error * last_error < 0.0:
                reason = 'overshoot'
                break

            if stop_on_marker_id is not None:
                if self.latest_visible_stamp != last_seen_stamp:
                    last_seen_stamp = self.latest_visible_stamp
                    fresh = (time.time() - self.latest_visible_stamp
                             < self.aruco_stale)
                    if fresh and stop_on_marker_id in self.latest_visible:
                        marker_streak += 1
                    else:
                        marker_streak = 0
                if marker_streak >= ARUCO_SEARCH_CONSECUTIVE:
                    self.get_logger().info(
                        f"Marker {stop_on_marker_id} spotted mid-rotation "
                        f"(streak={marker_streak}), stopping early "
                        f"(turned={math.degrees(turned):.1f}°)")
                    reason = 'marker'
                    break

            # P with deadband floor: never command below the motor's minimum
            # effective speed, otherwise wheels stall and error never closes.
            # Marker-search rotations cruise at search_cruise_cap, then drop
            # to search_decel_cap inside the decel zone for clean detection.
            if stop_on_marker_id is not None:
                if abs(error) < ARUCO_SEARCH_DECEL_ANGLE:
                    cap = search_decel_cap
                    floor = search_min_omega
                else:
                    cap = search_cruise_cap
                    floor = min(min_omega, search_cruise_cap)
            else:
                cap = speed_limit
                floor = min_omega
            desired = max(floor, min(cap, abs(error) * kp))
            target = desired if error > 0 else -desired
            delta = max(-max_delta, min(max_delta, target - current_cmd))
            current_cmd += delta

            self.get_logger().info(
                f"rotate_by_odom: target={math.degrees(delta_yaw):.1f}, "
                f"turned={math.degrees(turned):.1f}, "
                f"error={math.degrees(error):.1f}, "
                f"cmd={current_cmd:.3f}",
                throttle_duration_sec=0.5
            )

            twist.angular.z = current_cmd
            self.cmd_vel_pub.publish(twist)
            last_error = error
            time.sleep(dt)

        while abs(current_cmd) > 1e-3 and self.is_running:
            delta = max(-max_delta, min(max_delta, -current_cmd))
            current_cmd += delta
            twist.angular.z = current_cmd
            self.cmd_vel_pub.publish(twist)
            time.sleep(dt)

        self.stop_robot()
        elapsed = time.time() - start_time
        log = self.get_logger().warn if reason == 'timeout' else self.get_logger().info
        log(
            f"rotate_by_odom finished: reason={reason}, "
            f"elapsed={elapsed:.1f}/{timeout:.1f}s, "
            f"target={math.degrees(delta_yaw):.1f}°, "
            f"turned={math.degrees(turned):.1f}°, "
            f"error={math.degrees(error):.1f}°")
        return {
            'reason': reason,
            'turned': turned,
            'error': error,
            'elapsed': elapsed,
            'timeout': timeout,
        }

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
                return self.latest_visible[shelf_id]["angle"]
        self.get_logger().warn(
            f"Read failed: {frames_seen} fresh frames in "
            f"{self.aruco_read_timeout:.1f}s, "
            f"last visible={list(self.latest_visible.keys())}")
        return None

    def _read_marker_z(self, shelf_id):
        """Wait for a fresh post-stop reading; return marker forward z distance (m)."""
        settled_at = time.time()
        end = settled_at + self.aruco_read_timeout
        frames_seen = 0
        last_stamp = self.latest_visible_stamp

        while time.time() < end and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.latest_visible_stamp != last_stamp:
                frames_seen += 1
                last_stamp = self.latest_visible_stamp

            fresh = (
                self.latest_visible_stamp >= settled_at and
                time.time() - self.latest_visible_stamp < self.aruco_stale
            )

            if fresh and shelf_id in self.latest_visible:
                z = self.latest_visible[shelf_id].get("z")
                if z is not None:
                    return float(z)

        self.get_logger().warn(
            f"Distance read failed: {frames_seen} fresh frames in "
            f"{self.aruco_read_timeout:.1f}s, "
            f"last visible={list(self.latest_visible.keys())}"
        )
        return None

    def _pulse_rotate_for_marker(self, angle):
        speed = abs(self.aruco_rotation_speed)
        if speed < 1e-3:
            self.get_logger().warn("ArUco rotation speed too small, skipping")
            return

        duration = self.aruco_pulse_startup_time + abs(angle) / speed
        duration = max(self.aruco_pulse_min_duration, duration)
        duration = min(self.aruco_pulse_max_duration, duration)
        angular = speed if angle > 0.0 else -speed
        kick_speed = max(abs(self.aruco_pulse_kick_speed), speed)
        kick_angular = kick_speed if angle > 0.0 else -kick_speed
        kick_duration = min(max(self.aruco_pulse_kick_duration, 0.0), duration)
        hold_duration = max(0.0, duration - kick_duration)

        self.get_logger().info(
            f"Visual pulse rotate: angle={math.degrees(angle):.2f}°, "
            f"kick={kick_angular:.3f}rad/s for {kick_duration:.2f}s, "
            f"hold={angular:.3f}rad/s for {hold_duration:.2f}s")

        twist = Twist()
        dt = 0.02
        end_kick = time.time() + kick_duration
        while time.time() < end_kick and self.is_running:
            rclpy.spin_once(self, timeout_sec=0)
            twist.angular.z = kick_angular
            self.cmd_vel_pub.publish(twist)
            time.sleep(dt)

        end_hold = time.time() + hold_duration
        while time.time() < end_hold and self.is_running:
            rclpy.spin_once(self, timeout_sec=0)
            twist.angular.z = angular
            self.cmd_vel_pub.publish(twist)
            time.sleep(dt)

        self.stop_robot(duration=0.5)

    def align_to_marker(self, shelf_id, allow_sweep=True):
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
            if not allow_sweep:
                self.get_logger().warn(
                    f"Marker {shelf_id} not visible on re-align, skipping")
                return
            self.get_logger().warn(
                f"Marker {shelf_id} not visible, sweeping...")
            for delta in [self.aruco_sweep,
                          -2.0 * self.aruco_sweep,
                          self.aruco_sweep]:
                self.rotate_by_odom(delta, stop_on_marker_id=shelf_id)
                self._wait_for_aruco_settle()
                angle = self._read_marker_angle(shelf_id)
                if angle is not None:
                    break
            if angle is None:
                self.get_logger().warn(
                    f"Marker {shelf_id} not found, skipping align")
                return

        previous_abs_angle = abs(angle)
        no_progress_count = 0
        for i in range(self.aruco_max_iters):
            if abs(angle) < self.aruco_tol:
                self.get_logger().info(
                    f"Aligned: {math.degrees(angle):.2f}° "
                    f"(iter {i})")
                return

            self.get_logger().info(
                f"Align iter {i}: marker at {math.degrees(angle):.2f} deg, rotating")

            self._pulse_rotate_for_marker(angle)

            self._wait_for_aruco_settle()
            new_angle = self._read_marker_angle(shelf_id)
            if new_angle is None:
                self.get_logger().warn(
                    "Lost marker after rotation, stopping align")
                return
            progress = previous_abs_angle - abs(new_angle)
            if progress < math.radians(0.5):
                no_progress_count += 1
                self.get_logger().warn(
                    f"Marker angle did not improve enough: "
                    f"{math.degrees(angle):.2f}° -> "
                    f"{math.degrees(new_angle):.2f}°")
                if no_progress_count >= 2:
                    self.get_logger().warn(
                        "Visual pulse made no progress twice; stopping align")
                    return
            else:
                no_progress_count = 0
            previous_abs_angle = abs(new_angle)
            angle = new_angle
        if abs(angle) < self.aruco_tol:
            self.get_logger().info(
                f"Aligned: {math.degrees(angle):.2f}° "
                f"(after {self.aruco_max_iters} iters)")
            return
        self.get_logger().warn(
            f"Max iters reached, residual {math.degrees(angle):.2f}°")

    def adjust_distance_to_marker(self, shelf_id):
        if shelf_id is None:
            self.get_logger().info("No shelf_id, skipping distance adjustment")
            return

        speed = abs(self.aruco_distance_speed)
        if speed < 1e-3:
            self.get_logger().warn("Distance speed too small, skipping")
            return

        self.get_logger().info(
            f"Adjusting distance to shelf marker {shelf_id}, "
            f"target_z={self.aruco_target_z:.2f}m"
        )

        last_error = None
        for i in range(self.aruco_max_iters):
            self._wait_for_aruco_settle()
            z = self._read_marker_z(shelf_id)
            if z is None:
                self.get_logger().warn(
                    f"Lost marker {shelf_id} during distance adjust "
                    f"(iter {i}), stopping")
                return

            error = z - self.aruco_target_z
            if abs(error) <= self.aruco_distance_tolerance:
                self.get_logger().info(
                    f"Distance OK: z={z:.2f}m, error={error:.2f}m "
                    f"(iter {i})")
                return

            move_dist = max(
                -self.aruco_distance_max_step,
                min(self.aruco_distance_max_step, error)
            )
            duration = abs(move_dist) / speed
            linear = speed if move_dist > 0.0 else -speed

            self.get_logger().info(
                f"Distance adjust iter {i}: z={z:.2f}m, error={error:.2f}m, "
                f"move={move_dist:.2f}m, linear={linear:.2f}m/s, "
                f"duration={duration:.2f}s"
            )

            self.move(linear, 0.0, duration)
            self.stop_robot(duration=0.5)
            last_error = error

        self.get_logger().warn(
            f"Distance adjust max iters reached, residual "
            f"{last_error:.2f}m" if last_error is not None
            else "Distance adjust max iters reached")

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
                    self.get_logger().info(
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

            # Rotate to face target direction(roughly towards shelf). If we
            # spot the shelf marker mid-rotation, stop early — the nominal
            # map yaw is just an estimate and AMCL drift can put it well
            # outside FOV; the marker itself is the ground truth.
            self.rotate_to_yaw(oz, ow, stop_on_marker_id=shelf_id)

            # Fine alignment using ArUco marker on shelf: yaw, then close
            # the standoff distance, then re-yaw since driving forward can
            # shift the marker's bearing slightly.
            self.align_to_marker(shelf_id)
            self.adjust_distance_to_marker(shelf_id)
            self.align_to_marker(shelf_id, allow_sweep=False)

            self.get_logger().info("Triggering shelf scan")
            shelf_msg = Int32()
            shelf_msg.data = int(shelf_id) if shelf_id is not None else -1
            self.pub_current_shelf.publish(shelf_msg)
            if self.scan_client.wait_for_service(timeout_sec=2.0):
                future = self.scan_client.call_async(Trigger.Request())
                while not future.done():
                    rclpy.spin_once(self, timeout_sec=0.1)
                self.get_logger().info("Shelf scan complete")
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
