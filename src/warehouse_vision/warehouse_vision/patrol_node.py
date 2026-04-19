import math
import os
import time
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


def load_waypoints_from_yaml(path: str) -> list:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Waypoints file not found: {expanded}")
    with open(expanded) as f:
        data = yaml.safe_load(f) or {}
    waypoints = data.get('waypoints', [])
    if not waypoints:
        raise ValueError("Waypoints file is empty or has no 'waypoints' key")
    return [(wp['x'], wp['y'], wp['oz'], wp['ow']) for wp in waypoints]


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        self.declare_parameter('waypoints_file', '~/waypoints.yaml')
        self.declare_parameter('rotation_speed', 0.15)
        self.declare_parameter('yaw_tolerance', 0.1)

        waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value
        self.rotation_speed = self.get_parameter('rotation_speed').get_parameter_value().double_value
        self.yaw_tolerance = self.get_parameter('yaw_tolerance').get_parameter_value().double_value

        try:
            self.waypoints_data = load_waypoints_from_yaml(waypoints_file)
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(str(e))
            raise SystemExit(1)

        self.navigator = BasicNavigator()
        self.current_yaw = None

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            10
        )

        self.get_logger().info(f"Patrol node started with {len(self.waypoints_data)} waypoints")
        self.get_logger().info("Waiting for Nav2 action server...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Nav2 ready! Starting patrol...")

        self.run_patrol()

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        oz = msg.pose.pose.orientation.z
        ow = msg.pose.pose.orientation.w
        self.current_yaw = 2.0 * math.atan2(oz, ow)

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

    def rotate_to_yaw(self, target_oz, target_ow):
        target_yaw = 2.0 * math.atan2(target_oz, target_ow)
        self.get_logger().info(f"Aligning to yaw: {math.degrees(target_yaw):.1f}°")

        twist = Twist()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_yaw is None:
                continue

            diff = math.atan2(
                math.sin(target_yaw - self.current_yaw),
                math.cos(target_yaw - self.current_yaw)
            )

            if abs(diff) < self.yaw_tolerance:
                break

            twist.angular.z = self.rotation_speed if diff > 0 else -self.rotation_speed
            self.cmd_vel_pub.publish(twist)

        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info("Yaw aligned.")

    def run_patrol(self):
        try:
            for i, (x, y, oz, ow) in enumerate(self.waypoints_data):
                self.get_logger().info(f"Navigating to waypoint {i + 1}/{len(self.waypoints_data)}")

                self.navigator.goToPose(self.create_pose(x, y, oz, ow))

                reached = False
                while not self.navigator.isTaskComplete():
                    rclpy.spin_once(self, timeout_sec=0)
                    feedback = self.navigator.getFeedback()
                    if feedback:
                        remaining = feedback.distance_remaining
                        self.get_logger().info(
                            f"Waypoint {i + 1}: {remaining:.2f}m remaining",
                            throttle_duration_sec=3
                        )
                        if remaining <= 0.05:
                            self.navigator.cancelTask()
                            reached = True
                            break
                    time.sleep(0.5)

                if not reached:
                    result = self.navigator.getResult()
                    if result != TaskResult.SUCCEEDED:
                        self.get_logger().error(f"Failed to reach waypoint {i + 1}: {result}")
                        continue

                self.get_logger().info(f"Reached waypoint {i + 1}, aligning yaw...")
                self.rotate_to_yaw(oz, ow)
                self.get_logger().info("Holding position for 5 seconds...")
                time.sleep(5.0)

        except KeyboardInterrupt:
            self.get_logger().info("Ctrl+C — canceling navigation and stopping robot.")
            self.navigator.cancelTask()
            self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PatrolNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
