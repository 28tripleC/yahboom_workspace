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
        data = yaml.safe_load(f) or {}
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
            self.get_logger().warning("Patrol was canceled.")
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
        node.get_logger().info("Ctrl+C received — canceling navigation and stopping robot.")
        node.navigator.cancelTask()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
