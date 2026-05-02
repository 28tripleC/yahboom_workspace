import glob
import os
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import yaml


def delete_all_waypoints(directory: str):
    for f in glob.glob(os.path.join(directory, 'waypoints*.yaml')):
        os.remove(f)


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
        self.declare_parameter('append', False)

        self.waypoints_file = self.get_parameter('waypoints_file').get_parameter_value().string_value
        append_mode = self.get_parameter('append').get_parameter_value().bool_value

        if not append_mode:
            waypoints_dir = os.path.expanduser(os.path.dirname(self.waypoints_file) or '~')
            delete_all_waypoints(waypoints_dir)
            self.get_logger().info("New session: old waypoints files deleted.")
        else:
            self.get_logger().info("Append mode: adding to existing waypoints.")

        self.current_pose = None
        self.waypoint_count = 0

        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._pose_callback,
            10
        )

        self.startup_timer = self.create_timer(5.0, self._check_initial_pose)

        self.get_logger().info(f"Waypoint Recorder started. Saving to: {self.waypoints_file}")
        self.get_logger().info("Drive the car to each waypoint location, then press Enter to save.")
        self.get_logger().info("Press Ctrl+C to finish.")

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _pose_callback(self, msg: PoseWithCovarianceStamped):
        self.current_pose = msg.pose.pose

    def _check_initial_pose(self):
        if self.current_pose is None:
            self.get_logger().warning(
                "No /amcl_pose received yet! "
                "Please set the 2D Pose Estimate in RViz before recording waypoints."
            )
        self.startup_timer.cancel()

    def _input_loop(self):
        while rclpy.ok():
            try:
                input("")
            except EOFError:
                break
            self._save_current_pose()

    def _save_current_pose(self):
        if self.current_pose is None:
            self.get_logger().warning(
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
        if rclpy.ok():
            node.get_logger().info(f"Done. {node.waypoint_count} waypoints saved to {node.waypoints_file}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
