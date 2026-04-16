import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import time


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        # 航点列表 (x, y, yaw角度)
        self.waypoints_data = [
            (8.90, -2.76, 0.663, 0.748),
            (8.72, -6.77, -0.837, 0.547),
        ]

        self.navigator = BasicNavigator()

        self.get_logger().info(f"Patrol node started with {len(self.waypoints)} waypoints")
        self.get_logger().info("Waiting for Nav2 action server...")
        self.navigator.wait_until_nav2_active()
        self.get_logger().info("Nav2 ready! Starting patrol...")

        # 开始巡检
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

        self.navigator.follow_waypoints(waypoints)

        while not self.navigator.is_task_complete():
            feedback = self.navigator.get_feedback()
            if feedback:
                current = feedback.current_waypoint
                total = len(waypoints)
                self.get_logger().info(
                    f"Progress: waypoint {current + 1}/{total}",
                    throttle_duration_sec=3)
            time.sleep(0.5)
        
        result = self.navigator.get_result()
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