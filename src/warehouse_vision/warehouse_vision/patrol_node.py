import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import time
import math


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 航点列表 (x, y, yaw角度)
        self.waypoints = [
            (8.90, -2.76, 0.0),
            (8.72, -6.77, 0.0),
        ]

        self.current_waypoint = 0
        self.scan_duration = 5.0  # 每个航点停留5秒扫描

        self.get_logger().info(f"Patrol node started with {len(self.waypoints)} waypoints")
        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.get_logger().info("Nav2 ready! Starting patrol...")

        # 开始巡检
        self.send_next_waypoint()

    def send_next_waypoint(self):
        if self.current_waypoint >= len(self.waypoints):
            self.get_logger().info("===== PATROL COMPLETE =====")
            self.get_logger().info(f"Visited all {len(self.waypoints)} waypoints")
            return

        x, y, yaw = self.waypoints[self.current_waypoint]
        self.get_logger().info(
            f"Navigating to waypoint {self.current_waypoint + 1}/"
            f"{len(self.waypoints)}: ({x:.2f}, {y:.2f})")

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.position.z = 0.0

        # yaw转四元数
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            return

        self.get_logger().info("Goal accepted, navigating...")
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        # 可以在这里打印导航进度
        pass

    def result_callback(self, future):
        result = future.result()
        wp = self.current_waypoint + 1

        self.get_logger().info(
            f"Arrived at waypoint {wp}/{len(self.waypoints)}")
        self.get_logger().info(
            f"Scanning for {self.scan_duration} seconds...")

        # 停留扫描（aruco_detector在后台自动检测）
        time.sleep(self.scan_duration)

        self.get_logger().info(f"Scan complete at waypoint {wp}")

        # 前往下一个航点
        self.current_waypoint += 1
        self.send_next_waypoint()


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