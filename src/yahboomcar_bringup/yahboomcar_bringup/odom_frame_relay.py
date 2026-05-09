#!/usr/bin/env python3

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class OdomFrameRelay(Node):
    def __init__(self):
        super().__init__('odom_frame_relay')

        self.declare_parameter('input_topic', '/odom_raw')
        self.declare_parameter('output_topic', '/odom_fixed')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.publisher = self.create_publisher(Odometry, output_topic, 50)
        self.subscription = self.create_subscription(
            Odometry,
            input_topic,
            self.odom_callback,
            50,
        )

    def odom_callback(self, msg):
        fixed_msg = Odometry()
        fixed_msg = msg
        fixed_msg.header.frame_id = self.odom_frame
        fixed_msg.child_frame_id = self.base_frame
        self.publisher.publish(fixed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomFrameRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
