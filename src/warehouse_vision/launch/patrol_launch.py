import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('warehouse_vision'),
        'params', 'patrol_params.yaml'
    )
    log_level = LaunchConfiguration('log_level')

    return LaunchDescription([
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level for patrol_node'),
        Node(
            package='warehouse_vision',
            executable='patrol_node',
            name='patrol_node',
            output='screen',
            parameters=[params],
            arguments=['--ros-args', '--log-level', log_level],
        )
    ])
