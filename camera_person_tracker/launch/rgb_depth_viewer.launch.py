from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="camera_person_tracker",
            executable="rgb_depth_viewer",
            name="rgb_depth_viewer",
            output="screen",
        )
    ])