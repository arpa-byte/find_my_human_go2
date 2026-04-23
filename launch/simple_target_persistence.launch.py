import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("camera_person_tracker"),
        "configs",
        "rviz",
        "simple_tracking.rviz",
    )

    return LaunchDescription([
        Node(
            package="camera_person_tracker",
            executable="camera_target_persistence_node",
            name="camera_target_persistence",
            output="screen",
        ),
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="rqt_image_view_camera_persistence",
            output="screen",
            arguments=["/camera_person_tracker/simple_output"],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_camera_target_persistence",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])