import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory("camera_person_tracker"),
        "configs", "rviz", "tracking.rviz",
    )

    return LaunchDescription([
        Node(
            package="camera_person_tracker",
            executable="rgb_person_detector",
            name="rgb_person_detector",
            output="screen",
        ),
        Node(
            package="camera_person_tracker",
            executable="target_selector",
            name="target_selector",
            output="screen",
        ),
        Node(
            package="camera_person_tracker",
            executable="target_tf_broadcaster",
            name="target_tf_broadcaster",
            output="screen",
        ),
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="rqt_image_view",
            arguments=["/camera_person_tracker/target_image"],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])