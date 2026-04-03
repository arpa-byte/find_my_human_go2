from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
            package="rqt_image_view",
            executable="rqt_image_view",
            name="rqt_image_view",
            arguments=["/camera_person_tracker/target_image"],
        ),
    ])