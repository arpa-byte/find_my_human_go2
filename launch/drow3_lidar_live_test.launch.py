import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("camera_person_tracker")
    livox_share = get_package_share_directory("livox_ros_driver2")

    slice_config = os.path.join(
        pkg_share,
        "configs",
        "lidar",
        "mid360_slice_to_scan.yaml",
    )

    drow3_config = os.path.join(
        pkg_share,
        "configs",
        "lidar",
        "drow3_detector.yaml",
    )

    rviz_config = os.path.join(
        pkg_share,
        "configs",
        "rviz",
        "drow3_lidar_live_test.rviz",
    )

    livox_user_config_path = os.path.join(
        livox_share,
        "config",
        "MID360_config.json",
    )

    livox_ros2_params = [
        {"xfer_format": 0},
        {"multi_topic": 0},
        {"data_src": 0},
        {"publish_freq": 10.0},
        {"output_data_type": 0},
        {"frame_id": "livox_frame"},
        {"lvx_file_path": "/home/livox/livox_test.lvx"},
        {"user_config_path": livox_user_config_path},
        {"cmdline_input_bd_code": "livox0000000001"},
    ]

    return LaunchDescription([
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=livox_ros2_params,
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="mid360_static_tf",
            output="screen",
            arguments=[
                "--x", "0",
                "--y", "0",
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", "0",
                "--frame-id", "map",
                "--child-frame-id", "livox_frame",
            ],
        ),

        Node(
            package="camera_person_tracker",
            executable="lidar_pointcloud_to_scan",
            name="lidar_pointcloud_to_scan",
            output="screen",
            parameters=[slice_config],
        ),

        Node(
            package="camera_person_tracker",
            executable="drow3_lidar_detector_node",
            name="drow3_lidar_detector",
            output="screen",
            parameters=[drow3_config],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_drow3_lidar_live_test",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])