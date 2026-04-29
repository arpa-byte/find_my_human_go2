import os
from datetime import datetime

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("camera_person_tracker")
    livox_share = get_package_share_directory("livox_ros_driver2")

    rviz_config = os.path.join(
        pkg_share,
        "configs",
        "rviz",
        "fused_track_b.rviz",
    )

    lidar_slice_config = os.path.join(
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

    lidar_track_config = os.path.join(
        pkg_share,
        "configs",
        "lidar",
        "lidar_track_manager.yaml",
    )

    fusion_config = os.path.join(
        pkg_share,
        "configs",
        "fusion",
        "fused_track_b.yaml",
    )

    livox_user_config_path = os.path.join(
        livox_share,
        "config",
        "MID360_config.json",
    )

    # Shared frame
    base_frame = LaunchConfiguration("base_frame")

    # LiDAR mount relative to base_link
    livox_frame = LaunchConfiguration("livox_frame")
    livox_x = LaunchConfiguration("livox_x")
    livox_y = LaunchConfiguration("livox_y")
    livox_z = LaunchConfiguration("livox_z")
    livox_roll = LaunchConfiguration("livox_roll")
    livox_pitch = LaunchConfiguration("livox_pitch")
    livox_yaw = LaunchConfiguration("livox_yaw")

    # Camera mount relative to base_link
    camera_mount_frame = LaunchConfiguration("camera_mount_frame")
    camera_x = LaunchConfiguration("camera_x")
    camera_y = LaunchConfiguration("camera_y")
    camera_z = LaunchConfiguration("camera_z")
    camera_roll = LaunchConfiguration("camera_roll")
    camera_pitch = LaunchConfiguration("camera_pitch")
    camera_yaw = LaunchConfiguration("camera_yaw")

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
        DeclareLaunchArgument("base_frame", default_value="base_link"),

        DeclareLaunchArgument("livox_frame", default_value="livox_frame"),
        DeclareLaunchArgument("livox_x", default_value="0.0"),
        DeclareLaunchArgument("livox_y", default_value="0.0"),
        DeclareLaunchArgument("livox_z", default_value="0.0"),
        DeclareLaunchArgument("livox_roll", default_value="0.0"),
        DeclareLaunchArgument("livox_pitch", default_value="0.0"),
        DeclareLaunchArgument("livox_yaw", default_value="0.0"),

        DeclareLaunchArgument("camera_mount_frame", default_value="camera_link"),
        DeclareLaunchArgument("camera_x", default_value="0.10"),
        DeclareLaunchArgument("camera_y", default_value="0.0"),
        DeclareLaunchArgument("camera_z", default_value="0.0"),
        DeclareLaunchArgument("camera_roll", default_value="0.0"),
        DeclareLaunchArgument("camera_pitch", default_value="0.0"),
        DeclareLaunchArgument("camera_yaw", default_value="0.0"),

        # 1) Camera persistence target node
        Node(
            package="camera_person_tracker",
            executable="camera_target_persistence_node",
            name="camera_target_persistence",
            output="screen",
        ),

        # 2) LiDAR driver
        Node(
            package="livox_ros_driver2",
            executable="livox_ros_driver2_node",
            name="livox_lidar_publisher",
            output="screen",
            parameters=livox_ros2_params,
        ),

        # 3) LiDAR slice -> scan
        Node(
            package="camera_person_tracker",
            executable="lidar_pointcloud_to_scan",
            name="lidar_pointcloud_to_scan",
            output="screen",
            parameters=[lidar_slice_config],
        ),

        # 4) DROW3 detector
        Node(
            package="camera_person_tracker",
            executable="drow3_lidar_detector_node",
            name="drow3_lidar_detector",
            output="screen",
            parameters=[drow3_config],
        ),

        # 5) LiDAR track manager
        Node(
            package="camera_person_tracker",
            executable="lidar_track_manager_node",
            name="lidar_track_manager",
            output="screen",
            parameters=[lidar_track_config],
        ),

        # 6) New fused target tracker
        Node(
            package="camera_person_tracker",
            executable="fused_target_tracker_node",
            name="fused_target_tracker",
            output="screen",
            parameters=[
                fusion_config,
                {
                    "world_frame": base_frame,
                },
            ],
        ),

        # 7) Static TF: base_link -> livox_frame
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_livox_tf_b",
            output="screen",
            arguments=[
                "--x", livox_x,
                "--y", livox_y,
                "--z", livox_z,
                "--roll", livox_roll,
                "--pitch", livox_pitch,
                "--yaw", livox_yaw,
                "--frame-id", base_frame,
                "--child-frame-id", livox_frame,
            ],
        ),

        # 8) Static TF: base_link -> camera_link
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_camera_tf_b",
            output="screen",
            arguments=[
                "--x", camera_x,
                "--y", camera_y,
                "--z", camera_z,
                "--roll", camera_roll,
                "--pitch", camera_pitch,
                "--yaw", camera_yaw,
                "--frame-id", base_frame,
                "--child-frame-id", camera_mount_frame,
            ],
        ),

        # 9) Camera image feedback
        Node(
            package="rqt_image_view",
            executable="rqt_image_view",
            name="rqt_image_view_fused_b",
            output="screen",
            arguments=["/camera_person_tracker/simple_output"],
        ),

        # 10) RViz
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_fused_track_b",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])