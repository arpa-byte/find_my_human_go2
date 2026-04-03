from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("pointcloud_slice_to_scan")
    config_file = os.path.join(pkg_share, "config", "mid360_slice_to_scan.yaml")

    return LaunchDescription([
        Node(
            package="pointcloud_slice_to_scan",
            executable="pointcloud_slice_to_scan_node",
            name="pointcloud_slice_to_scan",
            output="screen",
            parameters=[config_file],
        )
    ])