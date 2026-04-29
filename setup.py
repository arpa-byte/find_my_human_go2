from setuptools import find_packages, setup
import os
from glob import glob

package_name = "camera_person_tracker"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "models"),
            glob("models/*"),
        ),
        (
            os.path.join("share", package_name, "configs", "rviz"),
            glob("configs/rviz/*.rviz"),
        ),
        (
            os.path.join("share", package_name, "configs", "lidar"),
            glob("configs/lidar/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "configs", "fusion"),
            glob("configs/fusion/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="arpan",
    maintainer_email="arpan@example.com",
    description="Camera-based person detection, LiDAR preprocessing, and fusion prototype package.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_target_persistence_node = camera_person_tracker.camera_target_persistence_node:main",
            "lidar_pointcloud_to_scan = camera_person_tracker.lidar_pointcloud_to_scan_node:main",
            "drow3_lidar_detector_node = camera_person_tracker.drow3_lidar_detector_node:main",
            "lidar_track_manager_node = camera_person_tracker.lidar_track_manager_node:main",
            "fused_target_tracker_node = camera_person_tracker.fused_target_tracker_node:main",
        ],
    },
)