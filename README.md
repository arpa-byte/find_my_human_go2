# Lightweight Camera-LiDAR Fusion for Human Tracking under Partial Camera Occlusion

This package implements the main ROS 2 stack for a thesis project on **lightweight camera-LiDAR fusion for human tracking**. The system combines an Intel RealSense D435i depth camera and a Livox MID360 3D LiDAR to maintain a persistent human target identity even when the person temporarily leaves the camera frame or becomes partially occluded.

The camera and LiDAR detect the human target independently. When both sensors observe the same person, the system associates the detections using position and distance consistency. If the person leaves the camera field of view, the LiDAR continues tracking the target. When the person re-enters the camera frame, the LiDAR track helps preserve the same target identity.

---

## Hardware Used

- Intel RealSense D435i depth camera
- Livox MID360 3D LiDAR
- NVIDIA Jetson Nano
- Laptop running Ubuntu with ROS 2 Humble
- Ethernet connection between Jetson, LiDAR, and laptop

---

## ROS Setup Overview

The system is split across two machines:

### Jetson Nano

The Jetson runs the RealSense camera node. The D435i is physically connected to the Jetson.

### Laptop

The laptop runs:

- Livox MID360 driver
- camera person tracker package
- LiDAR scan conversion
- DROW3 LiDAR detector
- target persistence
- fusion/tracking stack
- RViz visualization

Both machines must use the same DDS/RMW setup so that ROS 2 topics are visible across the network.

Recommended:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
````

If a ROS domain is used, the same `ROS_DOMAIN_ID` must be set on both machines.

---

## External ROS Packages

### RealSense ROS

Repository:

```text
https://github.com/realsenseai/realsense-ros
```

Clone inside the workspace `src` directory:

```bash
cd ~/realsense_ros2_ws/src
git clone https://github.com/realsenseai/realsense-ros.git
```

Build:

```bash
cd ~/realsense_ros2_ws
source /opt/ros/foxy/setup.bash
colcon build
source install/setup.bash
```

The RealSense package is used on the Jetson Nano.

---

### Livox ROS Driver 2

Repository:

```text
https://github.com/Livox-SDK/livox_ros_driver2
```

Clone inside the workspace `src` directory:

```bash
cd ~/unitree2/ws_livox/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
```

Build:

```bash
cd ~/unitree2/ws_livox
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

The Livox driver is installed and run on the laptop.

---

## Package Location

Main package:

```bash
/home/arpan/masterthesis/ws/src/camera_person_tracker
```

Package structure:

```text
camera_person_tracker
├── camera_person_tracker
│   ├── camera_target_persistence_node.py
│   ├── drow3_lidar_detector_node.py
│   ├── fused_target_tracker_node.py
│   ├── fusion_coordinator_node.py
│   ├── lidar_pointcloud_to_scan_node.py
│   ├── lidar_track_manager_node.py
│   ├── rgb_depth_viewer_node.py
│   ├── rgb_person_detector_node.py
│   ├── simple_detector_node.py
│   └── topics.py
├── configs
│   ├── fusion
│   │   └── fused_track_b.yaml
│   ├── lidar
│   │   ├── drow3_detector.yaml
│   │   ├── lidar_track_manager.yaml
│   │   └── mid360_slice_to_scan.yaml
│   └── rviz
│       ├── drow3_lidar_live_test.rviz
│       ├── fused_track_a.rviz
│       ├── fused_track_b.rviz
│       ├── mid360_scan_system.rviz
│       └── simple_tracking.rviz
├── launch
│   ├── drow3_lidar_live_test.launch.py
│   ├── fused_track_b.launch.py
│   ├── mid360_scan_system.launch.py
│   └── simple_target_persistence.launch.py
├── models
│   └── yolov8n.pt
├── test_logs
└── package.xml
```

---

## Important Launch Files

### Main Full Fusion System

```bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

This is the main launch file for the full camera-LiDAR fusion system.

It launches the main target tracking and fusion pipeline.

---

### Camera-Only Target Persistence Stack

```bash
ros2 launch camera_person_tracker simple_target_persistence.launch.py
```

This launch file is used for camera-only testing.

It can be used to check:

* YOLO person detection
* RealSense depth usage
* camera target localization
* camera-only target persistence

---

### LiDAR-Only DROW3 Detection Stack

```bash
ros2 launch camera_person_tracker drow3_lidar_live_test.launch.py
```

This launch file is used to test the LiDAR-only human detection pipeline.

It visualizes the DROW3 detector output from the LiDAR-derived 2D laser scan.

Note: this launch file is for LiDAR detection visualization only. It does not perform full target tracking.

---

### MID360 Scan System

```bash
ros2 launch camera_person_tracker mid360_scan_system.launch.py
```

This launch file is used to bring up the MID360 point cloud to LaserScan conversion and related RViz visualization.

---

## Test Logs

Test run data is saved inside:

```bash
/home/arpan/masterthesis/ws/src/camera_person_tracker/test_logs
```

The logs include CSV files such as:

* `fusion_state_history.csv`
* `camera_candidates_history.csv`
* `camera_owner_history.csv`

These files are useful for analyzing fusion behavior, target ownership, camera candidate selection, and identity persistence over time.

---

## Jetson RealSense Bring-Up

The RealSense D435i is connected to the Jetson Nano. Before launching the main system on the laptop, the RealSense node must be started on the Jetson.

SSH into the Jetson:

```bash
ssh unitree@192.168.123.51
```

Go to the RealSense workspace:

```bash
cd ~/realsense_ros2_ws
```

Launch the RealSense node:

```bash
./launch_realsense_jetson.sh
```

The launch script used on the Jetson is:

```bash
#!/usr/bin/env bash
source /opt/ros/foxy/setup.bash
source ~/realsense_ros2_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
#export ROS_DOMAIN_ID=42

echo "=== Starting RealSense D435i on Jetson (CycloneDDS, Domain 42) ==="

ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    device_type:=d435i \
    enable_color:=true \
    enable_depth:=true \
    enable_gyro:=true \
    enable_accel:=true \
    align_depth.enable:=true \
    color_module.profile:=1280x720x30 \
    depth_module.profile:=848x480x30 \
    infra1.enable:=false \
    infra2.enable:=false \
    pointcloud.enable:=false \
    tf_publish_rate:=0.0 \
    enable_sync:=true
```

After this node starts, the laptop should be able to see RealSense topics such as:

```bash
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
/camera/camera/accel/sample
/camera/camera/gyro/sample
```

Check from the laptop:

```bash
ros2 topic list | grep camera
```

---

## Laptop Environment Setup

On the laptop, source ROS 2 Humble, the Livox workspace, and the thesis workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/unitree2/ws_livox/install/setup.bash
source ~/masterthesis/ws/install/setup.bash
```

Recommended environment:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

If using a ROS domain:

```bash
export ROS_DOMAIN_ID=42
```

The same RMW implementation and ROS domain must be used on the Jetson and laptop.

---

## Build Instructions

From the main workspace:

```bash
cd /home/arpan/masterthesis/ws
source /opt/ros/humble/setup.bash
source ~/unitree2/ws_livox/install/setup.bash
colcon build --packages-select camera_person_tracker
source install/setup.bash
```

---

## Full Bring-Up Sequence

### 1. Start the RealSense node on the Jetson

```bash
ssh unitree@192.168.123.51
cd ~/realsense_ros2_ws
./launch_realsense_jetson.sh
```

### 2. On the laptop, source all required workspaces

```bash
cd /home/arpan/masterthesis/ws
source /opt/ros/humble/setup.bash
source ~/unitree2/ws_livox/install/setup.bash
source install/setup.bash
```

### 3. Verify camera topics are visible

```bash
ros2 topic list | grep camera
```

### 4. Launch the main fusion system

```bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

---

## Useful Topic Checks

Check camera topics:

```bash
ros2 topic list | grep camera
```

Check Livox topics:

```bash
ros2 topic list | grep livox
```

Check LiDAR scan:

```bash
ros2 topic type /scan_knee
```

Expected:

```text
sensor_msgs/msg/LaserScan
```

Check raw or republished LiDAR cloud:

```bash
ros2 topic type /livox/lidar_pointcloud2
```

Expected:

```text
sensor_msgs/msg/PointCloud2
```

---

## Main System Concept

The system works in stages:

1. The RealSense camera detects humans using YOLO.
2. The aligned depth image estimates the camera-side target distance.
3. The camera target is localized in 3D using depth and camera intrinsics.
4. The Livox MID360 point cloud is converted into a 2D laser scan.
5. The DROW3 detector detects human candidates from the scan.
6. The LiDAR track manager maintains LiDAR-side human tracks.
7. The fusion tracker associates camera targets and LiDAR tracks.
8. If the camera temporarily loses the target, the LiDAR track preserves the target identity.
9. When the camera sees the person again, the fusion system re-associates the detection with the previous target identity.

---

## Notes

* The camera must be launched from the Jetson before running the camera or fusion stack on the laptop.
* The LiDAR driver is installed on the laptop.
* Both RealSense and Livox ROS stacks must be discoverable under the same ROS 2 network setup.
* The system currently targets MVP-level behavior and prioritizes end-to-end functionality over perfect detection accuracy.
* The `test_logs` directory stores recorded run data for later analysis and plotting.

