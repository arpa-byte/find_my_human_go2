# Camera-LiDAR Fusion for Human Tracking under Partial Camera Occlusion

This ROS 2 package implements a lightweight camera-LiDAR fusion framework for human target tracking. The system is designed for a mobile robotics use case where a target person may temporarily leave the camera frame or become partially occluded, while still being observable by a 3D LiDAR.

The goal is to maintain a consistent target identity across camera loss and re-detection by using LiDAR-based tracking as a supporting modality.

---

## 1. System Overview

The system uses two main sensors:

- **Intel RealSense D435i**
  - Provides RGB image, aligned depth image, camera intrinsics, and IMU data.
  - Physically connected to the Jetson Nano.
  - ROS topics are published from the Jetson and consumed on the laptop.

- **Livox MID360 3D LiDAR**
  - Provides 3D point cloud and IMU data.
  - Connected to the laptop.
  - Used for LiDAR-only human detection and tracking.

The final objective is to associate camera detections and LiDAR detections geometrically so that both sensors agree when they are observing the same person. When the person exits the camera frame, the LiDAR continues tracking the target. When the person re-enters the camera view, the camera detection is re-associated with the same target identity.

---

## 2. External Packages

This package depends on the following external sensor drivers.

### RealSense ROS

Repository:

```text
https://github.com/realsenseai/realsense-ros
````

The RealSense ROS package is installed and run on the Jetson Nano. It publishes the D435i camera topics over the ROS 2 network.

### Livox ROS Driver 2

Repository:

```text
https://github.com/Livox-SDK/livox_ros_driver2
```

The Livox driver is installed on the laptop in a separate workspace.

Example location:

```bash
/home/arpan/unitree2/ws_livox/src/livox_ros_driver2
```

Both the Jetson and laptop should use the same ROS middleware implementation. In the current setup, CycloneDDS is used:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

---

## 3. Workspace Layout

Main workspace:

```bash
/home/arpan/masterthesis/ws
```

Package location:

```bash
/home/arpan/masterthesis/ws/src/camera_person_tracker
```

Current package tree:

```text
camera_person_tracker
├── camera_person_tracker
│   ├── camera_target_persistence_node.py
│   ├── drow3_lidar_detector_node.py
│   ├── fused_target_tracker_node.py
│   ├── fusion_coordinator_node.py
│   ├── __init__.py
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
├── package.xml
├── resource
│   └── camera_person_tracker
├── setup.cfg
├── setup.py
├── test
└── test_logs
```

---

## 4. Main Nodes

### Camera Nodes

#### `rgb_depth_viewer_node.py`

Validates the RealSense RGB and aligned depth streams.

It subscribes to:

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

It is mainly used for checking that RGB and depth data are arriving correctly.

---

#### `rgb_person_detector_node.py`

Runs YOLO-based person detection on the RGB image stream.

It detects humans in the camera image and publishes detection results for downstream target selection and tracking.

---

#### `simple_detector_node.py`

A simplified camera-side detection/localization node used in the camera-only stack.

---

#### `camera_target_persistence_node.py`

Maintains the selected camera target and supports camera-side target persistence.

This is useful for testing camera-only target identity behavior before fusion with LiDAR.

---

### LiDAR Nodes

#### `lidar_pointcloud_to_scan_node.py`

Converts the Livox MID360 3D point cloud into a 2D LaserScan-like representation by slicing a horizontal section of the point cloud.

This is used because some human detection approaches, such as DROW-style detectors, operate on 2D laser scan data.

Input:

```text
/livox/lidar
```

Output:

```text
/scan_knee
```

The slice height and range settings are configured in:

```bash
configs/lidar/mid360_slice_to_scan.yaml
```

---

#### `drow3_lidar_detector_node.py`

Runs the LiDAR-only human detection pipeline using the generated 2D laser scan.

This is used to test LiDAR-only person detection.

---

#### `lidar_track_manager_node.py`

Maintains LiDAR-side tracks from LiDAR detections.

This node is responsible for handling target continuity on the LiDAR side.

---

### Fusion Nodes

#### `fusion_coordinator_node.py`

Coordinates the camera and LiDAR detections/tracks.

This node is responsible for associating camera and LiDAR observations.

---

#### `fused_target_tracker_node.py`

Maintains the fused target state.

This is the main fusion-level target tracker used in the full system.

---

## 5. Configuration Files

### Fusion Config

```bash
configs/fusion/fused_track_b.yaml
```

Configuration for the main fused tracking system.

---

### LiDAR Configs

```bash
configs/lidar/mid360_slice_to_scan.yaml
```

Controls conversion from 3D point cloud to 2D scan.

Important parameters include:

```yaml
slice_z_min
slice_z_max
range_min
range_max
num_beams
```

These control the height of the horizontal point cloud slice, distance limits, and angular scan resolution.

Other LiDAR configs:

```bash
configs/lidar/drow3_detector.yaml
configs/lidar/lidar_track_manager.yaml
```

---

### RViz Configs

```bash
configs/rviz/simple_tracking.rviz
configs/rviz/drow3_lidar_live_test.rviz
configs/rviz/fused_track_b.rviz
configs/rviz/mid360_scan_system.rviz
```

These files are used for visualizing the camera-only, LiDAR-only, and fused tracking pipelines.

---

## 6. Launch Files

### Main Full System Launch

```bash
launch/fused_track_b.launch.py
```

This is the main launch file for the complete camera-LiDAR fusion system.

It launches the fused tracking pipeline and is the primary launch file for full-system testing.

Run with:

```bash
cd /home/arpan/masterthesis/ws
source install/setup.bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

---

### Camera-Only Stack

```bash
launch/simple_target_persistence.launch.py
```

This launch file starts the camera-only detection and localization stack.

It is used to test:

* RGB human detection
* depth-based localization
* camera-only target persistence

Run with:

```bash
cd /home/arpan/masterthesis/ws
source install/setup.bash
ros2 launch camera_person_tracker simple_target_persistence.launch.py
```

---

### LiDAR-Only Detection Test

```bash
launch/drow3_lidar_live_test.launch.py
```

This launch file starts the LiDAR-only human detection stack.

It is used to visualize the DROW3 detector output from the LiDAR-derived laser scan.

Note:

```text
This launch file is for LiDAR detection visualization only.
No full LiDAR tracking is performed here.
```

Run with:

```bash
cd /home/arpan/masterthesis/ws
source install/setup.bash
ros2 launch camera_person_tracker drow3_lidar_live_test.launch.py
```

---

### MID360 Scan System

```bash
launch/mid360_scan_system.launch.py
```

This launch file is used for testing the Livox MID360 point cloud to laser scan conversion and visualization.

It launches the LiDAR driver, the pointcloud-to-scan conversion node, TF, and RViz visualization.

Run with:

```bash
cd /home/arpan/masterthesis/ws
source install/setup.bash
ros2 launch camera_person_tracker mid360_scan_system.launch.py
```

---

## 7. Bring-Up Sequence

The RealSense camera node must be started on the Jetson before launching camera-dependent nodes on the laptop.

The Livox MID360 node is launched from the laptop.

A typical bring-up sequence is:

1. SSH into the Jetson.
2. Start the RealSense D435i node on the Jetson.
3. On the laptop, source the master thesis workspace.
4. Launch the desired camera-only, LiDAR-only, or fused tracking stack.

---

## 8. Starting the RealSense Node on Jetson

SSH into the Jetson:

```bash
ssh unitree@192.168.123.51
```

Then go to the RealSense workspace:

```bash
cd ~/realsense_ros2_ws
```

The RealSense launch executable used on the Jetson is:

```bash
./launch_realsense_jetson.sh
```

Current content of the executable:

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

Expected RealSense topics on the laptop include:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/aligned_depth_to_color/camera_info
/camera/camera/accel/sample
/camera/camera/gyro/sample
```

---

## 9. Starting the Main Fused System

After the RealSense node is running on the Jetson, start the main fusion system on the laptop:

```bash
cd /home/arpan/masterthesis/ws
source /opt/ros/humble/setup.bash
source /home/arpan/unitree2/ws_livox/install/setup.bash
source install/setup.bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

This is the main launch for the current fused camera-LiDAR tracking system.

---

## 10. Useful ROS Checks

List active topics:

```bash
ros2 topic list
```

List active nodes:

```bash
ros2 node list
```

Check topic type:

```bash
ros2 topic type /camera/camera/color/image_raw
ros2 topic type /camera/camera/aligned_depth_to_color/image_raw
ros2 topic type /livox/lidar
ros2 topic type /scan_knee
```

Echo one message:

```bash
ros2 topic echo /scan_knee --once
```

Check topic frequency:

```bash
ros2 topic hz /livox/lidar
ros2 topic hz /scan_knee
```

---

## 11. Test Logs

The package saves test run data in:

```bash
test_logs/
```

Example files include:

```text
fusion_state_history.csv
camera_candidates_history.csv
camera_owner_history.csv
```

These logs are generated from fusion and persistence test runs and are useful for later analysis, plotting, and thesis evaluation.

The directory contains timestamped runs, for example:

```text
test_logs/run_2026-05-01_21-22-41/
test_logs/run_2026-05-01_21-22-43/
```

---

## 12. Current System Status

Current implemented capability:

* RealSense D435i camera stream is published from Jetson.
* YOLO-based camera human detection is available.
* Camera depth is used for human distance/localization.
* Livox MID360 point cloud is available on the laptop.
* 3D LiDAR point cloud can be sliced into a 2D LaserScan.
* LiDAR-only human detection can be visualized using DROW3-style detection.
* LiDAR tracking and fused target tracking are implemented in the package.
* Main fusion system is launched using `fused_track_b.launch.py`.

---

## 13. Thesis Context

This package supports the thesis topic:

```text
Lightweight camera-LiDAR fusion for human tracking in case of partial camera occlusion
```

The central idea is:

1. The camera and LiDAR detect a human target independently.
2. When both sensors see the same human, geometric consistency is used to associate them.
3. If the human leaves the camera frame, the LiDAR continues tracking the target.
4. When the human re-enters the camera frame, the system re-associates the camera detection with the LiDAR-maintained identity.
5. The target ID remains consistent across temporary camera loss.

This is intended as a lightweight, mobile-robot-oriented tracking framework rather than a heavy full-scene perception system.

---

## 14. Notes

* The RealSense node must be running on the Jetson before camera-side or fusion launch files are started on the laptop.
* The Jetson and laptop should use the same ROS middleware implementation.
* The current setup uses CycloneDDS.
* The Livox driver is installed on the laptop in a separate workspace.
* The full fusion system should be launched from the laptop.


