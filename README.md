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