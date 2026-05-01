# Camera-Person Tracker: Lightweight Camera-LiDAR Fusion for Human Tracking

ROS 2 package for lightweight camera-LiDAR fusion using an Intel RealSense D435i and Livox MID360.  
The system is designed for human target detection, LiDAR-assisted persistence, and camera re-identification after partial/temporary camera occlusion.

Hardware used in the implementation:
- Intel RealSense D435i (Jetson)
- Livox MID360 (Laptop)

---

## 1. Dependencies

- ROS 2 Humble (Laptop)
- ROS 2 Foxy (Jetson)
- Same RMW on both systems:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
````

---

## 2. External Packages

* RealSense ROS
  [https://github.com/realsenseai/realsense-ros](https://github.com/realsenseai/realsense-ros)

* Livox ROS Driver 2
  [https://github.com/Livox-SDK/livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2)

---

## 3. Workspace Setup

```bash
cd ~/masterthesis/ws/src
git clone <YOUR_REPO_URL> camera_person_tracker

cd ~/masterthesis/ws
source /opt/ros/humble/setup.bash
colcon build --packages-select camera_person_tracker
source install/setup.bash
```

Also source Livox workspace:

```bash
source ~/unitree2/ws_livox/install/setup.bash
```

---

## 4. Jetson (Camera Bringup)

```bash
ssh unitree@192.168.123.51
cd ~/realsense_ros2_ws
./launch_realsense_jetson.sh
```

Script used:

```bash
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    device_type:=d435i \
    enable_color:=true \
    enable_depth:=true \
    enable_gyro:=true \
    enable_accel:=true \
    align_depth.enable:=true \
    enable_sync:=true
```

---

## 5. Launch Files

### Main System (Fusion)

```bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

### Camera Only

```bash
ros2 launch camera_person_tracker simple_target_persistence.launch.py
```

### LiDAR Detection (DROW3)

```bash
ros2 launch camera_person_tracker drow3_lidar_live_test.launch.py
```

### LiDAR Scan System

```bash
ros2 launch camera_person_tracker mid360_scan_system.launch.py
```

---

## 6. Logs

Test runs are stored in:

```bash
camera_person_tracker/test_logs/
```

---

## 7. Bringup Sequence

1. Start camera (Jetson)
2. Source environments (Laptop)
3. Launch system:

```bash
ros2 launch camera_person_tracker fused_track_b.launch.py
```

---

## Notes

* Camera runs on Jetson, LiDAR runs on laptop
* Both must share network + same RMW
* Build and source before every run