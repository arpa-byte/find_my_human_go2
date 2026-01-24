# Setting Up Intel RealSense D435i & Human Tracker (ROS 2 Humble)

This repository contains instructions and code for integrating the Intel RealSense D435i with a YOLOv8-based human detection and persistent tracking system.

## Contents

**In this repo:**

1. **[Setting Up RealSense](https://www.google.com/search?q=%231-setting-up-realsense)**
* 1.a [Prerequisites](https://www.google.com/search?q=%231a-prerequisites)
* 1.b [Install librealsense SDK (v2.56.5)](https://www.google.com/search?q=%231b-install-librealsense-sdk-v2565)
* 1.c [Install realsense-ros (tag r/4.56.4)](https://www.google.com/search?q=%231c-install-realsense-ros-tag-r4564)
* 1.d [Launching the Camera](https://www.google.com/search?q=%231d-launching-the-camera)


2. **[Human Tracker with RealSense](https://www.google.com/search?q=%232-human-tracker-with-realsense)**
* 2.a [Current Features](https://www.google.com/search?q=%232a-current-features)
* 2.b [Dependencies](https://www.google.com/search?q=%232b-dependencies)
* 2.c [Directory Structure](https://www.google.com/search?q=%232c-directory-structure)
* 2.d [How to Run](https://www.google.com/search?q=%232d-how-to-run)
* 2.e [What to Expect](https://www.google.com/search?q=%232e-what-to-expect)


3. **[Troubleshooting & Versions](https://www.google.com/search?q=%233-troubleshooting--versions)**

---

## 1. Setting Up RealSense

### 1.a Prerequisites

* **OS:** Ubuntu 22.04 LTS
* **ROS 2:** Humble Hawksbill
* **Hardware:** Intel RealSense D435i (USB 3.0)
* **System Dependencies:**
```bash
sudo apt update && sudo apt install -y git cmake libusb-1.0-0-dev pkg-config libgtk-3-dev libglfw3-dev libssl-dev

```



### 1.b Install librealsense SDK (v2.56.5)

1. **Clone and Checkout:**
```bash
mkdir -p ~/masterthesis/ws/src
cd ~/masterthesis/ws/src
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.56.5

```


2. **Build and Install:**
```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=bool:ON -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true
make -j$(nproc)
sudo make install

```



### 1.c Install realsense-ros (tag r/4.56.4)

1. **Clone and Build:**
```bash
cd ~/masterthesis/ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout r/4.56.4
cd ~/masterthesis/ws
rosdep update
rosdep install -i --from-path src --rosdistro humble -y
colcon build

```


2. **Environment Setup:**
Add these to your `~/.bashrc`:
```bash
source /opt/ros/humble/setup.bash
source ~/masterthesis/ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source ~/unitree2/ws_fastlio/install/setup.bash

```



### 1.d Launching the Camera

```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=true

```

---

## 2. Human Tracker with RealSense

### 2.a Current Features

* Subscribes to RealSense RGB image (`/camera/camera/color/image_raw`).
* Uses **YOLOv8n** for detection + persistent tracking.
* Automatically locks onto the **largest/closest** person as the target.
* Visual Feedback:
* **Green boxes:** Detected people.
* **Red thick box + Crosshair:** Locked target.


* Publishes results to `/human_tracker/annotated`.

### 2.b Dependencies

* Python packages: `ultralytics==8.0.196`, `opencv-python==4.10.0.84`, `numpy==1.26.4`
* ROS packages: `cv_bridge`, `sensor_msgs`, `image_view`

### 2.c Directory Structure

```
masterthesis/ws/src/human_tracker/
├── human_tracker/
│   ├── human_detector.py      ← Main node
│   └── yolov8n.pt             ← Model file
├── launch/
│   └── human_tracker_launch.py
├── package.xml
└── setup.py

```

### 2.d How to Run

**Option 1: Launch File (Recommended)**

```bash
ros2 launch human_tracker human_tracker_launch.py

```

**Option 2: Manual Start (3 Terminals)**

1. **Camera:** `ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=true`
2. **Node:** `ros2 run human_tracker detector`
3. **View:** `ros2 run image_view image_view --ros-args -r image:=/human_tracker/annotated`

### 2.e What to Expect

The terminal will log `TARGET ACQUIRED. Locking onto ID: X` when a person enters the frame. The `image_view` window will highlight the target in red.

---

## 3. Troubleshooting & Versions

| Component | Version |
| --- | --- |
| **librealsense** | v2.56.5 |
| **realsense-ros** | r/4.56.4 |
| **OpenCV** | 4.5.4 |
| **Ultralytics** | 8.0.196 |

* **No Output:** Ensure `RMW_IMPLEMENTATION` is set to `rmw_fastrtps_cpp`.
* **Model Errors:** Ensure `yolov8n.pt` is located inside the `human_tracker/human_tracker/` directory.

---

