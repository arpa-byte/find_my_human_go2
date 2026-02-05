import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # --- 1. Realsense 摄像头节点 ---
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='camera',
            name='camera',
            parameters=[{
                'initial_reset': True,          # 每次启动重置硬件
                
                # [关键] 彻底关闭 RGB，腾出 USB 2.0 带宽
                'enable_color': False,
                
                # [关键] 开启深度流 (用于 3D 锁定)
                'enable_depth': True,
                'depth_module.profile': '640x480x15', # 15FPS 对 USB 2.0 很友好
                'align_depth.enable': True,           # 必须开启对齐，让深度匹配红外
                
                # [关键] 开启红外流 (用于 YOLO 视觉识别)
                'enable_infra1': True,
                'enable_infra2': False,               # 只需要左眼红外
                'infra_width': 640,
                'infra_height': 480,
                'infra_fps': 15,
                
                # [关键] 开启激光发射器 (Emitter)
                # 即使在全黑环境下，激光也能打出结构光，保证深度图有数据
                # 同时红外图里人会变得可见
                'emitter_enabled': True,
                'emitter_on_off': False, # 常亮
                'laser_power': 250,      # 适中功率，防止近距离过曝
                
                # 禁用不必要的传感器
                'enable_gyro': False,
                'enable_accel': False,
                
                # 宽容的超时设置
                'wait_for_device_timeout': 60.0,
                'reconnect_timeout': 10.0,
            }],
            output='screen'
        ),

        # --- 2. 深度+骨架追踪算法节点 ---
        Node(
            package='human_tracker',
            executable='depth_tracker',  # 对应 setup.py
            name='depth_skeleton_tracker',
            output='screen'
        ),

        # --- 3. 图像显示窗口 ---
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=['/human_tracker/output'],
            output='screen'
        )
    ])