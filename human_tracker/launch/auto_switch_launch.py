import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
 
def generate_launch_description():
    # 强制指定参数，不使用默认值
    # 使用 424x240 分辨率，这是 D435i 支持的最低分辨率，能极大降低 USB 压力
    realsense_params = {
        'enable_color': 'true',
        'rgb_camera.profile': '424x240x15',    # 强制 RGB 为 240p
        'enable_infra1': 'true',
        'enable_infra2': 'false',
        'depth_module.profile': '424x240x15',  # 强制 红外 为 240p
        'enable_depth': 'false',               # 关闭深度计算图，省流
        'align_depth.enable': 'false',
        'emitter_enabled': 'true',             # 开启激光发射器
        'emitter_on_off': 'false',
        'reliability_policy': 'best_effort',
        'initial_reset': 'true'                # [关键] 每次启动前强制复位摄像头
    }
 
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ]),
        launch_arguments=realsense_params.items()
    )
 
    tracker_node = Node(
        package='human_tracker',
        executable='auto_tracker',
        name='auto_switch_tracker',
        output='screen'
    )
 
    rqt_node = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        arguments=['/human_tracker/output']
    )
 
    return LaunchDescription([
        realsense_launch,
        tracker_node,
        rqt_node
    ])
