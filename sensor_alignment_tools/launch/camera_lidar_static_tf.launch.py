import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # -----------------------------
    # Launch arguments
    # -----------------------------
    parent_frame_arg = DeclareLaunchArgument(
        'parent_frame',
        default_value='livox_frame'
    )
    child_frame_arg = DeclareLaunchArgument(
        'child_frame',
        default_value='camera_link'
    )

    x_arg = DeclareLaunchArgument('x', default_value='0.0')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.0')

    roll_arg = DeclareLaunchArgument('roll', default_value='0.0')
    pitch_arg = DeclareLaunchArgument('pitch', default_value='0.0')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0.0')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true'
    )

    # -----------------------------
    # Paths
    # -----------------------------
    livox_share = get_package_share_directory('livox_ros_driver2')
    livox_config_path = os.path.join(livox_share, 'config', 'MID360_config.json')

    # -----------------------------
    # 1) Livox driver directly
    #    IMPORTANT: publish PointCloud2
    # -----------------------------
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=[{
            'xfer_format': 0,              # 0 = PointCloud2
            'multi_topic': 0,
            'data_src': 0,
            'publish_freq': 10.0,
            'output_data_type': 0,
            'frame_id': 'livox_frame',
            'lvx_file_path': '',
            'user_config_path': livox_config_path,
            'cmdline_input_bd_code': ''
        }]
    )

    # -----------------------------
    # 2) map -> livox_frame
    # -----------------------------
    map_to_livox_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_livox_static_tf',
        output='screen',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'map',
            '--child-frame-id', 'livox_frame',
        ]
    )

    # -----------------------------
    # 3) livox_frame -> camera_link
    #    (your guessed extrinsic)
    # -----------------------------
    camera_lidar_tf = Node(
        package='sensor_alignment_tools',
        executable='camera_lidar_static_tf_node',
        name='camera_lidar_static_tf_node',
        output='screen',
        parameters=[{
            'parent_frame': LaunchConfiguration('parent_frame'),
            'child_frame': LaunchConfiguration('child_frame'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'roll': LaunchConfiguration('roll'),
            'pitch': LaunchConfiguration('pitch'),
            'yaw': LaunchConfiguration('yaw'),
        }]
    )

    # -----------------------------
    # 4) Camera tracker stack
    #    (no extra RViz, no rqt)
    # -----------------------------
    rgb_person_detector = Node(
        package='camera_person_tracker',
        executable='rgb_person_detector',
        name='rgb_person_detector',
        output='screen'
    )

    target_selector = Node(
        package='camera_person_tracker',
        executable='target_selector',
        name='target_selector',
        output='screen'
    )

    target_tf_broadcaster = Node(
        package='camera_person_tracker',
        executable='target_tf_broadcaster',
        name='target_tf_broadcaster',
        output='screen'
    )

    # -----------------------------
    # 5) Single RViz window only
    # -----------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_alignment',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # -----------------------------
    # Delays for cleaner startup
    # -----------------------------
    delayed_camera_stack = TimerAction(
        period=1.5,
        actions=[
            rgb_person_detector,
            target_selector,
            target_tf_broadcaster,
        ]
    )

    delayed_rviz = TimerAction(
        period=2.5,
        actions=[rviz]
    )

    return LaunchDescription([
        parent_frame_arg,
        child_frame_arg,
        x_arg,
        y_arg,
        z_arg,
        roll_arg,
        pitch_arg,
        yaw_arg,
        use_rviz_arg,

        livox_driver,
        map_to_livox_tf,
        camera_lidar_tf,
        delayed_camera_stack,
        delayed_rviz,
    ])
