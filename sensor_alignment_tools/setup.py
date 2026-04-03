from setuptools import setup

package_name = 'sensor_alignment_tools'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/camera_lidar_static_tf.launch.py',
            'launch/alignment_mvp.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arpan',
    maintainer_email='arpan@todo.todo',
    description='Tools for camera-LiDAR frame alignment and static transform publishing',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_lidar_static_tf_node = sensor_alignment_tools.camera_lidar_static_tf_node:main',
        ],
    },
)
