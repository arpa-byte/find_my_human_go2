from setuptools import setup
import os
from glob import glob

package_name = 'human_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 包含 launch 文件夹
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Depth and IR based Human Tracker for Unitree Robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 这里的名字对应 ros2 run 的命令
            'depth_tracker = human_tracker.depth_skeleton_tracker:main',
        ],
    },
)
