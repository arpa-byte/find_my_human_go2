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
        
        # 这一行会自动包含 launch 文件夹下的所有 .py 文件
        # 包括我们新写的 auto_switch_launch.py
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arpan',
    maintainer_email='arpan@example.com',
    description='Auto-switching Human Pose Tracker for Go2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # --- 最终版入口 ---
            # 格式: '可执行命令 = 包名.文件名:主函数名'
            # 以后你只需要运行ros2 run human_tracker auto_tracker
            'auto_tracker = human_tracker.auto_switch_tracker:main',
        ],
    },
)