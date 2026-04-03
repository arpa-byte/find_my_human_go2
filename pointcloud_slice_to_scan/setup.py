from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pointcloud_slice_to_scan'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arpan',
    maintainer_email='arpan@todo.todo',
    description='Convert a horizontal slice of 3D PointCloud2 into 2D LaserScan.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_slice_to_scan_node = pointcloud_slice_to_scan.pointcloud_slice_to_scan_node:main',
        ],
    },
)