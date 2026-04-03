from setuptools import setup

package_name = 'lidar_human_cluster'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arpan',
    maintainer_email='arpan@todo.todo',
    description='Simple LiDAR human candidate detection using ROI filtering and DBSCAN clustering',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_cluster_node = lidar_human_cluster.lidar_cluster_node:main',
        ],
    },
)