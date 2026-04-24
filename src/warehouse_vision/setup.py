from setuptools import find_packages, setup

package_name = 'warehouse_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/patrol_launch.py']),
        ('share/' + package_name + '/params', ['params/patrol_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ccc',
    maintainer_email='ccc@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_detector = warehouse_vision.aruco_detector:main',
            'calibrate_camera = warehouse_vision.calibrate_camera:main',
            'patrol_node = warehouse_vision.patrol_node:main',
            'waypoint_recorder = warehouse_vision.waypoint_recorder:main',
        ],
    },
)
