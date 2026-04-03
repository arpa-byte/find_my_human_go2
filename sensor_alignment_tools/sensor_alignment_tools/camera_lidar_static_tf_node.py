import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """
    Convert roll, pitch, yaw (radians) to quaternion (x, y, z, w).
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw


class CameraLidarStaticTFNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_static_tf_node')

        # Parent and child frames
        self.declare_parameter('parent_frame', 'livox_frame')
        self.declare_parameter('child_frame', 'camera_link')

        # Translation in meters
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.0)

        # Rotation in radians
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)

        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        self.x = float(self.get_parameter('x').value)
        self.y = float(self.get_parameter('y').value)
        self.z = float(self.get_parameter('z').value)

        self.roll = float(self.get_parameter('roll').value)
        self.pitch = float(self.get_parameter('pitch').value)
        self.yaw = float(self.get_parameter('yaw').value)

        self.broadcaster = StaticTransformBroadcaster(self)

        self.publish_transform()

        self.get_logger().info('==========================================')
        self.get_logger().info('Camera-LiDAR static transform published')
        self.get_logger().info(f'Parent frame : {self.parent_frame}')
        self.get_logger().info(f'Child frame  : {self.child_frame}')
        self.get_logger().info(
            f'Translation  : x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f}'
        )
        self.get_logger().info(
            f'Rotation(rad): roll={self.roll:.4f}, pitch={self.pitch:.4f}, yaw={self.yaw:.4f}'
        )
        self.get_logger().info('==========================================')

    def publish_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = self.z

        qx, qy, qz, qw = euler_to_quaternion(self.roll, self.pitch, self.yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarStaticTFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received, shutting down node.')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()