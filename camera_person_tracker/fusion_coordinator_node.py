#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker


class FusionCoordinatorNode(Node):
    """
    Thin fusion/coordinator node for Track A.

    Responsibilities in this first version:
    - verify that the camera target TF exists
    - transform target_person into a shared frame (base_link)
    - monitor LiDAR scan availability
    - publish a fused marker in the shared frame
    - provide debug logs for alignment

    This node does NOT yet do real camera-vs-LiDAR human association.
    That will be added once LiDAR human detection is available.
    """

    def __init__(self) -> None:
        super().__init__("fusion_coordinator")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("world_frame", "base_link")
        self.declare_parameter("camera_target_frame", "target_person")
        self.declare_parameter("lidar_scan_topic", "/scan_knee")
        self.declare_parameter("fused_marker_topic", "/camera_person_tracker/fused_target_marker")
        self.declare_parameter("scan_timeout_sec", 0.75)
        self.declare_parameter("log_period_sec", 1.0)

        self.world_frame = self.get_parameter(
            "world_frame").get_parameter_value().string_value
        self.camera_target_frame = self.get_parameter(
            "camera_target_frame").get_parameter_value().string_value
        self.lidar_scan_topic = self.get_parameter(
            "lidar_scan_topic").get_parameter_value().string_value
        self.fused_marker_topic = self.get_parameter(
            "fused_marker_topic").get_parameter_value().string_value

        self.scan_timeout_sec = self.get_parameter(
            "scan_timeout_sec").get_parameter_value().double_value
        self.log_period_sec = self.get_parameter(
            "log_period_sec").get_parameter_value().double_value

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # LiDAR scan monitoring
        self.last_scan_recv_time = None
        self.last_scan_finite_count = 0

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.lidar_scan_topic,
            self.scan_callback,
            10,
        )

        self.fused_marker_pub = self.create_publisher(Marker, self.fused_marker_topic, 10)

        self.timer = self.create_timer(0.10, self.update)
        self.last_log_time = self.get_clock().now()

        self.get_logger().info("FusionCoordinatorNode started.")
        self.get_logger().info(f"World frame        : {self.world_frame}")
        self.get_logger().info(f"Camera target frame: {self.camera_target_frame}")
        self.get_logger().info(f"LiDAR scan topic   : {self.lidar_scan_topic}")
        self.get_logger().info(f"Fused marker topic : {self.fused_marker_topic}")

    def scan_callback(self, msg: LaserScan) -> None:
        self.last_scan_recv_time = self.get_clock().now()
        self.last_scan_finite_count = sum(math.isfinite(r) for r in msg.ranges)

    def lidar_scan_alive(self) -> bool:
        if self.last_scan_recv_time is None:
            return False

        now = self.get_clock().now()
        age_sec = (now - self.last_scan_recv_time).nanoseconds / 1e9
        return age_sec <= self.scan_timeout_sec

    def publish_fused_marker(self, x: float, y: float, z: float) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.world_frame

        marker.ns = "fusion"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 300_000_000

        self.fused_marker_pub.publish(marker)

    def maybe_log_status(
        self,
        camera_visible: bool,
        lidar_visible: bool,
        x: Optional[float],
        y: Optional[float],
        z: Optional[float],
    ) -> None:
        now = self.get_clock().now()
        elapsed = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed < self.log_period_sec:
            return

        if camera_visible and x is not None and y is not None and z is not None:
            self.get_logger().info(
                f"[Fusion] camera_target_in_{self.world_frame}: "
                f"x={x:.2f}, y={y:.2f}, z={z:.2f} | "
                f"lidar_scan_alive={lidar_visible} | "
                f"scan_bins={self.last_scan_finite_count}"
            )
        else:
            self.get_logger().info(
                f"[Fusion] camera_target_visible={camera_visible} | "
                f"lidar_scan_alive={lidar_visible} | "
                f"scan_bins={self.last_scan_finite_count}"
            )

        self.last_log_time = now

    def update(self) -> None:
        camera_visible = False
        lidar_visible = self.lidar_scan_alive()

        x = y = z = None

        try:
            tf_target = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_target_frame,
                Time(),
            )

            x = float(tf_target.transform.translation.x)
            y = float(tf_target.transform.translation.y)
            z = float(tf_target.transform.translation.z)

            camera_visible = True
            self.publish_fused_marker(x, y, z)

        except TransformException:
            camera_visible = False

        self.maybe_log_status(camera_visible, lidar_visible, x, y, z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusionCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()