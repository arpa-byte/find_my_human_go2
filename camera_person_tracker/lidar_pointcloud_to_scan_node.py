#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2


class LidarPointcloudToScanNode(Node):
    """
    Subscribes to a 3D PointCloud2 topic, extracts a horizontal Z slice,
    flattens that slice into a 2D LaserScan, and publishes the scan.

    Also republishes the incoming PointCloud2 to a clean topic so RViz
    can visualize it without Livox CustomMsg ambiguity.
    """

    def __init__(self) -> None:
        super().__init__("lidar_pointcloud_to_scan")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("input_cloud_topic", "/livox/lidar")
        self.declare_parameter("output_scan_topic", "/scan_knee")
        self.declare_parameter("output_cloud_topic", "/livox/lidar_pointcloud2")

        # Z slice in LiDAR frame (meters)
        self.declare_parameter("slice_z_min", -0.60)
        self.declare_parameter("slice_z_max", -0.05)

        # Angular coverage (radians)
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)

        # Number of scan beams
        self.declare_parameter("num_beams", 720)

        # Range limits (meters)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 10.0)

        # Output timing metadata
        self.declare_parameter("scan_time", 0.10)

        # Whether to fill empty bins with inf or range_max + small epsilon
        self.declare_parameter("use_inf", True)

        # Whether to log stats
        self.declare_parameter("verbose", True)

        self.input_cloud_topic = self.get_parameter(
            "input_cloud_topic").get_parameter_value().string_value
        self.output_scan_topic = self.get_parameter(
            "output_scan_topic").get_parameter_value().string_value
        self.output_cloud_topic = self.get_parameter(
            "output_cloud_topic").get_parameter_value().string_value

        self.slice_z_min = self.get_parameter(
            "slice_z_min").get_parameter_value().double_value
        self.slice_z_max = self.get_parameter(
            "slice_z_max").get_parameter_value().double_value

        self.angle_min = self.get_parameter(
            "angle_min").get_parameter_value().double_value
        self.angle_max = self.get_parameter(
            "angle_max").get_parameter_value().double_value

        self.num_beams = self.get_parameter(
            "num_beams").get_parameter_value().integer_value

        self.range_min = self.get_parameter(
            "range_min").get_parameter_value().double_value
        self.range_max = self.get_parameter(
            "range_max").get_parameter_value().double_value

        self.scan_time = self.get_parameter(
            "scan_time").get_parameter_value().double_value

        self.use_inf = self.get_parameter(
            "use_inf").get_parameter_value().bool_value
        self.verbose = self.get_parameter(
            "verbose").get_parameter_value().bool_value

        if self.num_beams < 2:
            raise ValueError("num_beams must be >= 2")

        if self.slice_z_max <= self.slice_z_min:
            raise ValueError("slice_z_max must be greater than slice_z_min")

        if self.angle_max <= self.angle_min:
            raise ValueError("angle_max must be greater than angle_min")

        self.angle_increment = (self.angle_max - self.angle_min) / (self.num_beams - 1)

        self.scan_pub = self.create_publisher(LaserScan, self.output_scan_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, 10)

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_cloud_topic,
            self.cloud_callback,
            10,
        )

        self._msg_count = 0

        self.get_logger().info("LidarPointcloudToScanNode started.")
        self.get_logger().info(f"Input cloud topic : {self.input_cloud_topic}")
        self.get_logger().info(f"Output scan topic : {self.output_scan_topic}")
        self.get_logger().info(f"Output cloud topic: {self.output_cloud_topic}")
        self.get_logger().info(
            f"Z slice          : [{self.slice_z_min:.3f}, {self.slice_z_max:.3f}] m")
        self.get_logger().info(
            f"Angle range      : [{self.angle_min:.3f}, {self.angle_max:.3f}] rad")
        self.get_logger().info(f"Num beams        : {self.num_beams}")
        self.get_logger().info(
            f"Range limits     : [{self.range_min:.3f}, {self.range_max:.3f}] m")

    def cloud_callback(self, msg: PointCloud2) -> None:
        """
        Convert PointCloud2 -> LaserScan using a horizontal Z slice.
        For each angular bin, keep the nearest valid point.
        Also republish the raw PointCloud2 to a clean RViz-safe topic.
        """
        self.cloud_pub.publish(msg)

        if self.use_inf:
            ranges = [float("inf")] * self.num_beams
        else:
            ranges = [self.range_max + 1.0] * self.num_beams

        total_points = 0
        slice_points = 0

        try:
            points_iter = point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )

            for p in points_iter:
                total_points += 1
                x, y, z = float(p[0]), float(p[1]), float(p[2])

                # Keep only the chosen horizontal slice
                if z < self.slice_z_min or z > self.slice_z_max:
                    continue

                slice_points += 1

                r = math.hypot(x, y)
                if r < self.range_min or r > self.range_max:
                    continue

                angle = math.atan2(y, x)
                if angle < self.angle_min or angle > self.angle_max:
                    continue

                bin_index = int((angle - self.angle_min) / self.angle_increment)

                if bin_index < 0 or bin_index >= self.num_beams:
                    continue

                if r < ranges[bin_index]:
                    ranges[bin_index] = r

            scan_msg = LaserScan()
            scan_msg.header = msg.header
            scan_msg.angle_min = float(self.angle_min)
            scan_msg.angle_max = float(self.angle_max)
            scan_msg.angle_increment = float(self.angle_increment)
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = float(self.scan_time)
            scan_msg.range_min = float(self.range_min)
            scan_msg.range_max = float(self.range_max)
            scan_msg.ranges = ranges

            self.scan_pub.publish(scan_msg)

            self._msg_count += 1
            if self.verbose and self._msg_count % 20 == 0:
                finite_count = sum(math.isfinite(r) for r in ranges)
                self.get_logger().info(
                    f"Published LaserScan | total_points={total_points} "
                    f"| slice_points={slice_points} "
                    f"| occupied_bins={finite_count}/{self.num_beams}"
                )

        except Exception as exc:
            self.get_logger().error(f"Failed to process point cloud: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarPointcloudToScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()