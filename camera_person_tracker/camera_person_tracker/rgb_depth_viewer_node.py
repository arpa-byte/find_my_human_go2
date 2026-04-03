#!/usr/bin/env python3

from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from camera_person_tracker.topics import (
    ALIGNED_DEPTH_TOPIC,
    COLOR_CAMERA_INFO_TOPIC,
    COLOR_IMAGE_TOPIC,
)


class RGBDepthViewerNode(Node):
    """
    Step 2 node for Phase 3A.

    This node validates the camera data flow:
    - subscribes to RGB image
    - subscribes to aligned depth image
    - subscribes to color camera info
    - converts ROS images to OpenCV
    - displays RGB and normalized depth
    - prints image sizes once
    - prints camera intrinsics once
    - prints timestamp differences at a throttled rate
    """

    def __init__(self) -> None:
        super().__init__("rgb_depth_viewer")

        self.bridge = CvBridge()

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None

        self.latest_rgb_stamp: Optional[float] = None
        self.latest_depth_stamp: Optional[float] = None

        self.rgb_shape_logged = False
        self.depth_shape_logged = False
        self.camera_info_logged = False

        self.last_timestamp_log_time_sec = 0.0
        self.timestamp_log_period_sec = 1.0

        self.rgb_sub = self.create_subscription(
            Image,
            COLOR_IMAGE_TOPIC,
            self.rgb_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            ALIGNED_DEPTH_TOPIC,
            self.depth_callback,
            10,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            COLOR_CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10,
        )

        # Timer for periodic display and throttled timestamp logging
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("RGBDepthViewerNode started.")
        self.get_logger().info(f"Subscribing to RGB topic: {COLOR_IMAGE_TOPIC}")
        self.get_logger().info(f"Subscribing to depth topic: {ALIGNED_DEPTH_TOPIC}")
        self.get_logger().info(f"Subscribing to camera info topic: {COLOR_CAMERA_INFO_TOPIC}")

    @staticmethod
    def stamp_to_sec(msg_stamp) -> float:
        return float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9

    def rgb_callback(self, msg: Image) -> None:
        try:
            # Convert to OpenCV BGR image for display with OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_rgb = cv_image
            self.latest_rgb_stamp = self.stamp_to_sec(msg.header.stamp)

            if not self.rgb_shape_logged:
                self.get_logger().info(
                    f"RGB image received. shape={cv_image.shape}, "
                    f"dtype={cv_image.dtype}, frame_id={msg.header.frame_id}, "
                    f"encoding={msg.encoding}"
                )
                self.rgb_shape_logged = True

        except CvBridgeError as exc:
            self.get_logger().error(f"Failed to convert RGB image: {exc}")

    def depth_callback(self, msg: Image) -> None:
        try:
            # Preserve raw depth values
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_raw = depth_image
            self.latest_depth_stamp = self.stamp_to_sec(msg.header.stamp)

            if not self.depth_shape_logged:
                self.get_logger().info(
                    f"Depth image received. shape={depth_image.shape}, "
                    f"dtype={depth_image.dtype}, frame_id={msg.header.frame_id}, "
                    f"encoding={msg.encoding}"
                )
                self.depth_shape_logged = True

        except CvBridgeError as exc:
            self.get_logger().error(f"Failed to convert depth image: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_info_logged:
            return

        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        self.get_logger().info("Camera intrinsics received from color camera_info:")
        self.get_logger().info(f"  frame_id: {msg.header.frame_id}")
        self.get_logger().info(f"  width:    {msg.width}")
        self.get_logger().info(f"  height:   {msg.height}")
        self.get_logger().info(f"  fx:       {fx:.6f}")
        self.get_logger().info(f"  fy:       {fy:.6f}")
        self.get_logger().info(f"  cx:       {cx:.6f}")
        self.get_logger().info(f"  cy:       {cy:.6f}")
        self.get_logger().info(f"  distortion_model: {msg.distortion_model}")
        self.get_logger().info(f"  d: {list(msg.d)}")

        self.camera_info_logged = True

    def normalize_depth_for_display(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Convert the raw depth image into a viewable colorized image.
        """
        if depth_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        depth = depth_image.copy()

        if np.issubdtype(depth.dtype, np.floating):
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        valid_mask = depth > 0

        if not np.any(valid_mask):
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        valid_values = depth[valid_mask]
        min_val = float(np.min(valid_values))
        max_val = float(np.max(valid_values))

        if max_val <= min_val:
            normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            depth_float = depth.astype(np.float32)
            depth_clipped = np.clip(depth_float, min_val, max_val)
            normalized = cv2.normalize(
                depth_clipped,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

        colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        colorized[~valid_mask] = (0, 0, 0)
        return colorized

    def timer_callback(self) -> None:
        if self.latest_rgb is not None:
            cv2.imshow("Phase3A RGB Viewer", self.latest_rgb)

        if self.latest_depth_raw is not None:
            depth_display = self.normalize_depth_for_display(self.latest_depth_raw)
            cv2.imshow("Phase3A Depth Viewer", depth_display)

        if self.latest_rgb_stamp is not None and self.latest_depth_stamp is not None:
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if (now_sec - self.last_timestamp_log_time_sec) >= self.timestamp_log_period_sec:
                delta = abs(self.latest_rgb_stamp - self.latest_depth_stamp)
                self.get_logger().info(
                    f"RGB-Depth timestamp difference: {delta:.6f} seconds"
                )
                self.last_timestamp_log_time_sec = now_sec

        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            self.get_logger().info("ESC pressed. Closing viewer node.")
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RGBDepthViewerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Shutting down.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()