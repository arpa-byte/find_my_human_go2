#!/usr/bin/env python3

import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from camera_person_tracker.topics import (
    ALIGNED_DEPTH_TOPIC,
    DETECTIONS_TOPIC,
)

TARGET_IMAGE_TOPIC = "/camera_person_tracker/target_image"
DEPTH_SCALE = 0.001


class TargetSelectorNode(Node):

    def __init__(self) -> None:
        super().__init__("target_selector")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_detections: Optional[Detection2DArray] = None
        self.latest_rgb: Optional[np.ndarray] = None

        self._last_logged_count: int = -1

        self._target_pub = self.create_publisher(Image, TARGET_IMAGE_TOPIC, 10)

        self._rgb_sub = self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._rgb_callback,
            10,
        )
        self._depth_sub = self.create_subscription(
            Image,
            ALIGNED_DEPTH_TOPIC,
            self._depth_callback,
            10,
        )
        self._detections_sub = self.create_subscription(
            Detection2DArray,
            DETECTIONS_TOPIC,
            self._detections_callback,
            10,
        )

        self.create_timer(0.033, self._display_timer_callback)

        self.get_logger().info("TargetSelectorNode started.")
        self.get_logger().info(f"Subscribing to detections: {DETECTIONS_TOPIC}")
        self.get_logger().info(f"Subscribing to depth: {ALIGNED_DEPTH_TOPIC}")
        self.get_logger().info(f"Publishing target image to: {TARGET_IMAGE_TOPIC}")

    def _rgb_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self.latest_rgb = frame
        except CvBridgeError as exc:
            self.get_logger().error(f"RGB convert error: {exc}")

    def _depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self.latest_depth = depth
        except CvBridgeError as exc:
            self.get_logger().error(f"Depth convert error: {exc}")

    def _detections_callback(self, msg: Detection2DArray) -> None:
        with self._lock:
            self.latest_detections = msg

    def _sample_median_depth_metres(
        self,
        depth_image: np.ndarray,
        cx: float,
        cy: float,
        w: float,
        h: float,
    ) -> float:
        """
        Samples the central 50% of the bounding box in the aligned depth image.
        Each pixel value is in millimetres (RealSense D435i uint16).
        Returns median depth in metres, or -1.0 if no valid pixels.

        We use the central 50% (25% inset on each side) because depth at
        the silhouette edges of a person is unreliable — the sensor often
        returns the background depth there due to mixed pixels.
        """
        x1 = int(cx - w * 0.25)
        y1 = int(cy - h * 0.25)
        x2 = int(cx + w * 0.25)
        y2 = int(cy + h * 0.25)

        img_h, img_w = depth_image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return -1.0

        patch = depth_image[y1:y2, x1:x2].astype(np.float32)
        valid = patch[patch > 0]

        if valid.size == 0:
            return -1.0

        return float(np.median(valid)) * DEPTH_SCALE

    def _display_timer_callback(self) -> None:
        with self._lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            frame = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
            detections = self.latest_detections

        if detections is None or len(detections.detections) == 0:
            self._publish_frame(frame)
            return

        # Sample depth fresh every display tick (33ms) so the distance
        # readout updates at 30 Hz even though YOLO only runs at 2 Hz
        depths = []
        for det in detections.detections:
            d = self._sample_median_depth_metres(
                depth,
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.size_x,
                det.bbox.size_y,
            )
            depths.append(d)

        # Select target: person with smallest valid (> 0) depth
        target_idx = -1
        best_depth = float("inf")
        for i, d in enumerate(depths):
            if 0.0 < d < best_depth:
                best_depth = d
                target_idx = i

        for i, det in enumerate(detections.detections):
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            is_target = (i == target_idx)
            color = (0, 0, 255) if is_target else (180, 180, 180)
            thickness = 3 if is_target else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if is_target:
                # Large distance readout inside the box at the top
                dist_text = f"{best_depth:.2f} m"
                cv2.putText(
                    frame, dist_text,
                    (x1 + 6, y1 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                )
                # Smaller TARGET label above the box
                cv2.putText(
                    frame, "TARGET",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                )
            else:
                label = f"{depths[i]:.2f}m" if depths[i] > 0 else "no depth"
                cv2.putText(
                    frame, label,
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                )

        # Log only on change in number of detected persons
        count = len(detections.detections)
        if count != self._last_logged_count:
            if target_idx >= 0:
                self.get_logger().info(
                    f"Target selected: person {target_idx} "
                    f"at {best_depth:.2f}m  ({count} person(s) in frame)"
                )
            else:
                self.get_logger().info("No valid target — depth unavailable.")
            self._last_logged_count = count

        self._publish_frame(frame)

    def _publish_frame(self, frame: np.ndarray) -> None:
        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_color_optical_frame"
            self._target_pub.publish(ros_img)
        except CvBridgeError as exc:
            self.get_logger().error(f"Publish error: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetSelectorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Shutting down.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()