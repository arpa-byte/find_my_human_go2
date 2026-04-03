#!/usr/bin/env python3

import threading
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
from vision_msgs.msg import Detection2DArray

from camera_person_tracker.topics import (
    ALIGNED_DEPTH_TOPIC,
    COLOR_CAMERA_INFO_TOPIC,
    DETECTIONS_TOPIC,
)

TARGET_MARKER_TOPIC = "/camera_person_tracker/target_marker"
DEPTH_SCALE = 0.001
PARENT_FRAME = "camera_color_optical_frame"
TARGET_FRAME = "target_person"


class TargetTFBroadcasterNode(Node):

    def __init__(self) -> None:
        super().__init__("target_tf_broadcaster")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_detections: Optional[Detection2DArray] = None

        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._intrinsics_ready = False

        self._tf_broadcaster = TransformBroadcaster(self)
        self._marker_pub = self.create_publisher(Marker, TARGET_MARKER_TOPIC, 10)

        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            COLOR_CAMERA_INFO_TOPIC,
            self._camera_info_callback,
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

        self.create_timer(0.033, self._broadcast_timer_callback)

        self.get_logger().info("TargetTFBroadcasterNode started.")
        self.get_logger().info(f"Subscribing to camera info: {COLOR_CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Subscribing to depth: {ALIGNED_DEPTH_TOPIC}")
        self.get_logger().info(f"Subscribing to detections: {DETECTIONS_TOPIC}")
        self.get_logger().info(f"Publishing marker to: {TARGET_MARKER_TOPIC}")
        self.get_logger().info(
            f"Broadcasting TF: {PARENT_FRAME} -> {TARGET_FRAME}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self._intrinsics_ready:
            return
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._intrinsics_ready = True
        self.get_logger().info(
            f"Intrinsics received: fx={self._fx:.4f} fy={self._fy:.4f} "
            f"cx={self._cx:.4f} cy={self._cy:.4f}"
        )

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

    def _select_target(self, detections, depth_image):
        """
        Returns (target_detection, depth_metres) for the closest person.
        Returns (None, -1.0) if no valid target found.
        """
        best_det = None
        best_depth = float("inf")

        for det in detections.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            d = self._sample_median_depth_metres(depth_image, cx, cy, w, h)
            if 0.0 < d < best_depth:
                best_depth = d
                best_det = det

        if best_det is None:
            return None, -1.0
        return best_det, best_depth

    def _backproject_to_3d(self, u: float, v: float, z: float):
        """
        Back-projects a pixel (u, v) with depth z into 3D camera coordinates.

        The camera model:
            u = fx * X/Z + cx  =>  X = (u - cx) * Z / fx
            v = fy * Y/Z + cy  =>  Y = (v - cy) * Z / fy
            Z = z (from depth sensor, in metres)

        Returns (X, Y, Z) in camera_color_optical_frame in metres.
        X: positive to the right
        Y: positive downward  (optical frame convention)
        Z: positive forward (depth)
        """
        x = (u - self._cx) * z / self._fx
        y = (v - self._cy) * z / self._fy
        return float(x), float(y), float(z)

    def _broadcast_timer_callback(self) -> None:
        if not self._intrinsics_ready:
            return

        with self._lock:
            depth = self.latest_depth
            detections = self.latest_detections

        if depth is None or detections is None or len(detections.detections) == 0:
            return

        target_det, target_depth = self._select_target(detections, depth)

        if target_det is None:
            return

        u = target_det.bbox.center.position.x
        v = target_det.bbox.center.position.y
        tx, ty, tz = self._backproject_to_3d(u, v, target_depth)

        now = self.get_clock().now().to_msg()

        # --- Broadcast TF transform ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = PARENT_FRAME
        tf_msg.child_frame_id = TARGET_FRAME
        tf_msg.transform.translation.x = tx
        tf_msg.transform.translation.y = ty
        tf_msg.transform.translation.z = tz
        # No rotation — target frame axes are aligned with the camera frame
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

        # --- Publish RViz Marker (red sphere at target position) ---
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = PARENT_FRAME
        marker.ns = "target_person"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = tx
        marker.pose.position.y = ty
        marker.pose.position.z = tz
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.85
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200_000_000  # 200ms — vanishes if target lost
        self._marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetTFBroadcasterNode()
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