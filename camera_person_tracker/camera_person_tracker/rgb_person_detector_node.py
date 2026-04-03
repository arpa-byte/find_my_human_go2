#!/usr/bin/env python3

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray

from camera_person_tracker.topics import (
    ALIGNED_DEPTH_TOPIC,
    COLOR_CAMERA_INFO_TOPIC,
    COLOR_IMAGE_TOPIC,
    DETECTIONS_TOPIC,
)

ANNOTATED_IMAGE_TOPIC = "/camera_person_tracker/annotated_image"


class RGBPersonDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__("rgb_person_detector")

        self.bridge = CvBridge()
        self._lock = threading.Lock()

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.latest_rgb_stamp: Optional[float] = None
        self.latest_depth_stamp: Optional[float] = None
        self.latest_detections: List[dict] = []

        self.rgb_shape_logged = False
        self.depth_shape_logged = False
        self.camera_info_logged = False

        self.timestamp_log_period_sec = 10.0
        self.inference_period_sec = 0.5

        self.person_class_id = 0
        self.conf_threshold = 0.40
        self.inference_imgsz = 640

        self.first_inference_done = False
        self._inference_running = False
        self._last_logged_count: int = -1

        self.model: Optional[YOLO] = None
        self._model_ready = threading.Event()
        threading.Thread(target=self._load_model_bg, daemon=True).start()

        self.rgb_sub = self.create_subscription(
            Image, COLOR_IMAGE_TOPIC, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, ALIGNED_DEPTH_TOPIC, self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, COLOR_CAMERA_INFO_TOPIC, self.camera_info_callback, 10)

        self._annotated_pub = self.create_publisher(
            Image, ANNOTATED_IMAGE_TOPIC, 10)
        self._detections_pub = self.create_publisher(
            Detection2DArray, DETECTIONS_TOPIC, 10)

        self.display_timer = self.create_timer(0.033, self.display_timer_callback)
        self.inference_timer = self.create_timer(
            self.inference_period_sec, self.inference_timer_callback)
        self.log_timer = self.create_timer(
            self.timestamp_log_period_sec, self.log_timer_callback)

        self.get_logger().info("RGBPersonDetectorNode started.")
        self.get_logger().info(f"Subscribing to RGB: {COLOR_IMAGE_TOPIC}")
        self.get_logger().info(f"Subscribing to depth: {ALIGNED_DEPTH_TOPIC}")
        self.get_logger().info(f"Subscribing to camera info: {COLOR_CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Publishing annotated image to: {ANNOTATED_IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing detections to: {DETECTIONS_TOPIC}")
        self.get_logger().info(
            f"Detector config: person_class_id={self.person_class_id}, "
            f"conf={self.conf_threshold:.2f}, imgsz={self.inference_imgsz}, "
            f"period={self.inference_period_sec:.2f}s"
        )

    def _load_model_bg(self) -> None:
        share_dir = Path(get_package_share_directory("camera_person_tracker"))
        model_path = share_dir / "models" / "yolov8n.pt"
        if not model_path.exists():
            self.get_logger().error(f"YOLO model not found: {model_path}")
            return
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(str(model_path))
        self._model_ready.set()
        self.get_logger().info("YOLO model loaded and ready.")

    @staticmethod
    def stamp_to_sec(msg_stamp) -> float:
        return float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9

    def rgb_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self.latest_rgb = cv_image
                self.latest_rgb_stamp = self.stamp_to_sec(msg.header.stamp)
            if not self.rgb_shape_logged:
                self.get_logger().info(
                    f"RGB image received. shape={cv_image.shape}, "
                    f"dtype={cv_image.dtype}, frame_id={msg.header.frame_id}"
                )
                self.rgb_shape_logged = True
        except CvBridgeError as exc:
            self.get_logger().error(f"RGB convert error: {exc}")

    def depth_callback(self, msg: Image) -> None:
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._lock:
                self.latest_depth_raw = depth_image
                self.latest_depth_stamp = self.stamp_to_sec(msg.header.stamp)
            if not self.depth_shape_logged:
                self.get_logger().info(
                    f"Depth image received. shape={depth_image.shape}, "
                    f"dtype={depth_image.dtype}, encoding={msg.encoding}"
                )
                self.depth_shape_logged = True
        except CvBridgeError as exc:
            self.get_logger().error(f"Depth convert error: {exc}")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_info_logged:
            return
        self.get_logger().info("Camera intrinsics received:")
        self.get_logger().info(
            f"  frame_id={msg.header.frame_id} size={msg.width}x{msg.height}")
        self.get_logger().info(
            f"  fx={msg.k[0]:.4f} fy={msg.k[4]:.4f} "
            f"cx={msg.k[2]:.4f} cy={msg.k[5]:.4f}")
        self.camera_info_logged = True

    def log_timer_callback(self) -> None:
        with self._lock:
            rgb_stamp = self.latest_rgb_stamp
            depth_stamp = self.latest_depth_stamp
        if rgb_stamp is not None and depth_stamp is not None:
            self.get_logger().info(
                f"RGB-Depth timestamp diff: {abs(rgb_stamp - depth_stamp):.6f}s")

    def display_timer_callback(self) -> None:
        with self._lock:
            if self.latest_rgb is None:
                return
            frame = self.latest_rgb.copy()
            detections = list(self.latest_detections)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = "camera_color_optical_frame"
            self._annotated_pub.publish(ros_img)
        except CvBridgeError as exc:
            self.get_logger().error(f"Display publish error: {exc}")

    def inference_timer_callback(self) -> None:
        if not self._model_ready.is_set() or self._inference_running:
            return
        with self._lock:
            if self.latest_rgb is None:
                return
            frame_copy = self.latest_rgb.copy()
        self._inference_running = True
        threading.Thread(
            target=self._run_inference_bg,
            args=(frame_copy,),
            daemon=True,
        ).start()

    def _run_inference_bg(self, image_bgr: np.ndarray) -> None:
        try:
            _, person_detections = self._run_person_detection(image_bgr)

            with self._lock:
                self.latest_detections = person_detections

            # Publish structured detections for target_selector_node
            det_array = Detection2DArray()
            det_array.header.stamp = self.get_clock().now().to_msg()
            det_array.header.frame_id = "camera_color_optical_frame"
            for d in person_detections:
                x1, y1, x2, y2 = d["bbox"]
                det = Detection2D()
                det.bbox.center.position.x = float((x1 + x2) / 2)
                det.bbox.center.position.y = float((y1 + y2) / 2)
                det.bbox.size_x = float(x2 - x1)
                det.bbox.size_y = float(y2 - y1)
                det_array.detections.append(det)
            self._detections_pub.publish(det_array)

            count = len(person_detections)
            if count != self._last_logged_count:
                if count > 0:
                    summary = ", ".join(
                        f"bbox={d['bbox']} conf={d['confidence']:.2f}"
                        for d in person_detections
                    )
                    self.get_logger().info(
                        f"Detected {count} person(s): {summary}")
                else:
                    self.get_logger().info("Detected 0 persons.")
                self._last_logged_count = count

            if not self.first_inference_done:
                self.get_logger().info("First YOLO inference completed.")
                self.first_inference_done = True

        finally:
            self._inference_running = False

    def _run_person_detection(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        annotated = image_bgr.copy()
        person_detections: List[dict] = []

        results = self.model.predict(
            source=image_bgr,
            verbose=False,
            conf=self.conf_threshold,
            imgsz=self.inference_imgsz,
            device="cpu",
        )

        if not results or results[0].boxes is None:
            return annotated, person_detections

        for box, cls_id, conf in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy(),
            results[0].boxes.conf.cpu().numpy(),
        ):
            if int(cls_id) != self.person_class_id:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            person_detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls_id),
            })

        return annotated, person_detections


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RGBPersonDetectorNode()
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