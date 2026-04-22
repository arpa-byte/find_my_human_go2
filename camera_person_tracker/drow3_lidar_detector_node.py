#!/usr/bin/env python3

import math
import os
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class Drow3LidarDetectorNode(Node):
    def __init__(self):
        super().__init__("drow3_lidar_detector")

        self.declare_parameter("input_scan_topic", "/scan_knee")
        self.declare_parameter("marker_topic", "/camera_person_tracker/lidar_detections_marker")
        self.declare_parameter("poses_topic", "/camera_person_tracker/lidar_detection_poses")

        self.declare_parameter(
            "checkpoint_path",
            "/home/arpan/masterthesis/models/drow3/checkpoints/drow_pretrained/ckpt_jrdb_ann_drow3_e40.pth",
        )
        self.declare_parameter(
            "third_party_path",
            "/home/arpan/masterthesis/third_party/2D_lidar_person_detection/dr_spaam",
        )

        self.declare_parameter("model_type", "DROW3")
        self.declare_parameter("confidence_threshold", 0.60)
        self.declare_parameter("max_detections", 5)
        self.declare_parameter("laser_fov_deg", 360.0)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("gpu", False)
        self.declare_parameter("stride", 1)

        self.declare_parameter("inference_period_sec", 0.10)
        self.declare_parameter("replace_invalid_with_range_max", True)

        self.declare_parameter("marker_z", 0.15)
        self.declare_parameter("center_marker_scale", 0.12)
        self.declare_parameter("ring_radius", 0.35)
        self.declare_parameter("ring_line_width", 0.04)
        self.declare_parameter("ring_points", 48)
        self.declare_parameter("text_z", 0.45)
        self.declare_parameter("text_scale", 0.22)
        self.declare_parameter("marker_lifetime_sec", 0.35)

        self.declare_parameter("publish_center_dot", True)
        self.declare_parameter("publish_confidence_text", True)
        self.declare_parameter("log_detections", True)
        self.declare_parameter("log_period_sec", 0.50)

        self.input_scan_topic = self.get_parameter("input_scan_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.poses_topic = self.get_parameter("poses_topic").value

        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.third_party_path = self.get_parameter("third_party_path").value

        self.model_type = self.get_parameter("model_type").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_detections = int(self.get_parameter("max_detections").value)
        self.laser_fov_deg = float(self.get_parameter("laser_fov_deg").value)        
        self.panoramic_scan = bool(self.get_parameter("panoramic_scan").value)
        self.gpu = bool(self.get_parameter("gpu").value)
        self.stride = int(self.get_parameter("stride").value)

        self.inference_period_sec = float(self.get_parameter("inference_period_sec").value)
        self.replace_invalid_with_range_max = bool(
            self.get_parameter("replace_invalid_with_range_max").value
        )

        self.marker_z = float(self.get_parameter("marker_z").value)
        self.center_marker_scale = float(self.get_parameter("center_marker_scale").value)
        self.ring_radius = float(self.get_parameter("ring_radius").value)
        self.ring_line_width = float(self.get_parameter("ring_line_width").value)
        self.ring_points = int(self.get_parameter("ring_points").value)
        self.text_z = float(self.get_parameter("text_z").value)
        self.text_scale = float(self.get_parameter("text_scale").value)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)

        self.publish_center_dot = bool(self.get_parameter("publish_center_dot").value)
        self.publish_confidence_text = bool(self.get_parameter("publish_confidence_text").value)
        self.log_detections = bool(self.get_parameter("log_detections").value)
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        self.detector = self.load_detector()

        self.last_inference_time: Optional[rclpy.time.Time] = None
        self.last_log_time = self.get_clock().now()

        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            scan_qos,
        )

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.poses_pub = self.create_publisher(PoseArray, self.poses_topic, 10)

        self.get_logger().info("Drow3LidarDetectorNode started.")
        self.get_logger().info(f"Input scan topic : {self.input_scan_topic}")
        self.get_logger().info(f"Marker topic     : {self.marker_topic}")
        self.get_logger().info(f"Poses topic      : {self.poses_topic}")
        self.get_logger().info(f"Checkpoint       : {self.checkpoint_path}")
        self.get_logger().info(f"Model            : {self.model_type}")
        self.get_logger().info(f"Max detections    : {self.max_detections}")
        self.get_logger().info(f"Laser FOV deg    : {self.laser_fov_deg}")
        self.get_logger().info(f"Panoramic scan   : {self.panoramic_scan}")

    def load_detector(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if self.third_party_path and os.path.exists(self.third_party_path):
            sys.path.insert(0, self.third_party_path)

        from dr_spaam.detector import Detector

        detector = Detector(
            self.checkpoint_path,
            model=self.model_type,
            gpu=self.gpu,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        detector.set_laser_fov(self.laser_fov_deg)
        return detector

    def should_run_inference(self) -> bool:
        now = self.get_clock().now()

        if self.last_inference_time is None:
            self.last_inference_time = now
            return True

        dt = (now - self.last_inference_time).nanoseconds / 1e9
        if dt >= self.inference_period_sec:
            self.last_inference_time = now
            return True

        return False

    def should_log(self) -> bool:
        now = self.get_clock().now()
        dt = (now - self.last_log_time).nanoseconds / 1e9

        if dt >= self.log_period_sec:
            self.last_log_time = now
            return True

        return False

    def scan_to_numpy(self, msg: LaserScan) -> np.ndarray:
        scan = np.asarray(msg.ranges, dtype=np.float32)

        if self.replace_invalid_with_range_max:
            scan = np.nan_to_num(
                scan,
                nan=msg.range_max,
                posinf=msg.range_max,
                neginf=msg.range_max,
            )

        scan = np.clip(scan, msg.range_min, msg.range_max)
        return scan

    def scan_age_ms(self, msg: LaserScan) -> float:
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        now_msg = self.get_clock().now().to_msg()
        now_sec = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9

        if stamp_sec <= 0.0:
            return -1.0

        return max(0.0, (now_sec - stamp_sec) * 1000.0)

    def scan_callback(self, msg: LaserScan):
        if not self.should_run_inference():
            return

        try:
            scan = self.scan_to_numpy(msg)

            t0 = time.perf_counter()
            dets_xy, dets_cls, _ = self.detector(scan)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            dets_xy = np.asarray(dets_xy)
            dets_cls = np.asarray(dets_cls).reshape(-1)

            if dets_xy.size == 0 or dets_cls.size == 0:
                self.publish_empty(msg)
                if self.log_detections and self.should_log():
                    self.get_logger().info(
                        f"DROW3 detections: 0 | inference={inference_ms:.1f} ms "
                        f"| scan_age={self.scan_age_ms(msg):.1f} ms"
                    )
                return

            keep = dets_cls >= self.confidence_threshold
            dets_xy = dets_xy[keep]
            dets_cls = dets_cls[keep]

            if len(dets_cls) == 0:
                self.publish_empty(msg)
                if self.log_detections and self.should_log():
                    self.get_logger().info(
                        f"DROW3 detections: 0 | "
                        f"threshold={self.confidence_threshold:.2f} | "
                        f"inference={inference_ms:.1f} ms | "
                        f"scan_age={self.scan_age_ms(msg):.1f} ms"
                    )
                return

            order = np.argsort(dets_cls)[::-1]

            if self.max_detections > 0:
                order = order[:self.max_detections]

            dets_xy = dets_xy[order]
            dets_cls = dets_cls[order]

            self.publish_outputs(msg, dets_xy, dets_cls)

            if self.log_detections and self.should_log():
                if len(dets_cls) > 0:
                    best_score = float(np.max(dets_cls))
                else:
                    best_score = 0.0

                self.get_logger().info(
                    f"DROW3 detections: {len(dets_xy)} | "
                    f"best_score={best_score:.3f} | "
                    f"inference={inference_ms:.1f} ms | "
                    f"scan_age={self.scan_age_ms(msg):.1f} ms"
                )

        except Exception as exc:
            self.get_logger().error(f"DROW3 inference failed: {exc}")

    def publish_empty(self, scan_msg: LaserScan):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header = scan_msg.header
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        self.marker_pub.publish(marker_array)

        pose_array = PoseArray()
        pose_array.header = scan_msg.header
        self.poses_pub.publish(pose_array)

    def make_lifetime(self, marker: Marker):
        marker.lifetime.sec = int(self.marker_lifetime_sec)
        marker.lifetime.nanosec = int(
            (self.marker_lifetime_sec - int(self.marker_lifetime_sec)) * 1e9
        )

    def add_detection_ring(
        self,
        marker_array: MarkerArray,
        header,
        marker_id: int,
        x: float,
        y: float,
        score: float,
    ):
        marker = Marker()
        marker.header = header
        marker.ns = "drow3_detection_rings"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = self.ring_line_width

        marker.color.r = 0.0
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = min(1.0, max(0.25, float(score)))

        for i in range(self.ring_points + 1):
            theta = 2.0 * math.pi * float(i) / float(self.ring_points)
            p = Pose().position
            p.x = x + self.ring_radius * math.cos(theta)
            p.y = y + self.ring_radius * math.sin(theta)
            p.z = self.marker_z
            marker.points.append(p)

        self.make_lifetime(marker)
        marker_array.markers.append(marker)

    def add_center_dot(
        self,
        marker_array: MarkerArray,
        header,
        marker_id: int,
        x: float,
        y: float,
        score: float,
    ):
        marker = Marker()
        marker.header = header
        marker.ns = "drow3_detection_centers"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = self.marker_z
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.center_marker_scale
        marker.scale.y = self.center_marker_scale
        marker.scale.z = self.center_marker_scale

        marker.color.r = 0.0
        marker.color.g = 0.2
        marker.color.b = 1.0
        marker.color.a = min(1.0, max(0.35, float(score)))

        self.make_lifetime(marker)
        marker_array.markers.append(marker)

    def add_confidence_text(
        self,
        marker_array: MarkerArray,
        header,
        marker_id: int,
        x: float,
        y: float,
        score: float,
    ):
        marker = Marker()
        marker.header = header
        marker.ns = "drow3_detection_scores"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = self.text_z
        marker.pose.orientation.w = 1.0

        marker.scale.z = self.text_scale
        marker.text = f"{score:.2f}"

        marker.color.r = 0.8
        marker.color.g = 0.9
        marker.color.b = 1.0
        marker.color.a = 0.95

        self.make_lifetime(marker)
        marker_array.markers.append(marker)

    def publish_outputs(self, scan_msg: LaserScan, dets_xy: np.ndarray, dets_cls: np.ndarray):
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header = scan_msg.header
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        pose_array = PoseArray()
        pose_array.header = scan_msg.header

        for idx, (xy, score) in enumerate(zip(dets_xy, dets_cls)):
            x = float(xy[0])
            y = float(xy[1])
            score = float(score)

            base_id = idx * 10

            self.add_detection_ring(marker_array, scan_msg.header, base_id + 0, x, y, score)

            if self.publish_center_dot:
                self.add_center_dot(marker_array, scan_msg.header, base_id + 1, x, y, score)

            if self.publish_confidence_text:
                self.add_confidence_text(marker_array, scan_msg.header, base_id + 2, x, y, score)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.marker_pub.publish(marker_array)
        self.poses_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = Drow3LidarDetectorNode()

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