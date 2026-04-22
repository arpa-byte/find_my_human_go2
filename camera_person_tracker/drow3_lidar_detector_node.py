#!/usr/bin/env python3

import math
import os
import sys
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
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
            "/home/arpan/masterthesis/models/drow3/checkpoints/drow_pretrained/cckpt_jrdb_ann_drow3_e40.pth",
        )
        self.declare_parameter(
            "third_party_path",
            "/home/arpan/masterthesis/third_party/2D_lidar_person_detection/dr_spaam",
        )

        self.declare_parameter("model_type", "DROW3")
        self.declare_parameter("confidence_threshold", 0.20)
        self.declare_parameter("laser_fov_deg", 360.0)
        self.declare_parameter("panoramic_scan", True)
        self.declare_parameter("gpu", False)
        self.declare_parameter("stride", 1)

        self.declare_parameter("inference_period_sec", 0.10)

        self.declare_parameter("marker_z", 0.20)
        self.declare_parameter("marker_scale", 0.25)
        self.declare_parameter("marker_lifetime_sec", 0.30)

        self.declare_parameter("replace_invalid_with_range_max", True)

        self.input_scan_topic = self.get_parameter("input_scan_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.poses_topic = self.get_parameter("poses_topic").value

        self.checkpoint_path = self.get_parameter("checkpoint_path").value
        self.third_party_path = self.get_parameter("third_party_path").value

        self.model_type = self.get_parameter("model_type").value
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.laser_fov_deg = float(self.get_parameter("laser_fov_deg").value)
        self.panoramic_scan = bool(self.get_parameter("panoramic_scan").value)
        self.gpu = bool(self.get_parameter("gpu").value)
        self.stride = int(self.get_parameter("stride").value)

        self.inference_period_sec = float(self.get_parameter("inference_period_sec").value)

        self.marker_z = float(self.get_parameter("marker_z").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)

        self.replace_invalid_with_range_max = bool(
            self.get_parameter("replace_invalid_with_range_max").value
        )

        self.detector = self.load_detector()
        self.last_inference_time: Optional[rclpy.time.Time] = None

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_scan_topic,
            self.scan_callback,
            10,
        )

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.poses_pub = self.create_publisher(PoseArray, self.poses_topic, 10)

        self.get_logger().info("Drow3LidarDetectorNode started.")
        self.get_logger().info(f"Input scan topic : {self.input_scan_topic}")
        self.get_logger().info(f"Marker topic     : {self.marker_topic}")
        self.get_logger().info(f"Poses topic      : {self.poses_topic}")
        self.get_logger().info(f"Checkpoint       : {self.checkpoint_path}")
        self.get_logger().info(f"Model            : {self.model_type}")
        self.get_logger().info(f"Confidence thresh: {self.confidence_threshold}")
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

    def scan_callback(self, msg: LaserScan):
        if not self.should_run_inference():
            return

        try:
            scan = self.scan_to_numpy(msg)

            dets_xy, dets_cls, _ = self.detector(scan)

            dets_xy = np.asarray(dets_xy)
            dets_cls = np.asarray(dets_cls).reshape(-1)

            if dets_xy.size == 0 or dets_cls.size == 0:
                self.publish_empty(msg)
                return

            keep = dets_cls > self.confidence_threshold
            dets_xy = dets_xy[keep]
            dets_cls = dets_cls[keep]

            self.publish_outputs(msg, dets_xy, dets_cls)

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

            marker = Marker()
            marker.header = scan_msg.header
            marker.ns = "drow3_lidar_detections"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = self.marker_z
            marker.pose.orientation.w = 1.0

            marker.scale.x = self.marker_scale
            marker.scale.y = self.marker_scale
            marker.scale.z = self.marker_scale

            marker.color.r = 0.0
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = min(1.0, max(0.2, float(score)))

            marker.lifetime.sec = int(self.marker_lifetime_sec)
            marker.lifetime.nanosec = int(
                (self.marker_lifetime_sec - int(self.marker_lifetime_sec)) * 1e9
            )

            marker_array.markers.append(marker)

            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.marker_pub.publish(marker_array)
        self.poses_pub.publish(pose_array)

        self.get_logger().info(
            f"DROW3 detections: {len(dets_xy)} | "
            f"best_score={float(np.max(dets_cls)):.3f}"
        )


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