#!/usr/bin/env python3

import math
from dataclasses import dataclass
from threading import Lock
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from ultralytics import YOLO
from visualization_msgs.msg import Marker


COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

OUTPUT_TOPIC = "/camera_person_tracker/simple_output"
MARKER_TOPIC = "/camera_person_tracker/target_marker"

CAMERA_FRAME = "camera_color_optical_frame"
TARGET_FRAME = "target_person"

DEPTH_SCALE = 0.001


@dataclass
class CandidateDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    depth_m: float
    xyz: Tuple[float, float, float]


class CameraOwnerPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_owner_pipeline")

        self.declare_parameter("color_topic", COLOR_TOPIC)
        self.declare_parameter("depth_topic", DEPTH_TOPIC)
        self.declare_parameter("camera_info_topic", CAMERA_INFO_TOPIC)

        self.declare_parameter("output_image_topic", OUTPUT_TOPIC)
        self.declare_parameter("target_marker_topic", MARKER_TOPIC)

        self.declare_parameter("camera_frame", CAMERA_FRAME)
        self.declare_parameter("target_frame", TARGET_FRAME)
        self.declare_parameter("world_frame", "base_link")
        self.declare_parameter("locked_lidar_target_frame", "locked_lidar_target")

        self.declare_parameter("model_path", "yolov8n.pt")

        self.declare_parameter("process_period_sec", 0.12)
        self.declare_parameter("viz_publish_period_sec", 0.20)
        self.declare_parameter("publish_debug_image", True)

        self.declare_parameter("target_hold_sec", 1.0)
        self.declare_parameter("max_match_distance_m", 1.0)
        self.declare_parameter("max_depth_jump_m", 1.0)
        self.declare_parameter("lidar_identity_gate_m", 0.50)
        self.declare_parameter("min_valid_depth_m", 0.2)
        self.declare_parameter("max_valid_depth_m", 8.0)

        self.declare_parameter("marker_scale_m", 0.25)
        self.declare_parameter("marker_lifetime_sec", 0.30)

        self.declare_parameter("log_period_sec", 0.75)
        self.declare_parameter("max_rgb_depth_skew_ms", 120.0)

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.output_image_topic = self.get_parameter("output_image_topic").value
        self.target_marker_topic = self.get_parameter("target_marker_topic").value

        self.camera_frame = self.get_parameter("camera_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.locked_lidar_target_frame = self.get_parameter(
            "locked_lidar_target_frame"
        ).value

        self.model_path = self.get_parameter("model_path").value

        self.process_period_sec = float(self.get_parameter("process_period_sec").value)
        self.viz_publish_period_sec = float(
            self.get_parameter("viz_publish_period_sec").value
        )
        self.publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )

        self.target_hold_sec = float(self.get_parameter("target_hold_sec").value)
        self.max_match_distance_m = float(
            self.get_parameter("max_match_distance_m").value
        )
        self.max_depth_jump_m = float(self.get_parameter("max_depth_jump_m").value)
        self.lidar_identity_gate_m = float(
            self.get_parameter("lidar_identity_gate_m").value
        )
        self.min_valid_depth_m = float(self.get_parameter("min_valid_depth_m").value)
        self.max_valid_depth_m = float(self.get_parameter("max_valid_depth_m").value)

        self.marker_scale_m = float(self.get_parameter("marker_scale_m").value)
        self.marker_lifetime_sec = float(
            self.get_parameter("marker_lifetime_sec").value
        )

        self.log_period_sec = float(self.get_parameter("log_period_sec").value)
        self.max_rgb_depth_skew_ms = float(
            self.get_parameter("max_rgb_depth_skew_ms").value
        )

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.frame_lock = Lock()
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.latest_rgb_stamp = None
        self.latest_depth_stamp = None
        self.latest_rgb_seq = 0
        self.latest_depth_seq = 0

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx0: Optional[float] = None
        self.cy0: Optional[float] = None
        self.camera_info_ready = False

        self.processing = False
        self.last_processed_rgb_seq = -1
        self.last_processed_depth_seq = -1

        self.target_id = 1
        self.target_active = False
        self.target_visible = False
        self.last_seen_sec: Optional[float] = None
        self.last_target_xyz: Optional[Tuple[float, float, float]] = None
        self.last_target_px: Optional[Tuple[int, int]] = None
        self.last_target_box: Optional[Tuple[int, int, int, int]] = None
        self.last_target_depth_m: Optional[float] = None

        self.last_log_time = self.get_clock().now()

        self.sync_count = 0
        self.sync_sum_ms = 0.0
        self.sync_max_ms = 0.0

        self.last_debug_frame: Optional[np.ndarray] = None

        self.create_subscription(
            Image, self.color_topic, self.rgb_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.marker_pub = self.create_publisher(Marker, self.target_marker_topic, 10)

        self.create_timer(self.process_period_sec, self.process_latest_frame)
        if self.publish_debug_image:
            self.create_timer(self.viz_publish_period_sec, self.publish_debug_frame)

        self.get_logger().info("CameraOwnerPipelineNode started.")
        self.get_logger().info(f"Color topic           : {self.color_topic}")
        self.get_logger().info(f"Depth topic           : {self.depth_topic}")
        self.get_logger().info(f"Camera info topic     : {self.camera_info_topic}")
        self.get_logger().info(f"Output image topic    : {self.output_image_topic}")
        self.get_logger().info(f"Target marker topic   : {self.target_marker_topic}")
        self.get_logger().info(f"Camera frame          : {self.camera_frame}")
        self.get_logger().info(f"Target frame          : {self.target_frame}")
        self.get_logger().info(f"YOLO model            : {self.model_path}")
        self.get_logger().info(f"Process period        : {self.process_period_sec:.2f} s")
        self.get_logger().info(f"Viz publish period    : {self.viz_publish_period_sec:.2f} s")
        self.get_logger().info(f"Publish debug image   : {self.publish_debug_image}")
        self.get_logger().info(f"Target hold           : {self.target_hold_sec:.2f} s")
        self.get_logger().info(f"Max match distance    : {self.max_match_distance_m:.2f} m")
        self.get_logger().info(f"Max depth jump        : {self.max_depth_jump_m:.2f} m")
        self.get_logger().info(f"Max RGB-depth skew    : {self.max_rgb_depth_skew_ms:.1f} ms")

    def rgb_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        with self.frame_lock:
            self.latest_rgb = frame
            self.latest_rgb_stamp = msg.header.stamp
            self.latest_rgb_seq += 1

    def depth_callback(self, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        with self.frame_lock:
            self.latest_depth = depth
            self.latest_depth_stamp = msg.header.stamp
            self.latest_depth_seq += 1

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_info_ready:
            return

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx0 = float(msg.k[2])
        self.cy0 = float(msg.k[5])
        self.camera_info_ready = True

        self.get_logger().info(
            f"Camera intrinsics ready: fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx0:.2f}, cy={self.cy0:.2f}"
        )

    def stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def now_sec(self) -> float:
        return self.stamp_to_sec(self.get_clock().now().to_msg())

    def current_sync_ms(self, rgb_stamp, depth_stamp) -> Optional[float]:
        if rgb_stamp is None or depth_stamp is None:
            return None
        rgb_t = self.stamp_to_sec(rgb_stamp)
        depth_t = self.stamp_to_sec(depth_stamp)
        return abs(rgb_t - depth_t) * 1000.0

    def update_sync_stats(self, sync_ms: float) -> None:
        self.sync_count += 1
        self.sync_sum_ms += sync_ms
        if sync_ms > self.sync_max_ms:
            self.sync_max_ms = sync_ms

    def avg_sync_ms(self) -> float:
        if self.sync_count == 0:
            return 0.0
        return self.sync_sum_ms / self.sync_count

    def target_age_sec(self) -> float:
        if self.last_seen_sec is None:
            return float("inf")
        return max(0.0, self.now_sec() - self.last_seen_sec)

    def target_is_remembered(self) -> bool:
        if not self.target_active or self.last_seen_sec is None:
            return False
        return self.target_age_sec() <= self.target_hold_sec

    def should_log(self) -> bool:
        now = self.get_clock().now()
        elapsed = (now - self.last_log_time).nanoseconds / 1e9
        if elapsed >= self.log_period_sec:
            self.last_log_time = now
            return True
        return False

    def get_depth(self, depth_img: np.ndarray, px: int, py: int) -> float:
        if depth_img is None:
            return -1.0

        h, w = depth_img.shape
        x = int(np.clip(px, 0, w - 1))
        y = int(np.clip(py, 0, h - 1))

        d = depth_img[y, x]
        if d == 0:
            return -1.0

        depth_m = float(d) * DEPTH_SCALE
        if depth_m < self.min_valid_depth_m or depth_m > self.max_valid_depth_m:
            return -1.0

        return depth_m

    def pixel_to_3d(self, u: int, v: int, z: float) -> Optional[Tuple[float, float, float]]:
        if not self.camera_info_ready or z <= 0.0:
            return None

        x = (u - self.cx0) * z / self.fx
        y = (v - self.cy0) * z / self.fy
        return float(x), float(y), float(z)

    def distance_xyz(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        return math.sqrt(
            (a[0] - b[0]) ** 2 +
            (a[1] - b[1]) ** 2 +
            (a[2] - b[2]) ** 2
        )

    def rotate_point_by_quaternion(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> Tuple[float, float, float]:
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)

        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)

        return rx, ry, rz

    def transform_point(
        self,
        source_frame: str,
        target_frame: str,
        xyz: Tuple[float, float, float],
    ) -> Optional[Tuple[float, float, float]]:
        if source_frame == target_frame:
            return (float(xyz[0]), float(xyz[1]), float(xyz[2]))

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
        except TransformException:
            return None

        qx = float(tf_msg.transform.rotation.x)
        qy = float(tf_msg.transform.rotation.y)
        qz = float(tf_msg.transform.rotation.z)
        qw = float(tf_msg.transform.rotation.w)

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        tz = float(tf_msg.transform.translation.z)

        rx, ry, rz = self.rotate_point_by_quaternion(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            qx,
            qy,
            qz,
            qw,
        )

        return (rx + tx, ry + ty, rz + tz)

    def get_locked_lidar_target_world_xyz(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf_locked = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.locked_lidar_target_frame,
                Time(),
            )
            return (
                float(tf_locked.transform.translation.x),
                float(tf_locked.transform.translation.y),
                float(tf_locked.transform.translation.z),
            )
        except TransformException:
            return None

    def get_lidar_consistent_candidates_only(
        self,
        candidates: List[CandidateDetection],
    ) -> List[CandidateDetection]:
        locked_world_xyz = self.get_locked_lidar_target_world_xyz()
        if locked_world_xyz is None:
            return []

        filtered: List[CandidateDetection] = []

        for cand in candidates:
            cand_world_xyz = self.transform_point(
                self.camera_frame,
                self.world_frame,
                cand.xyz,
            )
            if cand_world_xyz is None:
                continue

            dist = self.distance_xyz(cand_world_xyz, locked_world_xyz)
            if dist <= self.lidar_identity_gate_m:
                filtered.append(cand)

        return filtered

    def collect_candidates(self, frame: np.ndarray, depth_img: np.ndarray) -> List[CandidateDetection]:
        results = self.model(frame, verbose=False)[0]
        candidates: List[CandidateDetection] = []

        if results.boxes is None:
            return candidates

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            depth_m = self.get_depth(depth_img, cx, cy)
            if depth_m <= 0.0:
                continue

            xyz = self.pixel_to_3d(cx, cy, depth_m)
            if xyz is None:
                continue

            candidates.append(
                CandidateDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    cx=cx,
                    cy=cy,
                    depth_m=depth_m,
                    xyz=xyz,
                )
            )

        return candidates

    def select_initial_target(
        self,
        candidates: List[CandidateDetection],
    ) -> Optional[CandidateDetection]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.depth_m)

    def match_existing_target(
        self,
        candidates: List[CandidateDetection],
    ) -> Optional[CandidateDetection]:
        if not candidates:
            return None

        if self.last_target_xyz is None or self.last_target_depth_m is None:
            return self.select_initial_target(candidates)

        valid_matches: List[Tuple[float, CandidateDetection]] = []

        for cand in candidates:
            dist_3d = self.distance_xyz(cand.xyz, self.last_target_xyz)
            depth_jump = abs(cand.depth_m - self.last_target_depth_m)

            if dist_3d <= self.max_match_distance_m and depth_jump <= self.max_depth_jump_m:
                valid_matches.append((dist_3d, cand))

        if valid_matches:
            valid_matches.sort(key=lambda x: x[0])
            return valid_matches[0][1]

        if self.target_is_remembered():
            return None

        lidar_consistent_candidates = self.get_lidar_consistent_candidates_only(
            candidates
        )
        if not lidar_consistent_candidates:
            return None

        return self.select_initial_target(lidar_consistent_candidates)

    def set_target_from_candidate(self, cand: CandidateDetection) -> None:
        self.target_active = True
        self.target_visible = True
        self.last_seen_sec = self.now_sec()
        self.last_target_xyz = cand.xyz
        self.last_target_px = (cand.cx, cand.cy)
        self.last_target_box = (cand.x1, cand.y1, cand.x2, cand.y2)
        self.last_target_depth_m = cand.depth_m

    def mark_target_temporarily_lost(self) -> None:
        self.target_visible = False

    def clear_target(self) -> None:
        self.target_active = False
        self.target_visible = False
        self.last_seen_sec = None
        self.last_target_xyz = None
        self.last_target_px = None
        self.last_target_box = None
        self.last_target_depth_m = None

    def publish_target_tf(self, x: float, y: float, z: float) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def publish_target_marker(
        self,
        x: float,
        y: float,
        z: float,
        remembered: bool,
    ) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.camera_frame
        marker.ns = "person_tracker"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.marker_scale_m
        marker.scale.y = self.marker_scale_m
        marker.scale.z = self.marker_scale_m

        if remembered:
            marker.color.r = 1.0
            marker.color.g = 0.6
            marker.color.b = 0.0
            marker.color.a = 0.85
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.90

        marker.lifetime.sec = int(self.marker_lifetime_sec)
        marker.lifetime.nanosec = int(
            (self.marker_lifetime_sec - int(self.marker_lifetime_sec)) * 1e9
        )

        self.marker_pub.publish(marker)

    def annotate_candidates(
        self,
        frame: np.ndarray,
        candidates: List[CandidateDetection],
    ) -> None:
        for cand in candidates:
            cv2.rectangle(frame, (cand.x1, cand.y1), (cand.x2, cand.y2), (0, 0, 255), 2)
            label = f"{cand.depth_m:.2f} m"
            cv2.putText(
                frame,
                label,
                (cand.x1, max(20, cand.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    def annotate_target_state(self, frame: np.ndarray) -> None:
        if self.last_target_box is not None:
            x1, y1, x2, y2 = self.last_target_box

            if self.target_visible:
                color = (0, 255, 0)
                text = f"TARGET ID {self.target_id}"
            elif self.target_is_remembered():
                color = (0, 165, 255)
                text = f"TARGET ID {self.target_id} (remembered)"
            else:
                color = (120, 120, 120)
                text = f"TARGET ID {self.target_id} (lost)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                text,
                (x1, max(40, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        state_text = "NO TARGET"
        if self.target_visible:
            state_text = "STATE: ACTIVE"
        elif self.target_is_remembered():
            state_text = "STATE: REMEMBERING"
        elif self.target_active:
            state_text = "STATE: LOST"

        cv2.putText(
            frame,
            state_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

    def maybe_log_target(self, sync_ms: Optional[float]) -> None:
        if not self.should_log():
            return

        rounded_sync = round(sync_ms, 1) if sync_ms is not None else -1.0
        rounded_avg_sync = round(self.avg_sync_ms(), 1)
        rounded_max_sync = round(self.sync_max_ms, 1)

        if self.last_target_xyz is None or self.last_target_depth_m is None:
            self.get_logger().info(
                f"Camera owner: NONE | "
                f"sync_now={rounded_sync:.1f} ms | "
                f"sync_avg={rounded_avg_sync:.1f} ms | "
                f"sync_max={rounded_max_sync:.1f} ms"
            )
            return

        x, y, z = self.last_target_xyz
        age = self.target_age_sec()

        if self.target_visible:
            state = "ACTIVE"
        elif self.target_is_remembered():
            state = "REMEMBERING"
        else:
            state = "LOST"

        self.get_logger().info(
            f"Camera owner ID {self.target_id} | state={state} | "
            f"depth={self.last_target_depth_m:.2f} m | "
            f"x={x:.2f}, y={y:.2f}, z={z:.2f} | "
            f"age={age:.2f} s | "
            f"sync_now={rounded_sync:.1f} ms | "
            f"sync_avg={rounded_avg_sync:.1f} ms | "
            f"sync_max={rounded_max_sync:.1f} ms"
        )

    def publish_debug_frame(self) -> None:
        if not self.publish_debug_image:
            return
        if self.last_debug_frame is None:
            return

        msg = self.bridge.cv2_to_imgmsg(self.last_debug_frame, "bgr8")
        self.image_pub.publish(msg)

    def process_latest_frame(self) -> None:
        if self.processing:
            return
        if not self.camera_info_ready:
            return

        with self.frame_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return
            if self.latest_rgb_stamp is None or self.latest_depth_stamp is None:
                return

            rgb_seq = self.latest_rgb_seq
            depth_seq = self.latest_depth_seq

            if rgb_seq == self.last_processed_rgb_seq and depth_seq == self.last_processed_depth_seq:
                return

            frame = self.latest_rgb.copy()
            depth_img = self.latest_depth.copy()
            rgb_stamp = self.latest_rgb_stamp
            depth_stamp = self.latest_depth_stamp

        sync_ms = self.current_sync_ms(rgb_stamp, depth_stamp)
        if sync_ms is not None:
            self.update_sync_stats(sync_ms)
            if sync_ms > self.max_rgb_depth_skew_ms:
                self.get_logger().warn(
                    f"Skipping owner inference because RGB-depth skew is {sync_ms:.1f} ms "
                    f"(limit {self.max_rgb_depth_skew_ms:.1f} ms)"
                )
                return

        self.processing = True

        try:
            candidates = self.collect_candidates(frame, depth_img)
            self.annotate_candidates(frame, candidates)

            matched = None
            lidar_consistent_candidates = self.get_lidar_consistent_candidates_only(
                candidates
            )

            if not self.target_active:
                if lidar_consistent_candidates:
                    matched = self.select_initial_target(lidar_consistent_candidates)
                else:
                    matched = self.select_initial_target(candidates)
            else:
                if lidar_consistent_candidates:
                    matched = self.match_existing_target(lidar_consistent_candidates)
                else:
                    matched = self.match_existing_target(candidates)

            if matched is not None:
                self.set_target_from_candidate(matched)
            else:
                if self.target_active:
                    self.mark_target_temporarily_lost()
                    if not self.target_is_remembered():
                        self.clear_target()

            if self.target_active and self.last_target_xyz is not None:
                remembered = not self.target_visible
                x, y, z = self.last_target_xyz
                self.publish_target_tf(x, y, z)
                self.publish_target_marker(x, y, z, remembered)

            self.annotate_target_state(frame)
            self.last_debug_frame = frame
            self.maybe_log_target(sync_ms)

            self.last_processed_rgb_seq = rgb_seq
            self.last_processed_depth_seq = depth_seq

        except Exception as exc:
            self.get_logger().error(f"Camera owner pipeline inference error: {exc}")

        self.processing = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraOwnerPipelineNode()
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