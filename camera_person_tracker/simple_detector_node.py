#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO
from visualization_msgs.msg import Marker

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

OUTPUT_TOPIC = "/camera_person_tracker/simple_output"
MARKER_TOPIC = "/camera_person_tracker/target_marker"

CAMERA_FRAME = "camera_color_optical_frame"
TARGET_FRAME = "target_person"

DEPTH_SCALE = 0.001  # depth in mm -> meters


class SimplePersonDetector(Node):
    def __init__(self):
        super().__init__("simple_person_detector")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.rgb = None
        self.depth = None
        self.rgb_stamp = None
        self.depth_stamp = None

        self.fx = None
        self.fy = None
        self.cx0 = None
        self.cy0 = None
        self.camera_info_ready = False

        self.processing = False
        self.last_logged_distance = None

        self.sync_count = 0
        self.sync_sum_ms = 0.0
        self.sync_max_ms = 0.0

        self.create_subscription(Image, COLOR_TOPIC, self.rgb_callback, qos_profile_sensor_data)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, qos_profile_sensor_data)

        self.image_pub = self.create_publisher(Image, OUTPUT_TOPIC, 10)
        self.marker_pub = self.create_publisher(Marker, MARKER_TOPIC, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_timer(0.15, self.process)

        self.get_logger().info("SIMPLE detector + RViz marker running")

    def rgb_callback(self, msg):
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_stamp = msg.header.stamp

    def depth_callback(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_stamp = msg.header.stamp

    def camera_info_callback(self, msg: CameraInfo):
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

    def stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def current_sync_ms(self):
        if self.rgb_stamp is None or self.depth_stamp is None:
            return None
        rgb_t = self.stamp_to_sec(self.rgb_stamp)
        depth_t = self.stamp_to_sec(self.depth_stamp)
        return abs(rgb_t - depth_t) * 1000.0

    def update_sync_stats(self, sync_ms):
        self.sync_count += 1
        self.sync_sum_ms += sync_ms
        if sync_ms > self.sync_max_ms:
            self.sync_max_ms = sync_ms

    def avg_sync_ms(self):
        if self.sync_count == 0:
            return 0.0
        return self.sync_sum_ms / self.sync_count

    def get_depth(self, px, py):
        h, w = self.depth.shape
        x = int(np.clip(px, 0, w - 1))
        y = int(np.clip(py, 0, h - 1))

        d = self.depth[y, x]
        if d == 0:
            return -1.0

        return float(d) * DEPTH_SCALE

    def pixel_to_3d(self, u, v, z):
        if not self.camera_info_ready or z <= 0.0:
            return None

        x = (u - self.cx0) * z / self.fx
        y = (v - self.cy0) * z / self.fy
        return float(x), float(y), float(z)

    def publish_target_tf(self, x, y, z):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = CAMERA_FRAME
        t.child_frame_id = TARGET_FRAME

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def publish_target_marker(self, x, y, z):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = CAMERA_FRAME

        marker.ns = "person_tracker"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 300_000_000

        self.marker_pub.publish(marker)

    def process(self):
        if self.rgb is None or self.depth is None:
            return
        if self.rgb_stamp is None or self.depth_stamp is None:
            return
        if not self.camera_info_ready:
            return
        if self.processing:
            return

        self.processing = True
        frame = self.rgb.copy()

        try:
            sync_ms = self.current_sync_ms()
            if sync_ms is not None:
                self.update_sync_stats(sync_ms)

            results = self.model(frame, verbose=False)[0]

            detected_any = False
            closest_distance = float("inf")
            best_center = None
            best_box = None

            if results.boxes is not None:
                for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                    if int(cls) != 0:
                        continue  # person only

                    x1, y1, x2, y2 = map(int, box)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    depth_m = self.get_depth(cx, cy)

                    if depth_m > 0.0:
                        detected_any = True

                        if depth_m < closest_distance:
                            closest_distance = depth_m
                            best_center = (cx, cy)
                            best_box = (x1, y1, x2, y2)

                    # draw all detected people
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{depth_m:.2f} m" if depth_m > 0 else "no depth"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            if detected_any and best_center is not None:
                u, v = best_center
                xyz = self.pixel_to_3d(u, v, closest_distance)

                if xyz is not None:
                    x, y, z = xyz

                    # highlight selected target
                    if best_box is not None:
                        x1, y1, x2, y2 = best_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(
                            frame,
                            "TARGET",
                            (x1, max(40, y1 - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    self.publish_target_tf(x, y, z)
                    self.publish_target_marker(x, y, z)

                    rounded_distance = round(closest_distance, 2)
                    rounded_sync = round(sync_ms, 1) if sync_ms is not None else -1.0
                    rounded_avg_sync = round(self.avg_sync_ms(), 1)
                    rounded_max_sync = round(self.sync_max_ms, 1)

                    if (
                        self.last_logged_distance is None
                        or abs(rounded_distance - self.last_logged_distance) >= 0.05
                    ):
                        self.get_logger().info(
                            f"Target at {rounded_distance:.2f} m | "
                            f"x={x:.2f}, y={y:.2f}, z={z:.2f} | "
                            f"sync_now={rounded_sync:.1f} ms | "
                            f"sync_avg={rounded_avg_sync:.1f} ms | "
                            f"sync_max={rounded_max_sync:.1f} ms"
                        )
                        self.last_logged_distance = rounded_distance

            self.publish_image(frame)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

        self.processing = False

    def publish_image(self, frame):
        msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_pub.publish(msg)


def main():
    rclpy.init()
    node = SimplePersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
