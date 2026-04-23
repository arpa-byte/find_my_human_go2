#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class CameraObservation:
    x: float
    y: float
    z: float
    visible: bool
    remembered: bool
    stamp_sec: float


@dataclass
class LidarTrackObservation:
    track_id: int
    x: float
    y: float
    z: float
    stamp_sec: float


class FusedTargetTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("fused_target_tracker")

        self.declare_parameter("world_frame", "base_link")

        self.declare_parameter(
            "camera_target_marker_topic",
            "/camera_person_tracker/target_marker",
        )

        # IMPORTANT: use data topic, not RViz display markers
        self.declare_parameter(
            "lidar_tracks_data_topic",
            "/camera_person_tracker/lidar_tracks_data",
        )

        self.declare_parameter(
            "fused_marker_topic",
            "/camera_person_tracker/fused_target_marker_b",
        )
        self.declare_parameter("fused_target_frame", "fused_target")

        self.declare_parameter("publish_period_sec", 0.10)
        self.declare_parameter("camera_timeout_sec", 0.60)
        self.declare_parameter("lidar_timeout_sec", 5.0)

        # Slightly more forgiving in XY-only association
        self.declare_parameter("association_gate_m", 1.00)
        self.declare_parameter("reacquire_gate_m", 1.20)
        self.declare_parameter("lock_memory_sec", 1.50)

        self.declare_parameter("marker_scale_m", 0.24)
        self.declare_parameter("text_scale_m", 0.22)
        self.declare_parameter("marker_lifetime_sec", 0.30)

        self.declare_parameter("log_period_sec", 0.75)

        self.world_frame = self.get_parameter("world_frame").value
        self.camera_target_marker_topic = self.get_parameter("camera_target_marker_topic").value
        self.lidar_tracks_data_topic = self.get_parameter("lidar_tracks_data_topic").value
        self.fused_marker_topic = self.get_parameter("fused_marker_topic").value
        self.fused_target_frame = self.get_parameter("fused_target_frame").value

        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.camera_timeout_sec = float(self.get_parameter("camera_timeout_sec").value)
        self.lidar_timeout_sec = float(self.get_parameter("lidar_timeout_sec").value)

        self.association_gate_m = float(self.get_parameter("association_gate_m").value)
        self.reacquire_gate_m = float(self.get_parameter("reacquire_gate_m").value)
        self.lock_memory_sec = float(self.get_parameter("lock_memory_sec").value)

        self.marker_scale_m = float(self.get_parameter("marker_scale_m").value)
        self.text_scale_m = float(self.get_parameter("text_scale_m").value)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)

        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.latest_camera_obs: Optional[CameraObservation] = None
        self.latest_lidar_tracks: Dict[int, LidarTrackObservation] = {}

        self.locked_lidar_track_id: Optional[int] = None
        self.lock_last_seen_sec: Optional[float] = None

        self.current_state = "NO_TARGET"
        self.last_fused_xyz: Optional[Tuple[float, float, float]] = None

        self.last_log_time = self.get_clock().now()

        self.create_subscription(
            Marker,
            self.camera_target_marker_topic,
            self.camera_marker_callback,
            10,
        )

        self.create_subscription(
            MarkerArray,
            self.lidar_tracks_data_topic,
            self.lidar_tracks_data_callback,
            10,
        )

        self.fused_marker_pub = self.create_publisher(MarkerArray, self.fused_marker_topic, 10)

        self.timer = self.create_timer(self.publish_period_sec, self.update)

        self.get_logger().info("FusedTargetTrackerNode started.")
        self.get_logger().info(f"World frame         : {self.world_frame}")
        self.get_logger().info(f"Camera marker topic : {self.camera_target_marker_topic}")
        self.get_logger().info(f"LiDAR data topic    : {self.lidar_tracks_data_topic}")
        self.get_logger().info(f"Fused marker topic  : {self.fused_marker_topic}")
        self.get_logger().info(f"Fused target frame  : {self.fused_target_frame}")
        self.get_logger().info(f"Association gate    : {self.association_gate_m:.2f} m")
        self.get_logger().info(f"Reacquire gate      : {self.reacquire_gate_m:.2f} m")
        self.get_logger().info(f"Lock memory         : {self.lock_memory_sec:.2f} s")

    def stamp_to_sec(self, stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    def now_sec(self) -> float:
        return self.stamp_to_sec(self.get_clock().now().to_msg())

    def should_log(self) -> bool:
        now = self.get_clock().now()
        dt = (now - self.last_log_time).nanoseconds / 1e9
        if dt >= self.log_period_sec:
            self.last_log_time = now
            return True
        return False

    def make_marker_lifetime(self, marker: Marker) -> None:
        marker.lifetime.sec = int(self.marker_lifetime_sec)
        marker.lifetime.nanosec = int(
            (self.marker_lifetime_sec - int(self.marker_lifetime_sec)) * 1e9
        )

    def distance_xy(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

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

    def transform_point_to_world(
        self,
        source_frame: str,
        x: float,
        y: float,
        z: float,
    ) -> Optional[Tuple[float, float, float]]:
        if source_frame == self.world_frame:
            return (x, y, z)

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"TF transform failed from {source_frame} to {self.world_frame}: {exc}"
            )
            return None

        qx = float(tf_msg.transform.rotation.x)
        qy = float(tf_msg.transform.rotation.y)
        qz = float(tf_msg.transform.rotation.z)
        qw = float(tf_msg.transform.rotation.w)

        tx = float(tf_msg.transform.translation.x)
        ty = float(tf_msg.transform.translation.y)
        tz = float(tf_msg.transform.translation.z)

        rx, ry, rz = self.rotate_point_by_quaternion(x, y, z, qx, qy, qz, qw)
        return (rx + tx, ry + ty, rz + tz)

    def marker_is_camera_visible(self, marker: Marker) -> bool:
        return marker.color.g >= 0.8 and marker.color.r <= 0.25

    def marker_is_camera_remembered(self, marker: Marker) -> bool:
        return marker.color.r >= 0.8 and 0.3 <= marker.color.g <= 0.8

    def camera_marker_callback(self, msg: Marker) -> None:
        if msg.action != Marker.ADD:
            return

        world_xyz = self.transform_point_to_world(
            msg.header.frame_id,
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

        if world_xyz is None:
            return

        stamp_sec = self.stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            stamp_sec = self.now_sec()

        visible = self.marker_is_camera_visible(msg)
        remembered = self.marker_is_camera_remembered(msg)

        self.latest_camera_obs = CameraObservation(
            x=world_xyz[0],
            y=world_xyz[1],
            z=world_xyz[2],
            visible=visible,
            remembered=remembered,
            stamp_sec=stamp_sec,
        )

    def lidar_tracks_data_callback(self, msg: MarkerArray) -> None:
        new_tracks: Dict[int, LidarTrackObservation] = {}

        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue

            if marker.ns != "lidar_tracks_data":
                continue

            track_id = int(marker.id)

            world_xyz = self.transform_point_to_world(
                marker.header.frame_id,
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                float(marker.pose.position.z),
            )

            if world_xyz is None:
                continue

            stamp_sec = self.stamp_to_sec(marker.header.stamp)
            if stamp_sec <= 0.0:
                stamp_sec = self.now_sec()

            new_tracks[track_id] = LidarTrackObservation(
                track_id=track_id,
                x=world_xyz[0],
                y=world_xyz[1],
                z=world_xyz[2],
                stamp_sec=stamp_sec,
            )

        self.latest_lidar_tracks = new_tracks

    def get_fresh_camera_observation(self) -> Optional[CameraObservation]:
        if self.latest_camera_obs is None:
            return None

        age = self.now_sec() - self.latest_camera_obs.stamp_sec
        if age > self.camera_timeout_sec:
            return None

        return self.latest_camera_obs

    def get_fresh_lidar_tracks(self) -> Dict[int, LidarTrackObservation]:
        now_sec = self.now_sec()
        fresh_tracks: Dict[int, LidarTrackObservation] = {}

        for track_id, track in self.latest_lidar_tracks.items():
            if (now_sec - track.stamp_sec) <= self.lidar_timeout_sec:
                fresh_tracks[track_id] = track

        return fresh_tracks

    def nearest_lidar_track_to_point_xy(
        self,
        x: float,
        y: float,
        tracks: Dict[int, LidarTrackObservation],
    ) -> Tuple[Optional[int], float]:
        best_track_id = None
        best_dist = float("inf")

        for track_id, track in tracks.items():
            dist = self.distance_xy((x, y), (track.x, track.y))
            if dist < best_dist:
                best_dist = dist
                best_track_id = track_id

        return best_track_id, best_dist

    def clear_lock(self) -> None:
        self.locked_lidar_track_id = None
        self.lock_last_seen_sec = None

    def update_lock_timestamp(self) -> None:
        self.lock_last_seen_sec = self.now_sec()

    def lock_is_recent(self) -> bool:
        if self.lock_last_seen_sec is None:
            return False
        return (self.now_sec() - self.lock_last_seen_sec) <= self.lock_memory_sec

    def publish_fused_tf(self, x: float, y: float, z: float) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.fused_target_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def publish_fused_markers(
        self,
        state: str,
        fused_xyz: Optional[Tuple[float, float, float]],
        camera_xyz: Optional[Tuple[float, float, float]],
        lidar_xyz: Optional[Tuple[float, float, float]],
        locked_track_id: Optional[int],
    ) -> None:
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.header.frame_id = self.world_frame
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if fused_xyz is None:
            self.fused_marker_pub.publish(marker_array)
            return

        x, y, z = fused_xyz

        sphere = Marker()
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.header.frame_id = self.world_frame
        sphere.ns = "fusion_b"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = x
        sphere.pose.position.y = y
        sphere.pose.position.z = z
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = self.marker_scale_m
        sphere.scale.y = self.marker_scale_m
        sphere.scale.z = self.marker_scale_m

        if state == "FUSED":
            sphere.color.r = 0.0
            sphere.color.g = 1.0
            sphere.color.b = 1.0
            sphere.color.a = 0.95
        elif state == "CAMERA_ONLY":
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 1.0
            sphere.color.a = 0.95
        elif state == "LIDAR_ONLY":
            sphere.color.r = 1.0
            sphere.color.g = 1.0
            sphere.color.b = 0.0
            sphere.color.a = 0.95
        else:
            sphere.color.r = 0.6
            sphere.color.g = 0.6
            sphere.color.b = 0.6
            sphere.color.a = 0.80

        self.make_marker_lifetime(sphere)
        marker_array.markers.append(sphere)

        text = Marker()
        text.header.stamp = self.get_clock().now().to_msg()
        text.header.frame_id = self.world_frame
        text.ns = "fusion_b_text"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = z + 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = self.text_scale_m
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 0.98

        lidar_id_text = "None" if locked_track_id is None else str(locked_track_id)
        text.text = f"{state} | lidar_id={lidar_id_text}"

        self.make_marker_lifetime(text)
        marker_array.markers.append(text)

        if camera_xyz is not None:
            cam = Marker()
            cam.header.stamp = self.get_clock().now().to_msg()
            cam.header.frame_id = self.world_frame
            cam.ns = "fusion_b_debug"
            cam.id = 10
            cam.type = Marker.SPHERE
            cam.action = Marker.ADD
            cam.pose.position.x = camera_xyz[0]
            cam.pose.position.y = camera_xyz[1]
            cam.pose.position.z = camera_xyz[2]
            cam.pose.orientation.w = 1.0
            cam.scale.x = 0.12
            cam.scale.y = 0.12
            cam.scale.z = 0.12
            cam.color.r = 1.0
            cam.color.g = 0.0
            cam.color.b = 0.0
            cam.color.a = 0.85
            self.make_marker_lifetime(cam)
            marker_array.markers.append(cam)

        if lidar_xyz is not None:
            lid = Marker()
            lid.header.stamp = self.get_clock().now().to_msg()
            lid.header.frame_id = self.world_frame
            lid.ns = "fusion_b_debug"
            lid.id = 11
            lid.type = Marker.SPHERE
            lid.action = Marker.ADD
            lid.pose.position.x = lidar_xyz[0]
            lid.pose.position.y = lidar_xyz[1]
            lid.pose.position.z = lidar_xyz[2]
            lid.pose.orientation.w = 1.0
            lid.scale.x = 0.12
            lid.scale.y = 0.12
            lid.scale.z = 0.12
            lid.color.r = 1.0
            lid.color.g = 1.0
            lid.color.b = 0.0
            lid.color.a = 0.85
            self.make_marker_lifetime(lid)
            marker_array.markers.append(lid)

        self.fused_marker_pub.publish(marker_array)

    def maybe_log_status(
        self,
        state: str,
        camera_obs: Optional[CameraObservation],
        locked_track: Optional[LidarTrackObservation],
        fused_xyz: Optional[Tuple[float, float, float]],
    ) -> None:
        if not self.should_log():
            return

        camera_status = "NONE"
        if camera_obs is not None:
            if camera_obs.visible:
                camera_status = "ACTIVE"
            elif camera_obs.remembered:
                camera_status = "REMEMBERED"

        locked_id = "None" if self.locked_lidar_track_id is None else str(self.locked_lidar_track_id)

        if fused_xyz is None:
            self.get_logger().info(
                f"[FusionB] state={state} | camera={camera_status} | locked_lidar_id={locked_id}"
            )
            return

        fx, fy, fz = fused_xyz
        msg = (
            f"[FusionB] state={state} | fused=({fx:.2f}, {fy:.2f}, {fz:.2f}) "
            f"| camera={camera_status} | locked_lidar_id={locked_id}"
        )

        if camera_obs is not None:
            msg += f" | cam=({camera_obs.x:.2f}, {camera_obs.y:.2f}, {camera_obs.z:.2f})"

        if locked_track is not None:
            msg += (
                f" | lidar=({locked_track.x:.2f}, {locked_track.y:.2f}, {locked_track.z:.2f})"
            )

        self.get_logger().info(msg)

    def update(self) -> None:
        camera_obs = self.get_fresh_camera_observation()
        lidar_tracks = self.get_fresh_lidar_tracks()

        camera_active = camera_obs is not None and camera_obs.visible
        camera_remembered = camera_obs is not None and camera_obs.remembered

        locked_track = None
        if self.locked_lidar_track_id is not None:
            locked_track = lidar_tracks.get(self.locked_lidar_track_id)

        fused_xyz: Optional[Tuple[float, float, float]] = None
        state = "NO_TARGET"

        if self.locked_lidar_track_id is None:
            if camera_active:
                best_track_id, best_dist = self.nearest_lidar_track_to_point_xy(
                    camera_obs.x, camera_obs.y, lidar_tracks
                )

                if best_track_id is not None and best_dist <= self.association_gate_m:
                    self.locked_lidar_track_id = best_track_id
                    self.update_lock_timestamp()
                    locked_track = lidar_tracks.get(best_track_id)
                    state = "FUSED"
                    fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)
                else:
                    state = "CAMERA_ONLY"
                    fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)

            elif camera_remembered:
                state = "LOST"
                fused_xyz = self.last_fused_xyz
            else:
                state = "NO_TARGET"

        else:
            if locked_track is not None:
                self.update_lock_timestamp()

                if camera_active:
                    dist = self.distance_xy(
                        (camera_obs.x, camera_obs.y),
                        (locked_track.x, locked_track.y),
                    )

                    if dist <= self.reacquire_gate_m:
                        state = "FUSED"
                        fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)
                    else:
                        state = "CAMERA_ONLY"
                        fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)

                else:
                    state = "LIDAR_ONLY"
                    fused_xyz = (locked_track.x, locked_track.y, locked_track.z)

            else:
                if camera_active:
                    best_track_id, best_dist = self.nearest_lidar_track_to_point_xy(
                        camera_obs.x, camera_obs.y, lidar_tracks
                    )

                    if best_track_id is not None and best_dist <= self.reacquire_gate_m:
                        self.locked_lidar_track_id = best_track_id
                        self.update_lock_timestamp()
                        locked_track = lidar_tracks.get(best_track_id)
                        state = "FUSED"
                        fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)
                    else:
                        state = "CAMERA_ONLY"
                        fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)

                elif self.lock_is_recent():
                    state = "LOST"
                    fused_xyz = self.last_fused_xyz
                else:
                    self.clear_lock()
                    if camera_remembered:
                        state = "LOST"
                        fused_xyz = self.last_fused_xyz
                    else:
                        state = "NO_TARGET"

        if fused_xyz is not None:
            self.last_fused_xyz = fused_xyz
            self.publish_fused_tf(fused_xyz[0], fused_xyz[1], fused_xyz[2])

        camera_xyz = None if camera_obs is None else (camera_obs.x, camera_obs.y, camera_obs.z)
        lidar_xyz = None if locked_track is None else (locked_track.x, locked_track.y, locked_track.z)

        self.publish_fused_markers(
            state=state,
            fused_xyz=fused_xyz,
            camera_xyz=camera_xyz,
            lidar_xyz=lidar_xyz,
            locked_track_id=self.locked_lidar_track_id,
        )

        self.maybe_log_status(
            state=state,
            camera_obs=camera_obs,
            locked_track=locked_track,
            fused_xyz=fused_xyz,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FusedTargetTrackerNode()
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