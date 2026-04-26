#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from visualization_msgs.msg import Marker


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
        self.declare_parameter(
            "lidar_tracks_data_topic",
            "/camera_person_tracker/lidar_tracks_data",
        )
        self.declare_parameter(
            "fused_marker_topic",
            "/camera_person_tracker/fused_target_marker_b",
        )
        self.declare_parameter("fused_target_frame", "fused_target")
        self.declare_parameter("locked_lidar_target_frame", "locked_lidar_target")
        self.declare_parameter("publish_locked_lidar_tf", True)

        self.declare_parameter("publish_period_sec", 0.10)
        self.declare_parameter("camera_timeout_sec", 0.60)
        self.declare_parameter("lidar_timeout_sec", 5.00)

        self.declare_parameter("association_gate_m", 1.00)
        self.declare_parameter("reacquire_gate_m", 1.20)
        self.declare_parameter("lock_memory_sec", 1.50)

        self.declare_parameter("marker_scale_m", 0.24)
        self.declare_parameter("text_scale_m", 0.22)
        self.declare_parameter("marker_lifetime_sec", 0.30)

        self.declare_parameter("log_period_sec", 0.75)
        self.declare_parameter("debug_match_logging", True)

        self.world_frame = self.get_parameter("world_frame").value
        self.camera_target_marker_topic = self.get_parameter(
            "camera_target_marker_topic"
        ).value
        self.lidar_tracks_data_topic = self.get_parameter(
            "lidar_tracks_data_topic"
        ).value
        self.fused_marker_topic = self.get_parameter("fused_marker_topic").value
        self.fused_target_frame = self.get_parameter("fused_target_frame").value
        self.locked_lidar_target_frame = self.get_parameter(
            "locked_lidar_target_frame"
        ).value
        self.publish_locked_lidar_tf_enabled = bool(
            self.get_parameter("publish_locked_lidar_tf").value
        )

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
        self.debug_match_logging = bool(self.get_parameter("debug_match_logging").value)

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
            Marker,
            self.lidar_tracks_data_topic,
            self.lidar_tracks_data_callback,
            10,
        )

        self.fused_marker_pub = self.create_publisher(Marker, self.fused_marker_topic, 10)
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

    def distance_3d(
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

        self.latest_camera_obs = CameraObservation(
            x=world_xyz[0],
            y=world_xyz[1],
            z=world_xyz[2],
            visible=self.marker_is_camera_visible(msg),
            remembered=self.marker_is_camera_remembered(msg),
            stamp_sec=stamp_sec,
        )

    def lidar_tracks_data_callback(self, msg: Marker) -> None:
        """
        Expects one marker per LiDAR track on /camera_person_tracker/lidar_tracks_data.
        Marker ID = track ID.
        """
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

        track_id = int(msg.id)

        self.latest_lidar_tracks[track_id] = LidarTrackObservation(
            track_id=track_id,
            x=world_xyz[0],
            y=world_xyz[1],
            z=world_xyz[2],
            stamp_sec=stamp_sec,
        )

    def get_fresh_camera_observation(self) -> Optional[CameraObservation]:
        if self.latest_camera_obs is None:
            return None
        if (self.now_sec() - self.latest_camera_obs.stamp_sec) > self.camera_timeout_sec:
            return None
        return self.latest_camera_obs

    def get_fresh_lidar_tracks(self) -> Dict[int, LidarTrackObservation]:
        now_sec = self.now_sec()
        fresh_tracks: Dict[int, LidarTrackObservation] = {}

        for track_id, track in self.latest_lidar_tracks.items():
            if (now_sec - track.stamp_sec) <= self.lidar_timeout_sec:
                fresh_tracks[track_id] = track

        return fresh_tracks

    def nearest_lidar_track_to_point(
        self,
        x: float,
        y: float,
        z: float,
        tracks: Dict[int, LidarTrackObservation],
    ) -> Tuple[Optional[int], float]:
        best_track_id = None
        best_dist = float("inf")

        for track_id, track in tracks.items():
            dist = self.distance_3d((x, y, z), (track.x, track.y, track.z))
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

    def publish_locked_lidar_tf(self, track: LidarTrackObservation) -> None:
        if not self.publish_locked_lidar_tf_enabled:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.locked_lidar_target_frame
        t.transform.translation.x = track.x
        t.transform.translation.y = track.y
        t.transform.translation.z = track.z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)
    

    def publish_fused_marker(
        self,
        state: str,
        fused_xyz: Optional[Tuple[float, float, float]],
        locked_track_id: Optional[int],
    ) -> None:
        if fused_xyz is None:
            return

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.world_frame
        marker.ns = "fusion_b"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = fused_xyz[0]
        marker.pose.position.y = fused_xyz[1]
        marker.pose.position.z = fused_xyz[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.marker_scale_m
        marker.scale.y = self.marker_scale_m
        marker.scale.z = self.marker_scale_m

        if state == "FUSED":
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.95
        elif state == "CAMERA_ONLY":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.95
        elif state == "LIDAR_ONLY":
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.95
        else:
            marker.color.r = 0.6
            marker.color.g = 0.6
            marker.color.b = 0.6
            marker.color.a = 0.8

        text = f"{state} | lidar_id={locked_track_id if locked_track_id is not None else 'None'}"
        marker.text = text

        self.make_marker_lifetime(marker)
        self.fused_marker_pub.publish(marker)

    def maybe_log_match_debug(
        self,
        camera_obs: Optional[CameraObservation],
        lidar_tracks: Dict[int, LidarTrackObservation],
    ) -> None:
        if not self.debug_match_logging:
            return
        if not self.should_log():
            return

        if camera_obs is None:
            self.get_logger().info("[FusionB DEBUG] camera_target=None")
        else:
            cam_state = "ACTIVE" if camera_obs.visible else ("REMEMBERED" if camera_obs.remembered else "UNKNOWN")
            self.get_logger().info(
                f"[FusionB DEBUG] camera_target state={cam_state} "
                f"pos=({camera_obs.x:.2f}, {camera_obs.y:.2f}, {camera_obs.z:.2f})"
            )

        if not lidar_tracks:
            self.get_logger().info("[FusionB DEBUG] lidar_tracks=None")
            return

        sorted_tracks = sorted(lidar_tracks.values(), key=lambda t: t.track_id)
        for track in sorted_tracks:
            self.get_logger().info(
                f"[FusionB DEBUG] lidar_track id={track.track_id} "
                f"pos=({track.x:.2f}, {track.y:.2f}, {track.z:.2f})"
            )

        if camera_obs is not None:
            best_track_id, best_dist = self.nearest_lidar_track_to_point(
                camera_obs.x,
                camera_obs.y,
                camera_obs.z,
                lidar_tracks,
            )

            if best_track_id is not None:
                best_track = lidar_tracks[best_track_id]
                gate_ok = best_dist <= self.association_gate_m
                self.get_logger().info(
                    f"[FusionB DEBUG] nearest_lidar_to_camera "
                    f"id={best_track_id} "
                    f"dist={best_dist:.3f} m "
                    f"gate={self.association_gate_m:.3f} "
                    f"pass={gate_ok} "
                    f"| cam=({camera_obs.x:.2f}, {camera_obs.y:.2f}, {camera_obs.z:.2f}) "
                    f"| lidar=({best_track.x:.2f}, {best_track.y:.2f}, {best_track.z:.2f})"
                )

    def update(self) -> None:
        camera_obs = self.get_fresh_camera_observation()
        lidar_tracks = self.get_fresh_lidar_tracks()

        self.maybe_log_match_debug(camera_obs, lidar_tracks)

        camera_active = camera_obs is not None and camera_obs.visible
        camera_remembered = camera_obs is not None and camera_obs.remembered

        locked_track = None
        if self.locked_lidar_track_id is not None:
            locked_track = lidar_tracks.get(self.locked_lidar_track_id)

        fused_xyz: Optional[Tuple[float, float, float]] = None
        state = "NO_TARGET"

        if self.locked_lidar_track_id is None:
            if camera_active:
                best_track_id, best_dist = self.nearest_lidar_track_to_point(
                    camera_obs.x, camera_obs.y, camera_obs.z, lidar_tracks
                )

                if best_track_id is not None and best_dist <= self.association_gate_m:
                    self.locked_lidar_track_id = best_track_id
                    self.update_lock_timestamp()
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
                    dist = self.distance_3d(
                        (camera_obs.x, camera_obs.y, camera_obs.z),
                        (locked_track.x, locked_track.y, locked_track.z),
                    )

                    if dist <= self.reacquire_gate_m:
                        state = "FUSED"
                        fused_xyz = (camera_obs.x, camera_obs.y, camera_obs.z)
                    else:
                        state = "LIDAR_ONLY"
                        fused_xyz = (locked_track.x, locked_track.y, locked_track.z)
                else:
                    state = "LIDAR_ONLY"
                    fused_xyz = (locked_track.x, locked_track.y, locked_track.z)

            else:
                if self.lock_is_recent():
                    state = "LOST"
                    fused_xyz = self.last_fused_xyz
                else:
                    self.clear_lock()

                    if camera_active:
                        best_track_id, best_dist = self.nearest_lidar_track_to_point(
                            camera_obs.x, camera_obs.y, camera_obs.z, lidar_tracks
                        )

                        if best_track_id is not None and best_dist <= self.reacquire_gate_m:
                            self.locked_lidar_track_id = best_track_id
                            self.update_lock_timestamp()
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

        if fused_xyz is not None:
            self.last_fused_xyz = fused_xyz
            self.publish_fused_tf(fused_xyz[0], fused_xyz[1], fused_xyz[2])
            self.publish_fused_marker(state, fused_xyz, self.locked_lidar_track_id)

        if self.locked_lidar_track_id is not None:
            locked_track = lidar_tracks.get(self.locked_lidar_track_id)
            if locked_track is not None:
                self.publish_locked_lidar_tf(locked_track)

        self.current_state = state

        locked_id = "None" if self.locked_lidar_track_id is None else str(self.locked_lidar_track_id)
        if fused_xyz is not None:
            self.get_logger().info(
                f"[FusionB] state={state} | "
                f"fused=({fused_xyz[0]:.2f}, {fused_xyz[1]:.2f}, {fused_xyz[2]:.2f}) | "
                f"camera={'ACTIVE' if camera_active else ('REMEMBERED' if camera_remembered else 'NONE')} | "
                f"locked_lidar_id={locked_id}"
            )
        else:
            self.get_logger().info(
                f"[FusionB] state={state} | camera={'ACTIVE' if camera_active else ('REMEMBERED' if camera_remembered else 'NONE')} | locked_lidar_id={locked_id}"
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