#!/usr/bin/env python3

from typing import Dict, List, Tuple
import math

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class Track:
    def __init__(self, track_id: int, x: float, y: float, stamp_sec: float):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.last_seen_sec = stamp_sec

        self.hits = 1
        self.missed = 0
        self.confirmed = False

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


class LidarTrackManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_track_manager")

        self.declare_parameter(
            "input_detection_topic",
            "/camera_person_tracker/lidar_detection_poses",
        )
        self.declare_parameter(
            "output_tracks_topic",
            "/camera_person_tracker/lidar_tracked_poses",
        )
        self.declare_parameter(
            "output_marker_topic",
            "/camera_person_tracker/lidar_tracks_marker",
        )

        # NEW: separate fusion/data topic with true track positions
        self.declare_parameter(
            "output_data_marker_topic",
            "/camera_person_tracker/lidar_tracks_data",
        )

        self.declare_parameter("association_distance_m", 0.75)
        self.declare_parameter("min_confirmed_hits", 3)
        self.declare_parameter("max_missed_frames", 5)
        self.declare_parameter("track_timeout_sec", 1.5)

        self.declare_parameter("track_marker_z", 0.20)
        self.declare_parameter("track_marker_scale", 0.20)
        self.declare_parameter("track_text_z", 0.55)
        self.declare_parameter("track_text_scale", 0.24)
        self.declare_parameter("marker_lifetime_sec", 0.45)

        # NEW: data marker settings for fusion
        self.declare_parameter("data_marker_z", 0.0)
        self.declare_parameter("data_marker_scale", 0.10)

        self.declare_parameter("publish_tentative_tracks", False)
        self.declare_parameter("log_period_sec", 1.0)

        self.input_detection_topic = self.get_parameter("input_detection_topic").value
        self.output_tracks_topic = self.get_parameter("output_tracks_topic").value
        self.output_marker_topic = self.get_parameter("output_marker_topic").value
        self.output_data_marker_topic = self.get_parameter("output_data_marker_topic").value

        self.association_distance_m = float(
            self.get_parameter("association_distance_m").value
        )
        self.min_confirmed_hits = int(self.get_parameter("min_confirmed_hits").value)
        self.max_missed_frames = int(self.get_parameter("max_missed_frames").value)
        self.track_timeout_sec = float(self.get_parameter("track_timeout_sec").value)

        self.track_marker_z = float(self.get_parameter("track_marker_z").value)
        self.track_marker_scale = float(self.get_parameter("track_marker_scale").value)
        self.track_text_z = float(self.get_parameter("track_text_z").value)
        self.track_text_scale = float(self.get_parameter("track_text_scale").value)
        self.marker_lifetime_sec = float(self.get_parameter("marker_lifetime_sec").value)

        self.data_marker_z = float(self.get_parameter("data_marker_z").value)
        self.data_marker_scale = float(self.get_parameter("data_marker_scale").value)

        self.publish_tentative_tracks = bool(
            self.get_parameter("publish_tentative_tracks").value
        )
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)

        self.pose_sub = self.create_subscription(
            PoseArray,
            self.input_detection_topic,
            self.detection_callback,
            10,
        )

        self.tracks_pub = self.create_publisher(PoseArray, self.output_tracks_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.output_marker_topic, 10)

        # NEW: fusion/data publisher
        self.data_marker_pub = self.create_publisher(
            Marker,
            self.output_data_marker_topic,
            10,
        )

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.last_log_time = self.get_clock().now()

        self.get_logger().info("LidarTrackManagerNode started.")
        self.get_logger().info(f"Input detections   : {self.input_detection_topic}")
        self.get_logger().info(f"Output tracks      : {self.output_tracks_topic}")
        self.get_logger().info(f"Output markers     : {self.output_marker_topic}")
        self.get_logger().info(f"Output data markers: {self.output_data_marker_topic}")
        self.get_logger().info(f"Association dist   : {self.association_distance_m:.2f} m")
        self.get_logger().info(f"Confirm hits       : {self.min_confirmed_hits}")
        self.get_logger().info(f"Max missed         : {self.max_missed_frames}")
        self.get_logger().info(f"Track timeout      : {self.track_timeout_sec:.2f} s")

    def stamp_to_sec(self, msg_stamp) -> float:
        return float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9

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

    def greedy_association(
        self,
        detections: List[Tuple[float, float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        track_ids = list(self.tracks.keys())

        candidate_pairs = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            for det_idx, (dx, dy) in enumerate(detections):
                dist = track.distance_to(dx, dy)
                if dist <= self.association_distance_m:
                    candidate_pairs.append((dist, track_id, det_idx))

        candidate_pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()
        matches: List[Tuple[int, int]] = []

        for dist, track_id, det_idx in candidate_pairs:
            if track_id in matched_tracks:
                continue
            if det_idx in matched_dets:
                continue

            matched_tracks.add(track_id)
            matched_dets.add(det_idx)
            matches.append((track_id, det_idx))

        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        unmatched_dets = [idx for idx in range(len(detections)) if idx not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets

    def prune_dead_tracks(self, now_sec: float) -> None:
        to_delete = []

        for track_id, track in self.tracks.items():
            age = now_sec - track.last_seen_sec
            if track.missed > self.max_missed_frames or age > self.track_timeout_sec:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

    def create_track(self, x: float, y: float, stamp_sec: float) -> None:
        track = Track(self.next_track_id, x, y, stamp_sec)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def update_track(self, track: Track, x: float, y: float, stamp_sec: float) -> None:
        track.x = x
        track.y = y
        track.last_seen_sec = stamp_sec
        track.hits += 1
        track.missed = 0

        if not track.confirmed and track.hits >= self.min_confirmed_hits:
            track.confirmed = True

    def build_tracks_pose_array(self, header, visible_tracks: List[Track]) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header = header

        for track in visible_tracks:
            pose = Pose()
            pose.position.x = track.x
            pose.position.y = track.y
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        return pose_array

    def build_display_marker_array(self, header, visible_tracks: List[Track]) -> MarkerArray:
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header = header
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for track in visible_tracks:
            base_id = track.track_id * 10

            center = Marker()
            center.header = header
            center.ns = "lidar_tracks"
            center.id = base_id + 0
            center.type = Marker.SPHERE
            center.action = Marker.ADD
            center.pose.position.x = track.x
            center.pose.position.y = track.y
            center.pose.position.z = self.track_marker_z
            center.pose.orientation.w = 1.0
            center.scale.x = self.track_marker_scale
            center.scale.y = self.track_marker_scale
            center.scale.z = self.track_marker_scale

            if track.confirmed:
                center.color.r = 1.0
                center.color.g = 1.0
                center.color.b = 0.0
                center.color.a = 0.95
            else:
                center.color.r = 1.0
                center.color.g = 0.5
                center.color.b = 0.0
                center.color.a = 0.70

            self.make_marker_lifetime(center)
            marker_array.markers.append(center)

            text = Marker()
            text.header = header
            text.ns = "lidar_track_ids"
            text.id = base_id + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = track.x
            text.pose.position.y = track.y
            text.pose.position.z = self.track_text_z
            text.pose.orientation.w = 1.0
            text.scale.z = self.track_text_scale
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95

            if track.confirmed:
                text.text = f"ID {track.track_id}"
            else:
                text.text = f"T{track.track_id}"

            self.make_marker_lifetime(text)
            marker_array.markers.append(text)

        return marker_array

    def detection_callback(self, msg: PoseArray) -> None:
        stamp_sec = self.stamp_to_sec(msg.header.stamp)
        if stamp_sec <= 0.0:
            now_msg = self.get_clock().now().to_msg()
            stamp_sec = self.stamp_to_sec(now_msg)

        detections = []
        for pose in msg.poses:
            detections.append((float(pose.position.x), float(pose.position.y)))

        matches, unmatched_tracks, unmatched_dets = self.greedy_association(detections)

        for track_id, det_idx in matches:
            dx, dy = detections[det_idx]
            self.update_track(self.tracks[track_id], dx, dy, stamp_sec)

        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1

        for det_idx in unmatched_dets:
            dx, dy = detections[det_idx]
            self.create_track(dx, dy, stamp_sec)

        self.prune_dead_tracks(stamp_sec)

        visible_tracks: List[Track] = []
        for track in self.tracks.values():
            if track.confirmed or self.publish_tentative_tracks:
                visible_tracks.append(track)

        visible_tracks.sort(key=lambda t: t.track_id)

        pose_array = self.build_tracks_pose_array(msg.header, visible_tracks)
        display_marker_array = self.build_display_marker_array(msg.header, visible_tracks)

        self.tracks_pub.publish(pose_array)
        self.marker_pub.publish(display_marker_array)

        # Publish one Marker per confirmed track so fused_target_tracker
        # receives individual Marker messages matching its subscription type
        for track in visible_tracks:
            data_marker = Marker()
            data_marker.header = msg.header
            data_marker.ns = "lidar_tracks_data"
            data_marker.id = track.track_id
            data_marker.type = Marker.SPHERE
            data_marker.action = Marker.ADD

            data_marker.pose.position.x = track.x
            data_marker.pose.position.y = track.y
            data_marker.pose.position.z = self.data_marker_z
            data_marker.pose.orientation.w = 1.0

            data_marker.scale.x = self.data_marker_scale
            data_marker.scale.y = self.data_marker_scale
            data_marker.scale.z = self.data_marker_scale

            data_marker.color.r = 0.2
            data_marker.color.g = 1.0
            data_marker.color.b = 0.2
            data_marker.color.a = 0.15

            self.make_marker_lifetime(data_marker)
            self.data_marker_pub.publish(data_marker)

        if self.should_log():
            confirmed_count = sum(1 for t in self.tracks.values() if t.confirmed)
            tentative_count = sum(1 for t in self.tracks.values() if not t.confirmed)
            ids = [t.track_id for t in visible_tracks]
            self.get_logger().info(
                f"LiDAR tracks | detections={len(detections)} | "
                f"confirmed={confirmed_count} | tentative={tentative_count} | "
                f"published_ids={ids}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarTrackManagerNode()
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