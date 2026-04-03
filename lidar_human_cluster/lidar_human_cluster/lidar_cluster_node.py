#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2


class LidarClusterNode(Node):
    def __init__(self):
        super().__init__('lidar_cluster_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('marker_topic', '/lidar_human_cluster/markers')

        # ROI limits
        self.declare_parameter('roi_x_min', 0.5)
        self.declare_parameter('roi_x_max', 8.0)
        self.declare_parameter('roi_y_min', -3.0)
        self.declare_parameter('roi_y_max', 3.0)
        self.declare_parameter('roi_z_min', -0.3)
        self.declare_parameter('roi_z_max', 2.2)

        # DBSCAN parameters
        self.declare_parameter('dbscan_eps', 0.25)
        self.declare_parameter('dbscan_min_samples', 8)

        # Cluster filtering parameters
        self.declare_parameter('cluster_min_points', 20)
        self.declare_parameter('cluster_height_min', 1.0)
        self.declare_parameter('cluster_height_max', 2.2)
        self.declare_parameter('cluster_width_min', 0.15)
        self.declare_parameter('cluster_width_max', 1.0)
        self.declare_parameter('cluster_depth_min', 0.15)
        self.declare_parameter('cluster_depth_max', 1.2)

        # Debug logging
        self.declare_parameter('verbose', True)

        self.input_topic = self.get_parameter('input_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10
        )

        self.get_logger().info('==========================================')
        self.get_logger().info('LiDAR Human Cluster Node started')
        self.get_logger().info(f'Input topic   : {self.input_topic}')
        self.get_logger().info(f'Marker topic  : {self.marker_topic}')
        self.get_logger().info('==========================================')

    def pointcloud_callback(self, msg: PointCloud2):
        # Convert PointCloud2 -> Nx3 numpy array
        points_xyz = self.pointcloud2_to_xyz(msg)

        if points_xyz.size == 0:
            self.publish_empty_markers(msg)
            self.get_logger().warn('Received empty point cloud after conversion', throttle_duration_sec=2.0)
            return

        # ROI filtering
        roi_points = self.filter_roi(points_xyz)

        if roi_points.shape[0] == 0:
            self.publish_empty_markers(msg)
            self.get_logger().info('No points left after ROI filtering', throttle_duration_sec=2.0)
            return

        # DBSCAN clustering
        cluster_labels = self.cluster_points(roi_points)

        if cluster_labels is None:
            self.publish_empty_markers(msg)
            self.get_logger().info('No clusters found', throttle_duration_sec=2.0)
            return

        # Extract human-like candidate clusters
        candidates = self.extract_human_candidates(roi_points, cluster_labels)

        # Publish RViz markers
        self.publish_candidate_markers(candidates, msg)

        verbose = self.get_parameter('verbose').value
        if verbose:
            self.get_logger().info(
                f'raw={points_xyz.shape[0]} roi={roi_points.shape[0]} candidates={len(candidates)}',
                throttle_duration_sec=1.0
            )

    def pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        """
        Convert ROS2 PointCloud2 to Nx3 float32 numpy array.
        Handles structured arrays returned by sensor_msgs_py.
        """
        points = point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        )

        # Convert generator/structured output into a plain Nx3 float32 array
        xyz_list = []
        for p in points:
            xyz_list.append([float(p[0]), float(p[1]), float(p[2])])

        if len(xyz_list) == 0:
            return np.empty((0, 3), dtype=np.float32)

        return np.array(xyz_list, dtype=np.float32)

    def filter_roi(self, points: np.ndarray) -> np.ndarray:
        """
        Keep only points inside the ROI box.
        """
        x_min = self.get_parameter('roi_x_min').value
        x_max = self.get_parameter('roi_x_max').value
        y_min = self.get_parameter('roi_y_min').value
        y_max = self.get_parameter('roi_y_max').value
        z_min = self.get_parameter('roi_z_min').value
        z_max = self.get_parameter('roi_z_max').value

        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )

        return points[mask]

    def cluster_points(self, points: np.ndarray):
        """
        Run DBSCAN clustering on ROI-filtered points.
        Noise points get label -1.
        """
        if points.shape[0] == 0:
            return None

        eps = self.get_parameter('dbscan_eps').value
        min_samples = self.get_parameter('dbscan_min_samples').value

        try:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(points)
            return labels
        except Exception as e:
            self.get_logger().error(f'DBSCAN failed: {e}')
            return None

    def extract_human_candidates(self, points: np.ndarray, labels: np.ndarray):
        """
        Build candidate list from clusters whose dimensions roughly match a person.
        """
        candidates = []

        unique_labels = set(labels.tolist())

        cluster_min_points = self.get_parameter('cluster_min_points').value
        height_min = self.get_parameter('cluster_height_min').value
        height_max = self.get_parameter('cluster_height_max').value
        width_min = self.get_parameter('cluster_width_min').value
        width_max = self.get_parameter('cluster_width_max').value
        depth_min = self.get_parameter('cluster_depth_min').value
        depth_max = self.get_parameter('cluster_depth_max').value

        for label in unique_labels:
            if label == -1:
                continue

            cluster_pts = points[labels == label]

            if cluster_pts.shape[0] < cluster_min_points:
                continue

            min_xyz = np.min(cluster_pts, axis=0)
            max_xyz = np.max(cluster_pts, axis=0)
            centroid = np.mean(cluster_pts, axis=0)

            width = float(max_xyz[0] - min_xyz[0])   # x span
            depth = float(max_xyz[1] - min_xyz[1])   # y span
            height = float(max_xyz[2] - min_xyz[2])  # z span

            # Human-like cluster filter
            if not (height_min <= height <= height_max):
                continue
            if not (width_min <= width <= width_max):
                continue
            if not (depth_min <= depth <= depth_max):
                continue

            candidates.append({
                'label': int(label),
                'num_points': int(cluster_pts.shape[0]),
                'centroid': centroid,
                'min_xyz': min_xyz,
                'max_xyz': max_xyz,
                'width': width,
                'depth': depth,
                'height': height,
            })

        return candidates

    def publish_empty_markers(self, msg: PointCloud2):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = msg.header.frame_id
        delete_marker.header.stamp = msg.header.stamp
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)
        self.marker_pub.publish(marker_array)

    def publish_candidate_markers(self, candidates, msg: PointCloud2):
        marker_array = MarkerArray()

        # First clear old markers
        delete_marker = Marker()
        delete_marker.header.frame_id = msg.header.frame_id
        delete_marker.header.stamp = msg.header.stamp
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0

        for i, candidate in enumerate(candidates):
            min_xyz = candidate['min_xyz']
            max_xyz = candidate['max_xyz']
            centroid = candidate['centroid']

            # -----------------------------
            # Bounding box marker
            # -----------------------------
            box_marker = Marker()
            box_marker.header.frame_id = msg.header.frame_id
            box_marker.header.stamp = msg.header.stamp
            box_marker.ns = 'human_candidate_boxes'
            box_marker.id = marker_id
            marker_id += 1
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD

            box_marker.pose.position.x = float((min_xyz[0] + max_xyz[0]) / 2.0)
            box_marker.pose.position.y = float((min_xyz[1] + max_xyz[1]) / 2.0)
            box_marker.pose.position.z = float((min_xyz[2] + max_xyz[2]) / 2.0)

            box_marker.pose.orientation.x = 0.0
            box_marker.pose.orientation.y = 0.0
            box_marker.pose.orientation.z = 0.0
            box_marker.pose.orientation.w = 1.0

            box_marker.scale.x = max(float(candidate['width']), 0.05)
            box_marker.scale.y = max(float(candidate['depth']), 0.05)
            box_marker.scale.z = max(float(candidate['height']), 0.05)

            box_marker.color.a = 0.35
            box_marker.color.r = 0.0
            box_marker.color.g = 1.0
            box_marker.color.b = 0.0

            marker_array.markers.append(box_marker)

            # -----------------------------
            # Centroid marker
            # -----------------------------
            centroid_marker = Marker()
            centroid_marker.header.frame_id = msg.header.frame_id
            centroid_marker.header.stamp = msg.header.stamp
            centroid_marker.ns = 'human_candidate_centroids'
            centroid_marker.id = marker_id
            marker_id += 1
            centroid_marker.type = Marker.SPHERE
            centroid_marker.action = Marker.ADD

            centroid_marker.pose.position.x = float(centroid[0])
            centroid_marker.pose.position.y = float(centroid[1])
            centroid_marker.pose.position.z = float(centroid[2])

            centroid_marker.pose.orientation.x = 0.0
            centroid_marker.pose.orientation.y = 0.0
            centroid_marker.pose.orientation.z = 0.0
            centroid_marker.pose.orientation.w = 1.0

            centroid_marker.scale.x = 0.20
            centroid_marker.scale.y = 0.20
            centroid_marker.scale.z = 0.20

            centroid_marker.color.a = 0.9
            centroid_marker.color.r = 1.0
            centroid_marker.color.g = 0.0
            centroid_marker.color.b = 0.0

            marker_array.markers.append(centroid_marker)

            # -----------------------------
            # Text marker
            # -----------------------------
            text_marker = Marker()
            text_marker.header.frame_id = msg.header.frame_id
            text_marker.header.stamp = msg.header.stamp
            text_marker.ns = 'human_candidate_labels'
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = float(centroid[0])
            text_marker.pose.position.y = float(centroid[1])
            text_marker.pose.position.z = float(max_xyz[2] + 0.25)

            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.25

            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.text = (
                f'cand_{i} '
                f'pts={candidate["num_points"]} '
                f'h={candidate["height"]:.2f}'
            )

            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarClusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()