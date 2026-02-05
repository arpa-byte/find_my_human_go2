#!/usr/bin/env python3
"""
bag_analyzer.py - Week 1 Phase 2
Reusable script to extract frames from ROS2 bags and prepare for metrics.
Focus: Extract annotated images from /human_tracker/output topic.
"""

import argparse
import os
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import numpy as np

class BagAnalyzer:
    def __init__(self, bag_path, output_dir):
        self.bag_path = bag_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.annotated_frames = []  # list of file paths for later use

    def extract_annotated_frames(self):
        """Extract images from /human_tracker/output topic and save them."""
        print(f"Opening bag: {self.bag_path}")

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader.open(storage_options, converter_options)

        topics = reader.get_all_topics_and_types()
        topic_names = [t.name for t in topics]
        print("Available topics:", topic_names)

        if '/human_tracker/output' not in topic_names:
            print("Warning: /human_tracker/output topic not found in bag!")
            return

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic == '/human_tracker/output':
                try:
                    msg = deserialize_message(data, Image)
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                    frame_path = os.path.join(
                        self.output_dir,
                        f"frame_{self.frame_count:06d}.jpg"
                    )
                    cv2.imwrite(frame_path, cv_img)
                    self.annotated_frames.append(frame_path)
                    self.frame_count += 1

                    if self.frame_count % 50 == 0:
                        print(f"Extracted {self.frame_count} frames so far...")

                except Exception as e:
                    print(f"Error processing frame {self.frame_count}: {e}")

        print(f"\nExtraction complete.")
        print(f"Total annotated frames saved: {self.frame_count}")
        print(f"Location: {self.output_dir}")

    def summarize_extraction(self):
        """Print summary stats (can expand with metrics later)."""
        if self.frame_count == 0:
            print("No frames extracted.")
            return

        print("\nSummary:")
        print(f"  - Frames extracted: {self.frame_count}")
        print(f"  - First frame: {self.annotated_frames[0] if self.annotated_frames else 'None'}")
        print(f"  - Last frame:  {self.annotated_frames[-1] if self.annotated_frames else 'None'}")
        # Placeholder for future metrics (MOTA, etc.)
        print("  - Metrics placeholder: Ready for ground truth comparison")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract annotated frames from ROS2 bag")
    parser.add_argument('--bag', required=True, help='Path to the rosbag directory (e.g. /path/to/my_bag)')
    parser.add_argument('--out', default='extracted_frames', help='Output directory for frames')
    args = parser.parse_args()

    analyzer = BagAnalyzer(args.bag, args.out)
    analyzer.extract_annotated_frames()
    analyzer.summarize_extraction()