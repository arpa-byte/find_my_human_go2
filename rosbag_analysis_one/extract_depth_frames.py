#!/usr/bin/env python3
"""
extract_depth_frames.py - Extract raw depth frames from rosbag
Saves as PNG for 16-bit precision
"""

import argparse
import os
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import numpy as np
from tqdm import tqdm

class DepthExtractor:
    def __init__(self, bag_path, output_dir):
        self.bag_path = bag_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.bridge = CvBridge()
        self.frame_count = 0
        self.depth_frames = []

    def extract_depth_frames(self):
        print(f"Opening bag: {self.bag_path}")

        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=self.bag_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(f"Failed to open bag: {e}")
            return

        topics = reader.get_all_topics_and_types()
        topic_names = [t.name for t in topics]
        print("Available topics:", topic_names)

        if '/camera/camera/depth/image_rect_raw' not in topic_names:
            print("Warning: /camera/camera/depth/image_rect_raw topic not found!")
            return

        # Approximate total for progress bar (from bag info)
        total_msgs = 1981  # adjust based on your bag info

        pbar = tqdm(total=total_msgs, desc="Extracting depth frames", unit="frame")

        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            pbar.update(1)

            if topic == '/camera/camera/depth/image_rect_raw':
                try:
                    msg = deserialize_message(data, Image)
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # 16-bit depth

                    frame_path = os.path.join(
                        self.output_dir,
                        f"depth_frame_{self.frame_count:06d}.png"
                    )
                    cv2.imwrite(frame_path, cv_img)
                    self.depth_frames.append(frame_path)
                    self.frame_count += 1

                except Exception as e:
                    print(f"Error processing frame {self.frame_count}: {e}")

        pbar.close()
        print(f"\nExtraction complete.")
        print(f"Total depth frames saved: {self.frame_count}")
        print(f"Location: {self.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract raw depth frames from ROS2 bag")
    parser.add_argument('--bag', required=True, help='Full path to the rosbag directory')
    parser.add_argument('--out', default='extracted_depth_dark', help='Output directory for depth PNGs')
    args = parser.parse_args()

    extractor = DepthExtractor(args.bag, args.out)
    extractor.extract_depth_frames()