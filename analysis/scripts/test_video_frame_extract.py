#!/usr/bin/env python3
"""
test_video_frame_extract.py - Simulate bag frame extraction using a regular video
For dry-run validation of bag_analyzer.py logic
"""

import cv2
import os
import argparse

def extract_frames_from_video(video_path, output_dir, every_nth=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % every_nth == 0:
            frame_path = os.path.join(output_dir, f"sim_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            print(f"Saved simulated frame: {frame_path}")

    cap.release()
    print(f"\nSimulation complete.")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Location: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate ROS bag frame extraction from video")
    parser.add_argument('--video', required=True, help='Path to MP4 video')
    parser.add_argument('--out', default='sim_extracted_frames', help='Output directory')
    parser.add_argument('--every', type=int, default=10, help='Save every Nth frame')
    args = parser.parse_args()

    extract_frames_from_video(args.video, args.out, args.every)