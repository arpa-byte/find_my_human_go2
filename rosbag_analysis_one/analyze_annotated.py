#!/usr/bin/env python3
"""
analyze_annotated.py - Basic stats from simulated annotated frames
"""

import os
import cv2
from tqdm import tqdm

def analyze_annotated_dir(annotated_dir):
    files = sorted([f for f in os.listdir(annotated_dir) if f.endswith('.jpg')])
    print(f"Analyzing {len(files)} annotated frames...")

    total_frames = len(files)
    total_humans = 0
    frame_with_humans = 0
    avg_conf = []

    for f in tqdm(files, desc="Processing frames"):
        img_path = os.path.join(annotated_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Count humans (simple: count "person" text or boxes — placeholder)
        # For real count, parse YOLO results or use text detection (future)
        # For now: assume every frame has at least 1 (adjust later)
        humans_in_frame = 1  # TODO: improve
        total_humans += humans_in_frame
        if humans_in_frame > 0:
            frame_with_humans += 1

        # Placeholder confidence (future: extract from YOLO)
        avg_conf.append(0.75)

    avg_humans = total_humans / total_frames if total_frames > 0 else 0
    detection_rate = frame_with_humans / total_frames * 100

    print("\nBasic Stats:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with humans: {frame_with_humans} ({detection_rate:.1f}%)")
    print(f"  Average humans per frame: {avg_humans:.2f}")
    # Add more: avg confidence, etc.

if __name__ == '__main__':
    dir_path = 'sim_annotated_dark'
    analyze_annotated_dir(dir_path)