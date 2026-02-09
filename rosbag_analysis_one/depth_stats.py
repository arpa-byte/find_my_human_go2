#!/usr/bin/env python3
"""
depth_stats.py - Basic depth statistics from extracted depth PNGs
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def depth_stats(depth_dir):
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    print(f"Processing {len(depth_files)} depth frames...")

    avg_depths = []
    valid_frames = 0

    pbar = tqdm(depth_files, desc="Computing depth stats", unit="frame")

    for f in pbar:
        depth_path = os.path.join(depth_dir, f)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit

        if depth_img is None:
            continue

        valid = depth_img > 0
        if np.any(valid):
            mean_d = np.mean(depth_img[valid])
            avg_depths.append(mean_d)
            valid_frames += 1

    if valid_frames == 0:
        print("No valid depth data.")
        return

    overall_avg = np.mean(avg_depths)
    overall_std = np.std(avg_depths)

    print("\nDepth Stats:")
    print(f"  Valid frames: {valid_frames}/{len(depth_files)}")
    print(f"  Average depth (mm): {overall_avg:.1f}")
    print(f"  Std dev of average depth: {overall_std:.1f}")
    print(f"  Min average depth: {np.min(avg_depths):.1f} mm")
    print(f"  Max average depth: {np.max(avg_depths):.1f} mm")

if __name__ == '__main__':
    depth_dir = 'extracted_depth_dark'
    depth_stats(depth_dir)