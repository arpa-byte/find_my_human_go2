#!/usr/bin/env python3
"""
depth_in_boxes.py - Compute average depth inside YOLO boxes (fixed for blue boxes)
Handles frame mismatch & looser box detection
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def depth_in_boxes(annotated_dir, depth_dir):
    annotated_files = sorted([f for f in os.listdir(annotated_dir) if f.startswith('annotated_ir_frame_') and f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith('depth_frame_') and f.endswith('.png')])

    print(f"Annotated IR frames: {len(annotated_files)}")
    print(f"Depth PNG frames: {len(depth_files)}")

    avg_depths = []
    frame_with_boxes = 0
    matched = 0

    pbar = tqdm(annotated_files, desc="Computing depth in boxes", unit="frame")

    for idx, f in enumerate(pbar):
        annotated_path = os.path.join(annotated_dir, f)
        # Use index-based matching (handle 1982 vs 1981)
        depth_idx = min(idx, len(depth_files) - 1)
        depth_f = depth_files[depth_idx]
        depth_path = os.path.join(depth_dir, depth_f)

        annotated_img = cv2.imread(annotated_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth_img is None or annotated_img is None:
            continue

        matched += 1

        # Blue box detection (from images: BGR [255, 0, 0] blue)
        lower_blue = np.array([180, 0, 0])
        upper_blue = np.array([255, 50, 50])
        mask = cv2.inRange(annotated_img, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_depths = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x, y = max(0, x-5), max(0, y-5)
            w, h = w+10, h+10
            roi_depth = depth_img[y:y+h, x:x+w]
            valid = roi_depth[roi_depth > 0]
            if len(valid) > 50:
                frame_depths.append(np.mean(valid))

        if frame_depths:
            avg_frame_depth = np.mean(frame_depths)
            avg_depths.append(avg_frame_depth)
            frame_with_boxes += 1
            pbar.set_postfix({'depth': f"{avg_frame_depth:.0f} mm"})

    if not avg_depths:
        print("No valid depth in boxes found after filtering.")
        return

    overall_avg = np.mean(avg_depths)
    overall_std = np.std(avg_depths)

    print("\nDepth Inside YOLO Boxes:")
    print(f"  Matched frames: {matched}/{len(annotated_files)}")
    print(f"  Frames with valid boxes & depth: {frame_with_boxes}")
    print(f"  Average depth inside boxes (mm): {overall_avg:.1f}")
    print(f"  Std dev: {overall_std:.1f}")
    print(f"  Min depth in boxes: {np.min(avg_depths):.1f} mm")
    print(f"  Max depth in boxes: {np.max(avg_depths):.1f} mm")

if __name__ == '__main__':
    annotated_dir = 'sim_annotated_dark'
    depth_dir = 'extracted_depth_dark'
    depth_in_boxes(annotated_dir, depth_dir)