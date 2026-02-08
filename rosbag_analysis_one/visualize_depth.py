#!/usr/bin/env python3
"""
visualize_depth.py - Visualize depth PNGs as color heatmaps
Saves heatmaps to disk
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def visualize_depth(depth_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    pbar = tqdm(depth_files, desc="Visualizing depth frames", unit="frame")

    for f in pbar:
        depth_path = os.path.join(depth_dir, f)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit

        if depth_img is None:
            pbar.write(f"Warning: Skipping {f}")
            continue

        # Normalize depth for visualization (0–65535 to 0–255, ignore 0)
        valid = depth_img > 0
        if np.any(valid):
            min_d = np.min(depth_img[valid])
            max_d = np.min([np.max(depth_img[valid]), 5000])  # clip to 5m
            normalized = (depth_img - min_d) / (max_d - min_d) * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        else:
            heatmap = np.zeros_like(depth_img, dtype=np.uint8)

        output_path = os.path.join(output_dir, f"heatmap_{f.replace('.png', '.jpg')}")
        cv2.imwrite(output_path, heatmap)

    print(f"\nDone! Heatmaps saved to: {output_dir}")

if __name__ == '__main__':
    depth_dir = 'extracted_depth_dark'
    output_dir = 'depth_heatmaps_dark'
    visualize_depth(depth_dir, output_dir)