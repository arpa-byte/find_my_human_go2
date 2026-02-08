#!/usr/bin/env python3
"""
fuse_ir_depth.py - Simple fusion of annotated IR and depth
Computes average depth in detected boxes
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def fuse_ir_depth(annotated_dir, depth_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    annotated_files = sorted([f for f in os.listdir(annotated_dir) if f.startswith('annotated_ir_frame_')])
    pbar = tqdm(annotated_files, desc="Fusing IR + Depth", unit="frame")

    for f in pbar:
        annotated_path = os.path.join(annotated_dir, f)
        depth_f = f.replace('annotated_ir_frame_', 'depth_frame_').replace('.jpg', '.png')
        depth_path = os.path.join(depth_dir, depth_f)

        if not os.path.exists(depth_path):
            pbar.write(f"Warning: Depth {depth_f} missing, skipping")
            continue

        annotated_img = cv2.imread(annotated_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16-bit

        # Simple: Overlay depth heatmap on annotated IR
        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        fused = cv2.addWeighted(annotated_img, 0.7, heatmap, 0.3, 0)

        output_path = os.path.join(output_dir, f"fused_{f}")
        cv2.imwrite(output_path, fused)

        # TODO: Compute avg depth in boxes (future)

    print(f"\nDone! Fused images saved to: {output_dir}")

if __name__ == '__main__':
    annotated_dir = 'sim_annotated_dark'
    depth_dir = 'extracted_depth_dark'
    output_dir = 'fused_ir_depth_dark'
    fuse_ir_depth(annotated_dir, depth_dir, output_dir)