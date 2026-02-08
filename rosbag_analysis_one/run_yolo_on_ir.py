#!/usr/bin/env python3
"""
run_yolo_on_ir.py - Run YOLOv8-pose on extracted IR frames to simulate node output
"""

import os
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def run_yolo_on_ir(ir_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith('.jpg')])
    pbar = tqdm(ir_files, desc="Running YOLO on IR frames", unit="frame")

    for f in pbar:
        img_path = os.path.join(ir_dir, f)
        cv_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Add error handling for failed image reads
        if cv_img is None:
            pbar.write(f"Warning: Failed to read {img_path}, skipping...")
            continue
        
        cv_img_color = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

        results = model.track(cv_img_color, persist=True, conf=0.5, verbose=False)
        # Draw results (boxes, skeletons) on the image
        annotated_img = results[0].plot()

        output_path = os.path.join(output_dir, f"annotated_{f}")
        cv2.imwrite(output_path, annotated_img)

    print(f"\nDone! Annotated frames saved to: {output_dir}")

if __name__ == '__main__':
    ir_dir = 'extracted_ir_dark'  # from previous extraction
    model_path = 'yolov8n.pt'
    output_dir = 'sim_annotated_dark'
    run_yolo_on_ir(ir_dir, model_path, output_dir)