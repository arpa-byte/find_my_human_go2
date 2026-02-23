#!/usr/bin/env python3
"""
Freiburg RGB-D People → YOLO depth-only dataset (person detection)
With 90° anticlockwise rotation + hole filling & noise reduction
Run from: datasets/frieburg_rgbd/preprocess/
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ========================= CONFIG =========================
BASE_DIR = Path("/Volumes/SamsungT7/thesis_ws_two/find_my_human_go2/ml_ws/datasets/frieburg_rgbd")
EXTRACTED = BASE_DIR / "extracted/mensa_seq0_1.1"
TRACK_DIR = EXTRACTED / "track_annotations"
DEPTH_DIR = EXTRACTED / "depth"

OUTPUT_ROOT = BASE_DIR / "preprocess"
ORIGINAL_SIZE = (640, 480)     # Original: width 640, height 480 (landscape)
ROTATED_SIZE = (480, 640)      # After 90° anticlockwise: width 480, height 640 (portrait)
CLIP_MAX_MM = 10000            # 10 meters - safe for indoor

# Create output structure if missing (no deletion)
for split in ["train", "val"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ========================= PARSE ANNOTATIONS =========================
print("Parsing all Track_*.txt files...")

image_to_boxes = {}   # image_stem → list of (x, y, w, h) in pixels (pre-rotation)

for track_file in sorted(TRACK_DIR.glob("Track_*.txt")):
    with open(track_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 11:
                continue

            img_name = parts[0]                    # e.g. seq0_0000_0
            try:
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                vis = int(parts[10])
            except:
                continue

            # Skip invalid boxes
            if x < 0 or y < 0 or w <= 0 or h <= 0 or vis == 0:
                continue

            # Clamp to original image bounds
            x = max(0, min(x, ORIGINAL_SIZE[0]-1))
            y = max(0, min(y, ORIGINAL_SIZE[1]-1))
            w = min(w, ORIGINAL_SIZE[0] - x)
            h = min(h, ORIGINAL_SIZE[1] - y)

            if img_name not in image_to_boxes:
                image_to_boxes[img_name] = []
            image_to_boxes[img_name].append((x, y, w, h))

print(f"Found {len(image_to_boxes)} unique images with valid person annotations")

# ========================= CONVERT DEPTH + CREATE LABELS =========================
all_images = sorted(image_to_boxes.keys())                     # sequential order
split_idx = int(len(all_images) * 0.8)                         # 80/20 split
train_images = all_images[:split_idx]
val_images   = all_images[split_idx:]

print(f"Train images: {len(train_images)} | Val images: {len(val_images)}")

def process_image(img_stem: str, split: str):
    # 1. Load 16-bit PGM
    pgm_path = DEPTH_DIR / f"{img_stem}.pgm"
    depth = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED)   # 16-bit

    if depth is None:
        return False

    # 2. Rotate 90° anticlockwise (original 640x480 → 480x640)
    depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 3. Clip + normalize to 0-255
    depth = np.clip(depth, 0, CLIP_MAX_MM).astype(np.float32)
    depth = (depth / CLIP_MAX_MM * 255).astype(np.uint8)

    # ─── NEW: Preprocessing ───────────────────────────────────────────────
    # Identify holes (pixels == 0)
    mask = (depth == 0).astype(np.uint8) * 255

    # Fill holes using inpainting (TELEA - good for smooth depth)
    depth_filled = cv2.inpaint(depth, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Noise reduction: bilateral filter (preserves edges, removes speckle-like noise)
    depth_clean = cv2.bilateralFilter(depth_filled, d=9, sigmaColor=75, sigmaSpace=75)

    # Use cleaned depth for final input
    depth_final = depth_clean
    # ──────────────────────────────────────────────────────────────────────

    # 4. Make 3-channel
    depth_3ch = cv2.cvtColor(depth_final, cv2.COLOR_GRAY2RGB)

    # 5. Save PNG
    png_path = OUTPUT_ROOT / "images" / split / f"{img_stem}.png"
    cv2.imwrite(str(png_path), depth_3ch)

    # 6. Transform and create YOLO label (adjust for rotation)
    label_lines = []
    for x, y, w, h in image_to_boxes[img_stem]:
        # Transform box for 90° anticlockwise rotation
        new_cx = y / ORIGINAL_SIZE[1]          # Original cy → new cx
        new_cy = 1 - (x / ORIGINAL_SIZE[0])    # 1 - original cx → new cy (flip)
        new_bw = h / ORIGINAL_SIZE[1]          # Original bh → new bw
        new_bh = w / ORIGINAL_SIZE[0]          # Original bw → new bh

        label_lines.append(f"0 {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

    txt_path = OUTPUT_ROOT / "labels" / split / f"{img_stem}.txt"
    txt_path.write_text("\n".join(label_lines) + "\n")

    return True

# ========================= RUN CONVERSION =========================
print("Converting depth images and creating labels...")

for split_name, image_list in [("train", train_images), ("val", val_images)]:
    for stem in tqdm(image_list, desc=f"Processing {split_name}"):
        process_image(stem, split_name)

# ========================= CREATE data.yaml =========================
data_yaml = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

nc: 1
names: ['person']
"""

(OUTPUT_ROOT / "data.yaml").write_text(data_yaml)

print("\n=== PREPROCESSING COMPLETE ===")
print(f"Dataset created at: {OUTPUT_ROOT}")
print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")
print("Ready for training!")