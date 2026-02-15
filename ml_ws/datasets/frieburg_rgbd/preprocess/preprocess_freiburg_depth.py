#!/usr/bin/env python3
"""
Freiburg RGB-D People → YOLO depth-only dataset (person detection)
Run from: datasets/frieburg_rgbd/
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# ========================= CONFIG =========================
BASE_DIR = Path("/Volumes/SamsungT7/thesis_ws_two/find_my_human_go2/ml_ws/datasets/frieburg_rgbd")
EXTRACTED = BASE_DIR / "extracted/mensa_seq0_1.1"
TRACK_DIR = EXTRACTED / "track_annotations"
DEPTH_DIR = EXTRACTED / "depth"

OUTPUT_ROOT = BASE_DIR / "preprocess"
IMG_SIZE = (640, 480)          # original size
CLIP_MAX_MM = 10000            # 10 meters - safe for indoor

# Create clean output structure
for split in ["train", "val"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# ========================= PARSE ANNOTATIONS =========================
print("Parsing all Track_*.txt files...")

image_to_boxes = {}   # image_stem → list of (x, y, w, h) in pixels

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

            # Clamp to image bounds
            x = max(0, min(x, IMG_SIZE[0]-1))
            y = max(0, min(y, IMG_SIZE[1]-1))
            w = min(w, IMG_SIZE[0] - x)
            h = min(h, IMG_SIZE[1] - y)

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

    # 2. Clip + normalize to 0-255
    depth = np.clip(depth, 0, CLIP_MAX_MM).astype(np.float32)
    depth = (depth / CLIP_MAX_MM * 255).astype(np.uint8)

    # 3. Make 3-channel
    depth_3ch = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

    # 4. Save PNG
    png_path = OUTPUT_ROOT / "images" / split / f"{img_stem}.png"
    cv2.imwrite(str(png_path), depth_3ch)

    # 5. Create YOLO label
    label_lines = []
    for x, y, w, h in image_to_boxes[img_stem]:
        cx = (x + w / 2) / IMG_SIZE[0]
        cy = (y + h / 2) / IMG_SIZE[1]
        bw = w / IMG_SIZE[0]
        bh = h / IMG_SIZE[1]
        label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

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