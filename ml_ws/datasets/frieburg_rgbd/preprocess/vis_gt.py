import cv2
from pathlib import Path
from tqdm import tqdm
import os
import signal
import sys

# Graceful Ctrl+C handling
def signal_handler(sig, frame):
    print("\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

BASE_DIR = Path("/Volumes/SamsungT7/thesis_ws_two/find_my_human_go2/ml_ws/datasets/frieburg_rgbd/preprocess")
VAL_IMAGES_DIR = BASE_DIR / "images" / "val"
VAL_LABELS_DIR = BASE_DIR / "labels" / "val"
VIS_OUTPUT_DIR = BASE_DIR / "visualized_val"

VIS_OUTPUT_DIR.mkdir(exist_ok=True)

def draw_gt_boxes(image_path, label_path, output_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to read image: {image_path}")
            return False

        h, w, _ = img.shape

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cls, cx, cy, bw, bh = map(float, parts)
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, 'person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except ValueError:
                            print(f"Invalid label line in {label_path}: {line.strip()}")

        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

# Get list of images
image_files = sorted(list(VAL_IMAGES_DIR.glob("*.png")))

print(f"Found {len(image_files)} images in {VAL_IMAGES_DIR}")

# Process with progress bar
success_count = 0
for img_path in tqdm(image_files, desc="Visualizing validation images"):
    stem = img_path.stem
    label_path = VAL_LABELS_DIR / f"{stem}.txt"
    output_path = VIS_OUTPUT_DIR / f"vis_{stem}.png"
    if draw_gt_boxes(img_path, label_path, output_path):
        success_count += 1

print(f"Done. Successfully visualized {success_count}/{len(image_files)} images in {VIS_OUTPUT_DIR}")
print("Open the 'visualized_val' folder in Finder to inspect.")