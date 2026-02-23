import cv2
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(".")
VAL_IMAGES_DIR = BASE_DIR / "images" / "val"
VAL_LABELS_DIR = BASE_DIR / "labels" / "val"
VIS_OUTPUT_DIR = BASE_DIR / "visualized_val"

VIS_OUTPUT_DIR.mkdir(exist_ok=True)

def draw_gt_boxes(image_path, label_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    h, w, _ = img.shape

    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, bw, bh = map(float, parts)
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, 'person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(str(output_path), img)

image_files = sorted(VAL_IMAGES_DIR.glob("*.png"))

for img_path in tqdm(image_files, desc="Visualizing val"):
    stem = img_path.stem
    label_path = VAL_LABELS_DIR / f"{stem}.txt"
    output_path = VIS_OUTPUT_DIR / f"vis_{stem}.png"
    draw_gt_boxes(img_path, label_path, output_path)

print(f"Done. Check visualized_val/ folder.")