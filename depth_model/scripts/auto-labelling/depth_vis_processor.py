# depth_vis_processor.py - Process depth images: copy + draw boxes from labels
import os
import cv2
from tqdm import tqdm

def read_yolo_label(label_path):
    """Read YOLO .txt file and return list of boxes: [class, x_center, y_center, w, h] normalized"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(p) for p in parts])
    return boxes

def draw_boxes_on_image(image, boxes, img_w, img_h, color=(0, 255, 0), thickness=2):
    for box in boxes:
        cls, xc, yc, w, h = box
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        # Optional: label class/conf (here just class 0 = person)
        cv2.putText(image, "person", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return image

def process_depth_visualization(raw_depth_jpg_dir, label_txt_dir, depth_copy_dir, depth_vis_dir, vis_color, vis_thickness):
    """
    1. Copy depth jpgs to depth_copy/
    2. Draw same bounding boxes from label_txt/ onto copies → save in depth_vis/
    """
    jpg_files = sorted([f for f in os.listdir(raw_depth_jpg_dir) if f.endswith('.jpg')])
    if not jpg_files:
        print("No depth jpg files found.")
        return

    os.makedirs(depth_copy_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    print(f"\nCopying and visualizing {len(jpg_files)} depth images...")

    pbar = tqdm(jpg_files, desc="Processing depth images", unit="image")

    for fname in pbar:
        # 1. Copy original depth jpg
        src_jpg = os.path.join(raw_depth_jpg_dir, fname)
        copy_jpg = os.path.join(depth_copy_dir, fname)
        img = cv2.imread(src_jpg)
        if img is None:
            print(f"Failed to read {src_jpg}")
            continue
        cv2.imwrite(copy_jpg, img)

        # 2. Load corresponding label
        base = fname.replace('.jpg', '')
        label_path = os.path.join(label_txt_dir, f"{base}.txt")
        boxes = read_yolo_label(label_path)

        if boxes:
            # Draw boxes
            img_vis = img.copy()
            h, w = img.shape[:2]
            img_vis = draw_boxes_on_image(img_vis, boxes, w, h, vis_color, vis_thickness)

            # Save visualized depth
            vis_path = os.path.join(depth_vis_dir, fname)
            cv2.imwrite(vis_path, img_vis)

        pbar.update(1)

    print(f"Depth processing complete.")
    print(f"Depth copies: {depth_copy_dir}")
    print(f"Depth visualizations: {depth_vis_dir}")