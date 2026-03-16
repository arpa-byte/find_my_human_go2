# label_yolo_processor.py
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def process_video(video_path, output_label_dir, vis_dir, labeled_video_path, model, conf_thres, iou_thres, target_classes, create_labeled_video=False):
    """
    Run YOLO on every frame of the video and save labels + visualizations.
    Optionally create a full labeled output video.
    Returns number of frames processed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video: {video_path} ({total_frames} frames @ {fps:.1f} fps)")

    frame_idx = 1
    os.makedirs(output_label_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # Prepare labeled video writer (if enabled)
    video_writer = None
    if create_labeled_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(labeled_video_path, fourcc, fps, (width, height))
        print(f"Creating labeled output video: {labeled_video_path}")

    pbar = tqdm(total=total_frames, desc="Labeling frames", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO
        results = model(frame, conf=conf_thres, iou=iou_thres, classes=target_classes, verbose=False)

        # Prepare YOLO txt label
        label_lines = []
        vis_frame = frame.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # YOLO format: class x_center y_center width height (normalized)
                img_h, img_w = frame.shape[:2]
                xc = (x1 + x2) / 2 / img_w
                yc = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                # Draw on vis frame
                if vis_dir or create_labeled_video:
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"{conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save label file
        label_path = os.path.join(output_label_dir, f"frame_{frame_idx:05d}.txt")
        with open(label_path, 'w') as f:
            f.write("\n".join(label_lines))

        # Save single visualization frame
        if vis_dir:
            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(vis_path, vis_frame)

        # Write to labeled video
        if create_labeled_video:
            video_writer.write(vis_frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if video_writer is not None:
        video_writer.release()

    print(f"Labeling complete: {frame_idx-1} frames processed")
    return frame_idx - 1