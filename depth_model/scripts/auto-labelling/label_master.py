# label_master.py - Launch script for auto-labeling
import os
import yaml
from label_yolo_processor import process_video
from ultralytics import YOLO

def main():
    config = yaml.safe_load(open("labeling_config.yaml"))

    raw_data_root = config["raw_data_root"]
    session = config["session_name"]
    labels_root = config["processed_labels_root"]

    # Input video
    video_path = os.path.join(raw_data_root, session, "rgb_video.mp4")
    if not os.path.isfile(video_path):
        print(f"Error: Video not found: {video_path}")
        return

    # Output directories
    session_label_dir = os.path.join(labels_root, session)
    label_txt_dir = os.path.join(session_label_dir, "label_txt")  # NEW: labels go here
    rgb_vis_dir = os.path.join(session_label_dir, config["rgb_vis_subfolder"]) if config["save_visualizations"] else None
    rgb_video_vis_dir = os.path.join(session_label_dir, config["labeled_video_subfolder"]) if config.get("create_labeled_video", False) else None

    # Labeled output video full path
    labeled_video_path = None
    if config.get("create_labeled_video", False):
        os.makedirs(rgb_video_vis_dir, exist_ok=True)
        labeled_video_path = os.path.join(rgb_video_vis_dir, config["labeled_video_name"])

    print(f"Labeling session: {session}")
    print(f"Input video: {video_path}")
    print(f"Labels directory: {label_txt_dir}")
    if rgb_vis_dir:
        print(f"RGB visualizations: {rgb_vis_dir}")
    if labeled_video_path:
        print(f"Labeled RGB video: {labeled_video_path}")

    # Load YOLO model
    model = YOLO(config["yolo_model"])
    print(f"Loaded model: {config['yolo_model']}")

    # Run processing
    num_frames = process_video(
        video_path=video_path,
        output_label_dir=label_txt_dir,                # ← changed to label_txt
        vis_dir=rgb_vis_dir,                           # ← now rgb_vis
        labeled_video_path=labeled_video_path,
        model=model,
        conf_thres=config["confidence_threshold"],
        iou_thres=config["iou_threshold"],
        target_classes=config["classes"],
        create_labeled_video=config.get("create_labeled_video", False)
    )

    print(f"Done. {num_frames} label files saved in: {label_txt_dir}")
    if rgb_vis_dir:
        print(f"RGB visualizations saved in: {rgb_vis_dir}")
    if labeled_video_path:
        print(f"Labeled RGB video created: {labeled_video_path}")

if __name__ == "__main__":
    main()