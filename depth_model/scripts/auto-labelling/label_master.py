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
    vis_dir = os.path.join(session_label_dir, config["vis_subfolder"]) if config["save_visualizations"] else None

    # Labeled output video
    labeled_video_path = None
    if config.get("create_labeled_video", False):
        labeled_video_path = os.path.join(session_label_dir, config["labeled_video_name"])

    print(f"Labeling session: {session}")
    print(f"Input video: {video_path}")
    print(f"Output labels: {session_label_dir}")
    if vis_dir:
        print(f"Visualizations: {vis_dir}")
    if labeled_video_path:
        print(f"Labeled video: {labeled_video_path}")

    # Load YOLO model
    model = YOLO(config["yolo_model"])
    print(f"Loaded model: {config['yolo_model']}")

    # Run processing
    num_frames = process_video(
        video_path=video_path,
        output_label_dir=session_label_dir,
        vis_dir=vis_dir,
        labeled_video_path=labeled_video_path,
        model=model,
        conf_thres=config["confidence_threshold"],
        iou_thres=config["iou_threshold"],
        target_classes=config["classes"],
        create_labeled_video=config.get("create_labeled_video", False)
    )

    print(f"Done. {num_frames} label files saved in: {session_label_dir}")
    if vis_dir:
        print(f"Visualizations saved in: {vis_dir}")
    if labeled_video_path:
        print(f"Labeled video created: {labeled_video_path}")


if __name__ == "__main__":
    main()