# label_master.py - Launch script for auto-labeling (RGB + Depth vis)
import os
import yaml
from label_yolo_processor import process_video as process_rgb_video
from depth_vis_processor import process_depth_visualization
from ultralytics import YOLO

def main():
    config = yaml.safe_load(open("labeling_config.yaml"))

    raw_data_root = config["raw_data_root"]
    session = config["session_name"]
    labels_root = config["processed_labels_root"]

    # === Input paths ===
    video_path = os.path.join(raw_data_root, session, "rgb_video.mp4")
    raw_depth_jpg_dir = os.path.join(raw_data_root, session, "depth_jpg")

    if not os.path.isfile(video_path):
        print(f"Error: RGB video not found: {video_path}")
        return
    if not os.path.isdir(raw_depth_jpg_dir):
        print(f"Error: Depth jpg dir not found: {raw_depth_jpg_dir}")
        return

    # === Output base ===
    session_label_dir = os.path.join(labels_root, session)
    label_txt_dir = os.path.join(session_label_dir, "label_txt")
    rgb_vis_dir = os.path.join(session_label_dir, config["rgb_vis_subfolder"]) if config["save_visualizations"] else None
    rgb_video_vis_dir = os.path.join(session_label_dir, config["labeled_video_subfolder"]) if config.get("create_labeled_video", False) else None

    # Depth output dirs
    depth_copy_dir = os.path.join(session_label_dir, config["depth_copy_subfolder"]) if config.get("save_depth_visualizations", False) else None
    depth_vis_dir = os.path.join(session_label_dir, config["depth_vis_subfolder"]) if config.get("save_depth_visualizations", False) else None

    print(f"Labeling session: {session}")
    print(f"Input RGB video: {video_path}")
    print(f"Input depth jpgs: {raw_depth_jpg_dir}")
    print(f"Labels directory: {label_txt_dir}")
    if rgb_vis_dir:
        print(f"RGB visualizations: {rgb_vis_dir}")
    if rgb_video_vis_dir:
        print(f"Labeled RGB video: {os.path.join(rgb_video_vis_dir, config['labeled_video_name'])}")
    if depth_copy_dir:
        print(f"Depth copy: {depth_copy_dir}")
    if depth_vis_dir:
        print(f"Depth visualizations: {depth_vis_dir}")

    # === 1. RGB Labeling ===
    model = YOLO(config["yolo_model"])
    print(f"Loaded model: {config['yolo_model']}")

    process_rgb_video(
        video_path=video_path,
        output_label_dir=label_txt_dir,
        vis_dir=rgb_vis_dir,
        labeled_video_path=os.path.join(rgb_video_vis_dir, config["labeled_video_name"]) if rgb_video_vis_dir else None,
        model=model,
        conf_thres=config["confidence_threshold"],
        iou_thres=config["iou_threshold"],
        target_classes=config["classes"],
        create_labeled_video=config.get("create_labeled_video", False)
    )

    # === 2. Depth Visualization (copy + draw same boxes) ===
    if config.get("save_depth_visualizations", False):
        process_depth_visualization(
            raw_depth_jpg_dir=raw_depth_jpg_dir,
            label_txt_dir=label_txt_dir,
            depth_copy_dir=depth_copy_dir,
            depth_vis_dir=depth_vis_dir,
            vis_color=config["vis_color"],
            vis_thickness=config["vis_thickness"]
        )

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()