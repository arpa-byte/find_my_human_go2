import datetime
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_depth_preprocessing(depth_raw, config, use_clahe=True):
    depth = depth_raw.astype(np.float32)
    depth = np.clip(depth, 300, 5000)

    if use_clahe and config.get('clahe', {}).get('enabled', False):
        clahe_cfg = config['clahe']
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_8bit = (depth_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_cfg.get('clip_limit', 2.0),
            tileGridSize=tuple(clahe_cfg.get('tile_grid_size', [8, 8]))
        )
        depth_enhanced = clahe.apply(depth_8bit)
        depth_processed = depth_enhanced.astype(np.float32) / 255.0 * 5000
    else:
        depth_processed = (depth - 300) / (5000 - 300 + 1e-8) * 255
        depth_processed = np.clip(depth_processed, 0, 255).astype(np.uint8)

    return depth_processed


def npy_to_jpg_conversion(depth_npy_dir, depth_jpg_dir, preprocess_config, use_clahe):
    npy_files = sorted([f for f in os.listdir(depth_npy_dir) if f.endswith('.npy')])
    if not npy_files:
        print("No .npy files found to convert.")
        return

    os.makedirs(depth_jpg_dir, exist_ok=True)
    print(f"\nConverting {len(npy_files)} depth .npy files to .jpg... (CLAHE: {'ON' if use_clahe else 'OFF'})")

    for fname in tqdm(npy_files, desc="Converting depth to jpg", unit="file"):
        npy_path = os.path.join(depth_npy_dir, fname)
        jpg_path = os.path.join(depth_jpg_dir, fname.replace('.npy', '.jpg'))

        depth_raw = np.load(npy_path)
        depth_processed = apply_depth_preprocessing(depth_raw, preprocess_config, use_clahe)

        depth_3ch = cv2.cvtColor(depth_processed.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(jpg_path, depth_3ch)

    print(f"Conversion done! {len(npy_files)} jpgs in: {depth_jpg_dir}")


def main():
    # Load configs
    capture_cfg = load_yaml("capture_config.yaml")
    preprocess_cfg = load_yaml("depth_preprocess.yaml")

    countdown_sec = capture_cfg.get('countdown_seconds', 5.0)
    record_sec = capture_cfg.get('record_seconds', 10.0)

    # === 1. Pipeline ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, capture_cfg["width"], capture_cfg["height"], rs.format.z16, capture_cfg["fps"])
    config.enable_stream(rs.stream.color, capture_cfg["width"], capture_cfg["height"], rs.format.bgr8, capture_cfg["fps"])

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    width = color_profile.width()
    height = color_profile.height()
    print(f"Camera started: {width} x {height} @ ~{capture_cfg['fps']} fps (preview mode)")

    # === 2. Variables (session only created when recording actually starts) ===
    session_dir = None
    video_writer = None
    depth_npy_dir = None
    depth_jpg_dir = None
    frame_idx = 0
    recording = False
    countdown_active = False
    countdown_start = 0.0
    record_start = 0.0
    prev_time = time.time()
    fps_counter = 0
    displayed_fps = 0.0

    # View mode: 0 = RGB (default), 1 = Depth
    view_mode = 0
    view_names = ["RGB (default)", "Depth (preprocessed)"]

    # CLAHE toggle (starts as per config)
    clahe_enabled = preprocess_cfg.get('clahe', {}).get('enabled', False)

    print("\nControls:")
    print(f"   S   → Start countdown → record for {record_sec}s (creates folder when recording begins)")
    print("   Q   → Quit (no folder if never started recording)")
    print("   R   → RGB view")
    print("   D   → Depth view (with preprocessing)")
    print("   C   → Toggle CLAHE preprocessing (only affects Depth view & final jpg)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Live FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                displayed_fps = fps_counter
                fps_counter = 0
                prev_time = current_time

            # Handle countdown
            if countdown_active:
                elapsed = current_time - countdown_start
                remaining = max(0, countdown_sec - elapsed)
                countdown_text = f"Countdown: {int(remaining)+1:.0f}"
                if remaining <= 0:
                    countdown_active = False
                    recording = True
                    record_start = current_time
                    print("Countdown finished → Recording STARTED")
                    # Create folder + writer now
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
                    session_name = f"{timestamp}_{capture_cfg.get('session_suffix', 'session01')}"
                    session_dir = os.path.join("../../../raw_data", session_name)
                    video_path = os.path.join(session_dir, "rgb_video.mp4")
                    depth_npy_dir = os.path.join(session_dir, "depth_npy")
                    depth_jpg_dir = os.path.join(session_dir, "depth_jpg")

                    os.makedirs(session_dir, exist_ok=True)
                    os.makedirs(depth_npy_dir, exist_ok=True)
                    print(f"Session folder created: {session_dir}")

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, capture_cfg["fps"], (width, height))

                    intr = color_profile.get_intrinsics()
                    camera_params = {
                        'fx': intr.fx, 'fy': intr.fy,
                        'ppx': intr.ppx, 'ppy': intr.ppy,
                        'width': width, 'height': height,
                        'depth_scale': depth_scale
                    }
                    with open(os.path.join(session_dir, 'intrinsics.json'), 'w') as f:
                        import json
                        json.dump(camera_params, f, indent=2)

            # Auto-stop after record_seconds
            if recording:
                elapsed_record = current_time - record_start
                if elapsed_record >= record_sec:
                    recording = False
                    print(f"Recording duration reached ({record_sec}s) → Recording STOPPED")

            # Prepare display
            if view_mode == 0:  # RGB
                display_img = rgb.copy()
            else:  # Depth
                depth_processed = apply_depth_preprocessing(depth_raw, preprocess_cfg, clahe_enabled)
                display_img = cv2.cvtColor(depth_processed.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Overlay
            cv2.putText(display_img, f"FPS: {displayed_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, f"View: {view_names[view_mode]}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            clahe_status = f"CLAHE: {'ON' if clahe_enabled else 'OFF'}"
            cv2.putText(display_img, clahe_status, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255) if clahe_enabled else (128, 128, 128), 2)

            if countdown_active:
                cv2.putText(display_img, countdown_text, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            else:
                status = "RECORDING" if recording else "PAUSED"
                cv2.putText(display_img, status, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if recording else (255, 255, 0), 2)

            cv2.imshow("Live View - S: Rec | Q: Quit | R/D: View | C: CLAHE", display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            elif key == ord('s') or key == ord('S'):
                if countdown_active:
                    countdown_active = False
                    recording = False
                    print("Countdown cancelled")
                elif recording:
                    recording = False
                    print("Recording manually PAUSED")
                else:
                    countdown_active = True
                    countdown_start = time.time()
                    print(f"Countdown started ({countdown_sec}s) → recording will begin after")

            elif key == ord('r') or key == ord('R'):
                view_mode = 0
                print("Switched to RGB view")
            elif key == ord('d') or key == ord('D'):
                view_mode = 1
                print("Switched to Depth view")
            elif key == ord('c') or key == ord('C'):
                clahe_enabled = not clahe_enabled
                print(f"CLAHE preprocessing {'ENABLED' if clahe_enabled else 'DISABLED'}")

            if recording:
                frame_idx += 1
                video_writer.write(rgb)

                npy_path = os.path.join(depth_npy_dir, f"frame_{frame_idx:05d}.npy")
                np.save(npy_path, depth_raw)

                if frame_idx % 30 == 0:
                    print(f"  Saved frame {frame_idx:05d}")

    finally:
        if video_writer is not None:
            video_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()

        if session_dir is not None and frame_idx > 0:
            print("\n" + "="*80)
            print("Recording finished!")
            print(f"Session       : {session_dir}")
            print(f"RGB video     : rgb_video.mp4")
            print(f"Depth npy     : depth_npy/  ({frame_idx} frames)")
            print(f"Live FPS      : {displayed_fps:.1f}")
            print(f"Final CLAHE state for conversion: {'ON' if clahe_enabled else 'OFF'}")
            print("="*80)

            npy_to_jpg_conversion(depth_npy_dir, depth_jpg_dir, preprocess_cfg, clahe_enabled)
        else:
            print("\nNo recording completed → no files saved.")

if __name__ == "__main__":
    main()