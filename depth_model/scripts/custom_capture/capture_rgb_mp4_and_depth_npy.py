import datetime
import os
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm


def npy_to_jpg_conversion(depth_npy_dir, depth_jpg_dir, min_depth_mm=300, max_depth_mm=5000):
    """
    Convert all .npy depth files to 8-bit 3-channel jpg images for YOLO training.
    - Clips to min/max depth
    - Normalizes to 0-255
    - Duplicates to 3 channels
    """
    npy_files = sorted([f for f in os.listdir(depth_npy_dir) if f.endswith('.npy')])
    if not npy_files:
        print("No .npy files found to convert.")
        return

    os.makedirs(depth_jpg_dir, exist_ok=True)
    print(f"\nConverting {len(npy_files)} depth .npy files to .jpg...")

    for fname in tqdm(npy_files, desc="Converting depth to jpg", unit="file"):
        npy_path = os.path.join(depth_npy_dir, fname)
        jpg_path = os.path.join(depth_jpg_dir, fname.replace('.npy', '.jpg'))

        depth = np.load(npy_path).astype(np.float32)  # shape: (H, W)

        # Clip to realistic range (in mm)
        depth = np.clip(depth, min_depth_mm, max_depth_mm)

        # Normalize to 0-255
        depth_norm = (depth - min_depth_mm) / (max_depth_mm - min_depth_mm + 1e-8)
        depth_8bit = (depth_norm * 255).astype(np.uint8)

        # Duplicate to 3 channels (YOLO expects 3-channel input)
        depth_3ch = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(jpg_path, depth_3ch)

    print(f"Conversion finished! {len(npy_files)} jpg files saved to:\n  {depth_jpg_dir}")


def main():
    # === 1. Start pipeline ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    # Get actual resolution
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    width = color_profile.width()
    height = color_profile.height()
    print(f"Camera started with resolution: {width} x {height}")

    # === 2. Session folder ===
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    session_name = f"{timestamp}_session01"
    session_dir = os.path.join("../../../raw_data", session_name)
    video_path = os.path.join(session_dir, "rgb_video.mp4")
    depth_npy_dir = os.path.join(session_dir, "depth_npy")
    depth_jpg_dir = os.path.join(session_dir, "depth_jpg")   # new folder for converted images

    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(depth_npy_dir, exist_ok=True)
    print(f"Session folder created: {session_dir}")

    # === 3. Video writer ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    # === 4. Intrinsics ===
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

    # === 5. Recording loop ===
    frame_idx = 0
    recording = False
    prev_time = time.time()
    fps_counter = 0
    displayed_fps = 0.0

    print("\nControls:")
    print("   S   → Start / Pause recording")
    print("   Q   → Quit and start conversion")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Live FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                displayed_fps = fps_counter
                fps_counter = 0
                prev_time = current_time

            display_img = rgb.copy()
            cv2.putText(display_img, f"FPS: {displayed_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            status = "RECORDING" if recording else "PAUSED"
            cv2.putText(display_img, status, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if recording else (255, 255, 0), 2)

            cv2.imshow("Live RGB - S: Start/Pause | Q: Quit", display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break

            elif key == ord('s') or key == ord('S'):
                recording = not recording
                print(f"Recording {'STARTED' if recording else 'PAUSED'}")

            if recording:
                frame_idx += 1
                video_writer.write(rgb)

                npy_path = os.path.join(depth_npy_dir, f"frame_{frame_idx:05d}.npy")
                np.save(npy_path, depth)

                if frame_idx % 30 == 0:
                    print(f"  Saved frame {frame_idx:05d}")

    finally:
        video_writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()

        print("\n" + "="*80)
        print("Recording finished!")
        print(f"Session       : {session_dir}")
        print(f"RGB video     : rgb_video.mp4")
        print(f"Depth npy     : depth_npy/  ({frame_idx} frames)")
        print(f"Live FPS      : {displayed_fps:.1f}")
        print("="*80)

        # === 6. Convert .npy to .jpg for YOLO training ===
        if frame_idx > 0:
            npy_to_jpg_conversion(depth_npy_dir, depth_jpg_dir)
        else:
            print("No frames recorded → skipping conversion.")


if __name__ == "__main__":
    main()
