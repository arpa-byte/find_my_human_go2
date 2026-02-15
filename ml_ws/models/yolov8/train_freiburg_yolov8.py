#!/usr/bin/env python3
"""
Fixed YOLOv8 training script – correct separation of data.yaml and cfg.yaml
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# ─── PATHS ─────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Volumes/SamsungT7/thesis_ws_two/find_my_human_go2/ml_ws")
DATA_YAML = BASE_DIR / "datasets/frieburg_rgbd/preprocess/data.yaml"
CFG_YAML  = BASE_DIR / "models/yolov8/freiburg_train_config.yaml"

for p in [DATA_YAML, CFG_YAML]:
    if not p.is_file():
        print(f"Error: file missing → {p}", file=sys.stderr)
        sys.exit(1)

# ─── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("Starting YOLOv8 training on Freiburg depth-only dataset")
    print(f"  Data YAML: {DATA_YAML}")
    print(f"  Config YAML: {CFG_YAML}")
    print("  Device: CPU only (M2 Mac)")
    print("  Progress: per-epoch bar from Ultralytics\n")

    # Load pretrained model
    model = YOLO("yolov8n.pt")

    try:
        results = model.train(
            data=str(DATA_YAML),           # ← dataset definition
            cfg=str(CFG_YAML),             # ← hyperparameters
            device="cpu",
            exist_ok=True,
            verbose=True,
        )

        print("\nTraining completed!")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print(f"Results folder: {results.save_dir}")
        print("Check: results.csv, confusion_matrix.png, etc.")

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Latest weights saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()