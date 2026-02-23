import os
from pathlib import Path
from tqdm import tqdm
import signal
import sys

# Graceful Ctrl+C handling
def signal_handler(sig, frame):
    print("\nInterrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Configuration
BASE_DIR = Path(".")
LABEL_DIRS = [
    BASE_DIR / "labels" / "train",
    BASE_DIR / "labels" / "val"
]

# Offset values (from your test on seq0_0062_0)
CX_SHIFT = 0.08   # right shift
CY_SHIFT = -0.17  # upward shift (negative = higher in image)

def apply_offset_to_label_file(txt_path):
    try:
        lines = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, bw, bh = parts
                    cx_new = float(cx) + CX_SHIFT
                    cy_new = float(cy) + CY_SHIFT
                    # Clamp to valid 0-1 range
                    cx_new = max(0.0, min(1.0, cx_new))
                    cy_new = max(0.0, min(1.0, cy_new))
                    lines.append(f"{cls} {cx_new:.6f} {cy_new:.6f} {bw} {bh}\n")
                else:
                    # Keep invalid lines as-is
                    lines.append(line)

        with open(txt_path, 'w') as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"Error processing {txt_path}: {e}")
        return False

# Collect all label files
all_label_files = []
for label_dir in LABEL_DIRS:
    all_label_files.extend(sorted(label_dir.glob("*.txt")))

print(f"Found {len(all_label_files)} label files to update.")

# Process with progress bar
success_count = 0
for txt_path in tqdm(all_label_files, desc="Applying offset to labels"):
    if apply_offset_to_label_file(txt_path):
        success_count += 1

print(f"Done. Successfully updated {success_count}/{len(all_label_files)} label files.")
print("Backup of original labels is in 'labels_backup_...' folder.")
print("Now re-run vis_gt_train.py and vis_gt_val.py to verify.")