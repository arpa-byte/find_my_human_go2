#!/usr/bin/env python3
"""
Extract rgbd_people_unihall.tar with tqdm progress bar showing bytes extracted.
Run from inside: datasets/frieburg_rgbd
"""

import tarfile
import os
from tqdm import tqdm
import sys

# ─── Configuration ──────────────────────────────────────────────────────────────
TAR_PATH    = "rgbd_people_unihall.tar"
EXTRACT_TO  = "extracted"

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(TAR_PATH):
        print(f"Error: tar file not found → {TAR_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(EXTRACT_TO, exist_ok=True)

    print(f"Extracting {TAR_PATH}  →  {EXTRACT_TO}")

    with tarfile.open(TAR_PATH, "r") as tar:
        # Estimate total size for progress (sum of all member sizes)
        total_size = sum(m.size for m in tar.getmembers() if m.size > 0)

        extracted_bytes = 0

        with tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="Extracting",
            leave=True
        ) as pbar:
            for member in tar.getmembers():
                tar.extract(member, path=EXTRACT_TO)
                extracted_bytes += member.size
                pbar.update(member.size)

    print("\nExtraction finished.")
    print(f"Extracted to:  {os.path.abspath(EXTRACT_TO)}")

    # Quick summary
    total_files = sum(len(files) for _, _, files in os.walk(EXTRACT_TO))
    print(f"Total files extracted: {total_files}")

if __name__ == "__main__":
    main()