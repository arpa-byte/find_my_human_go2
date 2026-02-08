#!/usr/bin/env python3
"""
plot_stats.py - Visualize basic stats from annotated frames
Now with tqdm progress bar inside the loop
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def count_humans_in_frame(img_path):
    """
    Very simple heuristic: count green bounding boxes (BGR: [0,255,0])
    Returns approximate number of humans (boxes) in the annotated image
    """
    img = cv2.imread(img_path)
    if img is None:
        return 0

    # Green box color range (from YOLO plot defaults)
    lower_green = np.array([0, 200, 0])
    upper_green = np.array([50, 255, 50])

    mask = cv2.inRange(img, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise contours (area > 500 pixels)
    human_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
    return human_count

def generate_plots(annotated_dir, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(annotated_dir) if f.endswith('.jpg')])
    if not files:
        print("No annotated JPGs found.")
        return

    print(f"Analyzing {len(files)} frames...")

    frame_numbers = []
    human_counts = []

    # Main loop with tqdm progress bar
    pbar = tqdm(files, desc="Counting humans per frame", unit="frame", mininterval=1.0)

    for idx, f in enumerate(pbar):
        img_path = os.path.join(annotated_dir, f)
        count = count_humans_in_frame(img_path)

        frame_numbers.append(idx)
        human_counts.append(count)

        # Optional: update tqdm description with current count (nice feedback)
        pbar.set_postfix({'current': count})

    # Plot 1: Humans per frame over time
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, human_counts, color='blue', linewidth=1.5)
    plt.title('Number of Detected Humans per Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Human Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'humans_per_frame.png'), dpi=150)
    plt.close()

    # Plot 2: Histogram of human counts
    plt.figure(figsize=(10, 6))
    plt.hist(human_counts, bins=range(0, max(human_counts)+2), color='green', edgecolor='black')
    plt.title('Distribution of Detected Humans per Frame')
    plt.xlabel('Number of Humans')
    plt.ylabel('Number of Frames')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'human_count_histogram.png'), dpi=150)
    plt.close()

    # Summary stats
    avg_humans = np.mean(human_counts)
    max_humans = max(human_counts)
    frames_with_humans = sum(1 for c in human_counts if c > 0)
    detection_rate = (frames_with_humans / len(files)) * 100

    print("\nFinal Stats & Plots:")
    print(f"  Total frames analyzed: {len(files)}")
    print(f"  Average humans per frame: {avg_humans:.2f}")
    print(f"  Max humans in a frame: {max_humans}")
    print(f"  Detection rate: {detection_rate:.1f}%")
    print(f"  Plots saved to: {plots_dir}")
    print("  - humans_per_frame.png")
    print("  - human_count_histogram.png")

if __name__ == '__main__':
    annotated_dir = 'sim_annotated_dark'     # your current annotated output
    plots_dir = 'plots'                       # new folder
    generate_plots(annotated_dir, plots_dir)