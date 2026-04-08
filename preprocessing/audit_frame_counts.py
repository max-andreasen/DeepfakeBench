"""
Quick audit of video frame counts for FF++ sub-datasets.
Reports how many videos fall below various frame thresholds.

Usage:
    python audit_frame_counts.py [--dataset-root PATH] [--comp c23] [--subdataset DeepFakeDetection]
"""

import os
import sys
import glob
import argparse
import cv2
from pathlib import Path
from collections import defaultdict


def count_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def main():
    parser = argparse.ArgumentParser(description="Audit video frame counts for FF++ datasets")
    parser.add_argument("--dataset-root", type=str,
                        default="/home/max-andreasen/GitHub/DeepfakeBench/datasets/rgb/FaceForensics++",
                        help="Path to FaceForensics++ root")
    parser.add_argument("--comp", type=str, default="c23", choices=["raw", "c23", "c40"])
    parser.add_argument("--subdataset", type=str, default=None,
                        help="Process only this sub-dataset (e.g. DeepFakeDetection). Default: all.")
    args = parser.parse_args()

    all_sub_datasets = [
        "original_sequences/youtube",
        "original_sequences/actors",
        "manipulated_sequences/Deepfakes",
        "manipulated_sequences/Face2Face",
        "manipulated_sequences/FaceSwap",
        "manipulated_sequences/NeuralTextures",
        "manipulated_sequences/FaceShifter",
        "manipulated_sequences/DeepFakeDetection",
    ]

    if args.subdataset:
        # Match by suffix
        matched = [s for s in all_sub_datasets if s.endswith(args.subdataset)]
        if not matched:
            print(f"Unknown sub-dataset: {args.subdataset}")
            print(f"Available: {[s.split('/')[-1] for s in all_sub_datasets]}")
            sys.exit(1)
        all_sub_datasets = matched

    thresholds = [4, 8, 16, 32, 64, 96, 128, 256]

    for sub in all_sub_datasets:
        sub_path = os.path.join(args.dataset_root, sub, args.comp)
        videos = sorted(glob.glob(os.path.join(sub_path, "**/*.mp4"), recursive=True))

        if not videos:
            print(f"\n{'='*60}")
            print(f"{sub}/{args.comp}: no videos found")
            continue

        print(f"\n{'='*60}")
        print(f"{sub}/{args.comp}: scanning {len(videos)} videos...")

        frame_counts = []
        short_videos = defaultdict(list)  # threshold -> list of (name, count)

        for v in videos:
            n = count_frames(v)
            name = os.path.basename(v)
            frame_counts.append((name, n))
            for t in thresholds:
                if 0 < n < t:
                    short_videos[t].append((name, n))

        # Stats
        counts = [c for _, c in frame_counts if c > 0]
        if not counts:
            print("  No readable videos!")
            continue

        print(f"  Total videos:  {len(counts)}")
        print(f"  Frame counts:  min={min(counts)}, max={max(counts)}, "
              f"median={sorted(counts)[len(counts)//2]}, mean={sum(counts)/len(counts):.0f}")

        print(f"\n  Videos below threshold:")
        print(f"  {'Threshold':>10}  {'Count':>6}  {'% of total':>10}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*10}")
        for t in thresholds:
            n_below = len(short_videos[t])
            pct = 100.0 * n_below / len(counts)
            print(f"  {f'< {t}':>10}  {n_below:>6}  {pct:>9.1f}%")

        # List videos below 32 frames (most detectors need at least this)
        if short_videos[32]:
            print(f"\n  Videos with fewer than 32 frames:")
            for name, n in sorted(short_videos[32], key=lambda x: x[1]):
                print(f"    {name}: {n} frames")

        # List videos between 32-96 (summary only)
        between_32_96 = [(name, n) for name, n in frame_counts if 32 <= n < 96]
        if between_32_96:
            print(f"\n  Videos with 32-95 frames (processable but short): {len(between_32_96)}")
            if len(between_32_96) <= 20:
                for name, n in sorted(between_32_96, key=lambda x: x[1]):
                    print(f"    {name}: {n} frames")


if __name__ == "__main__":
    main()
