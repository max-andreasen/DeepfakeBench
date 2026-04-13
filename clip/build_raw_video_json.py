"""
Builds a JSON file mapping FF++ raw videos to their labels and splits.
Output format matches rearrange.py but uses "video_path" instead of "frames".

Usage:
    python clip/build_raw_video_json.py
"""

import os
import json
import glob
from pathlib import Path


def build_ff_raw_json(dataset_root, comp="c23", output_path=None):
    """
    Scans FaceForensics++ video directories and builds a JSON with video paths,
    labels, and train/val/test splits.
    """
    ff_path = os.path.join(dataset_root, "FaceForensics++")

    # Load official split files
    with open(os.path.join(ff_path, "train.json"), "r") as f:
        train_json = json.load(f)
    with open(os.path.join(ff_path, "val.json"), "r") as f:
        val_json = json.load(f)
    with open(os.path.join(ff_path, "test.json"), "r") as f:
        test_json = json.load(f)

    video_to_split = {}
    for d1, d2 in train_json:
        video_to_split[d1] = "train"
        video_to_split[d2] = "train"
        video_to_split[f"{d1}_{d2}"] = "train"
        video_to_split[f"{d2}_{d1}"] = "train"
    for d1, d2 in val_json:
        video_to_split[d1] = "val"
        video_to_split[d2] = "val"
        video_to_split[f"{d1}_{d2}"] = "val"
        video_to_split[f"{d2}_{d1}"] = "val"
    for d1, d2 in test_json:
        video_to_split[d1] = "test"
        video_to_split[d2] = "test"
        video_to_split[f"{d1}_{d2}"] = "test"
        video_to_split[f"{d2}_{d1}"] = "test"

    # Sub-datasets to scan
    sub_datasets = {
        ("original_sequences", "youtube"): "FF-real",
        ("manipulated_sequences", "Deepfakes"): "FF-DF",
        ("manipulated_sequences", "Face2Face"): "FF-F2F",
        ("manipulated_sequences", "FaceSwap"): "FF-FS",
        ("manipulated_sequences", "NeuralTextures"): "FF-NT",
    }

    dataset_dict = {"FaceForensics++": {}}

    for (seq_type, sub_name), label_cat in sub_datasets.items():
        videos_dir = os.path.join(ff_path, seq_type, sub_name, comp, "videos")
        if not os.path.isdir(videos_dir):
            print(f"WARNING: {videos_dir} not found, skipping")
            continue

        dataset_dict["FaceForensics++"][label_cat] = {
            "train": {}, "val": {}, "test": {}
        }

        for video_file in sorted(glob.glob(os.path.join(videos_dir, "*.mp4"))):
            video_name = Path(video_file).stem
            split = video_to_split.get(video_name)
            if split is None:
                continue
            dataset_dict["FaceForensics++"][label_cat][split][video_name] = {
                "label": label_cat,
                "video_path": video_file,
            }

        total = sum(len(v) for v in dataset_dict["FaceForensics++"][label_cat].values())
        print(f"  {label_cat}: {total} videos")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "raw_video_json", "FaceForensics++.json"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset_dict, f)
    print(f"\n{output_path} generated successfully.")


if __name__ == "__main__":
    dataset_root = "/home/max-andreasen/GitHub/DeepfakeBench/datasets/rgb"
    build_ff_raw_json(dataset_root)
