
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

"""
Loads per-video CLIP embeddings (.npz files) for training.

Joins a split CSV (dataset, label_cat, video_id, split) with a catalogue CSV
(dataset, label_cat, video_id, label, embedding_file, ...) on the composite
key, then filters to the requested split.

Train/val: samples a random contiguous window of num_frames from each video.
Test: splits the full T into non-overlapping windows for multi-segment aggregation.
"""

JOIN_KEYS = ["dataset", "label_cat", "video_id"]


class DeepfakeDataset(Dataset):
    def __init__(self, split_file, catalogue_file, split, num_frames=32):
        split_df = pd.read_csv(Path(split_file))
        catalogue_df = pd.read_csv(Path(catalogue_file))

        # Old catalogues carry a stale "split" column; the split CSV is authoritative now.
        catalogue_df = catalogue_df.drop(columns=["split"], errors="ignore")

        # Old catalogues (pre-dataset-column) won't have "dataset" — fall back to
        # joining on (label_cat, video_id) only. Safe when the catalogue covers a
        # single dataset, which is the current layout (one catalogue per embed run).
        join_keys = [k for k in JOIN_KEYS if k in catalogue_df.columns]
        df = split_df.merge(catalogue_df, on=join_keys, how="inner")
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows for split='{split}' after joining {split_file} with {catalogue_file}. "
                f"Check that (dataset, label_cat, video_id) keys match."
            )

        self.data = [str(Path(p)) for p in df["embedding_file"]]
        self.labels = [int(x) for x in df["label"]]
        self.split = split
        self.num_frames = num_frames

        print(f"DeepfakeDataset [{split}]: {len(self.data)} samples")
        with np.load(self.data[0], allow_pickle=False) as z:
            first_shape = z["embedding"].shape
        print(f"DeepfakeDataset: first embedding shape={first_shape}, embed_dim={first_shape[-1]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with np.load(self.data[index], allow_pickle=False) as z:
            video_features = torch.from_numpy(z["embedding"].astype(np.float32, copy=False))

        label = torch.tensor(self.labels[index], dtype=torch.long)
        total = video_features.shape[0]

        if self.split in ("train", "val"):
            start = torch.randint(0, total - self.num_frames + 1, (1,)).item()
            video_features = video_features[start : start + self.num_frames]  # [num_frames, D]
        else:
            n_windows = total // self.num_frames
            video_features = video_features[: n_windows * self.num_frames]
            video_features = video_features.reshape(n_windows, self.num_frames, -1)

        return video_features, label
