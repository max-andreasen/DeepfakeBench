"""
Test-time data loader.

One video per __getitem__. Returns:
    (x([n_windows, T, D]), label, video_id, label_cat)

    Basically, in almost all cases, we have [3, 32, D], label, video_id, label_cat.

where n_windows = total_frames // T
DataLoader batches into [B, n_windows, T, D].
The Tester flattens windows into the batch dim for forward, then averages
per-window logits back to one prediction per video.

label_cat is returned so per-video aggregation can key on (label_cat, video_id):
FF++ fake manipulations (FF-DF, FF-F2F, FF-FS, FF-NT) reuse the same video_id
strings (e.g. '000_003'), so grouping by video_id alone pools all 4 manipulations
into one "video" and over-averages.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

JOIN_KEYS = ["dataset", "label_cat", "video_id"]


class DeepfakeTestDataset(Dataset):
    def __init__(self, split_file, catalogue_file, num_frames=32, split="test", input_transform="none"):
        split_df = pd.read_csv(Path(split_file))
        catalogue_df = pd.read_csv(Path(catalogue_file))

        # Old catalogues may carry a stale "split" column — split CSV is authoritative.
        catalogue_df = catalogue_df.drop(columns=["split"], errors="ignore")

        join_keys = [k for k in JOIN_KEYS if k in catalogue_df.columns]
        df = split_df.merge(catalogue_df, on=join_keys, how="inner")
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows for split='{split}' after joining {split_file} with {catalogue_file}."
            )
        if "n_frames" not in df.columns:
            raise ValueError(
                f"Catalogue {catalogue_file} missing 'n_frames' column."
            )

        # Require constant n_frames across the dataset so default collate works.
        # (If this ever fails, switch back to a windows-as-units layout.)
        n_frames_unique = sorted(set(df["n_frames"].astype(int).tolist()))
        if len(n_frames_unique) != 1:
            raise ValueError(
                f"Videos have differing n_frames={n_frames_unique}; "
                "DeepfakeTestDataset requires a constant embedding length. "
                "Re-embed with a fixed T or switch to a windows-as-units loader."
            )
        total_frames = int(n_frames_unique[0])
        n_windows = total_frames // num_frames
        if n_windows < 1:
            raise ValueError(
                f"num_frames={num_frames} > total embedded frames={total_frames}."
            )

        self.num_frames = num_frames
        self.n_windows = n_windows
        self.usable_frames = n_windows * num_frames  # truncate any remainder
        # Stored on the dataset so Tester._forward_all can discover it and apply
        # the transform AFTER the optional shuffle. Do not apply in __getitem__.
        self.input_transform = input_transform

        if "label_cat" not in df.columns:
            raise ValueError(
                f"Catalogue {catalogue_file} missing 'label_cat' column; "
                "required for correct per-video aggregation."
            )

        self.video_paths = [str(Path(p)) for p in df["embedding_file"]]
        self.video_labels = [int(x) for x in df["label"]]
        self.video_ids = [str(v) for v in df["video_id"]]
        self.video_label_cats = [str(v) for v in df["label_cat"]]

        effective_T = num_frames - 1 if input_transform == "diff" else num_frames
        print(
            f"DeepfakeTestDataset: {len(self.video_paths)} videos, "
            f"{n_windows} windows/video (T={num_frames}, total_frames={total_frames}), "
            f"transform={input_transform}, effective_T={effective_T}"
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, i):
        with np.load(self.video_paths[i], allow_pickle=False) as z:
            arr = z["embedding"][: self.usable_frames]
        x = torch.from_numpy(arr.astype(np.float32, copy=False))
        x = x.reshape(self.n_windows, self.num_frames, -1)  # [n_windows, T, D]
        label = torch.tensor(self.video_labels[i], dtype=torch.long)
        video_id = self.video_ids[i]
        label_cat = self.video_label_cats[i]
        return x, label, video_id, label_cat
