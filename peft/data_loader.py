"""
Frame-level dataset for PEFT training. Reads PNG frames straight from disk
(via the rearrange JSON) and applies CLIP's preprocess transform in workers.

NO .npz cache reuse — the LN weights change every step, so the existing
clip/embeddings/ caches are not valid for PEFT. Optional L2-normalization, when
enabled, is applied inside the model after the trainable visual encoder.

See peft/IMPLEMENTATION_PLAN.md §5 Step 3.
"""

import json
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# Verbatim copy from clip/embed.py:LABEL_MAP. Kept inline so this module has
# no awkward import gymnastics with the `clip/` folder vs the `clip` PyPI pkg.
LABEL_MAP = {
    "CelebDFv3_real": 1, "CelebDFv3_fake": 0,
    "CelebDFv2_real": 1, "CelebDFv2_fake": 0,
    "CelebDFv1_real": 1, "CelebDFv1_fake": 0,
    "WDF_real":       1, "WDF_fake":       0,
    "FF-real":        1, "FF-DF": 0, "FF-F2F": 0, "FF-FS": 0, "FF-NT": 0,
    "DFDCP_Real":     1, "DFDCP_FakeA": 0, "DFDCP_FakeB": 0,
    "DFDC_Real":      1, "DFDC_Fake": 0,
    "UADFV_Real":     1, "UADFV_Fake": 0,
    "DF_real":        1, "DF_fake": 0,
}

JOIN_KEYS = ["dataset", "label_cat", "video_id"]


def _build_df(rearrange_json: Path, dataset_name: str) -> pd.DataFrame:
    """Mirrors clip/embed.py:build_df_from_repo_json. Handles FF++ nested
    compression layer (split -> c23 -> videos)."""
    with open(rearrange_json, "r") as f:
        data = json.load(f)
    if dataset_name not in data:
        raise ValueError(
            f"'{dataset_name}' not in {rearrange_json}. Keys: {list(data.keys())}"
        )

    rows = []
    for label_cat, splits in data[dataset_name].items():
        numeric = LABEL_MAP.get(label_cat)
        if numeric is None:
            raise ValueError(
                f"No LABEL_MAP entry for '{label_cat}'. Add it to peft/data_loader.py."
            )
        for videos_or_comp in splits.values():
            # FF++ has split -> compression -> videos; everything else has
            # split -> videos.
            first = next(iter(videos_or_comp.values()), None)
            if first is not None and isinstance(first, dict) and "frames" not in first:
                videos = {}
                for _comp, comp_videos in videos_or_comp.items():
                    videos.update(comp_videos)
            else:
                videos = videos_or_comp

            for video_id, info in videos.items():
                if "frames" not in info:
                    raise ValueError(
                        f"Video '{video_id}' has no 'frames' key in {rearrange_json}."
                    )
                rows.append({
                    "dataset":     dataset_name,
                    "label_cat":   label_cat,
                    "video_id":    video_id,
                    "label":       numeric,
                    "frame_paths": sorted(info["frames"]),
                })

    return pd.DataFrame(rows).drop_duplicates(subset=JOIN_KEYS)


def _join_with_split(
    df_videos: pd.DataFrame,
    split_file: str,
    split: str,
) -> pd.DataFrame:
    df_split = pd.read_csv(Path(split_file))
    df_split["video_id"] = df_split["video_id"].astype(str)
    df_videos = df_videos.copy()
    df_videos["video_id"] = df_videos["video_id"].astype(str)
    df = df_split.merge(df_videos, on=JOIN_KEYS, how="inner")
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"Empty split='{split}' after joining {split_file} with rearrange JSON. "
            f"Check (dataset, label_cat, video_id) keys match."
        )
    return df


class FramePEFTDataset(Dataset):
    """Train/val dataset. One random contiguous T-window per __getitem__."""

    def __init__(
        self,
        split_file: str,
        rearrange_json: str,
        dataset_name: str,
        split: str,                       # "train" | "val"
        num_frames: int,
        preprocess: Callable,             # open_clip preprocess (PIL -> [3, H, W])
        min_superset: int = 96,
        max_videos: Optional[int] = None,
    ):
        df_videos = _build_df(Path(rearrange_json), dataset_name)
        df = _join_with_split(df_videos, split_file, split)

        rows = df[["label", "frame_paths", "video_id", "label_cat"]].to_dict("records")

        # Fail fast on too-short videos (T=96 superset is load-bearing).
        short = [r for r in rows if len(r["frame_paths"]) < min_superset]
        if short:
            raise ValueError(
                f"{len(short)} videos in split={split} have <{min_superset} frames. "
                f"First: {short[0]['video_id']} ({len(short[0]['frame_paths'])} frames)."
            )

        if max_videos is not None and max_videos < len(rows):
            rows = rows[:max_videos]

        self.rows: List[dict] = rows
        self.num_frames = int(num_frames)
        self.preprocess = preprocess
        self.split = split

        print(
            f"FramePEFTDataset[{split}]: {len(self.rows)} videos, T={self.num_frames}"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        paths = r["frame_paths"]
        total = len(paths)
        start = int(np.random.randint(0, total - self.num_frames + 1))
        selected = paths[start : start + self.num_frames]
        frames = [self.preprocess(Image.open(p).convert("RGB")) for p in selected]
        x = torch.stack(frames, dim=0)                    # [T, 3, H, W]
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, y


class FramePEFTTestDataset(Dataset):
    """Test dataset. Yields all 3 non-overlapping T-windows per video.

    __getitem__ returns:
        x [n_windows, T, 3, H, W], label, video_id, label_cat
    Mirrors evaluation/data_loader.py::DeepfakeTestDataset semantics so the
    pooling/aggregation logic in evaluation/tester.py is reusable.
    """

    def __init__(
        self,
        split_file: str,
        rearrange_json: str,
        dataset_name: str,
        split: str,                        # usually "test"
        num_frames: int,
        preprocess: Callable,
        min_superset: int = 96,
    ):
        df_videos = _build_df(Path(rearrange_json), dataset_name)
        df = _join_with_split(df_videos, split_file, split)

        rows = df[["label", "frame_paths", "video_id", "label_cat"]].to_dict("records")

        # Require constant frame count (matches DeepfakeTestDataset's invariant
        # so default collate works).
        n_frames_unique = sorted({len(r["frame_paths"]) for r in rows})
        if len(n_frames_unique) != 1:
            raise ValueError(
                f"Videos have differing frame counts: {n_frames_unique}. "
                f"FramePEFTTestDataset requires a constant frame count."
            )
        total_frames = n_frames_unique[0]
        if total_frames < min_superset:
            raise ValueError(
                f"All videos have only {total_frames} frames (<{min_superset})."
            )

        n_windows = total_frames // num_frames
        if n_windows < 1:
            raise ValueError(
                f"num_frames={num_frames} > total_frames={total_frames}."
            )

        self.rows = rows
        self.num_frames = int(num_frames)
        self.n_windows = int(n_windows)
        self.usable_frames = self.n_windows * self.num_frames  # truncate remainder
        self.preprocess = preprocess
        self.split = split

        print(
            f"FramePEFTTestDataset[{split}]: {len(self.rows)} videos, "
            f"{self.n_windows} windows/video (T={self.num_frames}, "
            f"total_frames={total_frames})"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        paths = r["frame_paths"][: self.usable_frames]
        frames = [self.preprocess(Image.open(p).convert("RGB")) for p in paths]
        x = torch.stack(frames, dim=0)                              # [W*T, 3, H, W]
        x = x.reshape(self.n_windows, self.num_frames, *x.shape[1:])  # [W, T, 3, H, W]
        label = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, label, str(r["video_id"]), str(r["label_cat"])
