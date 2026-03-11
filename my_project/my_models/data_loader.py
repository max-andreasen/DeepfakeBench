
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

"""
Used to extract the data from the .npz files and prepare the tensors 
to be loaded into memory.

Loads a split file to store training, testing and validation data. 
Also handles the sampling of num_frames (e.g. 32) for the traning, 
and uses splts the full T / sequence into num_frames, e.g. 3*32, 
for the testing and validation. 
"""

class DeepfakeDataset(Dataset):
    def __init__(self, data_split_file, split, num_frames=32):
        data_split_file = Path(data_split_file)

        df = pd.read_csv(data_split_file)
        df = df[df["split"] == split]

        self.data = [str(Path(p)) for p in df["embedding_file"]]
        self.labels = [int(label) for label in df["label"]]
        self.split = split
        self.num_frames = num_frames

        print(f"DeepfakeDataset [{split}]: {len(self.data)} samples")
        with np.load(self.data[0], allow_pickle=False) as z:
            first_shape = z["embedding"].shape
        print(f"DeepfakeDataset: first embedding shape={first_shape}, embed_dim={first_shape[-1]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # each .npz contains one array key: "embedding" with shape [T_full, D].
        with np.load(self.data[index], allow_pickle=False) as z:
            video_features = torch.from_numpy(z["embedding"].astype(np.float32, copy=False))

        label = torch.tensor(self.labels[index], dtype=torch.long)

        total = video_features.shape[0]

        if self.split in ("train", "val"):
            # randomly sample a contiguous window of num_frames
            start = torch.randint(0, total - self.num_frames + 1, (1,)).item()
            video_features = video_features[start : start + self.num_frames]  # [num_frames, D]
        else:
            # test: split into non-overlapping windows for multi-segment aggregation
            n_windows = total // self.num_frames
            video_features = video_features[: n_windows * self.num_frames]
            video_features = video_features.reshape(n_windows, self.num_frames, -1)  # [n_windows, num_frames, D]

        return video_features, label
