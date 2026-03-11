import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from tqdm import tqdm
import time

from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn.functional as F
import open_clip

"""
This file contains the CLIP embedder class, used to embed a set of videos.

Outputs embedded videos into dir -->
Writes a config.json and catalogue.csv
"""

class CLIP_EMBEDDER():
    def __init__(self, model_name, pretrained="openai", device="cuda", T=32, micro_bs=16, seed=0,
                    align_faces=False, face_scale=0.75):
        self.model_name = model_name
        self.T = T
        self.pretrained = pretrained
        self.micro_bs = micro_bs
        self.seed = seed
        self.align_faces = align_faces
        self.face_scale = face_scale

        self.rng = np.random.default_rng(self.seed)

        self.device = device if (not device.startswith("cuda")
            or torch.cuda.is_available()) else "cpu"
        print(f"Using device: {self.device}")


        self.model, self.preprocess = self._load_model(
            model_name=self.model_name,
            device=self.device,
            pretrained=self.pretrained,
        )
        self.embedding_dim = getattr(self.model.visual, "output_dim", None)
        self.dataset_name = None # will be filled when calling 'run' function.

        # Lazy import: facenet_pytorch is only loaded when face alignment is
        # actually requested, so it cannot interfere with CUDA initialisation
        # when running the raw (no alignment) embedding pipeline.
        if align_faces:
            print("Using aligned faces.")
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from preprocessing.preprocess import FaceAligner
            self._aligner = FaceAligner(output_size=336, face_scale=self.face_scale, device=self.device)
        else:
            self._aligner = None



    def _load_model(self, model_name, device=None, pretrained="openai"):
        """
        Loads the CLIP model into memory.
        """
        if device is None:
            device = self.device
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,  # loading pretrained weights
            device=device
        )
        model.eval()  # inference mode: disables dropout etc.
        return model, preprocess



    def _save_embedding_to_file(self, embedding, out_dir, video_id: str):
        """
        Saves embedded video tensor [T, D] as .npz (numpy zip archive), for one video.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # tmp file for safer writes (avoids corrupted files if something goes wrong supposedly)
        out_path = out_dir / f"{video_id}.npz"
        tmp_path = out_dir / f"{video_id}.tmp.npz"

        # stores embeddings compressed (lossless here)
        np.savez_compressed(tmp_path, embedding=embedding.numpy().astype(np.float32, copy=False))
        os.replace(tmp_path, out_path)
        return True
    


    def _create_output_dir(self, base_dir: str | Path = "clip/embeddings") -> Path:
        """
        Creates a unique run directory under clip/embeddings and returns the files/ directory path.
        Example: clip/embeddings/celeb_df_ViT-L-14-336-quickgelu_dim768_T32_0/files
        """
        if not self.dataset_name:
            raise ValueError("dataset_name is not set. Call run(...) with a dataframe containing a 'dataset' column.")

        base_dir_path = Path(base_dir)
        base_dir_path.mkdir(parents=True, exist_ok=True)

        safe_dataset = str(self.dataset_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_model = str(self.model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        dim = self.embedding_dim if self.embedding_dim is not None else "unknown"

        inc = 0
        while True:
            align_tag = "_aligned" if self.align_faces else ""
            run_name = f"{safe_dataset}_{safe_model}_dim{dim}_T{self.T}{align_tag}_{inc}"
            run_dir = base_dir_path / run_name
            files_dir = run_dir / "files"
            if not run_dir.exists():
                files_dir.mkdir(parents=True, exist_ok=False)
                return files_dir
            inc += 1



    @torch.no_grad()
    def _embed_batch(self, frames):
        """
        Embeds a batch of frames using the loaded CLIP vision model. 
        Returns the batch as a raw torch tensor. 
        """
        frame_tensors = []
        for frame in frames: 
            frame_tensors.append(self.preprocess(frame.convert("RGB")))  # [3, H, W], float tensor
        
        # Construct a tensor from list
        batch_tensor = torch.stack(frame_tensors, dim=0).to(self.device) # [B, 3, H, W]

        batch_embedding = self.model.encode_image(batch_tensor)  # [B, D]
        batch_embedding = batch_embedding / batch_embedding.norm(dim=-1, keepdim=True)

        return batch_embedding.float().cpu()                # [B, D]



    def _load_frames(self, frame_paths: List[str]):
        """
        Loads T frames from pre-extracted frame paths (PNG files produced by preprocess.py).
        Randomly samples a contiguous window of T frames, consistent with _extract_frames logic.
        """
        n_frames = len(frame_paths)
        if n_frames <= 0:
            raise RuntimeError("Empty frame_paths list")

        if n_frames < self.T:
            return False  # discard short videos, consistent with original behaviour

        start = int(self.rng.integers(0, n_frames - self.T + 1))
        selected = frame_paths[start : start + self.T]
        indices = list(range(start, start + self.T))

        frames = [Image.open(p).convert("RGB") for p in selected]
        return frames, indices


    
    def _align_frames(self, frames):
        """
        Apply face alignment to all T frames, using same crop across all frames.
        
        Returns None if no face was found (the caller should then discard the video).
        """
        if not self.align_faces:
            return frames
        return self._aligner.preprocess(frames)



    def _run_embedding_loop(self, max_videos, df, out_dir: str | None = None):
        """
        Loops over videos in created dataframe for each dataset, extracts T frames, embeds them. 
        Also returns results in-memory.

        Dataframe must contain columns: video_id, label, frame_paths (list of PNG paths), split.
        Returns: list of dicts with video_id, label, indices, embedding (torch.FloatTensor [T, D])
        """
        required = {"video_id", "label", "frame_paths"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {missing}")
        
        failures = []       # keeps track of failures / discarded videos.
        catalogue_rows = [] 
        observed_labels = set()     # keeps track of all labels (should just be binary)
        video_embed_times = []

        N = len(df) if max_videos is None else min(len(df), max_videos)

        if out_dir is None:
            out_dir = str(self._create_output_dir().as_posix())
        out_dir_path = Path(out_dir) 

        # Loops through each row in the dataframe (df)
        for j, row in enumerate(
            tqdm(df.itertuples(index=False), total=N, desc="Embedding dataset", unit="video"),
            start=1,
        ):
            t0 = time.perf_counter() # for timing purposes
            if max_videos is not None and j > max_videos:
                break

            video_id = row.video_id
            frame_paths = list(row.frame_paths)
            label = int(row.label)
            split = getattr(row, "split", None)

            try:
                frame_result = self._load_frames(frame_paths)
                if frame_result is False:
                    failures.append((video_id, str(Path(frame_paths[0]).parent), f"Too few frames (<{self.T})"))
                    continue
                frames, indices = frame_result

                if self.align_faces:
                    frames = self._align_frames(frames)

                if frames is None:
                    failures.append((video_id, str(Path(frame_paths[0]).parent), "No face detected during alignment"))
                    continue

                # micro-batch embedding
                micro_batch_embs = []
                for i in range(0, len(frames), self.micro_bs):
                    chunk_of_frames = frames[i : i + self.micro_bs]
                    embedded_batch = self._embed_batch(chunk_of_frames)  # [b, D] on CPU
                    micro_batch_embs.append(embedded_batch)
                embedding = torch.cat(micro_batch_embs, dim=0)  # [T, D]

                # basic check to catch simple failures early
                if embedding.ndim != 2 or embedding.shape[0] != self.T:
                    raise RuntimeError(f"Bad embedding shape {tuple(embedding.shape)} expected ({self.T}, D)")
                if len(indices) != self.T:
                    raise RuntimeError(f"Bad indices length {len(indices)} expected {self.T}")

                # Savind embeddings to file 
                clean_video_id = video_id.replace("/", "__").replace("\\", "__") # sanitising video id for storage
                status = self._save_embedding_to_file(embedding, out_dir, clean_video_id)

                label_name = "real" if label == 1 else ("fake" if label == 0 else "unknown")
                catalogue_rows.append(
                    {
                        "video_id": video_id,
                        "label": label,
                        "label_name": label_name,
                        "video_dir": str(Path(frame_paths[0]).parent),
                        "split": split,
                        "embedding_file": str((out_dir_path / f"{clean_video_id}.npz").as_posix()),
                        "n_frames": int(self.T),
                        "embedding_dim": int(embedding.shape[1]),
                    }
                )
                observed_labels.add(label)

                elapsed = time.perf_counter() - t0
                video_embed_times.append(elapsed)

            except Exception as e:
                failures.append((video_id, str(Path(frame_paths[0]).parent) if frame_paths else "", str(e)))

        # Outside of loop -->
        metadata_dir = out_dir_path.parent if out_dir_path.name == "files" else out_dir_path
        metadata_dir.mkdir(parents=True, exist_ok=True)

        catalogue_path = metadata_dir / "catalogue.csv"
        config_path = metadata_dir / "run_config.json"

        catalogue_df = pd.DataFrame(catalogue_rows)
        if not catalogue_df.empty:
            catalogue_df = catalogue_df.sort_values("video_id").reset_index(drop=True)
        else:
            catalogue_df = pd.DataFrame(
                columns=["video_id", "label", "label_name", "video_dir", "split", "embedding_file", "n_frames", "embedding_dim"]
            )
        
        # Safe writing of catalogue file (using tmp).
        catalogue_tmp = catalogue_path.with_suffix(".tmp.csv")
        catalogue_df.to_csv(catalogue_tmp, index=False)
        os.replace(catalogue_tmp, catalogue_path)

        if not video_embed_times:
            print(f"WARNING: 0 videos embedded successfully out of {N}. "
                  "Check the failures list — all videos may have raised exceptions.")
        avg_video_emb_time = sum(video_embed_times) / len(video_embed_times) if video_embed_times else 0.0

        run_config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": self.dataset_name,
            "model_name": self.model_name,
            "avg_video_emb_time": avg_video_emb_time,
            "n_requested_videos": N,
            "n_saved_embeddings": len(catalogue_rows),
            "n_failures": len(failures),
            "out_dir": str(out_dir_path.as_posix()),
            "catalogue_file": str(catalogue_path.as_posix()),
            "observed_labels": sorted([int(x) for x in observed_labels]),
            "T": self.T,
            "micro_batch_size": self.micro_bs, # does not really matter though.
            "device": self.device,
            "embedding_dim": self.embedding_dim,
            "align_faces": self.align_faces,
            "face_scale": self.face_scale if self.align_faces else None,
        }
        # Safe writing of config file (using tmp).
        config_tmp = config_path.with_suffix(".tmp.json")
        with open(config_tmp, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        os.replace(config_tmp, config_path)

        return failures


    
    def run(self, max_videos, input_frame: pd.DataFrame, out_dir: str | None = None):
        """
        Sets the embedding process in motion. Automatically embeds the videos, stores them
        in .npz format along with a config.json and manifest.csv file outlining the 
        data structure. 

        Args: 
            input_frame: A pandas DF containing the a structured layout specified by ...
            out_dir: Where the outputted videos should be stored. Default is clip/embeddings/
            max_videos: set to None by default. Specifies a max amount of videos being inputted. 
        Return: 
            Returns a status message if the embedding and writing process was successful or not.
        """
        # TODO: Set self.dataset_name
        if "dataset" not in input_frame.columns:
            raise ValueError("Missing required column: dataset")
        datasets = input_frame["dataset"].dropna().unique()
        if len(datasets) != 1:
            raise ValueError(f"Expected exactly one dataset in a run, got: {datasets.tolist()}")
        self.dataset_name = str(datasets[0])

        failures = self._run_embedding_loop(df=input_frame, out_dir=out_dir, max_videos=max_videos)
        return failures
