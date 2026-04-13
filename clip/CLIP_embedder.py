from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union
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
    


    def _get_model_dir(self, base_dir: str | Path = "embeddings") -> Path:
        """
        Returns the model-level output directory.
        Example: embeddings/ViT-L-14-336-quickgelu/
        """
        safe_model = str(self.model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        dim = self.embedding_dim if self.embedding_dim is not None else "unknown"
        model_dir = Path(base_dir) / f"{safe_model}_dim{dim}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir



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


    def _load_frames_from_video(self, video_path: str):
        """
        Loads T consecutive frames directly from a video file (.mp4).
        Randomly samples a contiguous window of T frames.
        Returns (frames, indices) or False if the video has fewer than T frames.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.T:
            cap.release()
            return False

        start = int(self.rng.integers(0, total_frames - self.T + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(self.T):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if len(frames) < self.T:
            return False

        indices = list(range(start, start + self.T))
        return frames, indices


    
    def _align_frames(self, frames):
        """
        Apply face alignment to all T frames, using same crop across all frames.
        
        Returns None if no face was found (the caller should then discard the video).
        """
        if not self.align_faces:
            return frames
        return self._aligner.preprocess(frames)



    def _run_embedding_loop(self, max_videos, df, model_dir: Path):
        """
        Loops over videos in created dataframe for each dataset, extracts T frames, embeds them.

        Saves embeddings to: model_dir/{dataset}/{label_cat}/{video_id}.npz
        Dataframe must contain columns: video_id, label, label_cat, frame_paths (list of PNG paths), split.
        """
        required = {"video_id", "label", "label_cat"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {missing}")
        if "frame_paths" not in df.columns and "video_path" not in df.columns:
            raise ValueError("df must have either 'frame_paths' or 'video_path' column")

        failures = []
        catalogue_rows = []
        observed_labels = set()
        video_embed_times = []
        skipped_existing = 0

        N = len(df) if max_videos is None else min(len(df), max_videos)

        # Loops through each row in the dataframe (df)
        for j, row in enumerate(
            tqdm(df.itertuples(index=False), total=N, desc="Embedding dataset", unit="video"),
            start=1,
        ):
            t0 = time.perf_counter()
            if max_videos is not None and j > max_videos:
                break

            video_id = row.video_id
            frame_paths = getattr(row, "frame_paths", None)
            video_path = getattr(row, "video_path", None)
            label = int(row.label)
            label_cat = row.label_cat
            split = getattr(row, "split", None)

            # Build output path: model_dir/dataset/label_cat/video_id.npz
            clean_video_id = video_id.replace("/", "__").replace("\\", "__")
            video_out_dir = model_dir / self.dataset_name / label_cat
            out_file = video_out_dir / f"{clean_video_id}.npz"

            # Skip if already embedded
            if out_file.exists():
                skipped_existing += 1
                continue

            # Determine source path for error messages
            source_path = ""
            if frame_paths is not None:
                frame_paths = list(frame_paths)
                source_path = str(Path(frame_paths[0]).parent) if frame_paths else ""
            elif video_path is not None:
                source_path = str(video_path)

            try:
                # Load frames from pre-extracted PNGs or directly from video
                if frame_paths is not None:
                    frame_result = self._load_frames(frame_paths)
                elif video_path is not None:
                    frame_result = self._load_frames_from_video(str(video_path))
                else:
                    failures.append((video_id, "", "No frame_paths or video_path provided"))
                    continue

                if frame_result is False:
                    failures.append((video_id, source_path, f"Too few frames (<{self.T})"))
                    continue
                frames, indices = frame_result

                if self.align_faces:
                    frames = self._align_frames(frames)

                if frames is None:
                    failures.append((video_id, source_path, "No face detected during alignment"))
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

                self._save_embedding_to_file(embedding, video_out_dir, clean_video_id)

                label_name = "real" if label == 1 else ("fake" if label == 0 else "unknown")
                catalogue_rows.append(
                    {
                        "video_id": video_id,
                        "label": label,
                        "label_name": label_name,
                        "label_cat": label_cat,
                        "split": split,
                        "embedding_file": str(out_file.as_posix()),
                        "n_frames": int(self.T),
                        "embedding_dim": int(embedding.shape[1]),
                    }
                )
                observed_labels.add(label)

                elapsed = time.perf_counter() - t0
                video_embed_times.append(elapsed)

            except Exception as e:
                failures.append((video_id, source_path, str(e)))

        # --- Save catalogue and config at model level ---
        catalogue_path = model_dir / "catalogue.csv"
        config_path = model_dir / "config.json"

        # Append to existing catalogue if present
        new_df = pd.DataFrame(catalogue_rows)
        if catalogue_path.exists():
            existing_df = pd.read_csv(catalogue_path)
            catalogue_df = pd.concat([existing_df, new_df], ignore_index=True)
            catalogue_df = catalogue_df.drop_duplicates(subset="video_id", keep="last")
        else:
            catalogue_df = new_df

        if not catalogue_df.empty:
            catalogue_df = catalogue_df.sort_values("video_id").reset_index(drop=True)

        catalogue_tmp = catalogue_path.with_suffix(".tmp.csv")
        catalogue_df.to_csv(catalogue_tmp, index=False)
        os.replace(catalogue_tmp, catalogue_path)

        if not video_embed_times and skipped_existing > 0:
            print(f"All {skipped_existing} videos already embedded — nothing to do.")
        elif not video_embed_times:
            print(f"WARNING: 0 videos embedded successfully out of {N}. "
                  "Check the failures list — all videos may have raised exceptions.")
        elif skipped_existing > 0:
            print(f"Embedded {len(video_embed_times)} new videos, skipped {skipped_existing} already existing.")
        avg_video_emb_time = sum(video_embed_times) / len(video_embed_times) if video_embed_times else 0.0

        run_config = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "T": self.T,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "align_faces": self.align_faces,
            "face_scale": self.face_scale if self.align_faces else None,
        }
        config_tmp = config_path.with_suffix(".tmp.json")
        with open(config_tmp, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        os.replace(config_tmp, config_path)

        return failures


    
    def run(self, max_videos, input_frame: pd.DataFrame, base_dir: str | Path = "embeddings"):
        """
        Sets the embedding process in motion. Embeds videos and stores them as .npz files
        organized by model/dataset/sub_dataset/.

        Args:
            input_frame: DataFrame with columns: video_id, label, label_cat, frame_paths, split, dataset
            base_dir: Root embeddings directory (default: embeddings/)
            max_videos: Max videos to process (None = all)
        Returns:
            List of (video_id, path, error) failure tuples.
        """
        if "dataset" not in input_frame.columns:
            raise ValueError("Missing required column: dataset")
        datasets = input_frame["dataset"].dropna().unique()
        if len(datasets) != 1:
            raise ValueError(f"Expected exactly one dataset in a run, got: {datasets.tolist()}")
        self.dataset_name = str(datasets[0])

        model_dir = self._get_model_dir(base_dir)
        failures = self._run_embedding_loop(df=input_frame, model_dir=model_dir, max_videos=max_videos)
        return failures
