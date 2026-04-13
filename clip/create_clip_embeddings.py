import os
import json
import glob
import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_embedder_mod = _load_module_from_path(
    "local_clip_embedder",
    Path(__file__).resolve().parent / "CLIP_embedder.py",
)
CLIP_EMBEDDER = _embedder_mod.CLIP_EMBEDDER


# Maps the label category strings from rearrange.py JSON → numeric label.
# Convention: 1 = real, 0 = fake  (matches the rest of this pipeline).
LABEL_MAP = {
    "CelebDFv2_real": 1, "CelebDFv2_fake": 0,
    "CelebDFv1_real": 1, "CelebDFv1_fake": 0,
    "FF-real":        1, "FF-DF": 0, "FF-F2F": 0, "FF-FS": 0, "FF-NT": 0,
    "DFDCP_Real":     1, "DFDCP_FakeA": 0, "DFDCP_FakeB": 0,
    "DFDC_Real":      1, "DFDC_Fake": 0,
    "UADFV_Real":     1, "UADFV_Fake": 0,
    "DF_real":        1, "DF_fake": 0,
}



def build_df_from_repo_json(json_path: Path, dataset_name: str):
    """
    Reads a JSON file produced by rearrange.py and returns a DataFrame with columns:
        video_id, label, frame_paths (list of PNG paths), split, dataset

    The split column uses the repo's own train/val/test assignments, so data_split.py
    is no longer needed.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if dataset_name not in data:
        raise ValueError(f"'{dataset_name}' not found in {json_path}. Keys: {list(data.keys())}")

    rows = []
    for label_cat, splits in data[dataset_name].items():
        numeric_label = LABEL_MAP.get(label_cat)
        if numeric_label is None:
            raise ValueError(
                f"No label mapping for '{label_cat}'. Add it to LABEL_MAP in create_clip_embeddings.py."
            )
        for split_name, videos_or_comp in splits.items():
            # FF++ JSONs have an extra compression level: split → c23 → videos
            # Detect by checking if the first value has "frames" or not
            first_val = next(iter(videos_or_comp.values()), None)
            if first_val is not None and isinstance(first_val, dict) and "frames" not in first_val and "video_path" not in first_val:
                # Extra nesting (compression level) — flatten it
                videos = {}
                for _comp, comp_videos in videos_or_comp.items():
                    videos.update(comp_videos)
            else:
                videos = videos_or_comp

            for video_id, video_info in videos.items():
                row = {
                    "dataset":     dataset_name,
                    "video_id":    video_id,
                    "label":       numeric_label,
                    "label_cat":   label_cat,
                    "split":       split_name,
                }
                if "frames" in video_info:
                    row["frame_paths"] = sorted(video_info["frames"])
                if "video_path" in video_info:
                    row["video_path"] = video_info["video_path"]
                rows.append(row)

    df = pd.DataFrame(rows)
    split_counts = df["split"].value_counts().to_dict()
    print(f"Built DataFrame: {len(df)} videos — {split_counts}")
    return df



def main():
    # ---- manual run config ----
    json_path    = REPO_ROOT / "preprocessing" / "dataset_json_dlib" / "FaceForensics++.json"   # dlib preprocessed
    dataset_name = "FaceForensics++"
    model_name   = "ViT-L-14-336-quickgelu"
    pretrained   = "openai"
    device       = "cuda"
    T            = 96
    micro_bs     = 16
    seed         = 0
    base_dir     = REPO_ROOT / "clip" / "embeddings" / "dlib"
    max_videos   = None
    # ---------------------------

    df = build_df_from_repo_json(json_path, dataset_name)

    embedder = CLIP_EMBEDDER(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        T=T,
        micro_bs=micro_bs,
        seed=seed,
        align_faces=False,  # frames are already face-aligned by preprocess.py
    )

    failures = embedder.run(
        input_frame=df,
        base_dir=base_dir,
        max_videos=max_videos,
    )

    print(f"Embedding finished. Failures: {len(failures)}")
    if failures:
        print(f"First failure: {failures[0]}")


if __name__ == "__main__":
    main()
