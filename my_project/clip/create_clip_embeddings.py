import json
import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


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



def build_df_from_repo_json(json_path: Path, dataset_name: str) -> pd.DataFrame:
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
        for split_name, videos in splits.items():
            for video_id, video_info in videos.items():
                frame_paths = sorted(video_info["frames"])
                rows.append({
                    "dataset":     dataset_name,
                    "video_id":    video_id,
                    "label":       numeric_label,
                    "frame_paths": frame_paths,
                    "split":       split_name,
                })

    df = pd.DataFrame(rows)
    split_counts = df["split"].value_counts().to_dict()
    print(f"Built DataFrame: {len(df)} videos — {split_counts}")
    return df



def main() -> None:
    # ---- manual run config ----
    json_path    = REPO_ROOT / "datasets" / "Celeb-DF-v2.json"   # produced by rearrange.py
    dataset_name = "Celeb-DF-v2"
    model_name   = "ViT-L-14-336-quickgelu"
    pretrained   = "openai"
    device       = "cuda"
    T            = 32
    micro_bs     = 32
    seed         = 0
    out_dir      = None   # None → auto-create under my_project/clip/embeddings/
    max_videos   = None   # set e.g. 100 for a quick smoke-test
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
        out_dir=out_dir,
        max_videos=max_videos,
    )

    print(f"Embedding finished. Failures: {len(failures)}")
    if failures:
        print(f"First failure: {failures[0]}")


if __name__ == "__main__":
    main()
