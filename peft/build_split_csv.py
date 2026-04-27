"""
Synthesize a split CSV from a rearrange JSON.

Reads `preprocessing/rearrangements/dataset_json_mtcnn/<dataset>.json`,
writes `datasets/splits/<dataset>.csv` with columns:
    dataset, label_cat, video_id, split

The split tag is taken straight from the JSON's split-level keys
('train' / 'val' / 'test'), so this preserves the dataset's official splits.

If the JSON has no split-level structure (every video sits directly under
label_cat), the whole dataset is tagged as `split=test` (cross-dataset eval
default — see peft/IMPLEMENTATION_PLAN.md §5 Step 9).

Usage:
    python peft/build_split_csv.py --dataset Celeb-DF-v2
    python peft/build_split_csv.py --dataset Celeb-DF-v2 --json <path> --out <path>
"""

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flatten_json(json_path: Path, dataset_name: str):
    with open(json_path) as f:
        data = json.load(f)
    if dataset_name not in data:
        raise ValueError(
            f"'{dataset_name}' not in {json_path}. Keys: {list(data.keys())}"
        )

    rows = []
    for label_cat, splits in data[dataset_name].items():
        for split_name, videos_or_comp in splits.items():
            # FF++ has split -> compression -> videos. Detect by checking
            # whether the first inner value has a 'frames' key.
            first = next(iter(videos_or_comp.values()), None)
            if first is not None and isinstance(first, dict) and "frames" not in first:
                videos = {}
                for _comp, comp_videos in videos_or_comp.items():
                    videos.update(comp_videos)
            else:
                videos = videos_or_comp

            for video_id in videos.keys():
                rows.append({
                    "dataset":   dataset_name,
                    "label_cat": label_cat,
                    "video_id":  str(video_id),
                    "split":     split_name,
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Synthesize a split CSV from a rearrange JSON.")
    ap.add_argument("--dataset", type=str, required=True,
                    help="dataset_name as it appears in the rearrange JSON top-level key.")
    ap.add_argument("--json", type=str, default=None,
                    help="path to rearrange JSON. defaults to "
                         "preprocessing/rearrangements/dataset_json_mtcnn/<dataset>.json")
    ap.add_argument("--out", type=str, default=None,
                    help="output CSV path. defaults to datasets/splits/<dataset>.csv")
    ap.add_argument("--all_test", action="store_true",
                    help="ignore JSON splits, mark every row split=test "
                         "(cross-dataset eval fallback).")
    args = ap.parse_args()

    json_path = Path(args.json) if args.json else (
        REPO_ROOT / "preprocessing" / "rearrangements" / "dataset_json_mtcnn"
        / f"{args.dataset}.json"
    )
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "datasets" / "splits" / f"{args.dataset}.csv"
    )

    rows = _flatten_json(json_path, args.dataset)
    if args.all_test:
        for r in rows:
            r["split"] = "test"

    # Dedup must include split: some datasets (e.g. CDFv2) tag the same
    # video_id under multiple splits. Dropping on the 3-tuple silently loses
    # rows; the 4-tuple preserves split-level multiplicity, and the dataset
    # class's split filter selects the right rows downstream.
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "label_cat", "video_id", "split"]
    ).sort_values(["dataset", "split", "label_cat", "video_id"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"wrote {out_path}  ({len(df)} rows)")
    print(df.groupby(["split", "label_cat"]).size().to_string())


if __name__ == "__main__":
    main()
