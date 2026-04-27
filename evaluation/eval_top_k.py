"""
eval_top_k.py — evaluate the top-K trials from a param-search run on a
held-out embedding dataset (e.g. CDFv2 benchmarks).

For each of the K trials (ranked by param-search val AUC, high → low):
  1. Loads model from trial_XXXX/run_config.json + model.pth
  2. Runs Tester.evaluate() (no frame shuffling) on all videos in the catalogue
  3. Writes per-trial JSON to <out_dir>/trial_XXXX/

Outputs:
  <out_dir>/run_config.json   — eval metadata (inputs, settings, UTC timestamp)
  <out_dir>/results.csv       — one row per trial, sorted by test AUC desc
  <out_dir>/trial_XXXX/       — results.json + run_config.json copy per trial
  <out_dir>/eval_top_k.log    — full log

CDFv2 (and other cross-dataset benchmarks) have no pre-existing split CSV
because every video is a test video. This script derives the split from the
catalogue itself: all rows become split=test.

Usage (from repo root):
    python evaluation/eval_top_k.py \\
        --study_dir  training/searches/runs/transformer_search3 \\
        --catalogue_file clip/embeddings/benchmarks/cdfv2_layer16/ViT-L-14-336-quickgelu/block_16/catalogue.csv \\
        --out_dir    evaluation/results/cdfv2_top_k/transformer_search3 \\
        [--top_k 10] [--num_frames 32] [--batch_size 64] [--window_aggregation mean]
"""

import argparse
import csv
import gc
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent

# Make repo root importable (models/, logger.py)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Make evaluation/ importable (data_loader.py, tester.py)
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from data_loader import DeepfakeTestDataset   # type: ignore[import-not-found]
from tester import Tester                     # type: ignore[import-not-found]
from logger import create_logger              # type: ignore[import-not-found]
from models import MODELS                     # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate top-K param-search trials on a held-out dataset.'
    )
    p.add_argument('--study_dir', required=True,
                   help='Path to Optuna run dir (must contain all_trials.csv and trial_XXXX/)')
    p.add_argument('--catalogue_file', required=True,
                   help='Path to embedding catalogue CSV (the held-out benchmark)')
    p.add_argument('--out_dir', required=True,
                   help='Output directory (created if absent)')
    p.add_argument('--top_k', type=int, default=10,
                   help='Number of top trials to evaluate (default: 10)')
    p.add_argument('--num_frames', type=int, default=32,
                   help='Frames per sliding window; must match training T (default: 32)')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Eval batch size in videos (default: 64)')
    p.add_argument('--window_aggregation', default='mean',
                   choices=['mean', 'max', 'softmax'],
                   help='How to pool per-window logits to a per-video score (default: mean)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_top_k_trials(study_dir: Path, top_k: int) -> pd.DataFrame:
    """Return top-k COMPLETE trials from all_trials.csv, sorted value desc."""
    df = pd.read_csv(study_dir / 'all_trials.csv')
    complete = df[df['state'] == 'COMPLETE'].copy()
    complete = complete.sort_values('value', ascending=False).reset_index(drop=True)
    result = complete.head(top_k)
    if len(result) < top_k:
        print(f"[WARN] Only {len(result)} COMPLETE trials found (requested top_k={top_k})")
    return result


def make_split_df(catalogue_file: Path) -> pd.DataFrame:
    """Derive a split DataFrame from a catalogue: every video → split=test.

    Used for datasets (CDFv2, CDFv3, WDF) that have no pre-existing split CSV
    because all their videos are cross-dataset test material.
    """
    cat = pd.read_csv(catalogue_file)
    join_keys = [k for k in ['dataset', 'label_cat', 'video_id'] if k in cat.columns]
    split_df = cat[join_keys].copy()
    split_df['split'] = 'test'
    return split_df


def build_loader(
    catalogue_file: Path,
    split_csv_path: str,
    num_frames: int,
    batch_size: int,
    input_transform: str,
) -> DataLoader:
    dataset = DeepfakeTestDataset(
        split_file=split_csv_path,
        catalogue_file=str(catalogue_file),
        num_frames=num_frames,
        split='test',
        input_transform=input_transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(trial_dir: Path):
    """Load model + run_config from a trial directory.

    Returns (model, run_config_dict). Raises FileNotFoundError if either
    run_config.json or model.pth is missing.
    """
    run_config_path = trial_dir / 'run_config.json'
    model_path = trial_dir / 'model.pth'

    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json missing: {run_config_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"model.pth missing: {model_path} "
            "(trial was not in top-K, or checkpoint was pruned)"
        )

    with open(run_config_path) as f:
        run_config = json.load(f)

    model_type = run_config['model_type']
    model_kwargs = run_config.get('model_kwargs', {})
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'; known: {list(MODELS)}")

    model = MODELS[model_type](**model_kwargs)
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state)
    return model, run_config


# ---------------------------------------------------------------------------
# Per-trial evaluation
# ---------------------------------------------------------------------------

def eval_trial(
    trial_dir: Path,
    catalogue_file: Path,
    split_csv_path: str,
    num_frames: int,
    batch_size: int,
    window_aggregation: str,
    out_dir: Path,
    logger,
) -> dict:
    """Evaluate one trial. Returns the Tester result dict (per_video / per_window).

    Writes trial_XXXX/results.json and trial_XXXX/run_config.json into out_dir.
    Frees GPU memory after evaluation.
    """
    model, run_config = load_model(trial_dir)
    input_transform = run_config.get('input_transform', 'none')

    loader = build_loader(catalogue_file, split_csv_path, num_frames, batch_size, input_transform)

    eval_cfg = {'window_aggregation': window_aggregation}
    tester = Tester(eval_cfg, model, logger)
    result = tester.evaluate(loader, shuffle_frames=False)

    trial_out = out_dir / trial_dir.name
    trial_out.mkdir(parents=True, exist_ok=True)
    with open(trial_out / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    with open(trial_out / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    del model, tester, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'final_rank', 'trial', 'val_auc',
    'test_auc', 'test_acc', 'test_acc_at_best',
    'test_f1', 'test_precision', 'test_recall',
    'best_thresh', 'num_videos',
]

def _row_from_result(rank, trial_num, val_auc, result) -> dict:
    pv = result['per_video']
    return {
        'final_rank': rank,
        'trial': trial_num,
        'val_auc': round(val_auc, 6),
        'test_auc': round(pv['auc'], 6),
        'test_acc': round(pv['accuracy'], 6),
        'test_acc_at_best': round(pv['acc_at_best_thresh'], 6),
        'test_f1': round(pv['f1'], 6),
        'test_precision': round(pv['precision'], 6),
        'test_recall': round(pv['recall'], 6),
        'best_thresh': round(pv['best_thresh'], 6),
        'num_videos': result['num_videos'],
    }


def _skipped_row(rank, trial_num, val_auc) -> dict:
    return {
        'final_rank': rank,
        'trial': trial_num,
        'val_auc': round(val_auc, 6),
        'test_auc': '', 'test_acc': '', 'test_acc_at_best': '',
        'test_f1': '', 'test_precision': '', 'test_recall': '',
        'best_thresh': '', 'num_videos': 0,
    }


def write_results_csv(rows: list, out_path: Path):
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    study_dir = Path(args.study_dir)
    catalogue_file = Path(args.catalogue_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(str(out_dir / 'eval_top_k.log'))
    logger.info("=== eval_top_k ===")
    logger.info(f"  study_dir        : {study_dir}")
    logger.info(f"  catalogue_file   : {catalogue_file}")
    logger.info(f"  out_dir          : {out_dir}")
    logger.info(f"  top_k={args.top_k}  num_frames={args.num_frames}  "
                f"batch_size={args.batch_size}  aggregation={args.window_aggregation}")

    top_k_df = get_top_k_trials(study_dir, args.top_k)
    logger.info(f"Trials to evaluate: {top_k_df['number'].tolist()}")

    split_df = make_split_df(catalogue_file)
    logger.info(f"Derived split from catalogue: {len(split_df)} videos (all → split=test)")

    # Write temp split CSV once; reused by all trials to avoid repeated writes.
    # Cleaned up automatically when the TemporaryDirectory is closed.
    rows_collected = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        split_csv_path = os.path.join(tmp_dir, '_split.csv')
        split_df.to_csv(split_csv_path, index=False)

        for seq_rank, (_, trial_row) in enumerate(top_k_df.iterrows(), start=1):
            trial_num = int(trial_row['number'])
            val_auc = float(trial_row['value'])
            trial_dir = study_dir / f'trial_{trial_num:04d}'

            logger.info(f"--- [{seq_rank}/{len(top_k_df)}] trial_{trial_num:04d}  val_auc={val_auc:.4f} ---")

            try:
                result = eval_trial(
                    trial_dir=trial_dir,
                    catalogue_file=catalogue_file,
                    split_csv_path=split_csv_path,
                    num_frames=args.num_frames,
                    batch_size=args.batch_size,
                    window_aggregation=args.window_aggregation,
                    out_dir=out_dir,
                    logger=logger,
                )
                rows_collected.append(_row_from_result(seq_rank, trial_num, val_auc, result))
            except (FileNotFoundError, ValueError) as exc:
                logger.warning(f"Skipping trial_{trial_num:04d}: {exc}")
                rows_collected.append(_skipped_row(seq_rank, trial_num, val_auc))

    # Sort collected rows by test_auc desc; skipped rows (empty string) go last.
    def sort_key(r):
        v = r['test_auc']
        return float(v) if v != '' else -1.0

    rows_collected.sort(key=sort_key, reverse=True)

    # Re-assign final_rank after sorting.
    for final_rank, row in enumerate(rows_collected, start=1):
        row['final_rank'] = final_rank

    results_csv = out_dir / 'results.csv'
    write_results_csv(rows_collected, results_csv)
    logger.info(f"Wrote {results_csv}")

    run_config_out = {
        'evaluated_utc': datetime.now(timezone.utc).isoformat(),
        'study_dir': str(study_dir),
        'catalogue_file': str(catalogue_file),
        'out_dir': str(out_dir),
        'top_k': args.top_k,
        'num_frames': args.num_frames,
        'batch_size': args.batch_size,
        'window_aggregation': args.window_aggregation,
        'trials_evaluated': [r['trial'] for r in rows_collected if r['test_auc'] != ''],
        'trials_skipped':   [r['trial'] for r in rows_collected if r['test_auc'] == ''],
    }
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(run_config_out, f, indent=2)
    logger.info(f"Wrote run_config.json")

    logger.info("=== Top-5 by test AUC ===")
    for r in rows_collected[:5]:
        if r['test_auc'] != '':
            logger.info(
                f"  rank={r['final_rank']}  trial={r['trial']}  "
                f"val_auc={r['val_auc']:.4f}  test_auc={r['test_auc']:.4f}  "
                f"acc={r['test_acc']:.4f}"
            )
        else:
            logger.info(f"  rank={r['final_rank']}  trial={r['trial']}  SKIPPED")


if __name__ == '__main__':
    main()
