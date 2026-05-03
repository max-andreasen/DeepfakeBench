"""
benchmark_inference.py — run all retrained seed models across multiple benchmarks.

Reads a retrain_top_k output directory, finds every trial_XXXX/seed_YYYY/model.pth,
and evaluates each model on every benchmark listed in the config. Results are
saved in a clean nested structure ready for statistical_analysis.py.

Output layout:
  <out_dir>/
    summary.csv                              — one row per (trial, seed, benchmark)
    trial_XXXX/
      seed_YYYY/
        <benchmark_name>/
          results.json                       — full per-window + per-video metrics
    run_config.json                          — metadata + benchmark list

Usage (from repo root):
    python evaluation/benchmark_inference.py \\
        --config evaluation/configs/benchmark_inference.yaml
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
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR  = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
    p = argparse.ArgumentParser(description='Run saved seed models across multiple benchmarks.')
    p.add_argument('--config', required=True, help='Path to benchmark_inference.yaml')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading (mirrors eval_top_k.load_model)
# ---------------------------------------------------------------------------
def load_model(seed_dir: Path):
    run_config_path = seed_dir / 'run_config.json'
    model_path      = seed_dir / 'model.pth'

    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json missing: {run_config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth missing: {model_path}")

    with open(run_config_path) as f:
        run_config = json.load(f)

    model_type   = run_config['model_type']
    model_kwargs = run_config.get('model_kwargs', {})
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type '{model_type}'; known: {list(MODELS)}")

    model = MODELS[model_type](**model_kwargs)
    state = torch.load(str(model_path), map_location='cpu')
    model.load_state_dict(state)
    return model, run_config


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def build_loader(catalogue_file: str, split_csv: str, split: str,
                 num_frames: int, batch_size: int, input_transform: str) -> DataLoader:
    dataset = DeepfakeTestDataset(
        split_file=split_csv,
        catalogue_file=catalogue_file,
        num_frames=num_frames,
        split=split,
        input_transform=input_transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def make_temp_split_csv(catalogue_file: str, tmp_dir: str, suffix: str = "") -> str:
    """All videos in catalogue → split=test (for cross-dataset benchmarks).

    The catalogue file is conventionally named `catalogue.csv` regardless of
    the dataset, so the suffix (typically the benchmark name) is required to
    keep per-benchmark temp split files from clobbering each other.
    """
    cat = pd.read_csv(catalogue_file)
    join_keys = [k for k in ['dataset', 'label_cat', 'video_id'] if k in cat.columns]
    split_df = cat[join_keys].copy()
    split_df['split'] = 'test'
    stem = Path(catalogue_file).stem
    name = f"{stem}_{suffix}" if suffix else stem
    path = os.path.join(tmp_dir, f'_split_{name}.csv')
    split_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Discovery: find all seed dirs with a model.pth
# ---------------------------------------------------------------------------
def find_seed_dirs(retrain_dir: Path) -> list[tuple[int, int, Path]]:
    """Return list of (trial_num, seed_num, seed_dir) for every saved model."""
    results = []
    for trial_dir in sorted(retrain_dir.glob('trial_*')):
        try:
            trial_num = int(trial_dir.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        for seed_dir in sorted(trial_dir.glob('seed_*')):
            if (seed_dir / 'model.pth').exists():
                try:
                    seed_num = int(seed_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue
                results.append((trial_num, seed_num, seed_dir))
    return results


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = ['trial', 'seed', 'benchmark', 'test_auc', 'test_acc',
               'test_acc_at_best', 'test_f1', 'num_videos']

def append_summary_row(rows: list, trial: int, seed: int,
                       benchmark: str, result: dict):
    pv = result['per_video']
    rows.append({
        'trial':           trial,
        'seed':            seed,
        'benchmark':       benchmark,
        'test_auc':        round(pv['auc'], 6),
        'test_acc':        round(pv['accuracy'], 6),
        'test_acc_at_best': round(pv['acc_at_best_thresh'], 6),
        'test_f1':         round(pv['f1'], 6),
        'num_videos':      result['num_videos'],
    })


def write_summary_csv(rows: list, out_path: Path):
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retrain_dir = Path(cfg['retrain_dir'])
    out_dir     = Path(cfg['out_dir'])
    benchmarks  = cfg['benchmarks']
    num_frames  = int(cfg.get('num_frames', 32))
    batch_size  = int(cfg.get('batch_size', 64))
    aggregation = cfg.get('window_aggregation', 'mean')

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(out_dir / 'benchmark_inference.log'))
    logger.info("=== benchmark_inference ===")
    logger.info(f"  retrain_dir : {retrain_dir}")
    logger.info(f"  out_dir     : {out_dir}")
    logger.info(f"  benchmarks  : {[b['name'] for b in benchmarks]}")

    seed_entries = find_seed_dirs(retrain_dir)
    logger.info(f"  Found {len(seed_entries)} seed models across "
                f"{len(set(t for t,_,_ in seed_entries))} trials")

    summary_rows = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pre-build split CSVs for each benchmark once.
        split_csvs = {}
        for b in benchmarks:
            name = b['name']
            if 'split_file' in b:
                split_csvs[name] = (str(REPO_ROOT / b['split_file']),
                                    b.get('split', 'test'))
            else:
                csv_path = make_temp_split_csv(
                    str(REPO_ROOT / b['catalogue_file']), tmp_dir, suffix=name
                )
                split_csvs[name] = (csv_path, 'test')

        for trial_num, seed_num, seed_dir in seed_entries:
            logger.info(f"--- trial_{trial_num:04d} / seed_{seed_num:04d} ---")

            try:
                model, run_config = load_model(seed_dir)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"  Skipping: {e}")
                continue

            input_transform = run_config.get('input_transform', 'none')
            eval_cfg = {'window_aggregation': aggregation}

            for b in benchmarks:
                name            = b['name']
                catalogue_file  = str(REPO_ROOT / b['catalogue_file'])
                split_csv, split = split_csvs[name]

                result_dir = out_dir / f'trial_{trial_num:04d}' / f'seed_{seed_num:04d}' / name
                result_dir.mkdir(parents=True, exist_ok=True)
                result_file = result_dir / 'results.json'

                if result_file.exists():
                    logger.info(f"  [{name}] already done, skipping")
                    with open(result_file) as f:
                        result = json.load(f)
                    append_summary_row(summary_rows, trial_num, seed_num, name, result)
                    continue

                try:
                    loader = build_loader(catalogue_file, split_csv, split,
                                          num_frames, batch_size, input_transform)
                    tester = Tester(eval_cfg, model, logger)
                    result = tester.evaluate(loader, shuffle_frames=False)

                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)

                    append_summary_row(summary_rows, trial_num, seed_num, name, result)
                    logger.info(f"  [{name}] auc={result['per_video']['auc']:.4f}")

                    del tester, loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"  [{name}] failed: {e}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_summary_csv(summary_rows, out_dir / 'summary.csv')
    logger.info(f"Wrote summary.csv ({len(summary_rows)} rows)")

    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump({
            'evaluated_utc': datetime.now(timezone.utc).isoformat(),
            'retrain_dir':   str(retrain_dir),
            'out_dir':       str(out_dir),
            'benchmarks':    [b['name'] for b in benchmarks],
            'num_frames':    num_frames,
            'batch_size':    batch_size,
            'window_aggregation': aggregation,
        }, f, indent=2)


if __name__ == '__main__':
    main()
