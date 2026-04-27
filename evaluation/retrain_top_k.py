"""
retrain_top_k.py — statistically robust cross-dataset evaluation of top-K
param-search configurations.

For each of the K best hyperparameter configs (ranked by FF++ val AUC from
the param search):
  1. Retrains the model N times from scratch, each with a different seed.
  2. After each training run, evaluates on the held-out benchmark (CDFv2).
  3. Aggregates per-config: mean ± std AUC across seeds.

This allows proper statistical claims about which config generalises better.

Output:
  <out_dir>/results.csv               — K rows sorted by mean test AUC
  <out_dir>/run_config.json           — eval metadata (inputs, seeds, UTC)
  <out_dir>/eval_top_k.log
  <out_dir>/trial_XXXX/seed_YYYY/     — model.pth, run_config.json,
                                         training.log, test_results.json
  <out_dir>/trial_XXXX/summary.json   — mean/std/min/max AUC for this config

Usage (from repo root):
    python evaluation/retrain_top_k.py \\
        --study_dir  training/searches/runs/transformer_search3 \\
        --catalogue_file clip/embeddings/benchmarks/cdfv2_layer16/ViT-L-14-336-quickgelu/block_16/catalogue.csv \\
        --out_dir    evaluation/results/retrain_top_k/transformer_search3 \\
        --top_k 10 --n_seeds 5
"""

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]

# Repo root: models/, logger.py
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# training/: train.py, data_loader.py (training version), trainer.py
TRAINING_DIR = REPO_ROOT / 'training'
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

# Evaluation data_loader and tester loaded in isolation to avoid shadowing
# training/data_loader.py (both are named data_loader).
def _load_isolated(alias, path):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

_eval_dl     = _load_isolated('eval_data_loader',  REPO_ROOT / 'evaluation' / 'data_loader.py')
_eval_tester = _load_isolated('eval_tester',        REPO_ROOT / 'evaluation' / 'tester.py')
DeepfakeTestDataset = _eval_dl.DeepfakeTestDataset
Tester              = _eval_tester.Tester

from train import train_from_config  # type: ignore[import-not-found]
from logger import create_logger     # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Retrain top-K param-search configs N times and evaluate cross-dataset.'
    )
    p.add_argument('--study_dir', required=True,
                   help='Path to Optuna run dir (contains all_trials.csv + trial_XXXX/)')
    p.add_argument('--catalogue_file', required=True,
                   help='Held-out benchmark catalogue CSV (e.g. CDFv2 block_16)')
    p.add_argument('--out_dir', required=True,
                   help='Output directory (created if absent)')
    p.add_argument('--top_k', type=int, default=10,
                   help='Number of top configs to evaluate (default: 10)')
    p.add_argument('--n_seeds', type=int, required=True,
                   help='Number of training runs (seeds) per config')
    p.add_argument('--base_seed', type=int, default=0,
                   help='Seeds used: base_seed, base_seed+1, ..., base_seed+n_seeds-1 (default: 0)')
    p.add_argument('--num_epochs', type=int, default=None,
                   help='Override training epochs (default: use value from run_config.json)')
    p.add_argument('--batch_size_eval', type=int, default=64,
                   help='Eval batch size in videos (default: 64)')
    p.add_argument('--window_aggregation', default='mean',
                   choices=['mean', 'max', 'softmax'])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_top_k_trials(study_dir: Path, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(study_dir / 'all_trials.csv')
    complete = df[df['state'] == 'COMPLETE'].copy()
    complete = complete.sort_values('value', ascending=False).reset_index(drop=True)
    result = complete.head(top_k)
    if len(result) < top_k:
        print(f"[WARN] Only {len(result)} COMPLETE trials available (requested {top_k})")
    return result


def make_split_csv(catalogue_file: Path, tmp_dir: str) -> str:
    """Write a temp split CSV where every video in the catalogue is split=test."""
    cat = pd.read_csv(catalogue_file)
    join_keys = [k for k in ['dataset', 'label_cat', 'video_id'] if k in cat.columns]
    split_df = cat[join_keys].copy()
    split_df['split'] = 'test'
    path = os.path.join(tmp_dir, '_split.csv')
    split_df.to_csv(path, index=False)
    return path


def build_test_loader(catalogue_file: Path, split_csv: str, num_frames: int,
                      batch_size: int, input_transform: str) -> DataLoader:
    dataset = DeepfakeTestDataset(
        split_file=split_csv,
        catalogue_file=str(catalogue_file),
        num_frames=num_frames,
        split='test',
        input_transform=input_transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# ---------------------------------------------------------------------------
# Config reconstruction
# ---------------------------------------------------------------------------

def reconstruct_train_config(run_config: dict, seed: int, log_dir: str,
                              num_epochs_override=None) -> dict:
    """Map a saved run_config.json back to the dict shape train_from_config expects.

    run_config.json uses a curated nested layout (trainer.save_run_config);
    train_from_config reads a flat-ish config dict with different key names.
    """
    model_type = run_config['model_type']
    opt_type   = run_config['optimizer']['type']

    cfg = {
        # Model
        'model_type': model_type,
        'model':      {model_type: run_config['model_kwargs']},

        # Optimizer
        'optimizer': {
            'type':  opt_type,
            opt_type: run_config['optimizer']['params'],
        },

        # Scheduler
        'lr_scheduler':  run_config['lr_scheduler'],
        'warmup_epochs': run_config.get('warmup_epochs', 5),

        # Training loop
        'num_epochs':      num_epochs_override or run_config['num_epochs'],
        'batchSize':       run_config['batch_size'],
        'num_frames':      run_config['num_frames'],
        'metric_scoring':  run_config.get('metric_scoring', 'auc'),
        'input_transform': run_config.get('input_transform', 'none'),

        # Seed (overridden per run)
        'seed': seed,

        # Data
        'root_dir':    run_config['data'].get('root_dir', ''),
        'split_file':  run_config['data']['split_file'],
        'catalogue_file': run_config['data']['catalogue_file'],
        'train_dataset':  run_config['data'].get('train_dataset'),
        'val_dataset':    run_config['data'].get('val_dataset'),
        'compression':    run_config['data'].get('compression'),

        # Augmentation
        'use_data_augmentation': run_config.get('use_data_augmentation', False),
        'data_aug':              run_config.get('data_aug'),

        # Misc
        'workers':   4,
        'save_ckpt': True,
        'log_dir':   log_dir,  # unused when log_path is explicit; kept for completeness
    }
    return cfg


# ---------------------------------------------------------------------------
# Per-seed: train + evaluate
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    train_config: dict,
    seed_dir: Path,
    catalogue_file: Path,
    split_csv: str,
    num_frames_eval: int,
    batch_size_eval: int,
    window_aggregation: str,
    logger,
) -> float:
    """Train one model, evaluate on benchmark, return per-video AUC."""
    seed_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    result = train_from_config(train_config, log_path=str(seed_dir))
    model = result['model']
    logger.info(f"  Seed {seed}: best_val_auc={result['best_val_auroc']:.4f}  "
                f"final_val_auc={result['final_val_auroc']:.4f}  "
                f"epochs={result['epochs_completed']}")

    # --- Evaluate on held-out benchmark ---
    input_transform = train_config.get('input_transform', 'none')
    loader = build_test_loader(catalogue_file, split_csv, num_frames_eval,
                               batch_size_eval, input_transform)
    eval_cfg = {'window_aggregation': window_aggregation}
    tester = Tester(eval_cfg, model, logger)
    test_result = tester.evaluate(loader, shuffle_frames=False)

    test_auc = test_result['per_video']['auc']
    logger.info(f"  Seed {seed}: test_auc={test_auc:.4f}")

    with open(seed_dir / 'test_results.json', 'w') as f:
        json.dump(test_result, f, indent=2)

    # --- Cleanup ---
    del model, tester, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_auc


# ---------------------------------------------------------------------------
# Per-config: loop over seeds
# ---------------------------------------------------------------------------

def run_config_seeds(
    trial_num: int,
    val_auc_search: float,
    seeds: list,
    study_dir: Path,
    out_dir: Path,
    catalogue_file: Path,
    split_csv: str,
    num_epochs_override,
    batch_size_eval: int,
    window_aggregation: str,
    logger,
) -> dict:
    """Retrain a config N times. Returns per-config summary dict."""
    trial_dir    = study_dir / f'trial_{trial_num:04d}'
    trial_out    = out_dir / f'trial_{trial_num:04d}'
    trial_out.mkdir(parents=True, exist_ok=True)

    run_config_path = trial_dir / 'run_config.json'
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json missing: {run_config_path}")

    with open(run_config_path) as f:
        saved_run_config = json.load(f)

    num_frames_eval = saved_run_config['num_frames']['val']
    aucs = []

    for seed in seeds:
        seed_dir = trial_out / f'seed_{seed:04d}'
        logger.info(f"  [trial_{trial_num:04d} | seed={seed}]")

        train_cfg = reconstruct_train_config(
            saved_run_config, seed=seed,
            log_dir=str(trial_out),
            num_epochs_override=num_epochs_override,
        )
        try:
            auc = run_seed(
                seed=seed,
                train_config=train_cfg,
                seed_dir=seed_dir,
                catalogue_file=catalogue_file,
                split_csv=split_csv,
                num_frames_eval=num_frames_eval,
                batch_size_eval=batch_size_eval,
                window_aggregation=window_aggregation,
                logger=logger,
            )
            aucs.append(auc)
        except Exception as exc:
            logger.warning(f"  Seed {seed} failed: {exc}")

    summary = {
        'trial': trial_num,
        'val_auc_search': val_auc_search,
        'seeds': seeds,
        'aucs': aucs,
        'n_completed': len(aucs),
        'mean_test_auc': float(np.mean(aucs)) if aucs else float('nan'),
        'std_test_auc':  float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float('nan'),
        'min_test_auc':  float(np.min(aucs)) if aucs else float('nan'),
        'max_test_auc':  float(np.max(aucs)) if aucs else float('nan'),
    }
    with open(trial_out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    'final_rank', 'trial', 'val_auc_search',
    'mean_test_auc', 'std_test_auc', 'min_test_auc', 'max_test_auc',
    'n_completed',
]

def write_results_csv(summaries: list, out_path: Path):
    # Sort by mean_test_auc desc; failed configs (nan) go last
    def sort_key(s):
        v = s['mean_test_auc']
        return v if not np.isnan(v) else -1.0

    summaries = sorted(summaries, key=sort_key, reverse=True)
    for rank, s in enumerate(summaries, start=1):
        s['final_rank'] = rank

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for s in summaries:
            row = {k: ('' if isinstance(s.get(k), float) and np.isnan(s.get(k)) else s.get(k, ''))
                   for k in _CSV_FIELDS}
            writer.writerow(row)

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    study_dir      = Path(args.study_dir)
    catalogue_file = Path(args.catalogue_file)
    out_dir        = Path(args.out_dir)
    seeds          = list(range(args.base_seed, args.base_seed + args.n_seeds))

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(out_dir / 'retrain_top_k.log'))
    logger.info("=== retrain_top_k ===")
    logger.info(f"  study_dir      : {study_dir}")
    logger.info(f"  catalogue_file : {catalogue_file}")
    logger.info(f"  out_dir        : {out_dir}")
    logger.info(f"  top_k={args.top_k}  n_seeds={args.n_seeds}  seeds={seeds}")
    logger.info(f"  num_epochs_override={args.num_epochs}  "
                f"batch_size_eval={args.batch_size_eval}  aggregation={args.window_aggregation}")

    top_k_df = get_top_k_trials(study_dir, args.top_k)
    logger.info(f"Trials selected: {top_k_df['number'].tolist()}")

    summaries = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        split_csv = make_split_csv(catalogue_file, tmp_dir)
        logger.info(f"Derived split CSV from catalogue (all → split=test): {split_csv}")

        for _, trial_row in top_k_df.iterrows():
            trial_num   = int(trial_row['number'])
            val_auc     = float(trial_row['value'])
            logger.info(f"=== trial_{trial_num:04d}  val_auc_search={val_auc:.4f} ===")

            try:
                summary = run_config_seeds(
                    trial_num=trial_num,
                    val_auc_search=val_auc,
                    seeds=seeds,
                    study_dir=study_dir,
                    out_dir=out_dir,
                    catalogue_file=catalogue_file,
                    split_csv=split_csv,
                    num_epochs_override=args.num_epochs,
                    batch_size_eval=args.batch_size_eval,
                    window_aggregation=args.window_aggregation,
                    logger=logger,
                )
            except FileNotFoundError as exc:
                logger.warning(f"Skipping trial_{trial_num:04d}: {exc}")
                summary = {
                    'trial': trial_num, 'val_auc_search': val_auc,
                    'seeds': seeds, 'aucs': [], 'n_completed': 0,
                    'mean_test_auc': float('nan'), 'std_test_auc': float('nan'),
                    'min_test_auc': float('nan'), 'max_test_auc': float('nan'),
                }

            summaries.append(summary)
            logger.info(
                f"  → mean={summary['mean_test_auc']:.4f}  "
                f"std={summary['std_test_auc']:.4f}  "
                f"n={summary['n_completed']}"
            )

    ranked = write_results_csv(summaries, out_dir / 'results.csv')
    logger.info(f"Wrote results.csv")

    run_cfg_out = {
        'evaluated_utc':    datetime.now(timezone.utc).isoformat(),
        'study_dir':        str(study_dir),
        'catalogue_file':   str(catalogue_file),
        'out_dir':          str(out_dir),
        'top_k':            args.top_k,
        'n_seeds':          args.n_seeds,
        'seeds':            seeds,
        'base_seed':        args.base_seed,
        'num_epochs_override': args.num_epochs,
        'batch_size_eval':  args.batch_size_eval,
        'window_aggregation': args.window_aggregation,
    }
    with open(out_dir / 'run_config.json', 'w') as f:
        json.dump(run_cfg_out, f, indent=2)

    logger.info("=== Top-5 by mean test AUC ===")
    for s in ranked[:5]:
        logger.info(
            f"  rank={s['final_rank']}  trial={s['trial']}  "
            f"val_auc_search={s['val_auc_search']:.4f}  "
            f"mean_test_auc={s['mean_test_auc']:.4f}  "
            f"std={s['std_test_auc']:.4f}  n={s['n_completed']}"
        )


if __name__ == '__main__':
    main()
