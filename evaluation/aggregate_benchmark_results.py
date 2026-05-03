"""
aggregate_benchmark_results.py — compute per-benchmark mean ± std across seeds.

Discovers every model subdirectory under --results_dir that contains a
summary.csv, then computes mean and std (ddof=1) across all seeds for AUC,
accuracy, acc_at_best, and F1. Writes one tidy CSV ready for Excel or plotting.

Output columns:
  model, benchmark, n_seeds,
  mean_auc, std_auc,
  mean_acc, std_acc,
  mean_acc_at_best, std_acc_at_best,
  mean_f1, std_f1

Usage (from repo root):
    python evaluation/aggregate_benchmark_results.py
    python evaluation/aggregate_benchmark_results.py \\
        --results_dir evaluation/results/benchmark_inference \\
        --out evaluation/results/benchmark_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


METRIC_MAP = {
    'test_auc':         'auc',
    'test_acc':         'acc',
    'test_acc_at_best': 'acc_at_best',
    'test_f1':          'f1',
}

BENCHMARK_ORDER = [
    'FF++_test', 'CDFv1', 'CDFv2', 'CDFv3',
    'DFD', 'FaceShifter', 'WildDF',
]


def parse_args():
    p = argparse.ArgumentParser(description='Aggregate benchmark_inference summary CSVs.')
    p.add_argument(
        '--results_dir',
        default='evaluation/results/benchmark_inference',
        help='Parent directory containing one subdir per model type (default: %(default)s)',
    )
    p.add_argument(
        '--out',
        default='evaluation/results/benchmark_summary.csv',
        help='Output CSV path (default: %(default)s)',
    )
    return p.parse_args()


def load_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_file = model_dir / 'summary.csv'
        if not summary_file.exists():
            print(f"  [skip] no summary.csv in {model_dir.name}/")
            continue
        df = pd.read_csv(summary_file)
        df.insert(0, 'model', model_dir.name)
        frames.append(df)
        print(f"  [ok]   {model_dir.name}/summary.csv — {len(df)} rows")
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, benchmark), grp in df.groupby(['model', 'benchmark'], sort=False):
        row = {
            'model':     model,
            'benchmark': benchmark,
            'n_seeds':   len(grp),
        }
        for src_col, short in METRIC_MAP.items():
            if src_col in grp.columns:
                row[f'mean_{short}'] = grp[src_col].mean()
                row[f'std_{short}']  = grp[src_col].std(ddof=1)
        rows.append(row)

    out = pd.DataFrame(rows)

    # Sort: model alphabetically, benchmark by canonical order (unknowns go last).
    bench_rank = {b: i for i, b in enumerate(BENCHMARK_ORDER)}
    out['_bench_rank'] = out['benchmark'].map(lambda b: bench_rank.get(b, 999))
    out = out.sort_values(['model', '_bench_rank']).drop(columns='_bench_rank')

    col_order = [
        'model', 'benchmark', 'n_seeds',
        'mean_auc', 'std_auc',
        'mean_acc', 'std_acc',
        'mean_acc_at_best', 'std_acc_at_best',
        'mean_f1', 'std_f1',
    ]
    return out[[c for c in col_order if c in out.columns]].reset_index(drop=True)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path    = Path(args.out)

    print(f"Reading from: {results_dir}")
    df = load_summaries(results_dir)
    print(f"Total rows loaded: {len(df)}")

    summary = aggregate(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, float_format='%.6f')

    print(f"\nWritten: {out_path} ({len(summary)} rows)\n")
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
