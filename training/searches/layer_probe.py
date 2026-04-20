"""Layer probing orchestrator.

One invocation = one model (linear / transformer / bigru), trained and
evaluated on every CLIP layer under --embeddings-dir. For each layer:
    * train on FF++ train split,
    * evaluate on FF++ test split (standard + frame-shuffled),
    * write run_dir/results.json.

After the grid, writes summary.csv and emits two PNGs:
    * plots/auc_<model_type>.png          per-window + per-video AUC
    * plots/temporal_gap_<model_type>.png std − shuffled per-video AUC

Plot-only mode regenerates plots from one or more existing summary.csv
files; with >1 distinct model_type it additionally emits
auc_combined.png and temporal_gap_combined.png for cross-model comparison.

No modifications to training/train.py, training/trainer.py,
evaluation/test.py, or evaluation/tester.py — this is a pure orchestrator.

Usage:
    python training/searches/layer_probe.py \\
        --base-config    training/configs/linear.yaml \\
        --embeddings-dir clip/embeddings/probing/ViT-L-14-336-quickgelu \\
        --out-dir        logs/probing/

    python training/searches/layer_probe.py --plot-only \\
        --from-summaries logs/probing/<ts>_linear/summary.csv \\
                         logs/probing/<ts>_transformer/summary.csv \\
                         logs/probing/<ts>_bigru/summary.csv \\
        --plot-out-dir   logs/probing/combined/plots/
"""

import argparse
import copy
import csv
import gc
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# ---- path setup (same pattern as parameter_search.py) ----
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'training'))

from train import train_from_config  # noqa: E402
from logger import create_logger  # noqa: E402


def _load_isolated(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_eval_dl = _load_isolated('eval_data_loader', REPO_ROOT / 'evaluation' / 'data_loader.py')
_eval_tester = _load_isolated('eval_tester', REPO_ROOT / 'evaluation' / 'tester.py')
DeepfakeTestDataset = _eval_dl.DeepfakeTestDataset
Tester = _eval_tester.Tester


# Forward-pass depth order used for the x-axis in plots.
LAYER_ORDER = [
    'block_4', 'block_8', 'block_12', 'block_16', 'block_20', 'block_22',
    'pre_proj', 'final',
]

SUMMARY_FIELDS = [
    'layer', 'model_type', 'clip_embed_dim',
    'best_val_auc',
    'test_auc_std_window', 'test_auc_std_video',
    'test_auc_shuf_window', 'test_auc_shuf_video',
    'temporal_gap_video',
    'wallclock_s',
]


# ---- layer discovery + config building ----

def discover_layers(embeddings_dir, requested):
    available = sorted(
        p.name for p in embeddings_dir.iterdir()
        if p.is_dir() and (p / 'catalogue.csv').is_file()
    )
    if not available:
        raise FileNotFoundError(
            f"No <layer>/catalogue.csv under {embeddings_dir}"
        )
    if requested:
        missing = [l for l in requested if l not in available]
        if missing:
            raise ValueError(
                f"Requested layers not found under {embeddings_dir}: {missing}. "
                f"Available: {available}"
            )
        selected = list(requested)
    else:
        selected = available
    known = [l for l in LAYER_ORDER if l in selected]
    unknown = sorted(l for l in selected if l not in LAYER_ORDER)
    return known + unknown


def clip_embed_dim_for(layer):
    return 768 if layer == 'final' else 1024


def build_cell_config(base_config, embeddings_dir, layer):
    cfg = copy.deepcopy(base_config)
    # Absolute catalogue path; os.path.join with root_dir keeps the abs path.
    cfg['catalogue_file'] = str(embeddings_dir / layer / 'catalogue.csv')
    cfg['model'][cfg['model_type']]['clip_embed_dim'] = clip_embed_dim_for(layer)
    # Per-cell log_dir isn't used (train_from_config receives explicit log_path),
    # but keep a plausible value in the snapshot.
    cfg['save_ckpt'] = True
    return cfg


# ---- evaluation ----
def evaluate_on_split(model, config, split, shuffle_frames, logger):
    """Tester.evaluate on one split. num_workers=0 because DeepfakeTestDataset
    was loaded via importlib under a custom module name (can't pickle to
    worker processes)."""
    root = config.get('root_dir', '')
    dataset = DeepfakeTestDataset(
        split_file=os.path.join(root, config['split_file']),
        catalogue_file=os.path.join(root, config['catalogue_file']),
        num_frames=config['num_frames']['val'],
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=config['batchSize']['val'],
        shuffle=False,
        num_workers=0,
    )
    tester = Tester({'window_aggregation': 'mean'}, model, logger)
    if shuffle_frames:
        # Seed before the shuffled pass so the permutation is reproducible
        # (mirrors evaluation/test.py:160).
        torch.manual_seed(config.get('seed', 0))
    return tester.evaluate(loader, shuffle_frames=shuffle_frames)


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_orch_logger(log_path):
    """Dedicated logger for the orchestrator that won't be hijacked by
    create_logger() calls inside train_from_config / Tester. Uses a distinct
    logger name so its handlers survive when the shared 'deepfakebench'
    logger gets reset per cell."""
    log = logging.getLogger("layer_probe")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):  # idempotent if run() is re-entered
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ---- summary.csv ----

def summary_row_from_results(layer, model_type, payload):
    std = payload.get('test_std') or {}
    shuf = payload.get('test_shuf') or {}
    std_video = (std.get('per_video') or {}).get('auc')
    shuf_video = (shuf.get('per_video') or {}).get('auc')
    gap = (std_video - shuf_video) if (std_video is not None and shuf_video is not None) else None
    return {
        'layer': layer,
        'model_type': model_type,
        'clip_embed_dim': clip_embed_dim_for(layer),
        'best_val_auc': (payload.get('train') or {}).get('best_val_auroc'),
        'test_auc_std_window': (std.get('per_window') or {}).get('auc'),
        'test_auc_std_video': std_video,
        'test_auc_shuf_window': (shuf.get('per_window') or {}).get('auc'),
        'test_auc_shuf_video': shuf_video,
        'temporal_gap_video': gap,
        'wallclock_s': payload.get('wallclock'),
    }


def write_summary(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_summary_rows(paths):
    numeric = [k for k in SUMMARY_FIELDS if k not in ('layer', 'model_type')]
    out = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in numeric:
                    v = r.get(k)
                    if v in (None, ''):
                        r[k] = None
                    else:
                        try:
                            r[k] = int(float(v)) if k == 'clip_embed_dim' else float(v)
                        except ValueError:
                            r[k] = None
                out.append(r)
    # Dedup on (model_type, layer), last wins — lets a fresh run override
    # older values for the same cell when replotting.
    keyed = {}
    dups = []
    for r in out:
        key = (r['model_type'], r['layer'])
        if key in keyed:
            dups.append(key)
        keyed[key] = r
    if dups:
        print(f"WARN: {len(dups)} duplicate (model_type, layer) rows "
              f"resolved to last-seen: {sorted(set(dups))}", file=sys.stderr)
    return list(keyed.values())


def group_by_model(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r['model_type'], []).append(r)
    return by_model


# ---- plotting ----

def _nan(v):
    return float('nan') if v is None else float(v)


def _sort_by_layer(rows):
    key = {l: i for i, l in enumerate(LAYER_ORDER)}
    # Secondary key on layer name so unknown layers (all mapped to the same
    # sentinel index) sort alphabetically instead of by dict iteration order.
    return sorted(rows, key=lambda r: (key.get(r['layer'], len(LAYER_ORDER)), r['layer']))


def plot_per_model(rows, model_type, out_path):
    import matplotlib.pyplot as plt
    rows = _sort_by_layer(rows)
    xs = [r['layer'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, [_nan(r['test_auc_std_video']) for r in rows], marker='o', label='per-video')
    ax.plot(xs, [_nan(r['test_auc_std_window']) for r in rows], marker='s', label='per-window')
    ax.set_xlabel('CLIP layer')
    ax.set_ylabel('AUC (FF++ test)')
    ax.set_title(f"{model_type}: per-window vs per-video AUC across CLIP layers (FF++ test)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_temporal_gap(rows, model_type, out_path):
    import matplotlib.pyplot as plt
    rows = _sort_by_layer(rows)
    xs = [r['layer'] for r in rows]
    ys = [_nan(r['temporal_gap_video']) for r in rows]
    if all(math.isnan(y) for y in ys):
        # No shuffled results available for this model — nothing to plot.
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker='o', label=model_type)
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlabel('CLIP layer')
    ax.set_ylabel('AUC(std) − AUC(shuffled), per-video')
    ax.set_title(f"{model_type}: temporal gap across CLIP layers (std − shuffled, per-video AUC)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_auc(rows_by_model, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for mt in sorted(rows_by_model):
        rows = _sort_by_layer(rows_by_model[mt])
        xs = [r['layer'] for r in rows]
        ys = [_nan(r['test_auc_std_video']) for r in rows]
        ax.plot(xs, ys, marker='o', label=mt)
    ax.set_xlabel('CLIP layer')
    ax.set_ylabel('per-video AUC (FF++ test)')
    ax.set_title('Per-video AUC across CLIP layers, all models (FF++ test)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_combined_gap(rows_by_model, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    drew_any = False
    for mt in sorted(rows_by_model):
        rows = _sort_by_layer(rows_by_model[mt])
        xs = [r['layer'] for r in rows]
        ys = [_nan(r['temporal_gap_video']) for r in rows]
        if all(math.isnan(y) for y in ys):
            continue
        ax.plot(xs, ys, marker='o', label=mt)
        drew_any = True
    if not drew_any:
        plt.close(fig)
        return
    ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
    ax.set_xlabel('CLIP layer')
    ax.set_ylabel('AUC(std) − AUC(shuffled), per-video')
    ax.set_title('Temporal gap across CLIP layers, all models (std − shuffled, per-video AUC)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def emit_plots(rows_by_model, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    for mt, rows in rows_by_model.items():
        plot_per_model(rows, mt, plot_dir / f'auc_{mt}.png')
        plot_temporal_gap(rows, mt, plot_dir / f'temporal_gap_{mt}.png')
    if len(rows_by_model) > 1:
        plot_combined_auc(rows_by_model, plot_dir / 'auc_combined.png')
        plot_combined_gap(rows_by_model, plot_dir / 'temporal_gap_combined.png')


# ---- main run ----

def run(args):
    embeddings_dir = Path(args.embeddings_dir).resolve()

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)
    model_type = base_config['model_type']

    layers = discover_layers(embeddings_dir, args.layers)

    if args.run_dir:
        # Resume into an existing run dir. Results from prior cells (those
        # with results.json) are picked up via the skip-if-exists branch
        # below; cells without results.json are re-trained.
        run_root = Path(args.run_dir).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        resumed = True
    else:
        out_dir_root = Path(args.out_dir).resolve()
        out_dir_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        run_root = out_dir_root / f'{timestamp}_{model_type}'
        run_root.mkdir(parents=True, exist_ok=True)
        resumed = False

    orch_log = _make_orch_logger(str(run_root / 'orchestrator.log'))
    orch_log.info(f"Layer probe: model_type={model_type}  layers={layers}")
    orch_log.info(f"Base config: {args.base_config}")
    orch_log.info(f"Embeddings dir: {embeddings_dir}")
    orch_log.info(f"Run root: {run_root}" + ("  (resumed)" if resumed else ""))

    rows = []
    for layer in layers:
        run_dir = run_root / layer
        results_path = run_dir / 'results.json'

        if results_path.is_file() and not args.force:
            orch_log.info(f"[{layer}] skip — results.json exists.")
            with open(results_path) as f:
                rows.append(summary_row_from_results(layer, model_type, json.load(f)))
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = build_cell_config(base_config, embeddings_dir, layer)
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2)

        orch_log.info(f"[{layer}] train (clip_embed_dim={cfg['model'][model_type]['clip_embed_dim']})...")
        t0 = time.perf_counter()
        train_result = train_from_config(cfg, log_path=str(run_dir))
        model = train_result['model']

        eval_logger = create_logger(str(run_dir / 'training.log'))
        test_std = evaluate_on_split(model, cfg, split='test',
                                     shuffle_frames=False, logger=eval_logger)
        test_shuf = None
        if not args.no_shuffle_test:
            test_shuf = evaluate_on_split(model, cfg, split='test',
                                          shuffle_frames=True, logger=eval_logger)
        wallclock = time.perf_counter() - t0

        payload = {
            'layer': layer,
            'model_type': model_type,
            'train': {
                'best_val_auroc': train_result['best_val_auroc'],
                'final_val_auroc': train_result['final_val_auroc'],
                'epochs_completed': train_result['epochs_completed'],
            },
            'test_std': test_std,
            'test_shuf': test_shuf,
            'wallclock': wallclock,
        }
        with open(results_path, 'w') as f:
            json.dump(payload, f, indent=2)

        row = summary_row_from_results(layer, model_type, payload)
        rows.append(row)

        del train_result, model
        _free_gpu()

        orch_log.info(
            f"[{layer}] done in {wallclock:.1f}s  "
            f"val_auc={row['best_val_auc']:.4f}  "
            f"test_video_std={row['test_auc_std_video']:.4f}"
            + (f"  shuf={row['test_auc_shuf_video']:.4f}  gap={row['temporal_gap_video']:+.4f}"
               if row['test_auc_shuf_video'] is not None else "")
        )

    write_summary(rows, run_root / 'summary.csv')
    emit_plots(group_by_model(rows), run_root / 'plots')
    orch_log.info(f"Wrote {run_root / 'summary.csv'}")
    orch_log.info(f"Wrote plots to {run_root / 'plots'}")


def replot(args):
    paths = [Path(p).resolve() for p in args.from_summaries]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    rows = load_summary_rows(paths)
    plot_out = Path(args.plot_out_dir).resolve()
    emit_plots(group_by_model(rows), plot_out)
    print(f"Wrote plots to {plot_out}")


# ---- CLI ----

def parse_args():
    p = argparse.ArgumentParser(
        description='Layer probing orchestrator (one model × all CLIP layers on FF++).',
    )
    p.add_argument('--base-config', type=str,
                   help='YAML path for ONE model (linear/transformer/bigru).')
    p.add_argument('--embeddings-dir', type=str,
                   help='Model-level dir containing <layer>/catalogue.csv subdirs.')
    p.add_argument('--out-dir', type=str, default='logs/probing/',
                   help='Parent dir; a timestamped <ts>_<model_type>/ subdir is created inside. '
                        'Ignored when --run-dir is set.')
    p.add_argument('--run-dir', type=str, default=None,
                   help='Resume into an existing run dir (skips timestamp creation). '
                        'Cells with results.json are picked up from cache; the rest are (re-)trained.')
    p.add_argument('--layers', nargs='+', default=None,
                   help='Subset of layers to run. Default: auto-discover.')
    p.add_argument('--no-shuffle-test', action='store_true',
                   help='Skip the shuffled-frames eval pass.')
    p.add_argument('--force', action='store_true',
                   help='Re-train cells even if results.json exists.')
    p.add_argument('--plot-only', action='store_true',
                   help='Regenerate plots from existing summary.csv files; no training.')
    p.add_argument('--from-summaries', nargs='+', default=None,
                   help='summary.csv paths for replot mode.')
    p.add_argument('--plot-out-dir', type=str, default=None,
                   help='Output dir for replot-mode plots.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_only:
        if not args.from_summaries or not args.plot_out_dir:
            raise SystemExit('--plot-only requires --from-summaries and --plot-out-dir.')
        replot(args)
    else:
        missing = [k for k in ('base_config', 'embeddings_dir') if not getattr(args, k)]
        if missing:
            raise SystemExit(f'Missing required args: {missing}')
        run(args)


if __name__ == '__main__':
    main()
