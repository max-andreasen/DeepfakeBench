"""
plot_retrain.py — per-epoch val AUC plots with mean ± std shading.

Reads retrain_top_k output and produces one PNG per trial config showing
the mean val AUC curve across seeds with a shaded ±1 std band.

Usage (from repo root):
    python evaluation/plot_retrain.py \\
        --retrain_dir evaluation/results/retrain_top_k/transformer_search3

Outputs:
    <retrain_dir>/plots/trial_XXXX.png   — one plot per config
    <retrain_dir>/plots/all_configs.png  — all configs overlaid (optional)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Plot mean ± std val AUC curves from retrain_top_k.')
    p.add_argument('--retrain_dir', required=True,
                   help='Path to retrain_top_k output directory')
    p.add_argument('--no_overlay', action='store_true',
                   help='Skip the all_configs overlay plot')
    p.add_argument('--std_multiplier', type=float, default=1.0,
                   help='Band width = std_multiplier × std (default: 1.0)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trial_curves(trial_dir: Path) -> dict:
    """Load per-seed epoch_aucs from a trial dir.

    Returns:
        {
          'trial': int,
          'summary': dict (from summary.json),
          'curves': list of lists  (one list per seed, each = [auc_epoch_0, ...]),
        }
    """
    summary_path = trial_dir / 'summary.json'
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    curves = []
    for seed_dir in sorted(trial_dir.glob('seed_*')):
        auc_file = seed_dir / 'epoch_aucs.json'
        if auc_file.exists():
            with open(auc_file) as f:
                curves.append(json.load(f))

    if not curves:
        return None

    return {
        'trial':   int(trial_dir.name.split('_')[1]),
        'summary': summary,
        'curves':  curves,
    }


def compute_stats(curves: list) -> tuple:
    """Return (epochs, mean, std) arrays. Trims all curves to shortest length."""
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])   # [n_seeds, n_epochs]
    epochs = np.arange(1, min_len + 1)
    mean   = arr.mean(axis=0)
    std    = arr.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros(min_len)
    return epochs, mean, std


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, title, n_seeds):
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val AUC (FF++)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]), top=min(1.0, ax.get_ylim()[1]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.text(0.98, 0.04, f'n_seeds={n_seeds}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='grey')


def plot_single(data: dict, out_path: Path, std_multiplier: float):
    """One figure: mean line + shaded band for a single trial config."""
    epochs, mean, std = compute_stats(data['curves'])
    n_seeds = len(data['curves'])
    trial   = data['trial']
    mean_final_auc = data['summary'].get('mean_test_auc', float('nan'))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(
        epochs,
        mean - std_multiplier * std,
        mean + std_multiplier * std,
        alpha=0.25,
        label=f'±{std_multiplier:.0f} std',
    )
    ax.plot(epochs, mean, linewidth=2, label='Mean val AUC')

    # Mark best epoch
    best_ep = int(np.argmax(mean)) + 1
    ax.axvline(best_ep, color='grey', linestyle=':', linewidth=1, alpha=0.7)
    ax.annotate(f'best: {mean[best_ep-1]:.4f}',
                xy=(best_ep, mean[best_ep-1]),
                xytext=(best_ep + 0.5, mean[best_ep-1] - 0.003),
                fontsize=8, color='grey')

    title = (f'Trial {trial} — mean CDFv2 test AUC: {mean_final_auc:.4f}'
             if not np.isnan(mean_final_auc) else f'Trial {trial}')
    _style_ax(ax, title, n_seeds)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_overlay(all_data: list, out_path: Path, std_multiplier: float):
    """All trial configs on one axes, each a distinct colour."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    cmap = plt.get_cmap('tab10')

    for i, data in enumerate(all_data):
        epochs, mean, std = compute_stats(data['curves'])
        colour = cmap(i % 10)
        label  = f"Trial {data['trial']} (CDFv2: {data['summary'].get('mean_test_auc', float('nan')):.4f})"

        ax.fill_between(
            epochs,
            mean - std_multiplier * std,
            mean + std_multiplier * std,
            alpha=0.15,
            color=colour,
        )
        ax.plot(epochs, mean, linewidth=1.8, color=colour, label=label)

    n_seeds = len(all_data[0]['curves']) if all_data else 0
    _style_ax(ax, 'All configs — mean val AUC (FF++) with ±std band', n_seeds)

    ax.legend(fontsize=7, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    retrain_dir = Path(args.retrain_dir)
    plots_dir   = retrain_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    trial_dirs = sorted(retrain_dir.glob('trial_*'))
    all_data   = []

    for td in trial_dirs:
        data = load_trial_curves(td)
        if data is None:
            print(f"  Skipping {td.name} (no epoch_aucs found)")
            continue
        all_data.append(data)
        out = plots_dir / f'{td.name}.png'
        plot_single(data, out, args.std_multiplier)

    if not args.no_overlay and len(all_data) > 1:
        plot_overlay(all_data, plots_dir / 'all_configs.png', args.std_multiplier)

    print(f"\nDone. {len(all_data)} plots written to {plots_dir}")


if __name__ == '__main__':
    main()
