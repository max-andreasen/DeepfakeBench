"""Optuna hyperparameter search orchestrator.

One trial = one training run on FF++ train/val (via train_from_config),
followed by a Tester.evaluate() on the val split that returns per-video AUC
as the objective. Per-epoch val AUC is reported to Optuna for median pruning.
Checkpoints are kept only for the top-K trials to bound disk usage.

Storage is SQLite, so runs are resumable with the same --study_name.

Post-study steps (report + final test eval on CDF-v2) are TODO — left as
stubs so the skeleton can be pilot-tested end-to-end before we add them.

Usage:
    python training/searches/parameter_search.py \
        --base_config training/configs/bigru.yaml \
        --study_name  bigru_pilot \
        --n_trials    5 \
        --search_epochs 15
"""

import argparse
import copy
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import optuna
import torch
import yaml
from torch.utils.data import DataLoader


# ---- path setup ----
# Both training/ and evaluation/ have a data_loader.py with DIFFERENT classes.
# We need DeepfakeDataset (train) via train_from_config, and DeepfakeTestDataset
# (eval) + Tester via the evaluation/ copies.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'training'))

from train import train_from_config
from logger import create_logger


# --- Loads dataloaders ---
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


# ---- search space ----
def search_space(trial, model_type):
    """Creates a dict containing the search space with hyperparameters for this trial.
    build_trial_config applies this onto a copy of the base YAML."""
    params = {}

    opt_type = trial.suggest_categorical('optimizer_type', ['adamw', 'adam'])
    params['optimizer_type'] = opt_type
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    sched = trial.suggest_categorical('lr_scheduler', ['constant', 'cosine', 'cosine_warmup'])
    params['lr_scheduler'] = sched
    if sched == 'cosine_warmup':
        params['warmup_epochs'] = trial.suggest_int('warmup_epochs', 3, 10)

    if model_type == 'transformer':
        params['model'] = {
            'num_layers':      trial.suggest_int('num_layers', 2, 12),
            'n_heads':         trial.suggest_categorical('n_heads', [4, 8, 16]),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [1024, 2048, 3072]),
            'attn_dropout':    trial.suggest_float('attn_dropout', 0.0, 0.5),
            'mlp_dropout':     trial.suggest_float('mlp_dropout', 0.1, 0.6),
        }
    elif model_type == 'bigru':
        params['model'] = {
            'hidden_dim':  trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024]),
            'num_layers':  trial.suggest_int('gru_num_layers', 1, 4),
            'gru_dropout': trial.suggest_float('gru_dropout', 0.0, 0.5),
            'mlp_dropout': trial.suggest_float('mlp_dropout', 0.1, 0.6),
        }
    elif model_type == 'linear':
        params['model'] = {}  # nothing model-specific to tune
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return params


def build_trial_config(base_config, overrides, search_epochs, trial_dir):
    cfg = copy.deepcopy(base_config)
    model_type = cfg['model_type']

    opt_type = overrides['optimizer_type']
    cfg['optimizer']['type'] = opt_type
    cfg['optimizer'][opt_type]['lr'] = overrides['lr']
    cfg['optimizer'][opt_type]['weight_decay'] = overrides['weight_decay']

    cfg['lr_scheduler'] = overrides['lr_scheduler']
    if 'warmup_epochs' in overrides:
        cfg['warmup_epochs'] = overrides['warmup_epochs']

    cfg['model'][model_type].update(overrides['model'])
    cfg['num_epochs'] = search_epochs
    cfg['log_dir'] = trial_dir
    cfg['save_ckpt'] = True  # per-trial; top-K retention prunes the losers
    return cfg


# ---- evaluation ----
def evaluate_on_split(model, config, split, logger):
    """Run Tester.evaluate on a given split. Returns result dict.
    num_workers=0 on the loader so DataLoader doesn't try to pickle an
    importlib-loaded dataset class across worker processes."""
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
    eval_cfg = {'window_aggregation': 'mean'}
    tester = Tester(eval_cfg, model, logger)
    return tester.evaluate(loader, shuffle_frames=False)


# ---- top-K checkpoint retention ----

def prune_checkpoints(study, top_k, study_dir):
    """Keep model.pth only for the current top-K completed trials by study
    value. Pruned / failed / running trials are not counted or touched."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    losers = completed[top_k:]
    for t in losers:
        ckpt = Path(study_dir) / f'trial_{t.number:04d}' / 'model.pth'
        if ckpt.exists():
            ckpt.unlink()


# ---- objective ----

def objective(trial, base_config, study_dir, search_epochs, top_k):
    trial_dir = str(Path(study_dir) / f'trial_{trial.number:04d}')
    os.makedirs(trial_dir, exist_ok=True)

    overrides = search_space(trial, base_config['model_type'])
    config = build_trial_config(base_config, overrides, search_epochs, trial_dir)

    with open(Path(trial_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # train — per-epoch pruning via train_from_config
    try:
        result = train_from_config(config, trial=trial, log_path=trial_dir)
    except optuna.TrialPruned:
        _free_gpu()
        raise

    # full Tester eval on val — this is the objective the study optimizes.
    # Reuse the trial's training.log so all messages land in one file.
    eval_logger = create_logger(str(Path(trial_dir) / 'training.log'))
    val_results = evaluate_on_split(result['model'], config, split='val', logger=eval_logger)

    with open(Path(trial_dir) / 'val_results.json', 'w') as f:
        json.dump(val_results, f, indent=2)

    objective_value = val_results['per_video']['auc']

    # Free the model before next trial.
    del result
    _free_gpu()

    # Top-K retention uses the study's current state (this trial not yet
    # committed, but Optuna records values post-return, so we call it here
    # knowing the current trial's value will be committed shortly. An extra
    # pass happens on next trial entry to catch up.)
    prune_checkpoints(trial.study, top_k, study_dir)

    return objective_value


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- study setup ----

def build_pruner(name):
    if name == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    if name == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    if name == 'none':
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown pruner: {name}")


def main():
    parser = argparse.ArgumentParser(description='Optuna hparam search for temporal CLIP models.')
    parser.add_argument('--base_config', type=str, required=True,
                        help='Path to base YAML (training/configs/{bigru,linear,transformer}.yaml).')
    parser.add_argument('--study_name', type=str, required=True,
                        help='Optuna study name; also subdir under training/searches/runs/.')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=None,
                        help='Optional wall-clock cap in seconds.')
    parser.add_argument('--search_epochs', type=int, default=30,
                        help='Epochs per trial. Fixed across the search; retrain winner longer after.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Keep model.pth for top K trials; prune ckpts from the rest.')
    parser.add_argument('--pruner', type=str, default='median',
                        choices=['median', 'hyperband', 'none'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--storage', type=str,
                        default=str(REPO_ROOT / 'training' / 'searches' / 'optuna_studies.db'),
                        help='SQLite path. Same path = same DB = resumable across study names.')
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    study_dir = REPO_ROOT / 'training' / 'searches' / 'runs' / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    pruner = build_pruner(args.pruner)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed)

    storage_url = f"sqlite:///{args.storage}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction='maximize',
        load_if_exists=True,
    )

    study.optimize(
        lambda t: objective(t, base_config, str(study_dir), args.search_epochs, args.top_k),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    # TODO: generate_report(study, study_dir)
    #       - all_trials.csv
    #       - best_config.json (full merged config of winner)
    #       - summary.md (top-5 table, pruning counts, wall-clock)
    #       - optuna HTML plots (optimization_history, param_importances,
    #         parallel_coordinate, slice_plot)

    # TODO: evaluate best trial on CDF-v2 TEST split
    #       - load trial_NNNN/model.pth
    #       - swap catalogue_file + split_file to CDF-v2
    #       - run Tester.evaluate
    #       - write best_trial_test_results.json

    print(f"\nBest trial: #{study.best_trial.number}  value={study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
