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
import logging
import os
import sys
from datetime import datetime
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
# Per-model optimizer / scheduler choices. Narrowed from the original universal
# lists based on top-10 distributions from the first-round studies:
#   - transformer top-10: adam 10/10; schedulers mixed → keep all three
#   - bigru      top-10: adam/adamw mix; cosine_warmup never won → drop
#   - linear     top-10: adamw 10/10 + cosine_warmup 10/10 → pin both
# Kept as categoricals even when single-valued so Optuna still logs the choice.
OPT_CHOICES = {
    'transformer': ['adam'],
    'bigru':       ['adam', 'adamw'],
    'linear':      ['adamw'],
}
SCHED_CHOICES = {
    'transformer': ['constant', 'cosine', 'cosine_warmup'],
    'bigru':       ['constant', 'cosine'],
    'linear':      ['constant', 'cosine_warmup'],
}


CLIP_LAYER_CHOICES = ['block_12', 'block_16', 'block_20', 'pre_proj']
CLIP_CATALOGUE_TEMPLATE = 'clip/embeddings/probing/ViT-L-14-336-quickgelu/{layer}/catalogue.csv'


def search_space(trial, model_type):
    """Creates a dict containing the search space with hyperparameters for this trial.
    build_trial_config applies this onto a copy of the base YAML."""
    params = {}

    # CLIP layer: late blocks + the pre-projection representation. All four
    # are 1024-dim so clip_embed_dim stays consistent; only the catalogue
    # path changes per trial.
    params['clip_layer'] = trial.suggest_categorical('clip_layer', CLIP_LAYER_CHOICES)

    opt_type = trial.suggest_categorical('optimizer_type', OPT_CHOICES[model_type])
    params['optimizer_type'] = opt_type

    sched = trial.suggest_categorical('lr_scheduler', SCHED_CHOICES[model_type])
    params['lr_scheduler'] = sched
    if sched == 'cosine_warmup':
        params['warmup_epochs'] = trial.suggest_int('warmup_epochs', 4, 14)

    # lr/weight_decay ranges are model-specific: the linear probe converges
    # near 1e-2, while transformer/BiGRU live in 1e-4..1e-3. Using a shared
    # [1e-5, 1e-2] range wastes budget for all three.
    if model_type == 'transformer':
        params['lr']           = trial.suggest_float('lr', 0.00001, 0.001, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        params['model'] = {
            'num_layers':      trial.suggest_int('num_layers', 6, 16),
            'n_heads':         trial.suggest_categorical('n_heads', [2, 4, 8, 16]),
            'dim_feedforward': trial.suggest_categorical('dim_feedforward', [256, 512, 1024, 2048]),
            'attn_dropout':    trial.suggest_float('attn_dropout', 0.0, 0.6),
            'mlp_dropout':     trial.suggest_float('mlp_dropout', 0.1, 0.6),
            'mlp_hidden_dim':  trial.suggest_categorical('mlp_hidden_dim', [64, 128, 256, 512, 1024]),
        }
    elif model_type == 'bigru':
        params['lr'] = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        # weight_decay pinned in the base YAML — the first-round study showed
        # near-zero correlation with AUC (+0.11) and the top-3 trials all used
        # WD < 1.3e-5. Not worth the search dimension.
        params['model'] = {
            'hidden_dim':  trial.suggest_categorical('hidden_dim', [256, 512, 2048]),
            'num_layers':  trial.suggest_int('gru_num_layers', 2, 5),
            'gru_dropout': trial.suggest_float('gru_dropout', 0.0, 0.4),
            'mlp_dropout': trial.suggest_float('mlp_dropout', 0.0, 0.4),
            'mlp_hidden_dim': trial.suggest_categorical('mlp_hidden_dim', [256, 512, 1024]),
        }
    elif model_type == 'linear':
        params['lr'] = trial.suggest_float('lr', 0.005, 0.1, log=True)
        # weight_decay pinned in the base YAML — top-5 clustered at WD ≈ 2e-5.
        # Correlation with AUC was +0.35 (weak); not worth the search dimension.
        params['model'] = {
            'mlp_hidden_dim': trial.suggest_categorical('mlp_hidden_dim', [512, 1024, 2048]),
            'mlp_dropout':    trial.suggest_float('mlp_dropout', 0.0, 0.4),
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return params


def build_trial_config(base_config, overrides, search_epochs, trial_dir):
    cfg = copy.deepcopy(base_config)
    model_type = cfg['model_type']

    if 'clip_layer' in overrides:
        cfg['catalogue_file'] = CLIP_CATALOGUE_TEMPLATE.format(layer=overrides['clip_layer'])

    opt_type = overrides['optimizer_type']
    cfg['optimizer']['type'] = opt_type
    cfg['optimizer'][opt_type]['lr'] = overrides['lr']
    if 'weight_decay' in overrides:
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
        input_transform=config.get('input_transform', 'none'),
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


# ---- post-study reporting ----
PLOT_VALUE_FLOOR = 0.7
"""Trials with value < PLOT_VALUE_FLOOR are dropped from param_importances,
parallel_coordinate, and slice_plot. A single collapsed trial (e.g. AUC~0.2)
compresses the color/axis scale so the bulk of good trials overlap into one
visual band. optimization_history keeps all trials so spike-downs stay visible."""


def _study_filtered_by_value(study, min_value):
    """Build an in-memory copy of `study` containing only COMPLETE trials
    with value >= min_value. Used solely for plotting — the original study
    (including failed/collapsed trials) is still persisted in SQLite and
    all_trials.csv."""
    filtered = optuna.create_study(
        study_name=study.study_name + '__plot_filter',
        sampler=optuna.samplers.RandomSampler(),  # unused, we only call .trials
        direction='maximize',
    )
    keep = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and t.value >= min_value]
    filtered.add_trials(keep)
    return filtered


def generate_report(study, study_dir):
    """Write study artifacts after study.optimize() completes:
    - all_trials.csv   : one row per trial (params, value, state, duration)
    - best_config.json : full merged config of the winner
    - summary.md       : top-5 table + state counts + wall-clock
    - plots/*.html     : Optuna's optimization_history, param_importances,
                         parallel_coordinate, slice_plot
    Each artifact is best-effort; a failure in one doesn't block the others."""
    study_dir = Path(study_dir)

    study.trials_dataframe().to_csv(study_dir / 'all_trials.csv', index=False)

    best_trial = study.best_trial
    best_cfg_src = study_dir / f'trial_{best_trial.number:04d}' / 'run_config.json'
    if best_cfg_src.exists():
        with open(best_cfg_src) as f:
            best_cfg = json.load(f)
        with open(study_dir / 'best_config.json', 'w') as f:
            json.dump(best_cfg, f, indent=2)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    completed.sort(key=lambda t: (t.value if t.value is not None else -float('inf')), reverse=True)
    top10 = completed[:10]
    total_wall_s = sum((t.duration.total_seconds() for t in study.trials if t.duration is not None), 0.0)

    lines = [
        f"# Study: {study.study_name}",
        "",
        f"- Total trials: {len(study.trials)}",
        f"- Completed: {len(completed)}",
        f"- Pruned: {len(pruned)}",
        f"- Failed: {len(failed)}",
        f"- Wall-clock: {total_wall_s/3600:.2f} h",
        "",
        "## Top 10",
        "",
        "| Rank | Trial | Value | Params |",
        "| --- | --- | --- | --- |",
    ]
    for i, t in enumerate(top10, 1):
        params_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
        lines.append(f"| {i} | {t.number} | {t.value:.4f} | {params_str} |")
    (study_dir / 'summary.md').write_text("\n".join(lines) + "\n")

    try:
        import optuna.visualization as vis
    except ImportError:
        print("[warn] plotly/optuna.visualization not available; skipping HTML plots")
        return

    plots_dir = study_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    plot_study = _study_filtered_by_value(study, PLOT_VALUE_FLOOR)
    plot_specs = [
        ('optimization_history', vis.plot_optimization_history,  study),
        ('param_importances',    vis.plot_param_importances,     plot_study),
        ('parallel_coordinate',  vis.plot_parallel_coordinate,   plot_study),
        ('slice_plot',           vis.plot_slice,                 plot_study),
    ]
    for name, fn, src in plot_specs:
        try:
            fn(src).write_html(str(plots_dir / f'{name}.html'))
        except Exception as e:
            print(f"[warn] {name} plot failed: {e}")


# ---- study setup ----
def build_pruner(name):
    if name == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    if name == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    if name == 'none':
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown pruner: {name}")


# ---- study-level logging & run_config ----
def setup_study_logger(study_dir):
    """File-only logger at <study_dir>/study.log. Stdout stays owned by
    the tqdm progress bar + per-trial print(), so nothing duplicates."""
    logger = logging.getLogger("optuna_search")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(Path(study_dir) / 'study.log'))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def describe_search_space(model_type):
    """Human-readable snapshot of the search space for a given model_type.
    Kept in sync with search_space() by hand — update both together."""
    space = {
        'common': {
            'clip_layer':     {'type': 'categorical', 'choices': list(CLIP_LAYER_CHOICES)},
            'optimizer_type': {'type': 'categorical', 'choices': list(OPT_CHOICES[model_type])},
            'lr_scheduler':   {'type': 'categorical', 'choices': list(SCHED_CHOICES[model_type])},
            'warmup_epochs':  {'type': 'int', 'range': [4, 14], 'conditional_on': "lr_scheduler=='cosine_warmup'"},
        },
    }
    if model_type == 'transformer':
        space['common']['lr']           = {'type': 'float', 'range': [1e-5, 1e-3], 'log': True}
        space['common']['weight_decay'] = {'type': 'float', 'range': [1e-6, 1e-2], 'log': True}
        space['model'] = {
            'num_layers':     {'type': 'int', 'range': [6, 16]},
            'n_heads':        {'type': 'categorical', 'choices': [2, 4, 8, 16]},
            'dim_feedforward':{'type': 'categorical', 'choices': [256, 512, 1024, 2048]},
            'attn_dropout':   {'type': 'float', 'range': [0.0, 0.6]},
            'mlp_dropout':    {'type': 'float', 'range': [0.1, 0.6]},
            'mlp_hidden_dim': {'type': 'categorical', 'choices': [64, 128, 256, 512, 1024]},
        }
    elif model_type == 'bigru':
        space['common']['lr'] = {'type': 'float', 'range': [1e-5, 5e-3], 'log': True}
        # weight_decay intentionally absent — pinned in base YAML.
        space['model'] = {
            'hidden_dim':     {'type': 'categorical', 'choices': [256, 512, 2048]},
            'num_layers':     {'type': 'int', 'range': [2, 5]},
            'gru_dropout':    {'type': 'float', 'range': [0.0, 0.4]},
            'mlp_dropout':    {'type': 'float', 'range': [0.0, 0.4]},
            'mlp_hidden_dim': {'type': 'categorical', 'choices': [256, 512, 1024]},
        }
    elif model_type == 'linear':
        space['common']['lr'] = {'type': 'float', 'range': [5e-3, 1e-1], 'log': True}
        # weight_decay intentionally absent — pinned in base YAML.
        space['model'] = {
            'mlp_hidden_dim': {'type': 'categorical', 'choices': [512, 1024, 2048]},
            'mlp_dropout':    {'type': 'float', 'range': [0.0, 0.4]},
        }
    return space


def save_search_run_config(study_dir, args, base_config):
    """Persist the search's CLI args + search space snapshot for reproducibility.
    Written once at study start; separate from per-trial config.json."""
    run_config = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'study_name': args.study_name,
        'base_config_path': os.path.abspath(args.base_config),
        'model_type': base_config['model_type'],
        'cli_args': {
            'n_trials':      args.n_trials,
            'search_epochs': args.search_epochs,
            'top_k':         args.top_k,
            'pruner':        args.pruner,
            'seed':          args.seed,
            'timeout':       args.timeout,
            'storage':       args.storage,
        },
        'sampler':      'TPESampler(multivariate=True)',
        'search_space': describe_search_space(base_config['model_type']),
    }
    with open(Path(study_dir) / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)


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

    # Silence Optuna's per-trial INFO line ("[I ...] Trial N finished...") —
    # we emit a compact equivalent in _trial_callback below, and tqdm shows
    # the bar. Keep WARNING+ so real problems still surface.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_logger = setup_study_logger(str(study_dir))
    save_search_run_config(str(study_dir), args, base_config)

    study_logger.info(f"=== Starting study '{args.study_name}' ===")
    study_logger.info(f"CLI args: {vars(args)}")
    study_logger.info(f"base_config: {os.path.abspath(args.base_config)}")
    study_logger.info(f"model_type: {base_config['model_type']}")
    study_logger.info(f"catalogue_file: {base_config.get('catalogue_file')}")
    study_logger.info(f"input_transform: {base_config.get('input_transform', 'none')}")

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

    def _trial_callback(study, trial):
        dur = trial.duration.total_seconds() if trial.duration is not None else 0.0
        val = f"{trial.value:.4f}" if trial.value is not None else "n/a"
        try:
            best_num = study.best_trial.number
            best_val = f"{study.best_value:.4f}"
        except ValueError:
            best_num, best_val = -1, "n/a"
        line = (f"[trial {trial.number:04d}] state={trial.state.name:9s} "
                f"value={val}  dur={dur:6.1f}s  best={best_val} (#{best_num})")
        print(line, flush=True)
        study_logger.info(line)

    study.optimize(
        lambda t: objective(t, base_config, str(study_dir), args.search_epochs, args.top_k),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
        callbacks=[_trial_callback],
    )

    generate_report(study, str(study_dir))

    # TODO: evaluate best trial on CDF-v2 TEST split
    #       - load trial_NNNN/model.pth
    #       - swap catalogue_file + split_file to CDF-v2
    #       - run Tester.evaluate
    #       - write best_trial_test_results.json

    try:
        best_trial = study.best_trial
        best_line = f"Best trial: #{best_trial.number}  value={study.best_value:.4f}"
        print(f"\n{best_line}")
        study_logger.info(best_line)
        print("Best params:")
        study_logger.info("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
            study_logger.info(f"  {k}: {v}")
    except ValueError:
        msg = "No completed trials."
        print(f"\n{msg}")
        study_logger.info(msg)
    study_logger.info(f"=== Study '{args.study_name}' complete ===")


if __name__ == '__main__':
    main()
