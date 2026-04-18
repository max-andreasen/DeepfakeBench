# Optuna Hyperparameter Search — Implementation Plan

Single-command Optuna search over training hyperparameters for Transformer,
BiGRU, and Linear models on CLIP embeddings. Persists to SQLite, prunes bad
trials early, generates a report, and re-evaluates the winner on a held-out
test dataset.

## Decisions (locked 2026-04-17)

| # | Decision | Value |
|---|----------|-------|
| 1 | Primary objective | Per-video AUC from `Tester.evaluate()` on val split |
| 2 | Pruning signal | Per-epoch val AUC from `Trainer.eval_epoch()` (cheaper, correlated) |
| 3 | `search_epochs` (per trial) | **30** (fixed; not searched — retrain winner longer after) |
| 4 | Val dataset (tuning signal) | **FaceForensics++** held-out val split |
| 5 | Test dataset (final eval) | **Celeb-DF-v2** (per-video AUC only) |
| 6 | Checkpoint retention | Top 5 trials by objective; others discarded |
| 7 | Pilot before full run | 5 trials × 15 epochs → check wall-clock + smoke |
| 8 | Optimizers in search | **AdamW, Adam** (SGD dropped — pilot showed it loses) |
| 9 | Transformer `n_heads` | **{4, 8, 16}** (all divide 768) |
| 10 | Pruner | `MedianPruner(n_startup=10, n_warmup=5)` |
| 11 | Sampler | `TPESampler(multivariate=True, seed=42)` |
| 12 | Storage | SQLite at `training/searches/optuna_studies.db` (resumable) |

## Prerequisites

- [x] **Fix `tester.py` per-video grouping.** Done 2026-04-17. Groups by
      `(label_cat, video_id)` instead of `video_id` alone. Previously all 4 FF++
      fake manipulations with the same `video_id` (e.g. `000_003`) pooled into
      one "video", over-averaging. Touched `evaluation/data_loader.py` and
      `evaluation/tester.py`; `DeepfakeTestDataset.__getitem__` now returns
      `(x, label, video_id, label_cat)`.
- [x] **Refactor `train.py`**: extracted `train_from_config(config, trial=None, log_path=None)`.
      `main()` owns argparse (moved out of module scope so `parameter_search.py`
      can `from train import train_from_config` without triggering CLI parsing).
      `run_config.json` is written via `finally` — captured even on pruning;
      `model.pth` is skipped on pruned trials. Model moved to CPU before return
      so GPU memory frees between trials.
- [x] **Parameterize `DeepfakeTestDataset`**: `split="test"` kwarg added,
      default preserves old behavior for `evaluation/test.py`.
- [x] **Logger handler leak fix**: `create_logger()` now clears + closes any
      existing handlers on the `"deepfakebench"` shared-name logger before
      adding new ones.
- [ ] **YAML updates**: change `val_dataset: [FaceForensics++]` in
      `training/configs/{bigru,linear}.yaml` (currently `[Celeb-DF-v2]`).
      User handles these manually.

## File changes

| File | Change | Status |
|------|--------|--------|
| `evaluation/data_loader.py` | Return `label_cat`; require it in catalogue | DONE |
| `evaluation/tester.py` | Group per-video by `(label_cat, video_id)` | DONE |
| `evaluation/data_loader.py` | Add `split="test"` kwarg | DONE |
| `training/train.py` | Extract `train_from_config(config, trial=None)`; keep `main()` as thin wrapper | DONE |
| `logger.py` | Clear + close existing handlers before adding new ones | DONE |
| `training/configs/bigru.yaml` | `val_dataset: [FaceForensics++]` | USER |
| `training/configs/linear.yaml` | `val_dataset: [FaceForensics++]` | USER |
| `training/configs/transformer.yaml` | Create + set val to FF++ | USER (if needed) |
| `training/searches/parameter_search.py` | Skeleton: CLI + study + objective + search_space + top-K retention + per-trial artifacts. Report + final CDF-v2 eval stubbed as TODO. | DONE (skeleton) |

## `train_from_config` contract

```python
def train_from_config(config: dict, trial: optuna.Trial | None = None) -> dict:
    """Run a full training loop from a merged config dict.

    Returns:
        {
          'model':             trained nn.Module (moved to CPU),
          'log_path':          output directory (str),
          'best_val_auroc':    float,
          'final_val_auroc':   float,
          'epochs_completed':  int,
        }

    If `trial` is not None, reports per-epoch val AUROC via `trial.report(...)`
    and raises `optuna.TrialPruned` when `trial.should_prune()` returns True.
    Checkpoint write is skipped on pruned runs.
    """
```

`main()` becomes:

```python
def main():
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # apply CLI overrides...
    train_from_config(config)
```

## `parameter_search.py` — CLI

```
python training/searches/parameter_search.py \
  --base_config  training/configs/bigru.yaml     \
  --study_name   bigru_sweep                      \
  --n_trials     100                              \
  --timeout      86400                            \
  --pruner       median                           \
  --seed         42
```

Pilot:
```
python training/searches/parameter_search.py \
  --base_config training/configs/bigru.yaml --study_name pilot \
  --n_trials 5 --search_epochs 15
```

## Objective flow (per trial)

1. `overrides = search_space(trial, base_config['model_type'])`.
2. `config = build_trial_config(base_config, overrides)` — deep-copy + merge.
   Override `config['num_epochs'] = 30`, `config['log_dir'] = trial_dir`.
3. `result = train_from_config(config, trial=trial)` — reports per-epoch AUC,
   prunes if below rolling median.
4. `val_results = evaluate_on_split(result['model'], config, split='val')` —
   `Tester.evaluate()` with `window_aggregation='mean'`. ~2s.
5. Save per-trial artifacts (`config.json`, `val_results.json`, `training.log`).
   Write `model.pth` only if this trial is currently in top 5.
6. `del result['model']; torch.cuda.empty_cache()`.
7. Return `val_results['per_video']['auc']`.

After `study.optimize()` completes:

1. Load best trial's checkpoint.
2. Re-evaluate on **Celeb-DF-v2** test split → `best_trial_test_results.json`.
3. Generate report (see below).

## Search spaces (placeholders — tune ranges after pilot)

### Shared
```python
opt_type = trial.suggest_categorical('optimizer_type', ['adamw', 'adam'])
cfg['optimizer']['type'] = opt_type
cfg['optimizer'][opt_type]['lr']           = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
cfg['optimizer'][opt_type]['weight_decay'] = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
# beta1/beta2/eps fixed

sched = trial.suggest_categorical('scheduler', ['constant', 'cosine', 'cosine_warmup'])
cfg['lr_scheduler'] = sched
if sched == 'cosine_warmup':
    cfg['warmup_epochs'] = trial.suggest_int('warmup_epochs', 3, 10)
```

### Transformer
```python
mcfg['num_layers']      = trial.suggest_int('num_layers', 2, 12)
mcfg['n_heads']         = trial.suggest_categorical('n_heads', [4, 8, 16])
mcfg['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', [1024, 2048, 3072])
mcfg['attn_dropout']    = trial.suggest_float('attn_dropout', 0.0, 0.5)
mcfg['mlp_dropout']     = trial.suggest_float('mlp_dropout', 0.1, 0.6)
```

### BiGRU
```python
mcfg['hidden_dim']  = trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024])
mcfg['num_layers']  = trial.suggest_int('gru_num_layers', 1, 4)
mcfg['gru_dropout'] = trial.suggest_float('gru_dropout', 0.0, 0.5)
mcfg['mlp_dropout'] = trial.suggest_float('mlp_dropout', 0.1, 0.6)
```

### Linear
Only shared hparams. No model-specific tuning.

## Output layout

```
training/searches/
  optuna_studies.db               # shared across all studies
  PLAN.md                         # this file
  parameter_search.py
  runs/
    <study_name>/
      trial_0000/
        config.json
        val_results.json
        training.log
        model.pth                 # only if in top-5
      trial_0001/
        ...
      best_config.json
      all_trials.csv
      best_trial_test_results.json
      summary.md
      optimization_history.html
      param_importances.html
      parallel_coordinate.html
      slice_plot.html
```

## Report (generated post-study)

- `best_config.json` — full merged config of winning trial.
- `all_trials.csv` — one row per trial: number, state, value, duration, all
  sampled params. Drives paper tables.
- `summary.md` — human-readable: best params, top-5 table, pruning counts,
  total wall-clock.
- Optuna HTML plots via `optuna.visualization`:
  - `optimization_history.html` — objective over trials.
  - `param_importances.html` — fANOVA-based hparam importance.
  - `parallel_coordinate.html` — interactions.
  - `slice_plot.html` — per-param marginal effect.
- `best_trial_test_results.json` — Tester output on Celeb-DF-v2 test split.

## Verification plan

1. **Smoke** — `--n_trials 1 --search_epochs 5`. Confirm one completed trial
   with `val_results.json`, `model.pth`, report files.
2. **Pruning** — `--n_trials 15 --search_epochs 15`. Confirm at least one
   trial has `state=PRUNED` in `all_trials.csv`.
3. **Resumability** — start with `--n_trials 10`, kill after 3. Re-run same
   command. Confirm it resumes at trial 3 (not 0).
4. **Test eval** — confirm `best_trial_test_results.json` exists with per-video
   and per-window metrics on Celeb-DF-v2.
5. **Existing CLI still works** — `python training/train.py --config ...`
   completes unchanged.

## Not in Optuna (post-hoc sweeps)

- **Aggregation** (`mean` / `max` / `softmax` with varying `softmax_temp`):
  cheap grid over winner's frozen checkpoint. Separate script.
- **Preprocessing** (dlib / mtcnn / raw CLIP): one full Optuna study per
  `catalogue_file`, compare winners across studies. Don't mix.
- **Retraining the winner for longer** (e.g. 100 epochs with the same
  hparams): run standalone `python training/train.py --config best_config.yaml`
  after the search completes.

## Open items / flagged

- **Label inversion**: catalogue has `FF-real: label=1, FF-fake*: label=0`,
  opposite of the YAML `label_dict`. Training path uses `df["label"]` directly
  so it's self-consistent, but final AUC interpretation flips. Decide which
  convention wins before publishing numbers.
- **Device plumbing**: `Tester.__init__` reads `torch.cuda.is_available()`
  directly and ignores `config['device']`. Harmless on the GPU box; clean up
  when touching `tester.py`.
- **Transformer YAML**: no `training/configs/transformer.yaml` currently — only
  `bigru.yaml` and `linear.yaml`. Create one before searching Transformer.

## Sub-agent strategy

Serial work (main thread):
- `train.py` refactor (`train_from_config` extraction). Small, load-bearing —
  keep in main thread to verify line-by-line.
- `data_loader.py` split= kwarg + `logger.py` fix. Both trivial.
- `parameter_search.py` skeleton (CLI, objective, search_space). Core logic —
  keep in main thread.

Parallelizable via agents once skeleton is stable:
- **Report generation helpers** (`generate_report`, `summary.md` writer, CSV
  exporter, HTML plot rendering). Self-contained, interface-bounded — good
  agent target.
- **Verification-style audit** after first pilot run: spawn an agent to read
  the final `parameter_search.py` + first pilot's `summary.md`, and sanity-check
  the objective-flow claims (no ckpt save on pruned trial, GPU freed, etc.).
- **Post-hoc aggregation sweep script**: orthogonal, can be written in
  parallel with the main parameter_search work once `Tester` + checkpoint
  loading is ironed out.
