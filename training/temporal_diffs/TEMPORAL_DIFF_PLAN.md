# Temporal-Difference Input Transform — Implementation Plan

Pilot experiment: does feeding frame-to-frame embedding differences
(`d_t = x_t − x_{t-1}`) into the temporal modules (linear / BiGRU /
transformer) recover any of the temporal signal that raw embeddings appear to
lack? Ships as a permanent `input_transform` config field, reusable for future
temporal transforms (`diff2`, `abs_diff`, etc.).

---

## 1. Overview

Add a single config field `input_transform: none | diff` that is threaded
through both the training and evaluation paths. When `diff` is set:

- Each window's T-axis is replaced with first-order temporal differences:
  `x[1:] - x[:-1]`, reducing window length by 1 (32 → 31 frames per window;
  3 windows per video unchanged).
- Raw diffs — no normalization. CLIP features through Adam/AdamW is a
  forgiving combination; scale differences are typically absorbed without
  explicit normalization.
- The transform is applied **after** any frame shuffling, so the shuffled
  temporal-gap measurement remains a valid "destroy temporal order" baseline.

One shared helper (`apply_temporal_transform`) is the single source of truth;
both train and eval paths import it. Training applies it inside the
DataLoader (no shuffle to coordinate with); evaluation applies it inside
`Tester._forward_all` after the optional shuffle, so ordering stays correct.

The `layer_probe.py` orchestrator's **training loop** is unchanged —
`input_transform` rides through the deep-copied config into `train_from_config`
without any change to the layer iteration or training logic. Minor bookkeeping
changes in §2.7 (output-dir suffix, extra CSV column, plot filename) keep raw
and diff runs from colliding.

Sweep cost: 3 models × 2 transforms = **6 invocations** of the script (one
invocation = 8 training runs across layers). The `none`-transform invocations
already exist from the prior layer probe, so only **3 new invocations** are
needed to complete the matrix.

---

## 2. Files changed and what changes each needs

### 2.1 New file — `training/utils/temporal_transforms.py`

Tiny shared utility so train and eval paths can't drift.

- Add function `apply_temporal_transform(x, kind)`:
  - Accepts a tensor with T as the second-to-last axis (works for
    `[T, D]` train shape and `[B, W, T, D]` eval shape).
  - `kind == 'none'`: return `x` unchanged.
  - `kind == 'diff'`: return `x[..., 1:, :] - x[..., :-1, :]`.
  - Raise `ValueError` for unknown kinds (cheap guard at the boundary).

### 2.2 `training/data_loader.py` (training path)

- `DeepfakeDataset.__init__`: accept new kwarg `input_transform='none'`, store
  on `self`. Keep existing ctor args otherwise unchanged.
- `DeepfakeDataset.__getitem__`: after the train/val window selection (lines
  65–66), apply `apply_temporal_transform(video_features, self.input_transform)`.
  Label/return shape stay the same except T shrinks by 1 when `diff`.
- Print line at line 52 (`first embedding shape=... embed_dim=...`): append
  the transform applied, e.g. `transform=diff → effective_T=31`. Helps catch
  config mistakes at dataset-init time.

### 2.3 `evaluation/data_loader.py` (eval path)

- `DeepfakeTestDataset.__init__`: accept new kwarg `input_transform='none'`,
  store on `self`. **Do not apply it in `__getitem__`** — the shuffle lives in
  the Tester, which has to run first.
- `DeepfakeTestDataset`: expose `self.input_transform` as a public attribute
  so the Tester can read it off the dataset without a second config plumb.

### 2.4 `evaluation/tester.py`

- `Tester._forward_all`: read `self.dataloader.dataset.input_transform` once
  at the top (or accept it via constructor — see below). After the existing
  shuffle block (lines 60–61), call
  `x = apply_temporal_transform(x, input_transform)` before the reshape to
  `[B*W, T, D]`. This is the single line that fixes the shuffle-then-diff
  ordering.
- `Tester.__init__` *or* `Tester.evaluate`: pick one place to discover the
  transform kind. Cleanest: read it from the dataset in `_forward_all` (one
  line, no constructor change). The dataset is the authoritative source; the
  Tester just honors it.
- `Tester.evaluate` result dict: add `input_transform` alongside the existing
  `shuffle_frames` key so `results.json` records exactly what was evaluated.

### 2.5 `training/train.py`

- `prepare_data`: pass `input_transform=config.get('input_transform', 'none')`
  into the `DeepfakeDataset` constructor.
- Log line added earlier (CLIP embeddings / Data split block): append one
  more line:
  `logger.info(f"  input_transform: {config.get('input_transform', 'none')}")`.
  Keeps the config visible at the top of every `training.log`.

### 2.6 `training/trainer.py`

- `save_run_config`: add `'input_transform': cfg.get('input_transform', 'none')`
  to the persisted dict, so `run_config.json` records the transform. Needed
  for post-hoc analysis and plot generation — without this, we can't tell a
  `none` run from a `diff` run just by reading the run dir.

### 2.7 `training/searches/layer_probe.py` (orchestrator)

- `build_cell_config`: already deep-copies the base YAML. No code change
  strictly needed — `input_transform` rides through untouched from the base
  YAML. But:
- Tag `out_root` directory name with the transform so sibling runs (raw vs
  diff on the same model) don't collide:
  `<timestamp>_<model_type>_<input_transform>/` when `input_transform != 'none'`,
  else stay `<timestamp>_<model_type>/` for backwards compatibility with
  existing linear/bigru/transformer runs.
- `summary.csv` row schema: add an `input_transform` column so merged CSVs in
  replot mode can distinguish raw from diff cells. Dedup key becomes
  `(model_type, layer, input_transform)` instead of `(model_type, layer)`.
- Plot titles/filenames: when `input_transform != 'none'`, append it to
  plot titles and filenames (`auc_<model_type>_<transform>.png`). Keeps
  outputs from colliding and makes plots self-identifying.

### 2.8 `training/configs/*.yaml` (all base configs)

- Add `input_transform: none` as a top-level field in every base config that
  goes through `train_from_config`:
  - `training/configs/linear.yaml`
  - `training/configs/bigru_1.yaml` (and any other bigru variants)
  - `training/configs/trans_proj.yaml/trans_proj.yaml`
  - `training/configs/bigru_probe.yaml`
  - `training/configs/transformer_probe.yaml`
- Default to `none` so existing behavior is preserved. The diff pilot uses
  copies of the probe configs with `input_transform: diff` set explicitly.

### 2.9 `training/searches/parameter_search.py` (forward compat, no code change)

- `build_trial_config` does a `deepcopy(base_config)`, so `input_transform`
  rides through untouched. Nothing to change now.
- If we later want the HP search to treat `input_transform` as a categorical
  search dimension, that's a 2-line addition to `search_space` — out of
  scope for this plan.

---

## 3. Configs for the pilot

Two new base configs that just flip `input_transform: diff` on top of the
existing probe configs. Same catalogue_file, same hparams, same everything
else — apples-to-apples comparison with the existing raw-embedding runs.

- `training/configs/linear_diff.yaml`     — copy of linear.yaml + `input_transform: diff`
- `training/configs/bigru_diff.yaml`      — copy of bigru_probe.yaml + `input_transform: diff`
- `training/configs/transformer_diff.yaml` — copy of transformer_probe.yaml + `input_transform: diff`

---

## 4. Running the pilot

Three `layer_probe.py` invocations — same pattern as the existing runs, just
pointing at the `_diff` configs:

```
python training/searches/layer_probe.py \
  --base-config training/configs/linear_diff.yaml \
  --embeddings-dir clip/embeddings/probing/ViT-L-14-336-quickgelu \
  --out-dir logs/probing/
# repeat for bigru_diff.yaml, transformer_diff.yaml
```

Output dirs: `logs/probing/<ts>_<model>_diff/` — sibling to the existing
`<ts>_<model>/` raw runs, no collision.

Then a single replot invocation to merge raw + diff summaries into one set of
plots per model (two lines per plot: raw and diff) and one combined plot
across models:

```
python training/searches/layer_probe.py --plot-only \
  --from-summaries logs/probing/<ts>_linear/summary.csv \
                   logs/probing/<ts>_linear_diff/summary.csv \
                   logs/probing/<ts>_bigru/summary.csv \
                   logs/probing/<ts>_bigru_diff/summary.csv \
                   logs/probing/<ts>_transformer/summary.csv \
                   logs/probing/<ts>_transformer_diff/summary.csv \
  --plot-out-dir logs/probing/combined_diff/plots/
```

Since `summary.csv` now carries `input_transform`, the replot's dedup and
line-grouping logic naturally treats raw and diff as separate series.

---

## 5. Expected artifacts (per pilot invocation)

Unchanged from the existing probe outputs, modulo the directory-name tag:

```
logs/probing/<ts>_<model>_diff/
├── summary.csv                  # +1 column: input_transform
├── plots/
│   ├── auc_<model>_diff.png
│   └── temporal_gap_<model>_diff.png
├── block_4/  block_8/  ...  final/
│   ├── config.json              # records input_transform: diff
│   ├── run_config.json          # records input_transform: diff
│   ├── results.json             # records input_transform in test_std/test_shuf
│   └── training.log             # logs input_transform at top
└── orchestrator.log
```

---

## 6. Verification steps before kicking off the full sweep

Cheap sanity checks to catch bugs before burning 6 hours of compute:

1. **Shape check**: Load one (layer, diff) config, instantiate
   `DeepfakeDataset`, fetch index 0, assert returned shape is
   `[num_frames - 1, D]`. One-line pytest-style assertion.
2. **Identity check**: With `input_transform='none'`, confirm a full run
   reproduces the existing linear-probe AUC at `block_16` to 4 decimals.
   Guards against an unintended side effect from the refactor.
3. **Shuffle-diff ordering**: Confirm `Tester._forward_all` applies diff
   strictly after shuffle. A unit test would set `shuffle_frames=True,
   input_transform='diff'` on a 2-frame fake input and compare the result
   against a manual reference. Tiny test, catches the most important bug.
4. **Config log line**: Confirm `training.log` prints the new
   `input_transform: ...` line. Trivial visual check.

Run 1, 2, 4 at minimum. Run 3 if we want belt-and-braces.

---

## 7. Effort estimate

- `temporal_transforms.py` (new): ~10 lines
- `training/data_loader.py`: ~4 lines
- `evaluation/data_loader.py`: ~3 lines
- `evaluation/tester.py`: ~3 lines
- `training/train.py`: ~2 lines
- `training/trainer.py`: ~1 line
- `training/searches/layer_probe.py`: ~10 lines (dir tagging, summary column,
  plot filename suffix)
- Base configs: ~1 line × N configs

**Total: ~35 lines of code + 3 copy-pasted pilot configs.** One short
afternoon of implementation + verification, well within the "cheap pilot"
budget.

---

## 8. Non-goals

- Second-order diffs, absolute diffs, or other temporal transforms — extend
  `apply_temporal_transform` later if first-order diffs show promise.
- Making `input_transform` an HP-search dimension — trivial to add later, not
  in scope now.
- Changing the upstream CLIP embedding extraction — all transforms happen
  downstream of the saved embeddings.
- Input normalization of any kind (z-score, L2, etc.) — raw diffs only.

---

# APPENDIX — Possible future extensions (DO NOT IMPLEMENT)

**For the implementing model: the sections above (§1–§8) are the complete
scope of this plan. Everything below is ideas for potential future work,
recorded here so they're not lost. Do not implement any of the following as
part of the current plan. Do not reference them in the code or configs.
Treat this appendix as read-only notes.**

### A.1 Dataset-wide z-score normalization

If raw diffs turn out to train unstably (val AUC stuck at 0.5, loss NaN,
gradient collapse), add a `normalize: none | z_score` config field:

- Offline stats pass: iterate the training split, compute per-dimension mean
  `μ` and std `σ` over all diffs, save as `stats.npz` in the catalogue
  directory.
- Load-time transform: after `apply_temporal_transform`, apply
  `(d - μ) / σ`. Stats from training split only — no leakage to val/test.
- Scoped config knob: default `none`; `z_score` opts in.

### A.2 Higher-order / alternative temporal transforms

- Second-order diffs: `d²_t = d_{t+1} - d_t` — captures acceleration in the
  embedding space.
- Absolute diffs: `|x_t - x_{t-1}|` — magnitude-only, drops direction.
- Optical-flow-like residuals between adjacent embeddings, perhaps after an
  L2-normalization step.
- Would be added by extending the `kind` values accepted by
  `apply_temporal_transform`.

### A.3 Mixing raw + diff inputs

- Concatenate `[x_t, d_t]` along the feature dimension → model sees both
  static and motion features per timestep. Doubles input dim but may give
  the best of both worlds.
- Would require a small model-side adaptation (double the input dim config)
  — no longer a pure input-transform experiment.

### A.4 HP search over `input_transform`

- Promote `input_transform` to a `trial.suggest_categorical` dimension in
  `parameter_search.py:search_space`. Two lines of code once the pilot has
  established that `diff` is a viable option worth tuning against raw.

### A.5 Combining with PEFT

- Run LN-tuning with `input_transform: diff` to test whether PEFT can
  amplify (or cancel) the temporal signal from diffs. Only worth considering
  if both the diff pilot and the PEFT experiment land positive signals
  independently.
