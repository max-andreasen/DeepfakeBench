# Temporal-diff pilot: code changes

Implements the plan in `TEMPORAL_DIFF_PLAN.md`: a single `input_transform`
config field drives whether the model trains on raw CLIP embeddings
(`none`) or first-order frame-to-frame diffs (`diff`), applied online in
the dataloader on the train side and in the tester on the eval side (so
the shuffled-frames baseline uses diffs of shuffled sequences, not
shuffles of already-diffed sequences).

Per-window shapes:
- `none` → `[T, D]` on train, `[B, W, T, D]` on eval (T = `num_frames`)
- `diff` → `[T-1, D]` on train, `[B, W, T-1, D]` on eval (T = `num_frames`)

For FF++ with `num_frames=32`: 31 diffs per window, 3 windows per video
(93 diffs total per video).

---

## 1. New: `training/utils/temporal_transforms.py`

Single source of truth for the temporal transform; imported by both the
training dataloader and the evaluation tester so they can't drift.

```python
VALID_KINDS = ("none", "diff")

def apply_temporal_transform(x, kind):
    if kind == "none":
        return x
    if kind == "diff":
        return x[..., 1:, :] - x[..., :-1, :]
    raise ValueError(...)
```

Operates on the T axis (second-to-last dim), so it works for train-path
`[T, D]` and eval-path `[B, W, T, D]` alike.

Also created empty `training/utils/__init__.py` to make `utils` a package.

---

## 2. `training/data_loader.py` (train-side)

- Added `from utils.temporal_transforms import apply_temporal_transform`.
- `DeepfakeDataset.__init__`: new kwarg `input_transform="none"`, stored as
  `self.input_transform`.
- Print line now reports `transform={input_transform}, effective_T={T or T-1}`.
- `__getitem__`: applies the transform right before returning
  (`video_features = apply_temporal_transform(video_features, self.input_transform)`).

There is no shuffle on the train path, so applying in `__getitem__` is
safe (unlike the eval path).

---

## 3. `evaluation/data_loader.py` (eval-side)

- `DeepfakeTestDataset.__init__`: new kwarg `input_transform="none"`, stored as
  `self.input_transform`.
- Print line updated to report `transform` and `effective_T`.
- **Transform is NOT applied in `__getitem__`.** The shuffled-frames baseline
  lives in `Tester._forward_all`; if we diffed here first, we'd be
  measuring "shuffle of diffs" instead of "diffs of shuffled frames."

---

## 4. `evaluation/tester.py`

- Added dual import path (both `layer_probe.py` and `evaluation/test.py`
  invoke this file, and they put different dirs on `sys.path`):

  ```python
  try:
      from utils.temporal_transforms import apply_temporal_transform
  except ImportError:
      from training.utils.temporal_transforms import apply_temporal_transform
  ```

- `_forward_all`: reads `input_transform` from the dataset
  (`getattr(dataloader.dataset, "input_transform", "none")`) and applies
  it *after* the optional shuffle step and *before* flattening
  `[B, W, T, D]` → `[B*W, T, D]`.
- `evaluate()` result dict now carries `'input_transform'` so the
  summary layer can distinguish raw/diff runs.

---

## 5. `training/train.py`

- `prepare_data`: forwards
  `input_transform=config.get('input_transform', 'none')` to
  `DeepfakeDataset`.
- Startup logs now report the transform:
  `logger.info(f"  input_transform: {config.get('input_transform', 'none')}")`.

---

## 6. `training/trainer.py`

- `save_run_config`: persists `'input_transform': cfg.get('input_transform', 'none')`
  into `run_config.json` for reproducibility.

---

## 7. `training/searches/layer_probe.py`

Raw and diff runs of the same model need to coexist in the same summary
layer and be plotted as separate series.

- `SUMMARY_FIELDS`: added `'input_transform'`.
- New helper `series_key(row)` → `f"{model_type}"` or `f"{model_type}_{transform}"`.
- `summary_row_from_results`: reads `input_transform` from the payload
  (falls back to `test_std.input_transform`, then `'none'` for old runs).
- `load_summary_rows`: dedup key is now
  `(model_type, input_transform, layer)`; old rows without the field
  default to `'none'`.
- `group_by_model` renamed to `group_by_series`, keyed on `series_key`.
- Plotting helpers renamed:
  - `plot_per_model` → `plot_per_series`
  - `plot_temporal_gap` → `plot_temporal_gap_series`
  - `plot_combined_auc` / `plot_combined_gap` now iterate
    `rows_by_series` (sorted by label).
- `emit_plots` uses `rows_by_series`; filenames are `auc_<series>.png` and
  `temporal_gap_<series>.png`. Combined plots only fire when there are
  ≥2 distinct series.
- `run()`:
  - Reads `input_transform` from the base config.
  - `run_root` directory naming is suffixed when not `'none'`:
    `<timestamp>_<model_type>` → `<timestamp>_<model_type>_<transform>`
    so raw and diff runs don't collide on disk.
  - `orchestrator.log` header now includes `input_transform=...`.
  - The per-layer `results.json` payload now carries `'input_transform'`.
- `run()` and `replot()` switched from `group_by_model(rows)` to
  `group_by_series(rows)`.
- Module docstring updated to reflect series-based (not model-based) plot
  filenames.

---

## 8. Configs

### Base configs — added `input_transform: none` to each

- `training/configs/linear.yaml`
- `training/configs/bigru_probe.yaml`
- `training/configs/transformer_probe.yaml`
- `training/configs/bigru_1.yaml`
- `training/configs/bigru_2.yaml`

The field sits right under `num_frames` with a short explanatory comment.

### New diff-variant configs — `input_transform: diff`

- `training/configs/linear_diff.yaml`
- `training/configs/bigru_diff.yaml`
- `training/configs/transformer_diff.yaml`

**transformer_diff subtlety:** the transformer's learned positional
encoding is sized `(1, num_frames + 1, D)` in the constructor (see
`models/transformer.py:117`). With `input_transform: diff` the model
receives T=31, so `config['model']['transformer']['num_frames']` is set
to `31` in `transformer_diff.yaml` while the top-level
`num_frames: {train: 32, val: 32}` keeps the sampled window size at 32
(which then becomes 31 diffs). A comment in the file documents this
coupling.

BiGRU and linear have no positional-encoding dependency and accept any
sequence length, so their diff configs only flip the `input_transform`
field.

---

## Post-review fixes

A first-pass implementation review found three places where
`input_transform` was never threaded into `DeepfakeTestDataset`, which
would have caused a diff-trained model to silently be evaluated on raw
embeddings (wrong T, collapsed AUC). All three fixed:

- `training/searches/layer_probe.py::evaluate_on_split` — now forwards
  `input_transform=config.get('input_transform', 'none')`.
- `training/searches/parameter_search.py::evaluate_on_split` — same.
- `evaluation/test.py::build_test_loader` — now takes `input_transform`
  kwarg; the CLI reads it from the trained model's `run_config.json`
  (not the eval YAML) so test-time input always matches what the model
  was trained on. Logged at startup.

Also tidied `evaluation/tester.py`: replaced the explicit
`x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3])` with
`x.reshape(-1, x.shape[-2], x.shape[-1])` (was verbose; the original
captured-then-stale `B,W,T,D` locals were the reason the inline shape
lookups appeared in the first place).

## Verification

- Shape + identity check on `apply_temporal_transform`:
  - `[32, 4]` → `none`: `[32, 4]` identity; `diff`: `[31, 4]` matches manual `x[1:]-x[:-1]`
  - `[2, 3, 32, 8]` → `diff`: `[2, 3, 31, 8]`
  - Unknown kind raises `ValueError`
- Every config parses; `input_transform` present in all six listed above;
  `transformer_diff.yaml` has `transformer.num_frames=31` as required.
- Both import paths for `apply_temporal_transform` resolve:
  - `from utils.temporal_transforms` (layer_probe with `training/` on path)
  - `from training.utils.temporal_transforms` (evaluation/test.py with repo root on path)

End-to-end smoke tests on real FF++ block_16 embeddings (96 frames/video):
- `DeepfakeDataset(split='train', input_transform='diff')` returns
  `[31, 1024]` per item, and matches `x_raw[1:] - x_raw[:-1]` for a
  seed-matched raw pull.
- `DeepfakeTestDataset(input_transform='diff')` returns `[W=3, T=32, D]`
  (diff NOT applied in `__getitem__`, as required).
- `Tester.evaluate` with a probe model that records the T it receives:
  - `input_transform='diff'`: model always sees `T=31` (whether shuffled or not).
  - `input_transform='none'`: model always sees `T=32`.
- `models.transformer` with `num_frames=31` accepts `T=31` and correctly
  raises `RuntimeError` on `T=32` (positional-encoding guard).
- Shuffle-then-diff vs diff-then-shuffle produce different values on a
  constructed sequence — confirms the ordering in `_forward_all` matters
  and the current implementation has it right.
