# Layer Probing Orchestrator — Implementation Plan

One-shot orchestrator that trains + tests a classifier on each CLIP layer's
embeddings and plots the resulting AUC curves. Last step of Pilot 1.

The script is a thin wrapper over the existing training/evaluation functions —
it does NOT modify `training/train.py`, `training/trainer.py`,
`evaluation/test.py`, or `evaluation/tester.py`.

---

## 1. Goal

Answer two questions, on FF++ only:

1. **Which CLIP layer gives the best downstream classifier AUC?** (per-video)
2. **How does temporal information flow through CLIP's depth?** Measured as
   `temporal_gap = per_video_auc_standard − per_video_auc_shuffled` per layer.
   A larger gap = the classifier relies more on frame ordering at that depth.

Cross-dataset (CDF-v2), Optuna tuning, and preprocessing-variant sweeps are
out of scope for this script.

---

## 2. Inputs

- **Pre-computed embeddings**, one subdir per layer:
  `clip/embeddings/probing/ViT-L-14-336-quickgelu/<layer>/catalogue.csv`
  where `<layer> ∈ {block_4, block_8, block_12, block_16, block_20, block_22, pre_proj, final}`.
- **Split file**: `datasets/splits/FaceForensics++.csv` (train/val/test).
- **Base configs**, one per classifier family:
  - `training/configs/linear.yaml`
  - `training/configs/transformer.yaml` (verify exists, else create — prerequisite)
  - `training/configs/bigru.yaml` (verify exists, else create — prerequisite)

Each base config already supports `catalogue_file` and
`model.<type>.clip_embed_dim`, which is what the orchestrator overrides per
layer. Nothing else in the YAML schema needs to change.

---

## 3. CLI

One invocation runs **one model** across all layers. For 3 models, call the
script 3 times. Combined-models plots are produced later via replot mode
from the 3 `summary.csv` files (see §7a).

```
python training/searches/layer_probe.py \
  --base-config training/configs/linear.yaml \
  --embeddings-dir clip/embeddings/probing/ViT-L-14-336-quickgelu \
  --out-dir logs/probing/ \
  [--layers pre_proj block_12 ...]      # optional subset
  [--no-shuffle-test]                   # skip shuffled eval (default runs it)
  [--force]                             # bypass skip-if-exists
```

Defaults:
- `--layers`: auto-discovered from `<embeddings-dir>/*/catalogue.csv`.
- Layer order (for plots): `block_4, block_8, block_12, block_16, block_20,
  block_22, pre_proj, final` — chronological through the ViT forward pass.

The `--out-dir` receives a timestamped subdir per invocation:
`logs/probing/<timestamp>_<model_type>/`. Multiple invocations never clobber
each other's outputs.

---

## 4. Algorithm

```
base_cfg = load_yaml(<base-config>)
model_type = base_cfg['model_type']
out_root = <out-dir>/<timestamp>_<model_type>/

for layer in layers:
    run_dir = out_root / layer
    if run_dir/'results.json' exists and not --force:
        skip (read cached metrics into summary)
        continue

    cfg = deepcopy(base_cfg)
    cfg['catalogue_file'] = <embeddings-dir>/<layer>/catalogue.csv
    cfg['model'][model_type]['clip_embed_dim'] = 768 if layer=='final' else 1024

    t0 = time.perf_counter()
    train_result = train_from_config(cfg, log_path=run_dir)
    test_std  = evaluate_on_split(cfg, run_dir, shuffle_frames=False)
    test_shuf = evaluate_on_split(cfg, run_dir, shuffle_frames=True)
    wallclock = time.perf_counter() - t0

    write run_dir/'results.json' {train, test_std, test_shuf, wallclock}
    append summary row

write out_root/'summary.csv'
generate plots -> out_root/plots/   # 2 PNGs per §7
```

`evaluate_on_split()` is adapted from `training/searches/parameter_search.py:127–146`:
build `DeepfakeTestDataset`, wrap in a DataLoader, call `Tester.evaluate()`,
return the result dict.

---

## 5. Integration points (no modifications)

| Called function | Source | Purpose |
|---|---|---|
| `train_from_config(cfg, log_path=run_dir)` | `training/train.py:214` | train one (model, layer) → best_val_auroc |
| `DeepfakeTestDataset(...)` | `evaluation/data_loader.py:30` | build FF++ test loader |
| `Tester.evaluate(shuffle_frames=bool)` | `evaluation/tester.py:166` | returns `{per_window: {auc}, per_video: {auc}}` |
| Checkpoint reload | mirror `evaluation/test.py:95` | load `run_dir/model.pth` before test |

---

## 6. Output artifacts

Each invocation writes one timestamped subdir (model_type in the name):

```
logs/probing/<timestamp>_<model_type>/
├── summary.csv
├── plots/
│   ├── auc_<model_type>.png
│   └── temporal_gap_<model_type>.png
├── block_4/
│   ├── config.json          # resolved config snapshot
│   ├── training.log
│   ├── model.pth
│   └── results.json         # {train, test_std, test_shuf, wallclock}
├── block_8/  ...
├── ...
└── final/    ...
```

Three invocations (linear / transformer / bigru) produce three sibling
`<timestamp>_<model_type>/` directories under `logs/probing/`. A replot
call (§7a) merges their `summary.csv` files into a separate
`logs/probing/combined/plots/` dir without touching the originals.

### `summary.csv` schema

```
layer, model_type, clip_embed_dim,
best_val_auc,
test_auc_std_window, test_auc_std_video,
test_auc_shuf_window, test_auc_shuf_video,
temporal_gap_video,                    # = std_video - shuf_video
wallclock_s
```

Per-window AUCs are captured even though plots use per-video — cheap to save
and useful if the pattern surprises us.

---

## 7. Plots

All plots read from `summary.csv` — never from in-memory data mid-training —
so every plot is fully regeneratable from saved results. See §7a for replot
mode.

Since one invocation trains exactly one model, each invocation produces
**2 PNGs** in `out_root/plots/`:

### `auc_<model_type>.png` — AUC across layers (2 lines)

- **x**: layer (categorical, ordered by forward-pass depth:
  `block_4, block_8, block_12, block_16, block_20, block_22, pre_proj, final`)
- **y**: AUC (data-driven range, e.g. 0.5–1.0)
- **2 lines**:
  - `test_auc_std_video` — per-video AUC
  - `test_auc_std_window` — per-window AUC
- **title**: `"<model_type>: per-window vs per-video AUC across CLIP layers (FF++ test)"`
- **legend**: "per-video", "per-window"

Purpose: shows how aggregation strategy interacts with layer depth for one
classifier family.

### `temporal_gap_<model_type>.png` — std − shuffled per-video AUC (1 line)

- **x**: same layer axis
- **y**: `test_auc_std_video − test_auc_shuf_video`
- **1 line** for this invocation's model_type
- **horizontal dashed line at y=0** for reference — values near zero mean
  frame ordering doesn't help at that layer.
- **title**: `"<model_type>: temporal gap across CLIP layers (std − shuffled, per-video AUC)"`
- **legend**: model_type

Purpose: answers question 2 (how temporal information flows through CLIP
depth) for this classifier family.

Cross-model comparison plots (one line per model overlaid) are **not**
produced by the default run — they come from replot mode (§7a) once you
have `summary.csv` files from multiple invocations.

---

## 7a. Replot mode

The script has two execution modes, sharing code:

1. **Train + plot** (default): runs one model across all layers, writes
   results, emits the 2 per-model PNGs at the end.
2. **Plot-only**: skips all training, regenerates plots from existing
   `summary.csv` files. Used to produce cross-model comparison plots
   after separately invoking the trainer once per model.

```
# re-emit the 2 per-model PNGs from a single prior run
python training/searches/layer_probe.py --plot-only \
  --from-summaries logs/probing/2026-04-18_15-30_linear/summary.csv \
  --plot-out-dir logs/probing/2026-04-18_15-30_linear/plots/

# combine 3 runs (one per model) into cross-model comparison plots
python training/searches/layer_probe.py --plot-only \
  --from-summaries logs/probing/<ts>_linear/summary.csv \
                   logs/probing/<ts>_transformer/summary.csv \
                   logs/probing/<ts>_bigru/summary.csv \
  --plot-out-dir logs/probing/combined/plots/
```

Behavior when `--from-summaries` receives multiple paths:
- Concatenate the CSVs.
- Dedup on `(model_type, layer)` keeping the last-seen row (so a re-run on
  one cell overrides the older value if both CSVs cover it).
- Warn if duplicates are found with differing values.
- Output plots depend on how many distinct `model_type` values are present
  in the merged frame:
  - **Single model**: emit the same 2 per-model PNGs as the default run.
  - **Multiple models**: additionally emit cross-model comparison PNGs:
    - `auc_combined.png` — per-video AUC, one line per model_type.
    - `temporal_gap_combined.png` — per-video std − shuffled, one line
      per model_type, with y=0 reference.

This means you can train the 3 models in 3 separate invocations (on
different days, different machines, whatever) and then produce the
unified cross-model plots from the three `summary.csv` files in a single
replot call.

---

## 8. Resume / skip-if-exists

Mirrors `clip/embed.py`'s ergonomics:

- **Default**: if `<run_dir>/results.json` exists, skip the (model, layer) pair
  and read the cached metrics into `summary.csv`.
- **`--force`**: re-train every (model, layer) pair, overwriting.
- Plots are always regenerated from the final `summary.csv` — they're cheap.

A partial prior run → idempotent re-invocation finishes the remaining cells
and produces the plots.

---

## 9. Non-goals

- Cross-dataset eval on CDF-v2 — embeddings don't exist yet. The orchestrator
  structure generalizes trivially later (swap `catalogue_file` root).
- Hyperparameter tuning per layer — fixed config from the base yaml. Optuna
  stays in `parameter_search.py`.
- Config-authoring for new models — base configs are inputs to the script,
  not generated by it.

---

## 10. Prerequisites / open to verify before writing

1. **`BiGRU` handles `clip_embed_dim=1024`** (it has to accept the arg and
   project/feed correctly). Quick scan of `models/bigru.py` before writing.
2. **`Tester.evaluate()` supports `shuffle_frames=True`** — confirmed in
   `evaluation/tester.py:166`.
3. **Split CSV coverage**: `datasets/splits/FaceForensics++.csv` has `split`
   values including `test`. If only train/val are present, we'd need to also
   run `datasets/create_split.py` first.
4. **Results aggregation vs CLI**: summary.csv keyed by (layer, model_type) —
   one invocation writes one model's rows. Each invocation creates a new
   `<timestamp>_<model_type>/` dir so prior results aren't clobbered.

Address these on script authorship; none should force design changes.
Base configs are user-supplied inputs (`linear.yaml`, `bigru_1.yaml`,
`trans_proj.yaml`, etc. — whatever already exists under
`training/configs/`), not generated.

---

## 11. Estimated effort

- Orchestrator loop + config overrides: ~35 lines (single model, not 3×)
- `evaluate_on_split()` helper (copy from parameter_search.py): ~25 lines
- Summary CSV writer: ~15 lines
- Plotting (matplotlib, 2 default PNGs + 2 optional combined PNGs,
  via 2 helpers): ~40 lines
- Replot mode + multi-CSV merge + model-count branching: ~30 lines
- CLI/arg parsing + layer discovery + skip-if-exists: ~35 lines

Target: **~180 lines total**, no modifications to any existing training or
evaluation module.
