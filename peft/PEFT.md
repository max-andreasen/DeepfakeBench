# PEFT — LN-tuned CLIP ViT-L/14 + temporal transformer

Parameter-efficient fine-tuning for deepfake detection. Unfreezes only the
LayerNorm γ/β of CLIP ViT-L/14-336 (OpenAI) — Yermakov et al. 2025 — and
trains a from-scratch temporal transformer head on the 1024-d
pre-projection CLS feature.

For the design rationale and step-by-step build/test plan, see
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## What this adds vs. the base repo

The existing pipeline (root `training/` + `clip/embed.py`) freezes CLIP, caches
embeddings to `.npz`, and trains only a small temporal head. PEFT runs CLIP
**inside** the training loop so its LayerNorm parameters can update — this
means:

- No `.npz` cache reuse — frames are read from disk each epoch.
- A new dataset class (`peft/data_loader.py`) that streams PNG frames.
- A composite model (`peft/models/clip_peft.py`) that owns frozen CLIP +
  trainable LNs + the temporal head together.
- Cross-dataset eval needs CLIP-in-the-loop too (existing CDFv2 `.npz` are
  from frozen CLIP and aren't valid for a tuned model).

Zero edits to existing files outside `peft/`.

## Layout

```
peft/
├── IMPLEMENTATION_PLAN.md     # the build spec (read this first)
├── PEFT.md                    # this file
├── models/clip_peft.py        # CompositePEFT
├── data_loader.py             # FramePEFTDataset, FramePEFTTestDataset
├── trainer.py                 # PEFTTrainer
├── train.py                   # training entry point
├── evaluation/
│   ├── tester.py              # PEFTTester
│   └── test.py                # eval entry point
├── configs/
│   ├── peft_ff_mtcnn.yaml         # main FF++ training config
│   ├── peft_smoke.yaml            # 1-epoch / 10-video smoke run
│   ├── peft_eval_ff_test.yaml     # FF++ test eval
│   └── peft_eval_cdfv2.yaml       # CDFv2 cross-dataset eval
├── build_split_csv.py         # one-shot: rearrange JSON → datasets/splits/<ds>.csv
└── trained/                   # run outputs land here
```

## Prerequisites

- Conda env `DeepfakeBench` with torch + open_clip (see repo root `install.sh`).
- MTCNN-preprocessed frames already on disk and rearrange JSON at
  `preprocessing/rearrangements/dataset_json_mtcnn/<dataset>.json`.
- For FF++: `datasets/splits/FaceForensics++.csv` (already in repo).
- For CDFv2: `datasets/splits/Celeb-DF-v2.csv` — generate once with
  ```bash
  python peft/build_split_csv.py --dataset "Celeb-DF-v2"
  ```

## Smoke run (verify the pipeline)

Sanity-check the full forward+backward path on 10 train + 10 val FF++ videos.
Useful before burning a real run.

```bash
python peft/train.py --config peft/configs/peft_smoke.yaml
```

Expected: training log shows `trainable=34.53M total=337.9M`, one epoch
completes, `model.pth` (~138 MB) and `run_config.json` written under
`peft/trained/peft_smoke_<timestamp>/`.

If it OOMs at `batchSize.train: 1` with `grad_checkpointing: true`, the GPU
is too small / contended; move to a larger card.

## Full FF++ training

```bash
python peft/train.py --config peft/configs/peft_ff_mtcnn.yaml
```

30 epochs, AdamW lr=2.0e-5, cosine schedule, fp16 + grad checkpointing,
effective batch 8 (B=1 train × accum=8). The temporal head is from scratch.

Outputs land in `peft/trained/peft_ln_ff_mtcnn_<YYYY-MM-DD-HH-MM-SS>/`:
- `training.log` — per-epoch train_loss + val AUC/ACC.
- `model.pth` — trainable state dict (LN γ/β + temporal head + input_proj
  + classifier). ~138 MB at fp32. Frozen CLIP weights are NOT saved.
- `run_config.json` — full config + `best_val_auc` + `per_epoch_val_auc[]`
  + completion flag. Eval scripts read this to rebuild the model.

## Evaluation

Both eval configs reference `trained_model_dir` — replace
`<FILL_IN_AFTER_STEP_7>` with the timestamp suffix of your training run dir
before running.

### FF++ test split

```bash
python peft/evaluation/test.py --config peft/configs/peft_eval_ff_test.yaml
```

### CDFv2 cross-dataset (518 official test videos)

```bash
python peft/evaluation/test.py --config peft/configs/peft_eval_cdfv2.yaml
```

Both write to `peft/evaluation/results/<trained_dir_basename>/<run_tag>/`:
- `results.json` — per-window + per-video metrics (acc, auc, precision,
  recall, f1, confusion matrix, best-threshold acc) for both standard and
  optional shuffled passes.
- `eval_config.json` — the resolved eval config.
- `test.log` — log of the eval run.

`<run_tag>` defaults to the config filename stem; override with `--run_tag`.

### Optional: temporal-shuffle ablation

Set `eval_shuffled: true` in the eval config. Shuffles the temporal axis per
window before forward — measures how much of the model's signal comes from
temporal ordering vs. per-frame appearance.

## Reading `run_config.json`

```json
{
  "saved_utc": "...",
  "config":   { ... full training config ... },
  "best_val_auc": 0.93,
  "metrics": {
    "per_epoch_val_auc": [0.71, 0.83, ..., 0.92],
    "completed": true
  },
  "amp_dtype": "float16",
  "grad_accum": 8,
  ...
}
```

`completed: false` means the loop crashed mid-training (e.g., OOM); the
checkpoint reflects the best-so-far AUC, not necessarily the converged model.

## Known gotchas

- **L2-norm is intentionally absent** in the PEFT path. The frozen-CLIP
  pipeline L2-normalizes; this one must not, or LN's learned γ is erased.
- **`visual.proj = None`** is how we get the 1024-d pre_proj feature. Don't
  monkey-patch hooks — open_clip supports this directly.
- **Optimizer is built over `model.trainable_parameters()`**, not
  `model.parameters()`. Building over all params creates AdamW state for the
  ~300 M frozen weights and burns ~2.4 GB VRAM for nothing.
- **T=96 frames per video is load-bearing** (feedback memory). Train-time
  sampling takes a random 32-window from this superset; do not change.
- **CDFv2's rearrange JSON** has overlapping video_ids across train/val/test;
  `build_split_csv.py` handles this with a 4-column dedup. Check the printed
  per-split counts after generating the CSV.
- **YAML floats** must be written `2.0e-5`, never `2e-5` — PyYAML 1.1 parses
  the latter as a string.
