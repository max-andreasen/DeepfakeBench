# PEFT Implementation Plan — LN-tuned CLIP ViT-L/14 + Temporal Transformer

> **Purpose of this document.** This is a self-contained build spec for adding
> PEFT (LayerNorm-tuning of CLIP ViT-L/14) on top of the existing temporal
> transformer head. It is written for *future-me* to read cold and execute
> step-by-step. Every step has a concrete acceptance test. Do not deviate from
> the locked decisions without re-reading §2.

---

## 1 · What we're building

A fine-tuning pipeline that unfreezes **only the LayerNorm γ/β** of the CLIP
ViT-L/14-336 (OpenAI) visual encoder and jointly trains a from-scratch
temporal transformer head on top of the **1024-d pre-projection CLS feature**.
Target dataset: FaceForensics++ (train + val on the existing split CSV).
Cross-dataset eval: Celeb-DF-v2. Approach follows **Yermakov et al. 2025**
(LN-tuning for CLIP on deepfake detection).

The existing offline-embedding pipeline (frozen CLIP → `.npz` → small head) is
untouched. PEFT runs CLIP *inside* the training loop, so it does not and
cannot reuse existing `.npz` caches.

## 2 · Locked decisions — do not re-litigate

| # | Decision | Why / source |
|---|----------|--------------|
| 1 | LN-tuning scope = **all 50 LayerNorms** in `model.visual` (`ln_pre`, `ln_post`, every `resblocks[i].ln_1` / `ln_2`). | Matches Yermakov 2025's canonical LN-tuning. |
| 2 | "CLIP pre-projection layer" = the existing `input_proj` inside `models/transformer.py` (Linear 1024→512 + LN). Already trainable. | User-confirmed. No new module needed. |
| 3 | Feature = **1024-d `pre_proj` CLS** (output of `visual.ln_post`). Bypass `visual.proj` by setting `model.visual.proj = None`. | Paper uses pre-projection. |
| 4 | Temporal head trained **from scratch** with the same hyperparams as the MTCNN pilot. | Isolates the LN-tuning delta. |
| 5 | **No L2-norm** on the CLS feature inside the training graph. | L2-norm cancels LN's learned γ — defeats LN-tuning. |
| 6 | Train on FF++ train, validate on FF++ val each epoch. Test on FF++ test + CDFv2 (cross-dataset). | User-approved scope. |
| 7 | Existing `.npz` caches are **not** reused anywhere in PEFT. LN weights change per-step so features must be recomputed live (and recomputed again at test time with the final weights). | Correctness. |
| 8 | Gradient checkpointing + AMP (fp16) **from day one**. Target GPU: RTX 5070 12 GB first, cluster if OOM. | Memory budget. |
| 9 | Zero edits to existing files. Everything lives under `peft/`. | User-approved. |
| 10 | T=96 frames per video is load-bearing in preprocessing. Train-time sampling takes a random 32-window from the 96-frame superset. Do not change T=96. | Hard feedback rule. |
| 11 | YAML scientific notation must use explicit decimal, e.g. `2.0e-5` not `2e-5`. | PyYAML 1.1 parses bare `2e-5` as string. |

## 3 · File layout (all new, under `peft/`)

```
peft/
├── IMPLEMENTATION_PLAN.md           # this file
├── PEFT.md                          # short user-facing readme (last step)
├── __init__.py
├── models/
│   ├── __init__.py
│   └── clip_peft.py                 # CompositePEFT (frozen CLIP + LN-trainable + temporal head)
├── data_loader.py                   # FramePEFTDataset, FramePEFTTestDataset
├── train.py                         # CLI entry point
├── trainer.py                       # PEFTTrainer
├── configs/
│   ├── peft_ff_mtcnn.yaml           # training config
│   ├── peft_eval_ff_test.yaml       # FF++ test eval
│   └── peft_eval_cdfv2.yaml         # CDFv2 cross-dataset eval
├── evaluation/
│   ├── __init__.py
│   ├── tester.py                    # PEFTTester (CLIP-in-the-loop eval)
│   └── test.py                      # eval entry point
└── trained/                         # run outputs (created at runtime)
```

## 4 · Do-not-touch list

Leave these files exactly as they are. Read-only imports only.

- `models/transformer.py` — import `Transformer`, instantiate with
  `clip_embed_dim=1024`. The learnable `input_proj` inside it IS the
  "pre-projection layer." Nothing to change.
- `models/__init__.py`, `models/linear_cls.py`, `models/bigru.py`
- Entire `training/` directory (it operates on cached `.npz`, not relevant here).
- Entire `clip/` directory (offline embedder — not used in PEFT).
- Entire `preprocessing/` directory (frames/JSON already on disk).
- `evaluation/tester.py` — we *import* its pooling and metric helpers from
  `peft/evaluation/tester.py`; we do not edit them.

---

## 5 · Implementation steps

Each step has: **Goal**, **Files**, **Notes** (design details that matter),
**Acceptance test** (a concrete thing to run that must pass), **Checkbox**.

### ☑ Step 1 — Skeleton (done 2026-04-24)

**Goal.** Directory + empty module files so later steps can import each other.

**Files.**
- `peft/__init__.py` (empty)
- `peft/models/__init__.py` (empty)
- `peft/evaluation/__init__.py` (empty)
- `peft/configs/` (directory, may hold a `.gitkeep`)
- `peft/trained/` (directory, `.gitkeep`)

**Notes.** Don't add `__init__` re-exports yet — keeps import errors localized
if later steps have bugs.

**Acceptance test.**
```bash
python -c "import peft; import peft.models; import peft.evaluation; print('ok')"
```
From repo root. Must print `ok`.

---

### ☑ Step 2 — Composite model (`peft/models/clip_peft.py`) (done 2026-04-25)

**Goal.** A single `nn.Module` that owns frozen-except-LN CLIP + temporal head,
with forward `[B, T, 3, 336, 336] → [B, 2]`.

**Files.** `peft/models/clip_peft.py`.

**Notes.**

Design skeleton (not verbatim — fill in types, docstring):

```python
import open_clip
import torch
import torch.nn as nn

# Repo root on sys.path expected; import the existing head unchanged.
from models.transformer import Transformer


class CompositePEFT(nn.Module):
    """Frozen CLIP ViT-L/14-336 with all LayerNorms trainable + temporal head.

    Forward: x [B, T, 3, 336, 336] -> logits [B, 2]
    """

    def __init__(
        self,
        clip_name: str = "ViT-L-14-336-quickgelu",
        clip_pretrained: str = "openai",
        ln_scope: str = "all",              # "all" | "ln_post" | "last_n:<int>"
        grad_checkpointing: bool = True,
        temporal_kwargs: dict | None = None,
    ):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            clip_name, pretrained=clip_pretrained
        )
        self.visual = model.visual              # drop text tower entirely
        self.visual.proj = None                 # skip 1024->768 projection in forward

        self._freeze_and_unfreeze_lns(ln_scope)

        if grad_checkpointing:
            # open_clip >=2.20 exposes this; wraps resblocks in torch.utils.checkpoint
            self.visual.set_grad_checkpointing(True)

        tk = dict(
            clip_embed_dim=1024,
            num_frames=32,
            num_classes=2,
            num_layers=8,
            n_heads=8,
            dim_feedforward=3072,
            attn_dropout=0.1,
            mlp_dropout=0.4,
            mlp_hidden_dim=512,
        )
        if temporal_kwargs:
            tk.update(temporal_kwargs)
        self.temporal = Transformer(**tk)

    def _freeze_and_unfreeze_lns(self, scope: str) -> None:
        for p in self.visual.parameters():
            p.requires_grad_(False)

        if scope == "all":
            targets = [m for m in self.visual.modules() if isinstance(m, nn.LayerNorm)]
        elif scope == "ln_post":
            targets = [self.visual.ln_post]
        elif scope.startswith("last_n:"):
            n = int(scope.split(":", 1)[1])
            blocks = self.visual.transformer.resblocks[-n:]
            targets = [m for b in blocks for m in b.modules() if isinstance(m, nn.LayerNorm)]
            targets += [self.visual.ln_post]  # always include ln_post
        else:
            raise ValueError(f"Unknown ln_scope: {scope!r}")

        for m in targets:
            for p in m.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self) -> dict:
        """Only save params that have requires_grad=True (LNs + head). ~140 MB at fp32."""
        full = self.state_dict()
        trainable_names = {
            name for name, p in self.named_parameters() if p.requires_grad
        }
        # Also include all sub-module buffers of the temporal head (it's fully trainable)
        # and nothing else. Keep keys stable so load_state_dict matches.
        keep = {k: v for k, v in full.items()
                if (k in trainable_names) or k.startswith("temporal.")}
        return keep

    def load_trainable_state_dict(self, sd: dict) -> None:
        self.load_state_dict(sd, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.visual(x)                  # [B*T, 1024] — pre_proj CLS, no L2-norm
        feats = feats.reshape(B, T, -1).float() # cast to fp32 before head
        return self.temporal(feats)             # [B, 2]
```

**Gotcha.** `self.visual(x)` with `proj=None` returns the ln_post CLS directly
(1024-d). Verify by reading `open_clip/model.py::VisionTransformer.forward` —
there's a `if self.proj is not None: x = x @ self.proj` guard. This is the
canonical open_clip way; don't monkey-patch.

**Acceptance test.** From repo root:
```python
import torch, sys
sys.path.insert(0, ".")
from peft.models.clip_peft import CompositePEFT
m = CompositePEFT(grad_checkpointing=False)  # checkpointing needs cuda
n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in m.parameters())
print(f"trainable={n_train/1e6:.2f}M total={n_total/1e6:.1f}M")
# Expect: trainable ~34.5M (102 k LN + 34.4 M temporal head), total ~338M.

# T must equal temporal.num_frames (default 32) so positional_encoding fits.
x = torch.randn(1, 32, 3, 336, 336)
y = m(x)
assert y.shape == (1, 2), y.shape
print("forward ok")
```
Must print a trainable count between 33M and 36M, total ~330–340M, and
`forward ok`.

---

### ☑ Step 3 — Dataset (`peft/data_loader.py`) (done 2026-04-25)

**Goal.** Stream face-cropped PNG frames from the rearrange JSON, apply CLIP's
preprocess transform in workers, yield `[T, 3, 336, 336]` tensors with labels.

**Files.** `peft/data_loader.py`.

**Notes.**

Two dataset classes — mirror the existing `training/data_loader.py` /
`evaluation/data_loader.py` split.

Key points:
- Build the per-video dataframe from rearrange JSON the same way
  `clip/embed.py::build_df_from_repo_json` does (flatten compression layer for
  FF++). Copy the logic; don't import the module (it's loaded via
  `_load_module_from_path` gymnastics, not a clean import).
- Reuse the `LABEL_MAP` dict from `clip/embed.py` — again, copy it inline.
- Join with the split CSV on `(dataset, label_cat, video_id)` and filter by
  `split`. Same keys as `training/data_loader.py::JOIN_KEYS`.
- Require `len(frame_paths) >= 96`; raise a clear error if any video is short.
- `preprocess` is passed in from the caller (obtained via
  `open_clip.create_model_and_transforms(...)[-1]`). The dataset does NOT
  import open_clip to avoid duplicate model loads.
- **Do not L2-normalize.** (The offline pipeline does; PEFT does not.)

Sketch:

```python
from pathlib import Path
import json
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_MAP = {
    # copy verbatim from clip/embed.py
    "FF-real": 1, "FF-DF": 0, "FF-F2F": 0, "FF-FS": 0, "FF-NT": 0,
    "CelebDFv2_real": 1, "CelebDFv2_fake": 0,
    # ... copy the full dict
}
JOIN_KEYS = ["dataset", "label_cat", "video_id"]


def _build_df(rearrange_json: Path, dataset_name: str) -> pd.DataFrame:
    """Replicates clip/embed.py:build_df_from_repo_json. Handles FF++ nested
    compression layer."""
    with open(rearrange_json) as f:
        data = json.load(f)
    if dataset_name not in data:
        raise ValueError(f"'{dataset_name}' not in {rearrange_json}")
    rows = []
    for label_cat, splits in data[dataset_name].items():
        numeric = LABEL_MAP.get(label_cat)
        if numeric is None:
            raise ValueError(f"No LABEL_MAP entry for {label_cat}")
        for videos_or_comp in splits.values():
            first = next(iter(videos_or_comp.values()), None)
            if first is not None and isinstance(first, dict) and "frames" not in first:
                videos = {}
                for _comp, comp_videos in videos_or_comp.items():
                    videos.update(comp_videos)
            else:
                videos = videos_or_comp
            for video_id, info in videos.items():
                rows.append({
                    "dataset": dataset_name,
                    "label_cat": label_cat,
                    "video_id": video_id,
                    "label": numeric,
                    "frame_paths": sorted(info["frames"]),
                })
    return pd.DataFrame(rows).drop_duplicates(subset=JOIN_KEYS)


class FramePEFTDataset(Dataset):
    """Train / val dataset. Samples a random contiguous T-window per video."""

    def __init__(
        self,
        split_file: str,
        rearrange_json: str,
        dataset_name: str,
        split: str,                  # "train" | "val"
        num_frames: int,
        preprocess: Callable,        # open_clip preprocess transform (PIL -> [3, H, W])
        min_superset: int = 96,      # guard: every video must have >=96 frames
    ):
        df_videos = _build_df(Path(rearrange_json), dataset_name)
        df_split = pd.read_csv(split_file)
        df_split["video_id"] = df_split["video_id"].astype(str)
        df_videos["video_id"] = df_videos["video_id"].astype(str)
        df = df_split.merge(df_videos, on=JOIN_KEYS, how="inner")
        df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"Empty split={split} after join")

        self.rows: List[dict] = df[["label", "frame_paths", "video_id", "label_cat"]].to_dict("records")
        self.num_frames = int(num_frames)
        self.preprocess = preprocess

        # Validate superset length upfront (fail fast, not in a worker).
        short = [r for r in self.rows if len(r["frame_paths"]) < min_superset]
        if short:
            raise ValueError(
                f"{len(short)} videos have <{min_superset} frames. "
                f"First: {short[0]['video_id']}"
            )

        print(f"FramePEFTDataset[{split}]: {len(self.rows)} videos, T={num_frames}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        paths = r["frame_paths"]
        total = len(paths)
        start = int(np.random.randint(0, total - self.num_frames + 1))
        selected = paths[start:start + self.num_frames]
        frames = [self.preprocess(Image.open(p).convert("RGB")) for p in selected]
        x = torch.stack(frames, dim=0)                  # [T, 3, H, W]
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        return x, y


class FramePEFTTestDataset(Dataset):
    """Eval dataset. Yields all 3 non-overlapping windows per video."""

    def __init__(
        self,
        split_file: str,
        rearrange_json: str,
        dataset_name: str,
        split: str,                  # "test" usually
        num_frames: int,
        preprocess: Callable,
    ):
        # Same build as FramePEFTDataset minus the random sampling.
        # Returns x [n_windows, T, 3, H, W], label, video_id, label_cat.
        # Compute n_windows from len(frame_paths) // num_frames, truncate remainder.
        # Mirrors evaluation/data_loader.py::DeepfakeTestDataset semantics.
        ...  # fill in following the pattern above
```

**Acceptance test.**
```python
import open_clip, sys
sys.path.insert(0, ".")
from peft.data_loader import FramePEFTDataset

_, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14-336-quickgelu", pretrained="openai")

ds = FramePEFTDataset(
    split_file="datasets/splits/FaceForensics++.csv",
    rearrange_json="preprocessing/rearrangements/dataset_json_mtcnn/FaceForensics++.json",
    dataset_name="FaceForensics++",
    split="train",
    num_frames=32,
    preprocess=preprocess,
)
x, y = ds[0]
print(x.shape, x.dtype, y)
# Expect: torch.Size([32, 3, 336, 336]) torch.float32 tensor(0 or 1)
assert x.shape == (32, 3, 336, 336)
assert x.dtype == torch.float32
assert y.dtype == torch.int64
```

---

### ☑ Step 4 — Trainer (`peft/trainer.py`) (done 2026-04-25)

**Goal.** Encapsulate one epoch of train + val. Handles AMP, grad accumulation,
best-ckpt saving, and run_config serialization. Mirrors
`training/trainer.py::Trainer` structure.

**Files.** `peft/trainer.py`.

**Notes.**

Key differences from the existing `Trainer`:
- `train_step` wraps CLIP forward in `torch.cuda.amp.autocast(dtype=fp16)`.
- `GradScaler` for the backward.
- Supports `grad_accum_steps > 1` by scaling loss and stepping every K batches.
- `save_ckpt(path)` saves **only** `model.trainable_state_dict()`, not the
  full state dict. Load side mirrors via `load_trainable_state_dict`.
- Val AUC computed with `sklearn.metrics.roc_auc_score` on softmax[:, 1].
- Save best-val-AUC checkpoint only (overwrite on improvement). Log each
  epoch's val AUC.

Sketch:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class PEFTTrainer:
    def __init__(self, config, model, optimizer, scheduler, logger):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = GradScaler()
        self.grad_accum = int(config.get("grad_accum_steps", 1))
        self.best_auc = 0.0

    def train_epoch(self, epoch, train_loader, val_loader):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"ep{epoch}", leave=False)):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = self.model(x)
                loss = F.cross_entropy(logits, y) / self.grad_accum
            self.scaler.scale(loss).backward()
            if (i + 1) % self.grad_accum == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.item()) * self.grad_accum
        avg = total / max(len(train_loader), 1)
        self.logger.info(f"epoch {epoch} train_loss={avg:.4f}")
        return self.eval_epoch(epoch, val_loader)

    @torch.no_grad()
    def eval_epoch(self, epoch, val_loader):
        self.model.eval()
        probs, labels = [], []
        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            with autocast(dtype=torch.float16):
                logits = self.model(x)
            p = torch.softmax(logits.float(), dim=1)[:, 1].cpu()
            probs.append(p); labels.append(y)
        probs = torch.cat(probs).numpy()
        labels = torch.cat(labels).numpy()
        auc = float(roc_auc_score(labels, probs))
        self.logger.info(f"epoch {epoch} val_auc={auc:.4f}")
        return auc

    def save_best(self, auc: float, out_path: str):
        if auc > self.best_auc:
            self.best_auc = auc
            torch.save(self.model.trainable_state_dict(), out_path)
            self.logger.info(f"saved best ckpt (auc={auc:.4f}) to {out_path}")

    def save_run_config(self, path: str, metrics: dict):
        payload = {
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "best_val_auc": self.best_auc,
            "metrics": metrics,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
```

**Acceptance test.** Deferred to Step 6 (end-to-end smoke) — Trainer is hard
to unit-test in isolation without the other pieces.

---

### ☑ Step 5 — Config + train entry (`peft/train.py`, `peft/configs/peft_ff_mtcnn.yaml`) (done 2026-04-25)

**Goal.** CLI `python peft/train.py --config peft/configs/peft_ff_mtcnn.yaml`
kicks off a full training run.

**Files.**
- `peft/configs/peft_ff_mtcnn.yaml`
- `peft/train.py`

**Notes.**

**Config (`peft_ff_mtcnn.yaml`):**

```yaml
tag: peft_ln_ff_mtcnn

# data
root_dir: /home/max-andreasen/GitHub/DeepfakeBench
split_file: datasets/splits/FaceForensics++.csv
rearrange_json: preprocessing/rearrangements/dataset_json_mtcnn/FaceForensics++.json
dataset_name: FaceForensics++
num_frames: { train: 32, val: 32 }

# model
clip:
  name: ViT-L-14-336-quickgelu
  pretrained: openai
  ln_scope: all                   # all | ln_post | last_n:<int>
  grad_checkpointing: true
  amp_dtype: fp16

temporal:
  clip_embed_dim: 1024
  num_frames: 32
  num_classes: 2
  num_layers: 8
  n_heads: 8
  dim_feedforward: 3072
  attn_dropout: 0.1
  mlp_dropout: 0.4
  mlp_hidden_dim: 512

# optim
optimizer:
  type: adamw
  lr: 2.0e-5
  weight_decay: 5.0e-4
  beta1: 0.9
  beta2: 0.999
  eps: 1.0e-8

lr_scheduler: cosine              # cosine | cosine_warmup | constant
num_epochs: 30
warmup_epochs: 2

# runtime
batchSize: { train: 1, val: 2 }
grad_accum_steps: 8
workers: 4
seed: 1024
device: cuda

# output
log_dir: peft/trained
save_ckpt: true
metric_scoring: auc
```

**`train.py` responsibilities (mirrors `training/train.py`):**
1. Parse `--config`, load YAML.
2. Build timestamped output dir under `cfg["log_dir"]` as
   `<log_dir>/<tag>_<YYYY-MM-DD-HH-MM-SS>/`. Create logger there
   (`logger.create_logger`).
3. Load `open_clip.create_model_and_transforms` once to get the `preprocess`
   transform, pass to both datasets. Build the composite model from the same
   config.
4. Build train + val DataLoaders using `FramePEFTDataset`.
5. Build optimizer over `model.trainable_parameters()` (NOT
   `model.parameters()` — that gives AdamW 300 M param states).
6. Build scheduler per `lr_scheduler` field.
7. Run `num_epochs` iterations; after each, call `trainer.save_best`.
8. After training, write `run_config.json` via `trainer.save_run_config`.

Reuse `logger.create_logger` from repo root (see `training/train.py:22`).

**Acceptance test.** Deferred to Step 6.

---

### ◐ Step 6 — Smoke run (1 epoch, ~20 videos) — BLOCKED on cluster

**Status (2026-04-25):** Pipeline plumbing verified end-to-end up to the
backward pass. CLIP loaded, model built (34.53 M trainable, matches Step 2
gate), dataset+loader built (`train batches: 10  val batches: 10`), forward
issued, `run_config.json` written. CUDA OOM'd inside the gradient-checkpoint
recompute on the first backward — but the cause is **environmental**: another
python process (PID 148927) was holding 7.06 GiB / 11.50 GiB at 100 %
GPU-util, leaving only ~3.4 GiB for the smoke run. The plan's Q6/§7 fallback
is "move to cluster"; that's the next action for the user.

When retrying on the cluster (or when the local GPU is free), the existing
`peft/configs/peft_smoke.yaml` config is unchanged-ready; just rerun
`python peft/train.py --config peft/configs/peft_smoke.yaml`.

**Goal.** Validate the entire train pipeline end-to-end without burning a full
run. Smoke config differs from main config by `num_epochs: 1` and a
`max_videos` field that caps the dataset.

**Files.** `peft/configs/peft_smoke.yaml` (temp). Add a `max_videos` cap to
`FramePEFTDataset.__init__` if not already there (truncate `self.rows`).

**Notes.** Lock the seed at 1024. Log must show:
- model built with ~34.5 M trainable params,
- one epoch completes (no OOM, no NaN loss),
- val AUC logged (value will be noisy on tiny data — not graded),
- `model.pth` and `run_config.json` on disk,
- best val AUC > 0 (loaded from the log).

If OOM at `batchSize.train: 1` with `grad_checkpointing: true`: this is the
signal to move to the cluster. Log the OOM and stop. Do not try `T<32` —
that's a real config change, not a band-aid.

**Acceptance test.**
```bash
python peft/train.py --config peft/configs/peft_smoke.yaml
# then:
ls peft/trained/peft_smoke_*/
# expect: model.pth  run_config.json  training.log
grep "val_auc=" peft/trained/peft_smoke_*/training.log
# expect at least one line with a float AUC.
```
Also manually verify the saved state dict is small:
```python
import torch
sd = torch.load("peft/trained/peft_smoke_<timestamp>/model.pth", map_location="cpu")
total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
print(f"ckpt size: {total_bytes/1e6:.2f} MB")
# expect ~135–145 MB (34.5M trainable params × 4 bytes fp32).
# NOT >300 MB — that would mean we saved frozen CLIP weights too.
```

---

### ◐ Step 7 — Full FF++ training (code ready, run blocked on Step 6)

**Status (2026-04-25):** No new code needed — uses `peft/train.py` +
`peft/configs/peft_ff_mtcnn.yaml`, both already verified-importing. Run
command:
```
python peft/train.py --config peft/configs/peft_ff_mtcnn.yaml
```
Blocked transitively on Step 6 (smoke).

**Goal.** Real 30-epoch run on FF++. Produces the PEFT checkpoint used for
both eval tasks.

**Notes.**
- Run with the main config (`peft_ff_mtcnn.yaml`).
- Expect ~5 h on RTX 5070 if batch=1 fits; faster on cluster.
- Log every epoch's train loss + val AUC. Final best val AUC should beat the
  frozen-CLIP pilot baseline (~0.86–0.91) — the expected delta from Yermakov
  is +2–5 AUC points.
- If training diverges (val AUC collapses to 0.5 or loss explodes): first
  check that `model.visual.proj is None` after init, then check AMP isn't
  producing fp16 NaNs in ln_post (if so, move `ln_post` forward to fp32 by
  keeping the last block's activations in fp32).

**Acceptance test.**
- `training.log` shows monotonically improving (noisy) val AUC that
  eventually exceeds 0.90.
- `model.pth` + `run_config.json` written.
- No NaN/Inf in training loss.

---

### ◐ Step 8 — Tester + FF++ test eval (`peft/evaluation/`) — code ready 2026-04-25

**Status:** all three files written and import-tested:
- `peft/evaluation/tester.py` — `PEFTTester`. Reuses
  `evaluation.tester.Tester._compute_metrics` and `._pool_windows_per_video`
  by calling them unbound and passing `self` (which carries `aggregation`
  and `softmax_temp`). Forward path: `[B, W, T, 3, H, W]` → reshape to
  `[B*W, T, 3, H, W]` → CompositePEFT → `[B*W, 2]` → repeat-interleave
  labels/keys → metrics → pool per video. AMP enabled in eval (no backward
  graph, so it's a free speedup).
- `peft/evaluation/test.py` — CLI entry point. Rebuilds `CompositePEFT`
  from the trained run's `run_config.json`, loads weights via
  `model.load_trainable_state_dict(...)`, runs `tester.evaluate`. Output dir:
  `<output_dir>/<trained_dir_basename>/<run_tag>/{results.json, eval_config.json, test.log}`.
- `peft/configs/peft_eval_ff_test.yaml` — eval config with placeholder
  `<FILL_IN_AFTER_STEP_7>` for the trained run's timestamp suffix.

Run command (after Step 7 produces a checkpoint):
```
python peft/evaluation/test.py --config peft/configs/peft_eval_ff_test.yaml
```
Blocked on Step 7's checkpoint.

**Goal.** Evaluate the trained checkpoint on FF++ test split using
3-windows-per-video aggregation. Produce `results.json`.

**Files.**
- `peft/evaluation/tester.py`
- `peft/evaluation/test.py`
- `peft/configs/peft_eval_ff_test.yaml`

**Notes.**

`tester.py` imports the existing pooling + metric helpers verbatim:

```python
# Importable as plain modules because repo root is on sys.path in test.py
from evaluation.tester import Tester as _BaseTester  # for _pool_windows_per_video, _compute_metrics
```

We do NOT subclass the existing Tester cleanly (its `_forward_all` assumes
cached features). Instead, implement a new `PEFTTester` that:
1. Takes a `CompositePEFT` model + test DataLoader.
2. For each batch `(x [B, W, T, 3, H, W], label, video_id, label_cat)`:
   - Reshape to `[B*W, T, 3, H, W]`, forward → `[B*W, 2]`.
   - Repeat-interleave labels / keys like the existing tester.
3. Calls `_pool_windows_per_video` and `_compute_metrics` from the existing
   Tester (copy them as free functions into `peft/evaluation/tester.py` if
   importing instance methods is awkward — either path works, be consistent).
4. `evaluate(dataloader, shuffle_frames=False)` returns the same result dict
   shape the existing Tester does. We run with `shuffle_frames=False` only
   for PEFT — the temporal-shuffle ablation is optional.

`test.py` responsibilities:
1. Parse `--config` + optional `--run_tag`.
2. Load run_config.json from `cfg["trained_model_dir"]`; rebuild
   `CompositePEFT` with the same clip/temporal kwargs.
3. Load `model.pth` via `model.load_trainable_state_dict(torch.load(...))`.
4. Build `FramePEFTTestDataset` + DataLoader.
5. Run tester.evaluate on `split=test`.
6. Write `results.json` + `eval_config.json` into
   `<output_dir>/<trained_dir_basename>/<run_tag>/`.

**Config (`peft_eval_ff_test.yaml`):**

```yaml
tag: peft_eval_ff_test

root_dir: /home/max-andreasen/GitHub/DeepfakeBench
split_file: datasets/splits/FaceForensics++.csv
rearrange_json: preprocessing/rearrangements/dataset_json_mtcnn/FaceForensics++.json
dataset_name: FaceForensics++
split: test

trained_model_dir: peft/trained/peft_ln_ff_mtcnn_<fill-in-after-step-7>
output_dir: peft/evaluation/results

num_frames: 32
batch_size: 1
workers: 4
seed: 1024
device: cuda
window_aggregation: mean
```

**Acceptance test.**
```bash
python peft/evaluation/test.py --config peft/configs/peft_eval_ff_test.yaml
# Check:
cat peft/evaluation/results/<trained_dir_basename>/peft_eval_ff_test/results.json \
    | python -c "import json,sys; d=json.load(sys.stdin); print(d['standard']['per_video']['auc'])"
# Expect: float AUC > 0.90.
```

---

### ◐ Step 9 — Cross-dataset CDFv2 eval — code ready 2026-04-25

**Status:** all artifacts in place:
- `peft/build_split_csv.py` — generic synthesizer: rearrange JSON →
  `datasets/splits/<dataset>.csv`. Dedup keys on the **4-tuple** (dataset,
  label_cat, video_id, split) — important because CDFv2's JSON tags the
  same video_id under multiple splits, and a 3-key dedup silently loses
  rows.
- `datasets/splits/Celeb-DF-v2.csv` — generated. Verified split breakdown:
  test = 178 real + 340 fake = 518 (the official CDFv2 test set);
  train = 5639 fake + 300 real; val = 340 fake + 178 real (val and test
  share rows; only `split=test` is consumed for cross-dataset eval).
- `peft/configs/peft_eval_cdfv2.yaml` — eval config with the same
  `<FILL_IN_AFTER_STEP_7>` placeholder.

Run command:
```
python peft/evaluation/test.py --config peft/configs/peft_eval_cdfv2.yaml
```

If `datasets/splits/Celeb-DF-v2.csv` is ever lost or needs regenerating:
```
python peft/build_split_csv.py --dataset "Celeb-DF-v2"
```
Blocked on Step 7's checkpoint.

**Goal.** Reuse Step 8's tester, different dataset.

**Files.** `peft/configs/peft_eval_cdfv2.yaml` (and maybe a CDFv2 split CSV if
one doesn't already exist — check `datasets/splits/` first).

**Notes.**
- Uses `preprocessing/rearrangements/dataset_json_mtcnn/Celeb-DF-v2.json`
  (confirmed present). No re-preprocessing needed.
- CDFv2 has no train/val/test split baked into a CSV the way FF++ does. Check
  `datasets/splits/` — if a CDFv2 CSV is missing, do NOT invent a split; use
  every video as `split=test`. Check the existing `datasets/splits/` layout
  before deciding.
- The `.npz` caches under `clip/embeddings/benchmarks/cdfv2_layer16/` are
  ignored here — they're frozen CLIP's output. PEFT runs CLIP live.

**Config (`peft_eval_cdfv2.yaml`):** Same as Step 8's eval config but:
```yaml
tag: peft_eval_cdfv2
split_file: datasets/splits/Celeb-DF-v2.csv     # or a synthesized all-test one
rearrange_json: preprocessing/rearrangements/dataset_json_mtcnn/Celeb-DF-v2.json
dataset_name: Celeb-DF-v2
split: test
```

**Acceptance test.**
```bash
python peft/evaluation/test.py --config peft/configs/peft_eval_cdfv2.yaml
# Check:
cat peft/evaluation/results/<trained_dir_basename>/peft_eval_cdfv2/results.json
# Expect: a results.json with per_video.auc field populated.
```
The actual AUC value is the experimental result, not a correctness gate —
cross-dataset deepfake AUC will be lower than FF++ test (often 0.60–0.80).
The correctness gate is that the file exists and all keys are populated
(no NaN, non-zero num_videos).

---

### ☑ Step 10 — Short readme (`peft/PEFT.md`) (done 2026-04-25)

**Goal.** User-facing doc (not this file). How to run train + eval, where
outputs land, how to read `run_config.json`.

**Notes.** Keep to ~1 page. Link to this IMPLEMENTATION_PLAN.md for deeper
context. Include the exact CLI commands for the two flows.

**Acceptance test.** File exists; `wc -l peft/PEFT.md` returns a reasonable
number (50–150 lines).

---

## 6 · Gotchas (read before implementing)

1. **`visual.proj = None` is the clean pre_proj hook.** Don't register forward
   hooks the way `clip/CLIP_embedder.py` does — hooks don't participate in
   autograd the way module outputs do, and you'd end up with two conventions.
   The `proj = None` trick is supported by open_clip by design.

2. **L2-norm is a trap.** The existing offline pipeline L2-normalizes. If you
   copy-paste from `CLIP_embedder.py::_embed_batch`, you'll inherit the
   `t / t.norm(dim=-1, keepdim=True)` line. Remove it for PEFT.

3. **`trainable_state_dict()` must be a strict subset.** If you accidentally
   include frozen CLIP params, `model.pth` balloons past 1 GB and disk fills
   fast. Use the exact whitelist pattern in Step 2 and verify the ckpt is
   ~135–145 MB in Step 6 (34.5 M trainable params × 4 bytes fp32).

4. **Optimizer must receive filtered params.** `AdamW(model.parameters(), ...)`
   creates state tensors for all 300 M params, wasting ~2.4 GB of VRAM for
   nothing. Use `AdamW(model.trainable_parameters(), ...)`.

5. **AMP + LN-tuning interactions.** ln_post's γ is trainable; ln_post runs in
   the autocast region as fp16. That's usually fine, but watch the first
   epoch's loss — if it's NaN, hoist ln_post out of autocast by running it in
   fp32 (wrap `self.visual.ln_post.forward` in a `with autocast(enabled=False):`
   inside a thin subclass, or patch open_clip).

6. **YAML floats.** Write `2.0e-5`, not `2e-5`. PyYAML 1.1 parses the latter
   as the string `"2e-5"`, and optimizer lr becomes a string, and AdamW
   silently accepts it (Python's numerics are liberal), producing bizarre
   behavior. Stored in `feedback_yaml_float_notation.md`.

7. **T=96 frames is load-bearing.** Videos are preprocessed at T=96 so
   training can sample a random 32-window. Do not change T=96 anywhere.
   See `feedback_T96_is_load_bearing.md`.

8. **Don't chain destructive commands into tests.** No
   `rm -rf peft/trained/* && python peft/train.py ...`. See
   `feedback_destructive_actions.md`.

9. **Grad checkpointing API.** `self.visual.set_grad_checkpointing(True)`
   works on open_clip >= 2.20. If the installed version is older (check with
   `pip show open_clip_torch`), fall back to manually wrapping resblocks in
   `torch.utils.checkpoint.checkpoint` inside a subclass.

10. **CLIP preprocess normalization.** open_clip's preprocess Compose includes
    mean/std normalization with CLIP's stats (NOT ImageNet's). Feed in raw
    PIL — don't pre-normalize anywhere else.

11. **FF++ label_cat multiplicity.** Fake manipulations reuse video_id
    strings (`000_003` exists in FF-DF, FF-F2F, FF-FS, FF-NT). The join key
    `(dataset, label_cat, video_id)` is mandatory — never key on `video_id`
    alone. See `evaluation/tester.py::_pool_windows_per_video` for the
    canonical pooling.

12. **rearrange JSON nested compression layer for FF++.** The compression
    level (`c23`) sits between split and videos. `clip/embed.py::build_df_
    from_repo_json` has the flatten logic; copy it.

## 7 · Risk register + fallbacks

| Risk | Trigger | Fallback |
|------|---------|----------|
| OOM on 12 GB 5070 at B=1 | CUDA OOM in first forward | Move to cluster. Bump B to 2–4 there. |
| Training diverges (val AUC ≈ 0.5) | epoch 1–3 log | Check: `visual.proj is None`, L2-norm removed, `trainable_parameters()` count sane. Lower LR to `5.0e-6`. |
| AMP NaNs in ln_post | loss = NaN in epoch 1 | Disable AMP (`amp_dtype: fp32`) for a smoke epoch to confirm; then hoist ln_post to fp32. |
| Checkpoint > 200 MB | step 6 test | `trainable_state_dict()` not filtering correctly (frozen CLIP weights leaking in). Re-verify the whitelist. Expected ~135–145 MB. |
| FF++ val AUC ≤ frozen-CLIP baseline | full run | Try `ln_scope: last_n:12`; bump LR; try `cosine_warmup`. |
| CDFv2 split CSV missing | Step 9 | Build a synthetic all-test CSV from the rearrange JSON: one row per (dataset=Celeb-DF-v2, label_cat, video_id, split=test). |

## 8 · Sanity pass before calling this "done"

1. Every step's acceptance test passes.
2. `git status` shows only new files under `peft/` — nothing modified
   outside it.
3. `peft/trained/` contains exactly one successful training run + one FF++
   eval + one CDFv2 eval.
4. `results.json` for FF++ test has a per-video AUC reported.
5. `results.json` for CDFv2 has a per-video AUC reported (value not graded).
6. IMPLEMENTATION_PLAN.md checkboxes in §5 all ticked.

---

## 9 · Extension: BiGRU temporal head

> **Goal.** Swap the temporal Transformer for a BiGRU under the *same*
> LN-tuned CLIP backbone, training pipeline, and evaluation protocol. This
> mirrors the frozen-backbone Transformer-vs-BiGRU comparison at the PEFT
> level, so the four-model story (Linear / BiGRU / Transformer frozen + PEFT
> variants) stays consistent.

### 9.1 · Locked decisions

| # | Decision | Why / source |
|---|----------|--------------|
| B1 | Temporal head class = `models.bigru.BiGRU` (existing, untouched). | Re-use the same head used in the frozen pipeline so PEFT-Transformer vs PEFT-BiGRU is a clean head-only contrast. |
| B2 | Backbone, LN scope, T=32 sampling window, AMP, grad checkpointing — all identical to the Transformer PEFT path. | Isolates the head as the only varying factor. |
| B3 | New config field `temporal_type: "transformer" | "bigru"` (default `"transformer"` to keep existing configs working). | Backwards-compatible; existing `peft_ff_mtcnn.yaml` etc. unchanged. |
| B4 | BiGRU hyperparameters seeded from the rank-1 frozen-pipeline BiGRU search (`bigru_search2`); re-tune later if needed. | Reasonable starting point; comparable to the frozen baseline. |
| B5 | Zero edits to `models/bigru.py`, the trainer, the data loader, or the entry script's training loop. Only `clip_peft.py` and configs change. | Smallest surface area; same do-not-touch policy as §4. |

### 9.2 · Files touched

```
peft/
├── models/clip_peft.py             # +temporal_type dispatch
├── train.py                        # pass temporal_type through to CompositePEFT
└── configs/
    ├── peft_bigru_ff_mtcnn.yaml    # NEW — BiGRU + FF++ val
    ├── peft_bigru_ff_cdfv2val.yaml # NEW — BiGRU + CDFv2 val (overfit-onset run)
    └── peft_bigru_smoke.yaml       # NEW — 1-epoch / 10-video smoke
```

No new module file; `BiGRU` is imported directly from `models.bigru`.

### 9.3 · Implementation steps

#### Step B1 — Dispatch in `CompositePEFT`

In `peft/models/clip_peft.py`:

- Add `from models.bigru import BiGRU` next to the existing
  `from models.transformer import Transformer` import.
- Add a `temporal_type: str = "transformer"` argument to
  `CompositePEFT.__init__`.
- Replace the unconditional `self.temporal = Transformer(**defaults)` block
  with a dispatch that picks defaults by type and instantiates the right
  class:
  - `"transformer"` → existing defaults + `Transformer`.
  - `"bigru"` → `dict(clip_embed_dim=1024, num_classes=2, hidden_dim=512,
    num_layers=2, gru_dropout=0.1, mlp_dropout=0.4, mlp_hidden_dim=512)` +
    `BiGRU`.
  - Any other value → `ValueError`.
- `temporal_kwargs.update(...)` still merges user overrides on top, same as
  for the Transformer path.

**Acceptance test (offline):**
```python
m1 = CompositePEFT(temporal_type="transformer")
m2 = CompositePEFT(temporal_type="bigru")
x  = torch.randn(2, 32, 3, 336, 336)
assert m1(x).shape == (2, 2)
assert m2(x).shape == (2, 2)
assert isinstance(m2.temporal, BiGRU)
```

#### Step B2 — Wire `temporal_type` through `train.py`

In `peft/train.py`:

- Read `cfg["temporal_type"]` (default `"transformer"`) and pass it as the
  `temporal_type=` kwarg to `CompositePEFT(...)`. Single line.
- No other changes — the trainer doesn't care which head it's training.

#### Step B3 — Smoke config + run

Create `peft/configs/peft_bigru_smoke.yaml`. Identical to `peft_smoke.yaml`
except:
- `tag: peft_bigru_smoke`
- `temporal_type: bigru`
- `temporal:` block uses BiGRU keys (`hidden_dim`, `num_layers`,
  `gru_dropout`, `mlp_dropout`, `mlp_hidden_dim`) — drop the
  Transformer-only keys.

**Acceptance test:**
```bash
python peft/train.py --config peft/configs/peft_bigru_smoke.yaml
```
Expected: one epoch completes, `model.pth` written, `trainable=` line in the
log shows roughly the same LN parameter count as the Transformer path plus
the BiGRU's own trainable params (lower than Transformer's ~30 M head, since
a 2-layer BiGRU at hidden=512 is ~6 M params).

#### Step B4 — Full configs

Create `peft/configs/peft_bigru_ff_mtcnn.yaml` and
`peft/configs/peft_bigru_ff_cdfv2val.yaml`. Both are direct copies of their
Transformer counterparts with:
- `tag` updated.
- `temporal_type: bigru` added.
- `temporal:` block replaced with BiGRU defaults (B4 above), seeded from
  the rank-1 frozen `bigru_search2` config.

Same `optimizer`, `lr_scheduler`, `num_epochs`, `batchSize`,
`grad_accum_steps`, `seed`, etc. — keep the optimisation regime identical to
the Transformer PEFT runs so the head is the *only* varying factor.

**Acceptance test:** training launches without config errors and reaches
epoch 1 val AUC > 0.5.

### 9.4 · Out of scope for this extension

- Optuna search over BiGRU PEFT hyperparameters. (Run only if the seeded
  config is clearly worse than the Transformer PEFT baseline; otherwise the
  shared optimisation regime is sufficient for the comparison.)
- Any change to `peft/evaluation/`. The evaluator already calls
  `model(x)` and reads `pred_dict`-style outputs; head class is irrelevant.
- Any change to `peft/data_loader.py`. The loader returns `[B, T, 3, H, W]`
  regardless of head.

### 9.5 · Acceptance summary

The BiGRU PEFT extension is "done" when:

1. `CompositePEFT(temporal_type="bigru")` round-trips a forward pass.
2. `peft_bigru_smoke.yaml` completes one epoch end-to-end.
3. `peft_bigru_ff_mtcnn.yaml` (or the CDFv2-val variant) trains to
   completion and produces a `model.pth` + `run_config.json`.
4. The Transformer PEFT path still works unchanged — `peft_ff_mtcnn.yaml`
   continues to train without any config edits.

---

## 10 · Extension: Linear temporal head

> **Goal.** Same as §9, but with the LinearClassifier (mean-pool + MLP) as
> the temporal head. Completes the four-model story at the PEFT level
> (Linear / BiGRU / Transformer + a frozen baseline). The Linear PEFT run
> is also the cheapest to train, which makes it a useful sanity check
> against the more elaborate heads.

### 10.1 · Locked decisions

| # | Decision | Why / source |
|---|----------|--------------|
| L1 | Temporal head class = `models.linear_cls.LinearClassifier` (existing, untouched). | Same head as the frozen-pipeline Linear baseline — clean head-only contrast. |
| L2 | Backbone, LN scope, T=32 sampling window, AMP, grad checkpointing — identical to the Transformer/BiGRU PEFT paths. | Isolates the head as the only varying factor. |
| L3 | `temporal_type` gains a third value: `"linear"`. Default still `"transformer"` for backwards compatibility. | Same dispatch pattern as §9. |
| L4 | Linear hyperparameters seeded from the rank-1 frozen `linear_search3_1` config. | Reasonable starting point; comparable to the frozen baseline. |
| L5 | **Pooling stays inside `LinearClassifier`** (`x.abs().mean(dim=1)`). Do not pre-pool in `CompositePEFT.forward` — the head expects `[B, T, D]`. | Matches the frozen-pipeline Linear path; see `feedback_linear_pool_change.md`. |
| L6 | Zero edits to `models/linear_cls.py`, the trainer, the data loader, or the entry script's training loop. Only `clip_peft.py` and configs change. | Smallest surface area. |

### 10.2 · Files touched

```
peft/
├── models/clip_peft.py              # +"linear" branch in dispatch
├── train.py                         # already wired in §9 — no further change
└── configs/
    ├── peft_linear_ff_mtcnn.yaml    # NEW — Linear + FF++ val
    ├── peft_linear_ff_cdfv2val.yaml # NEW — Linear + CDFv2 val (overfit-onset run)
    └── peft_linear_smoke.yaml       # NEW — 1-epoch / 10-video smoke
```

### 10.3 · Implementation steps

#### Step L1 — Extend dispatch in `CompositePEFT`

In `peft/models/clip_peft.py`:

- Add `from models.linear_cls import LinearClassifier`.
- Add a third branch to the `temporal_type` dispatch:
  - `"linear"` → defaults `dict(clip_embed_dim=1024, num_classes=2,
    mlp_dropout=0.2, mlp_hidden_dim=512)` + `LinearClassifier`.
- `temporal_kwargs.update(...)` merges user overrides as for the other
  heads.

**Acceptance test (offline):**
```python
m = CompositePEFT(temporal_type="linear")
x = torch.randn(2, 32, 3, 336, 336)
assert m(x).shape == (2, 2)
assert isinstance(m.temporal, LinearClassifier)
```

#### Step L2 — Smoke config + run

Create `peft/configs/peft_linear_smoke.yaml`. Identical to `peft_smoke.yaml`
except:
- `tag: peft_linear_smoke`
- `temporal_type: linear`
- `temporal:` block uses Linear keys only (`clip_embed_dim`, `num_classes`,
  `mlp_dropout`, `mlp_hidden_dim`). Drop the Transformer-only keys.

**Acceptance test:**
```bash
python peft/train.py --config peft/configs/peft_linear_smoke.yaml
```
Expected: one epoch completes, `model.pth` written. The `trainable=` line
should be roughly *LN params + ~0.6 M head params* — i.e. clearly smaller
than the Transformer PEFT run, since the LinearClassifier is just two
`nn.Linear` layers around a GELU.

#### Step L3 — Full configs

Create `peft/configs/peft_linear_ff_mtcnn.yaml` and
`peft/configs/peft_linear_ff_cdfv2val.yaml`. Direct copies of the
Transformer counterparts with:
- `tag` updated.
- `temporal_type: linear` added.
- `temporal:` block replaced with the Linear defaults (L4 above), seeded
  from the rank-1 frozen `linear_search3_1` config.

Same `optimizer`, `lr_scheduler`, `num_epochs`, `batchSize`,
`grad_accum_steps`, `seed`, etc. as the other PEFT configs — keep the
optimisation regime identical so the head is the only varying factor.

**Acceptance test:** training launches without config errors and reaches
epoch 1 val AUC > 0.5.

### 10.4 · Out of scope for this extension

- Optuna search over Linear PEFT hyperparameters.
- Any change to `peft/evaluation/`, `peft/data_loader.py`, `peft/trainer.py`,
  or `peft/train.py` beyond what §9 already added.
- Any change to the pooling rule (`abs().mean()` stays in `LinearClassifier`).

### 10.5 · Acceptance summary

The Linear PEFT extension is "done" when:

1. `CompositePEFT(temporal_type="linear")` round-trips a forward pass.
2. `peft_linear_smoke.yaml` completes one epoch end-to-end.
3. `peft_linear_ff_mtcnn.yaml` (or the CDFv2-val variant) trains to
   completion and produces a `model.pth` + `run_config.json`.
4. The Transformer and BiGRU PEFT paths still work unchanged — their
   configs continue to train without any edits.

---

*End of plan.*
