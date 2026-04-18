# Pilot Tests — Pre-Optuna Plan

Three pilots to run before starting the full Optuna hyperparameter search. These
answer architectural/foundational questions whose answers change the Optuna
search space itself. Running Optuna before settling these wastes trials on a
suboptimal foundation.

**Order matters.** Pilot 1 shifts the entire embedding foundation; Pilot 2
depends on 1's outcome; Pilot 3 is independent and can run in parallel with 2.

| # | Pilot | Time | Depends on |
|---|-------|------|------------|
| 1 | CLIP layer probing (multi-layer + 1024-dim pre-projection) | ~2 days | — |
| 2 | Transformer input projection | ~0.5 day | 1 |
| 3 | Feature-level augmentation | ~1 day | — (run in parallel with 2) |

Total: ~3 days elapsed if 2 and 3 run in parallel.

## Completed pilots (for reference)

- **MLP head depth** (2026-04-17): 1–2 layers with GELU is optimal. Depth=8
  collapses temporal models. Transformer peak 0.845, BiGRU 0.886, Linear 0.895.
  → Optuna search space should use `head_depth ∈ {1, 2, 3}` and GELU only.

- **Input projection (2026-04-18)**: A (identity, 768) → per-video AUC ~0.845.
  C (Linear + LN → 512) → 0.891. Did not test D/E under matched LR; committed
  to C as default based on this pilot's margin. Caveat: LR not swept, so part
  of the gap may be LR-related.

## Known priors (informing the pilots below)

- **Temporal shuffle-invariance with frozen CLIP CLS features**:
  across *all* pilots to date, `shuffle_frames=True` at eval time produces
  identical per-video AUC to `shuffle_frames=False` (e.g. standard = 0.8906,
  shuffled = 0.8908 for transformer 512+LN on FF++ test). This means the
  temporal models (Transformer, BiGRU) are not actually using frame order —
  they're behaving as bag-of-frames pooled classifiers. Consistent with the
  observation that Linear ties Transformer/BiGRU at the top of the results
  table.

  **Implication**: the frozen final-projected CLIP CLS layer is temporally
  uninformative for this task. This is the central motivating finding for
  Pilot 1 (earlier/pre-projection layers may retain temporally-informative
  texture features) and Appendix B.2 (PEFT may let CLIP learn to extract
  temporal signal).

  **Therefore**: the `standard` vs `shuffled` AUC gap is the primary
  diagnostic for whether temporal modeling is actually working. It should be
  tracked as a top-level metric alongside AUC in Pilot 1 and Appendix B.2.
  A non-zero gap = temporal signal is being used.

---

## Pilot 1 — CLIP layer probing (unified with 1024-dim pre-projection)

### Hypothesis

The current embedding uses the final 768-dim post-projection CLS output, which
is optimized for **text-image alignment** (semantics), not forensic signal.
Deepfake artifacts live in low-level texture/frequency features that the
text-alignment projection likely discards.

Extracting features from earlier transformer blocks (or the 1024-dim
pre-projection hidden state) may retain more artifact-relevant information. The
probing sweep tells us which layer is best; the 1024-dim pre-projection test
falls out as a special case.

### Goal

Produce a single AUC-vs-layer-depth curve. Pick the layer (or layers) that
maximize per-video AUC on FF++ val **and** CDF-v2. Everything downstream
(Optuna search, final benchmarking) is rebuilt on top of the winning layer.

### Implementation

#### Step 1: Modify `CLIP_embedder.py` to hook intermediate blocks

Current code (`clip/CLIP_embedder.py:127`):
```python
batch_embedding = self.model.encode_image(batch_tensor)  # [B, 768]
```

This calls `encode_image` which runs the full ViT and the final projection.
Replace with a hook-based version that captures block outputs.

**Approach:** register forward hooks on `self.model.visual.transformer.resblocks[i]`
for each target layer `i`, run a forward pass, collect hook outputs, and save
each layer separately. Also keep the final 768-dim projected output so we have
a direct apples-to-apples comparison with current embeddings.

**Target layers** (ViT-L/14 has 24 blocks):
- `{8, 12, 16, 20, 22, 24}` hidden states (all 1024-dim)
- `projected_768` (final CLS after projection — what we have today)

This gives 7 embedding variants per preprocessing variant, spaced densely
in the region where deepfake signal is most likely (middle to late blocks).

**Storage layout**:
```
clip/embeddings/mtcnn/ViT-L-14-336-quickgelu_dim768/      # current (keep as-is)
clip/embeddings/mtcnn/ViT-L-14-336-quickgelu_layer08/     # new
clip/embeddings/mtcnn/ViT-L-14-336-quickgelu_layer12/
...
clip/embeddings/mtcnn/ViT-L-14-336-quickgelu_layer24/     # pre-projection 1024
```

Each subdir has its own `catalogue.csv`. Downstream code (training data loader)
doesn't change — `config.catalogue_file` just points to the layer subdir you
want to train on.

**Float16 storage.** Halves disk cost with no measurable precision loss for
downstream probe/training. Change the `np.savez_compressed` call to use
`embedding.numpy().astype(np.float16)` per-layer.

#### Step 2: CLS token extraction

For hidden-state layers, extract `hidden_state[:, 0, :]` (CLS token position)
just as the final projection does. Mean-pooling over patch tokens is also an
option but CLS is more directly comparable to current pipeline.

#### Step 3: Re-embed once

One re-embedding pass writes all 7 variants for one preprocessing variant.
Cost: same GPU time as the original embedding run (one forward pass), more disk
I/O and disk space.

**For the pilot, only re-embed one preprocessing variant** — the cheapest and
best-performing one from the preprocessing A/B test (likely MTCNN). Once the
best layer is known, re-embed the other preprocessing variants at that layer
only.

#### Step 4: Probe training

For each layer:
1. Create a new YAML config pointing to the layer's `catalogue_file` and
   setting `clip_embed_dim` correctly (768 for projected, 1024 for hidden).
2. Train a **linear probe** (simplest model, cheapest). Use the `linear` model
   config. ~5–10 min per layer on GPU.
3. Also train a transformer and BiGRU on the top 2 layers from the linear
   probe, to confirm the ranking holds with bigger models. ~1 hour each.

**Scoring:**
- Per-video AUC on FF++ val (tuning signal).
- Per-video AUC on CDF-v2 test (generalization signal — this is the one that
  matters for the study).
- Report both. A layer that wins FF++ but loses CDF-v2 is not the winner.

#### Deliverable

`runs/layer_probing/results.csv` with columns:
```
layer, clip_embed_dim, model,
  ff_val_auc_std,  ff_val_auc_shuf,  ff_val_temporal_gap,
  cdf2_test_auc_std, cdf2_test_auc_shuf, cdf2_test_temporal_gap,
  wallclock_s
```

`temporal_gap = auc_std - auc_shuf`. For temporal models (Transformer, BiGRU)
this is the key metric — a meaningfully positive gap (e.g. > 0.005) means
the model is actually using frame order. For Linear it should always be zero
(sanity check).

Two plots required:
1. **AUC vs layer** for each dataset → pick best-performing layer overall.
2. **Temporal gap vs layer** for Transformer/BiGRU → identify which layer
   (if any) produces temporally-informative features.

The two plots may disagree (e.g. layer 20 has highest AUC but zero temporal
gap; layer 16 has lower AUC but +0.02 gap). If so, that's the core finding
for the paper — document both and discuss which regime to pursue.

Expected literature prior: block 20–22 for raw AUC; earlier blocks (12–16)
may be where temporal gap emerges if at all. Let the data decide.

### Risks / flags

- **Disk space**: 7 variants × ~30 GB per preprocessing variant = ~200 GB.
  Check free space on the embedding drive before starting.
- **Hooks on a frozen model**: make sure `model.eval()` is set and hooks don't
  accidentally enable gradient tracking. Use `torch.no_grad()` around the
  forward pass (already done in `_embed_batch`).
- **CLS token position**: verify `open_clip`'s ViT actually prepends a CLS
  token at position 0. For `ViT-L-14-336-quickgelu` it does, but confirm with
  a small sanity check before re-embedding everything.

---

## Pilot 2 — Transformer input projection

### Hypothesis

The current transformer (`models/transformer.py:108-113`) has no explicit input
projection — CLIP embeddings go directly into `cls_token + positional_encoding
+ TransformerEncoder`. Adding a projection layer could help the transformer
adapt CLIP's representation to a more attention-friendly shape, stabilize
training, or (via dimensionality reduction) reduce overfitting.

Depends on Pilot 1 — `d_model` is determined by the winning layer (768 for
projected, 1024 for hidden-state layers).

### Goal

Pick one of 4 input-projection configurations for the transformer. Run only on
the transformer — BiGRU has its own input handling (GRU cell does internal
projection), and Linear doesn't have a "projection" concept.

### Configurations

The pilot tests two orthogonal axes simultaneously: **projection type** (linear
vs MLP) and **d_model reduction** (none / moderate / aggressive). Option B
(`Linear(d_in, d_in)` with no LN) was dropped — its output is almost entirely
absorbed by the first encoder block's pre-LN + Q/K/V projections, so it's
near-identical to A in practice while spending d² parameters.

| # | Projection | d_model | What it isolates |
|---|-----------|---------|------------------|
| A | None (identity) | d_in | Baseline — does any projection help? |
| C | `Linear(d_in, 512) + LayerNorm` | 512 | Moderate dim reduction + LN at input. |
| D | `Linear(d_in, 512) → GELU → Dropout → Linear(512, 512) + LayerNorm` | 512 | Nonlinearity vs C at matched target dim. |
| E | `Linear(d_in, 256) + LayerNorm` | 256 | Aggressive reduction (KDD paper's choice). |

`d_in` is the CLIP embedding dim from Pilot 1's winning layer (768 for the
projected output, 1024 for hidden-state layers).

**Interpretation axes:**
- **A vs C**: does any projection help at all?
- **C vs D**: does nonlinearity help (at matched d_model=512)?
- **A / C / E**: dim-reduction ladder (d_in → 512 → 256). If monotonic, tells
  you which direction to push d_model in the Optuna search.

**Hparam compatibility notes:**
- `n_heads` must divide `d_model`. For 768 → 8 heads (96-dim) works; for 512 →
  {8, 16} work; for 256 → 8 works but 16 does not.
- `dim_feedforward` should scale ~4× d_model. Default 3072 is fine for 768,
  should drop to ~2048 for 512 and ~1024 for 256. Update the config
  accordingly when switching options.

### Implementation

Implemented in `models/transformer.py` as a commented-block swap — each option
is a ~5-line block at the top of `__init__`, with one active and the rest
commented. Switch by uncommenting the desired block and commenting the active
one. `proj_out_dim` drives `self.d_model`, which automatically resizes
`cls_token`, `positional_encoding`, the encoder, and the classifier.

`forward` applies `x = self.input_proj(x)` before prepending the CLS token;
with option A (identity), behavior is unchanged from the original model.

When switching options, also remember to adjust `dim_feedforward` in the
active YAML config (3072 for d_model=768; ~2048 for 512; ~1024 for 256).

### Run plan

4 configs × 1 model (transformer) × 1 run each = 4 runs. ~30–45 min per run
with 30 epochs. **Total: ~3–4 hours.**

Use the winning layer from Pilot 1. All other hparams held constant (use a
reasonable default: lr=1e-4, wd=1e-4, cosine warmup, mlp_dropout=0.4, 8 heads,
6 layers, 30 epochs).

### Deliverable

`runs/input_projection/results.csv` + short comparison note. Pick the winner
and make it the default for Optuna (still searchable if time permits, but
default is fine).

### Risks / flags

- **Single seed.** As with Pilot 1, differences < 0.01 AUC are noise. If the
  top 2 configs are within 0.005, pick the simpler one (B over C, C over D).
- **Pilot 1 winner interaction.** If the winning layer is 1024-dim
  pre-projection, a `LayerNorm` input projection is more valuable (those
  hidden states are not normalized to the text-aligned space). If the winner
  is the projected 768-dim, identity is more likely to win.

---

## Pilot 3 — Feature-level augmentation

### Hypothesis

For a cross-dataset generalization study, regularization/robustness
augmentations are arguably the *most* important knob. Feature-level
augmentations can be applied on-the-fly to pre-computed CLIP embeddings,
avoiding the cost of re-embedding with pixel-level augmentations.

**Critical framing**: augmentation that improves FF++ val AUC is not
automatically good. What matters is whether it closes the FF++ → CDF-v2 gap.
An aug that drops FF++ val AUC by 0.01 but raises CDF-v2 test AUC by 0.03 is a
win.

### Configurations

| # | Aug | Description |
|---|-----|-------------|
| 0 | None | Baseline |
| 1 | Feature dropout | `nn.Dropout(p=0.1)` on the entire embedding vector (per-dim zero-out with rescaling). Applied to input of transformer/BiGRU/linear, after input proj if present. |
| 2 | Gaussian noise | Add `N(0, σ²)` noise to embeddings at train time. σ ∈ {0.01, 0.05, 0.1}. |
| 3 | Temporal frame dropping | Randomly mask `p` fraction of the T frames (replace with zeros or repeat prev). `p ∈ {0.1, 0.25}`. |
| 4 | Mixup in feature space | Interpolate two random videos' embeddings and their labels. `α ∈ {0.1, 0.4}`. |

**Select 2–3 configs to test**, not all 12 parameter combinations. Start with
representative defaults:
- Feature dropout `p=0.1`
- Gaussian noise `σ=0.05`
- Temporal frame dropping `p=0.25`
- Mixup `α=0.4`

4 aug variants × 3 models = 12 runs. ~6–8 hours total.

### Implementation

Augmentations live in the training data loader (`training/data_loader.py`) OR
as a `transform` applied inside the training loop. Simplest: add a small
module or helper functions in a new file `training/augmentations.py`, and
wire it into the training step.

Example scaffold:
```python
# training/augmentations.py
import torch

def feature_dropout(x, p=0.1, training=True):
    if not training: return x
    return torch.nn.functional.dropout(x, p=p, training=True)

def gaussian_noise(x, sigma=0.05, training=True):
    if not training: return x
    return x + sigma * torch.randn_like(x)

def temporal_frame_drop(x, p=0.25, training=True):
    # x: [B, T, D]
    if not training: return x
    B, T, D = x.shape
    mask = (torch.rand(B, T, device=x.device) > p).float().unsqueeze(-1)
    return x * mask

def feature_mixup(x, y, alpha=0.4, training=True):
    # x: [B, T, D]; y: [B] (labels)
    if not training: return x, y, None, None, None
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    return x_mixed, y, y[idx], lam, idx
```

Mixup requires a loss modification (`lam * loss(y_a) + (1-lam) * loss(y_b)`),
which is a small change in `trainer.py`.

Wire via a YAML field:
```yaml
augmentation:
  type: "feature_dropout"   # or "gaussian_noise", "frame_drop", "mixup", "none"
  params:
    p: 0.1
```

### Run plan

**Stage 1 — single model (BiGRU, best so far) × 4 augs + baseline = 5 runs**.
~3–4 hours. Pick the winning aug per model based on **CDF-v2 test AUC**, not
FF++ val.

**Stage 2 — top aug × 3 models = 3 runs** to check the winning aug generalizes
across architectures. ~2–3 hours.

### Deliverable

`runs/feature_aug/results.csv` with columns:
```
aug_type, params, model, ff_val_auc, cdf2_test_auc, ff_minus_cdf_gap
```

The `ff_minus_cdf_gap` column is the key metric. Smaller gap = better
generalization. Winning aug is the one that minimizes the gap while keeping
FF++ val AUC within 0.01 of the no-aug baseline.

### Risks / flags

- **Scoring on CDF-v2 test during pilot is a form of test-set leakage.** This
  is mitigated by the fact that the final benchmark uses the same test split,
  and by running only 4 configs (low risk of overfitting to it). But when
  reporting, acknowledge that aug choice was informed by CDF-v2. If more
  rigor is needed, split CDF-v2 into dev/test halves.
- **Mixup with per-video AUC**: mixup violates "one label per video" during
  training. The val/test eval still uses clean videos, so this isn't a
  measurement problem. But the per-window training loss interpretation gets
  murky. Document it.

---

## Appendix A — Pilot 4 (optional): Number of frames (T)

### Hypothesis

Larger T = more temporal context per window = potentially better temporal
modeling. The current `T=32` with 3 windows/video (total 96 frames embedded)
was chosen ad-hoc. Testing `T ∈ {16, 32, 48, 64, 96}` with the same total
frame budget could reveal the optimal context length.

### Why it's optional

- **Low leverage.** Expected gain: 0.005–0.015 AUC. Order of magnitude less
  than Pilot 1.
- **Preprocessing change needed.** Current embeddings already are 96
  continuous frames per video (see `clip/create_clip_embeddings.py:101`,
  `T=96`). So actually **no re-embedding needed** — the constraint is just in
  the data loader, which reshapes into `n_windows × num_frames`. Set
  `num_frames=48` in the config and you get 2 windows of 48 instead of 3
  windows of 32. Easy.
- Can be added as a **cheap Optuna search parameter** (`num_frames ∈ {16, 24,
  32, 48, 96}`) without any new pilot, since it's just a data-loader reshape.

### Recommendation

Skip the dedicated pilot. Add `num_frames` to the Optuna categorical search
space instead. If the search lingers over a single value, that's your answer.

### Caveat

Re-verify that the current embedding files really contain 96 continuous frames
(not 3 separately-sampled 32-frame windows). Check `CLIP_EMBEDDER._load_frames`
(`clip/CLIP_embedder.py:134`) — it samples **one contiguous T-length window**
per video, so with `T=96` it is 96 continuous frames. Good.

---

## Appendix B — Post-Optuna experiments

These run **after** the main hyperparameter search wraps, on the winning
configuration. Each is large enough to be its own mini-study.

### B.1 Pixel-level augmentation

JPEG compression, color jitter, resolution down/up, motion blur, Gaussian
blur. Targeted at closing the domain gap between FF++ (high quality) and
CDF-v2 (YouTube-sourced, degraded).

- **Cost**: re-embed the entire dataset per augmentation config.
- **Scope**: pick 3–4 targeted augs, run each as a new preprocessing variant.
- **Scoring**: per-video AUC on FF++ val AND CDF-v2 test, gap metric.

### B.2 PEFT (LoRA / adapters on CLIP)

Adapt CLIP itself to the deepfake task with LoRA on later layers. Compare
against frozen-CLIP winner on both FF++ and CDF-v2.

- **Primary hypothesis**: if Pilot 1 shows *no* layer has a meaningful
  temporal gap (all layers shuffle-invariant), then frozen CLIP
  fundamentally can't provide temporal signal and PEFT is the only way to
  obtain one. The `standard` vs `shuffled` gap is the key metric here too —
  success is measured not only by AUC gain but by whether PEFT induces a
  temporal gap that frozen CLIP lacked.
- **Risk**: LoRA may improve FF++ while hurting CDF-v2 (overfits to FF++
  artifacts). This itself is an interesting finding for the paper — a
  "generalization cost of in-domain adaptation" figure.
- **Cost**: one full training run per LoRA config. Can't cache embeddings
  — slower by ~10–100×.
- **Scope**: one clean comparison, not a sweep. Target the winning CLIP layer
  from Pilot 1; apply LoRA to layers ≤ winner-layer with rank 8, 16.
- **Scoring**: report `auc_std`, `auc_shuf`, and `temporal_gap` on both FF++
  and CDF-v2 test. A successful PEFT outcome produces a larger
  `temporal_gap` than any frozen-CLIP layer from Pilot 1.

### B.3 Label inversion → canonical convention

Current catalogues have `FF-real: label=1, FF-fake*: label=0`. YAML convention
is `real=0, fake=1`. AUC is invariant to the flip, so this doesn't affect the
search. But before paper numbers: align to a single convention. Script
available at `clip/invert_catalogue_labels.py`.

### B.4 Retrain winner for longer

Optuna uses `search_epochs=30`. After finding the winner, retrain with
`num_epochs=100` to get the final benchmark numbers.

---

## Checklist

Pilot 1 — CLIP layer probing
- [ ] Add forward hooks in `CLIP_embedder.py`
- [ ] Add float16 save path
- [ ] Verify ViT CLS token position
- [ ] Check disk space (~200 GB)
- [ ] Re-embed MTCNN variant with all target layers
- [ ] Write per-layer catalogue CSVs
- [ ] Train linear probe per layer
- [ ] Train transformer + BiGRU on top 2 layers
- [ ] Produce AUC-vs-layer plot (FF++ val + CDF-v2 test)
- [ ] Pick winner, document

Pilot 2 — Transformer input projection
- [ ] Add `input_projection` param to `Transformer.__init__`
- [ ] Add wiring in `forward`
- [ ] Run 4 configs on Pilot-1 winner layer
- [ ] Document winner

Pilot 3 — Feature-level augmentation
- [ ] Create `training/augmentations.py`
- [ ] Wire into `trainer.py` (incl. mixup loss)
- [ ] Add `augmentation:` config block to YAMLs
- [ ] Stage 1: BiGRU × 5 configs
- [ ] Stage 2: top aug × 3 models
- [ ] Report FF++ / CDF-v2 / gap table
- [ ] Document winner

Then → Optuna.
