# PEFT Experiment Context

This is the canonical context file for the PEFT workstream. Keep the root
`PROJECT_CONTEXT.md` brief; put PEFT architecture details, paper context,
search-space decisions, run commands, and interpretation notes here.

For the implementation build notes, see [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Purpose

The frozen-backbone models in this project train temporal heads on cached CLIP
features. The PEFT experiment asks whether updating a very small part of CLIP
itself, specifically the visual LayerNorm affine parameters, improves
cross-dataset deepfake generalization on Celeb-DF-v2.

The working hypothesis is that LN-tuned CLIP should beat the frozen-backbone
ceiling around 72% CDFv2 AUC and move toward the much stronger results reported
by Yermakov et al. for GenD.

## Current PEFT Architecture

```text
FF++ videos
  -> MTCNN face detection -> frame PNGs
  -> CLIP ViT-L/14-336-quickgelu, OpenAI weights
     - CLIP backbone mostly frozen
     - visual LayerNorm affine params trainable
     - optional feature layer: pre_proj or block_16
  -> optional per-frame CLS L2 normalization
  -> temporal Transformer head over 32 frame features
  -> binary video logits
```

Implemented switches:

- `clip.feature_layer`: `pre_proj` or `block_16`.
- `clip.l2_normalize_features`: whether the temporal head receives normalized
  frame features.
- `loss.ua_enabled`: whether to add GenD-style uniformity/alignment losses on
  normalized per-frame CLS features.
- `loss.alignment_weight`: default `0.1`.
- `loss.uniformity_weight`: default `0.5`.

The trainer always builds a normalized feature copy for UA when UA is enabled,
so UA remains meaningful even if the temporal head receives unnormalized
features.

## Key Difference From Frozen Models

The frozen pipeline under `training/` reads cached `.npz` CLIP features. PEFT
cannot reuse those caches because CLIP LayerNorm weights change during
training. PEFT therefore runs CLIP inside the training loop and streams frames
from disk through `peft/data_loader.py`.

Important files:

- [`models/clip_peft.py`](models/clip_peft.py): CLIP wrapper, LN unfreezing,
  feature extraction, optional L2, and temporal head wiring.
- [`trainer.py`](trainer.py): PEFT training loop, AMP, UA loss, val loss/AUC
  logging, checkpointing.
- [`train.py`](train.py): single-run training entry point.
- [`optuna_search.py`](optuna_search.py): PEFT-specific Optuna search launcher.
- [`search_configs/peft_gend_pilot12.yaml`](search_configs/peft_gend_pilot12.yaml):
  self-contained Optuna, PEFT training, and search-space config for the first
  12-trial pilot.
- [`PILOT_SEARCH.md`](PILOT_SEARCH.md): exact pilot command, outputs, and
  decision rules.

## Yermakov et al. / GenD Context

Paper: Yermakov, Cech, Matas, and Fritz, "Deepfake Detection that Generalizes
Across Benchmarks" (WACV 2026).

Useful links:

- Accepted paper PDF: https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf
- arXiv page: https://arxiv.org/abs/2508.06248
- Official GitHub repo: https://github.com/yermandy/GenD

GenD architecture:

- Frame-level detector, not a temporal model.
- Encoder: CLIP ViT-L/14, PEcoreL, or DINOv3 ViT-L/16.
- Input: 32 frames per video, face detection/alignment, 1.3x face crop margin.
- Feature: CLS token only; patch tokens discarded.
- Feature normalization: L2-normalized CLS feature.
- Head: binary linear classifier per frame.
- Trainable params: all LayerNorm affine parameters plus classifier weights.
- Loss: cross-entropy plus uniformity/alignment on normalized features.
- Video inference: average per-frame softmax probabilities.

Reported CLIP training details from the paper:

- Optimizer: Adam, betas 0.9/0.999, no weight decay.
- Precision: bfloat16.
- LR schedule: cosine cyclic; each 10-epoch cycle warms up for one epoch from
  `1e-5` to `3e-4`, then decays over nine epochs to `1e-5`.
- Batch size: 128 frame samples.
- Most runs stopped improving after about two cycles, around 20 epochs.
- Reported UA weights for CLIP: alignment alpha `0.1`, uniformity beta `0.5`.

Important GenD findings:

- The large gain comes mainly from LN tuning on top of the frozen foundation
  encoder.
- Uniformity/alignment adds a smaller but useful improvement.
- Their simple temporal self-attention ablation did not noticeably improve over
  averaging frame probabilities.
- Reported FF++-trained GenD cross-dataset CDFv2 AUC is in the low/mid 90s,
  far above this repo's frozen-backbone results around 72% CDFv2 AUC.

Implication for this repo: the temporal Transformer is the experimental part.
The GenD-style baseline is LN tuning + L2-normalized CLS + linear frame
classifier + average per-frame probabilities.

## Current Status

The earlier 20-epoch PEFT run with the temporal Transformer reached
`best_val_auc=0.8374` on CDFv2-clean val. It was already near `0.81` by epoch 4,
but the best value occurred later. This supports using 5 epochs for coarse
Optuna pruning, but not for final model selection.

Current state:

- PEFT model supports `pre_proj` and `block_16` features.
- Optional model-side L2 normalization is implemented.
- Optional GenD-style UA loss is implemented.
- PEFT Optuna pilot launcher is implemented.
- Pilot search has not been run yet.

## Pilot Search

The first PEFT search is intentionally narrow. It is designed to determine
whether L2 and UA are worth keeping before spending many GPU hours on a larger
search.

Command:

```bash
python peft/optuna_search.py \
  --search-config peft/search_configs/peft_gend_pilot12.yaml
```

The search config is self-contained. It owns the Optuna orchestration settings,
the PEFT model/training defaults, and the `training.search_space` block. Its
`training.root_dir` is `null`, so paths resolve relative to the repo root on the
machine running the search.

The first four trials are fixed anchors:

- L2 on, UA on
- L2 on, UA off
- L2 off, UA on
- L2 off, UA off

Anchor settings:

- `clip.feature_layer: pre_proj`
- `temporal.num_layers: 1`
- `temporal.dim_feedforward: 1024`
- `temporal.attn_dropout: 0.0`
- `temporal.mlp_dropout: 0.25`
- `temporal.mlp_hidden_dim: 256`
- `optimizer.lr: 2.0e-5`
- `optimizer.weight_decay: 1.0e-4`

The remaining trials are TPE-sampled from the `training.search_space` block in
[`search_configs/peft_gend_pilot12.yaml`](search_configs/peft_gend_pilot12.yaml).
The median pruner waits two epochs and starts after the four anchor trials, as
configured in the same file.

Pilot outputs:

- `study.log`: compact study-level progress.
- `optuna_internal.log`: Optuna sampler/pruner/storage warnings and errors.
- `study_manifest.json`: search config values, anchors, and search space.
- `all_trials.csv`: full Optuna trials dataframe.
- `summary.md`: state counts, wall-clock, and top-10 trials.
- `plots/*.html`: Optuna plots when Plotly is available.
- `trial_XXXX/trial_config.yaml`: exact merged config.
- `trial_XXXX/trial_params.json`: Optuna overrides plus config.
- `trial_XXXX/epoch_metrics.csv`: train loss, CE, UA components, val loss,
  val AUC, accuracies, threshold, and LR per epoch.
- `trial_XXXX/training.log`: full training log.
- `trial_XXXX/run_config.json`: exact saved config and metrics.
- `trial_XXXX/model.pth`: best trainable PEFT weights for that trial.

Decision rules after the pilot:

- If both L2 anchors beat both non-L2 anchors, keep L2 fixed on.
- If both UA anchors lag their matched non-UA anchors, drop UA.
- If UA helps only with L2, keep the combined switch and stop searching UA
  independently.
- If sampled trials do not beat the best anchor meaningfully, narrow around the
  anchor settings instead of widening the search.
- Use best validation AUC, not final epoch AUC, because overfitting after early
  convergence is plausible.

## Current Search Space

Defined in [`search_configs/peft_gend_pilot12.yaml`](search_configs/peft_gend_pilot12.yaml)
under `training.search_space`:

```yaml
clip.feature_layer: [pre_proj, block_16]
clip.l2_normalize_features: [true, false]
loss.ua_enabled: [true, false]
optimizer.lr: log-uniform [1.0e-5, 4.0e-5]
optimizer.weight_decay: [0.0, 1.0e-5, 1.0e-4, 5.0e-4]
temporal.num_layers: [1, 2, 4]
temporal.dim_feedforward: [1024, 2048]
temporal.attn_dropout: [0.0, 0.1]
temporal.mlp_dropout: uniform [0.1, 0.4]
temporal.mlp_hidden_dim: [256, 512]
```

Fixed for the first pilot:

- `ln_scope: all`
- `temporal.n_heads: 8`
- `num_frames.train/val: 32`
- `batchSize.train/val: 8`
- `grad_accum_steps: 2`
- `lr_scheduler: cosine_warmup`
- `warmup_epochs: 1`

## Actual Search And Selection Plan

The 12-trial run is a pilot, not the final selection step. Use it to prune the
space and decide whether L2 and UA should remain searchable.

After the pilot:

1. Keep/drop `clip.l2_normalize_features` and `loss.ua_enabled` based on the
   four anchors.
2. Narrow any weak dimensions from the TPE trials, especially temporal depth
   and dropout.
3. Run a focused search or extension pass on the reduced space. The current
   expected extension is the top few pilot configurations trained for 10-20
   epochs, selected by best CDFv2-clean val AUC.
4. Treat the selected PEFT configuration as provisional until it has been
   rerun over multiple seeds.
5. Compare final PEFT seed results against the strongest frozen-backbone model
   using the separate PEFT-vs-frozen Welch test described in
   [`../PROJECT_CONTEXT.md`](../PROJECT_CONTEXT.md).

Do not select a PEFT config from final-epoch AUC alone. The earlier 20-epoch
run showed early gains followed by later fluctuations, so best validation AUC
and the full per-epoch curve should be inspected.

## Recommended Next Steps

1. Run the 12-trial pilot above.
2. Inspect the four L2 x UA anchors before reading TPE importances.
3. Extend the top few configurations to 10-20 epochs.
4. Add a true GenD-style baseline if time allows:
   L2-normalized per-frame CLS -> linear frame classifier -> average frame
   softmax probabilities.
5. Retrain the selected PEFT configuration with multiple seeds before the final
   PEFT vs frozen-baseline statistical comparison.

## Gotchas

- L2 normalization is model-side and configurable. Do not apply it in the
  dataset or cached feature files.
- Optimizers must use `model.trainable_parameters()`, otherwise AdamW state is
  allocated for the frozen CLIP weights.
- `pre_proj` is exposed by disabling `visual.proj`; do not replace it with an
  ad hoc hook unless needed.
- YAML floats should be written as `2.0e-5`, not `2e-5`, because older PyYAML
  parsing can treat the latter as a string.
- CDFv2 raw val and test overlap in the original split. Use
  `datasets/splits/Celeb-DF-v2-clean.csv` for validation/test separation.
