# PEFT Pilot Search

This is the first narrow Optuna run after the GenD/Yermakov context review.

## Command

Run from the repository root:

```bash
python peft/optuna_search.py \
  --search-config peft/search_configs/peft_gend_pilot12.yaml
```

The search config is self-contained. It defines the PEFT training config, search
space, study name, trial count, epoch count, sampler, pruner, storage, output
directory, and anchor behavior.
For this pilot those values are:

- `study_name: peft_gend_pilot12`
- `n_trials: 12`
- `epochs: 5`
- `storage: peft/searches/peft_optuna.db`
- `output_dir: peft/searches/runs`
- `sampler.type: tpe`
- `sampler.startup_trials: 4`
- `pruner.type: median`
- `pruner.warmup_epochs: 2`
- `anchors.enabled: true`
- `training.root_dir: null`, so paths resolve relative to the current repo root

Trial outputs are written under `peft/searches/runs/peft_gend_pilot12/`.

## Pilot Structure

The first four trials are fixed anchors:

- L2 on, UA on
- L2 on, UA off
- L2 off, UA on
- L2 off, UA off

All four anchors use a small temporal head:

- `clip.feature_layer: pre_proj`
- `temporal.num_layers: 1`
- `temporal.dim_feedforward: 1024`
- `temporal.attn_dropout: 0.0`
- `temporal.mlp_dropout: 0.25`
- `temporal.mlp_hidden_dim: 256`
- `optimizer.lr: 2.0e-5`
- `optimizer.weight_decay: 1.0e-4`

The remaining trials are TPE-sampled from the `training.search_space` block in
`peft/search_configs/peft_gend_pilot12.yaml`. The median pruner starts after the
anchor trials and waits two epochs before pruning.

## Outputs

- `study.log`: study-level Optuna progress, callbacks, and best-trial updates.
- `optuna_internal.log`: Optuna library warnings/errors from the sampler,
  pruner, storage, or visualization layer.
- `study_manifest.json`: search config values, anchors, and search space.
- `trial_XXXX/trial_config.yaml`: exact config for a trial.
- `trial_XXXX/trial_params.json`: Optuna overrides plus the merged config.
- `trial_XXXX/training.log`: epoch metrics.
- `trial_XXXX/epoch_metrics.csv`: per-epoch `train_loss`, `train_ce`,
  `train_align`, `train_uniform`, `val_loss`, `val_auc`, accuracies, threshold,
  and LR.
- `trial_XXXX/model.pth`: best trainable PEFT weights for that trial.
- `trial_XXXX/run_config.json`: exact saved config and metrics.
- `all_trials.csv`: full Optuna trials dataframe.
- `summary.md`: state counts, wall-clock, and top-10 trials.
- `best_config.json`: copied `run_config.json` for the best trial.
- `plots/*.html`: Optuna optimization history, parameter importances, parallel
  coordinate, and slice plots when Plotly is available.

## Decision Rules After The Pilot

Use the four anchors first. They are the cleanest read on whether L2
normalization and UA are worth keeping.

- If both L2 anchors beat both non-L2 anchors, keep L2 fixed on.
- If both UA anchors lag their matched non-UA anchors, drop UA for the next
  search.
- If UA helps only with L2, keep the combined switch and stop searching UA
  independently.
- If the sampled trials do not beat the best anchor by a meaningful margin,
  narrow around the anchor settings instead of widening the search.
- If the best trial peaks by epoch 3-4 and then drops, keep 5 epochs for search
  but select by best validation AUC, not final epoch AUC.
