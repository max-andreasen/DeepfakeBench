# eval_top_k — cross-dataset evaluation of param-search top-K

Run from **repo root**.

## What it does

1. Reads `all_trials.csv` from a param-search run dir.
2. Picks the top-K COMPLETE trials by val AUC (param-search was on FF++ val).
3. For each trial (sequentially, not in parallel):
   - Loads `trial_XXXX/run_config.json` + `trial_XXXX/model.pth`
   - Builds a test loader from the supplied embedding catalogue (CDFv2 or other)
   - Runs `Tester.evaluate()` (no frame shuffle)
   - Saves per-trial `results.json` + `run_config.json` copy
4. Writes a summary CSV sorted by test AUC and a top-level `run_config.json`.

## Prerequisites

- CDFv2 (or target) CLIP embeddings must be complete.
  Check: `clip/embeddings/benchmarks/cdfv2_layer16/ViT-L-14-336-quickgelu/block_16/catalogue.csv`
  exists and has rows for both `CelebDFv2_real` and `CelebDFv2_fake`.
- Top-K model checkpoints must be present.
  Check: `find training/searches/runs/transformer_search3 -name model.pth | wc -l`
  should be >= top_k (parameter_search.py keeps only top-k on disk).

## Command

```bash
python evaluation/eval_top_k.py \
    --study_dir  training/searches/runs/transformer_search3 \
    --catalogue_file clip/embeddings/benchmarks/cdfv2_layer16/ViT-L-14-336-quickgelu/block_16/catalogue.csv \
    --out_dir    evaluation/results/cdfv2_top_k/transformer_search3 \
    --top_k 10 \
    --num_frames 32 \
    --batch_size 64 \
    --window_aggregation mean
```

For the BiGRU search:
```bash
python evaluation/eval_top_k.py \
    --study_dir  training/searches/runs/bigru_search2_1 \
    --catalogue_file clip/embeddings/benchmarks/cdfv2_layer16/ViT-L-14-336-quickgelu/block_16/catalogue.csv \
    --out_dir    evaluation/results/cdfv2_top_k/bigru_search2_1 \
    --top_k 10 \
    --num_frames 32 \
    --batch_size 64
```

## Output layout

```
evaluation/results/cdfv2_top_k/transformer_search3/
  run_config.json          ← eval metadata (inputs, settings, UTC timestamp)
  results.csv              ← one row per trial, sorted test AUC desc
  eval_top_k.log           ← full log
  trial_0433/
    results.json           ← per-window + per-video metrics
    run_config.json        ← copy of trial's training config
  trial_0750/
    ...
```

## results.csv columns

| column | description |
|---|---|
| `final_rank` | rank by test AUC (1 = best) |
| `trial` | trial number from Optuna study |
| `val_auc` | AUC on FF++ val (param search objective) |
| `test_auc` | AUC on CDFv2 test (cross-dataset generalisation) |
| `test_acc` | accuracy at threshold 0.5 |
| `test_acc_at_best` | accuracy at Youden's J threshold |
| `test_f1` / `test_precision` / `test_recall` | F1 and its components |
| `best_thresh` | Youden's J threshold |
| `num_videos` | total videos evaluated |

Skipped trials (missing model.pth) appear with empty metric fields.

## Notes

- No split CSV is needed. The script derives `split=test` for every video in
  the catalogue, so it works for any benchmark dataset without a split file.
- `num_frames=32` must match what the model was trained on (the search used 32).
- `input_transform` is read from each trial's `run_config.json` automatically,
  so diff-trained models get the correct transform.
- GPU memory is freed between trials (`gc.collect()` + `cuda.empty_cache()`).
- If a trial is missing `model.pth` (pruned from top-K by a later search run),
  it's skipped with a warning and appears at the bottom of results.csv.
