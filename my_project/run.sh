#!/bin/bash
# =============================================================================
# Run script for the CLIP Transformer deepfake detector.
#
# Usage:
#   bash run.sh embed       # extract CLIP embeddings from preprocessed frames
#   bash run.sh train       # train the Transformer
#   bash run.sh eval        # evaluate on test split (+ frame shuffle test)
#   bash run.sh all         # run embed → train → eval in sequence
#
# Edit the CONFIG block below before each run.
# =============================================================================

set -e  # exit immediately on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# CONFIG — edit these before each run
# =============================================================================

# --- Embedding ---
JSON_PATH="$SCRIPT_DIR/../datasets/rgb/FaceForensics++.json"   # produced by rearrange.py
DATASET_NAME="Celeb-DF-v2"
MODEL_NAME="ViT-L-14-336-quickgelu"
PRETRAINED="openai"
T=32
MICRO_BS=32
SEED=0
MAX_VIDEOS=""           # leave empty to embed all videos

# --- Training ---
CATALOGUE="$SCRIPT_DIR/clip/embeddings/YOUR_RUN_DIR/catalogue.csv"
CLIP_EMBED_DIM=768
NUM_FRAMES=32
LR=1e-4
WEIGHT_DECAY=0.0
BATCH_SIZE=32
NUM_EPOCHS=50
ATTN_DROPOUT=0.1
MLP_DROPOUT=0.4
LR_SCHEDULER="cosine_warmup"   # constant | cosine_warmup | step
WARMUP_EPOCHS=5
WEIGHTED_LOSS=false

# --- Evaluation ---
MODEL_DIR="$SCRIPT_DIR/my_models/trained/YOUR_MODEL_DIR"

# =============================================================================

stage="${1:-all}"

run_embed() {
    echo "=== Embedding ==="
    python3 - <<EOF
import sys
sys.path.insert(0, "$SCRIPT_DIR/clip")
from create_clip_embeddings import build_df_from_repo_json, CLIP_EMBEDDER
from pathlib import Path

df = build_df_from_repo_json(Path("$JSON_PATH"), "$DATASET_NAME")

embedder = CLIP_EMBEDDER(
    model_name="$MODEL_NAME",
    pretrained="$PRETRAINED",
    device="cuda",
    T=$T,
    micro_bs=$MICRO_BS,
    seed=$SEED,
    align_faces=False,
)

max_videos = ${MAX_VIDEOS:-None} if "${MAX_VIDEOS}" == "" else int("${MAX_VIDEOS}")

failures = embedder.run(input_frame=df, out_dir=None, max_videos=None)
print(f"Done. Failures: {len(failures)}")
EOF
}

run_train() {
    echo "=== Training ==="
    cd "$SCRIPT_DIR/my_models"
    python3 - <<EOF
import sys
sys.path.insert(0, ".")
from train_transformer import Trainer

trainer = Trainer(
    data_split_file="$CATALOGUE",
    clip_embed_dim=$CLIP_EMBED_DIM,
    num_frames=$NUM_FRAMES,
    lr=$LR,
    weight_decay=$WEIGHT_DECAY,
    batch_size=$BATCH_SIZE,
    num_epochs=$NUM_EPOCHS,
    attn_dropout=$ATTN_DROPOUT,
    mlp_dropout=$MLP_DROPOUT,
    lr_scheduler="$LR_SCHEDULER",
    warmup_epochs=$WARMUP_EPOCHS,
    weighted_loss=$( [ "$WEIGHTED_LOSS" = true ] && echo "True" || echo "False" ),
)
trainer.train()
EOF
}

run_eval() {
    echo "=== Evaluation ==="
    cd "$SCRIPT_DIR/my_models"
    python3 - <<EOF
import sys
sys.path.insert(0, ".")
from run_transformer import Evaluator

evaluator = Evaluator(
    model_dir="$MODEL_DIR",
    data_split_file="$CATALOGUE",
)
evaluator.run(frame_shuffle_test=True)
EOF
}

case "$stage" in
    embed) run_embed ;;
    train) run_train ;;
    eval)  run_eval  ;;
    all)
        run_embed
        run_train
        run_eval
        ;;
    *)
        echo "Unknown stage: $stage"
        echo "Usage: bash run.sh [embed|train|eval|all]"
        exit 1
        ;;
esac

echo "=== Done ==="
