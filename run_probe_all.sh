#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

EMB_DIR=clip/embeddings/probing/ViT-L-14-336-quickgelu
OUT=logs/probing/

mkdir -p "$OUT"

launch() {
    local cfg=$1
    local tag=$2
    python training/searches/layer_probe.py \
        --base-config "$cfg" \
        --embeddings-dir "$EMB_DIR" \
        --out-dir "$OUT" \
        > "$OUT/${tag}.out" 2>&1 &
    echo "  $tag → PID $!  (log: $OUT/${tag}.out)"
}

echo "Launching concurrent runs..."
launch training/configs/linear.yaml                       linear
launch training/configs/bigru_1.yaml                      bigru
launch training/configs/trans_proj.yaml/trans_proj.yaml   transformer

echo
echo "Waiting for all three to finish. Tail logs with:"
echo "  tail -f $OUT/{linear,bigru,transformer}.out"
wait
echo "All done."
