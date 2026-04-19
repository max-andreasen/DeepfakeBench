#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python training/searches/layer_probe.py \
  --base-config training/configs/linear.yaml \
  --embeddings-dir clip/embeddings/probing/ViT-L-14-336-quickgelu \
  --out-dir logs/probing/
