#!/usr/bin/env bash
set -Eeuo pipefail

# Lightweight install for the PEFT pipeline only.
#
# This intentionally skips the original DeepfakeBench detector/preprocessing
# stack: dlib, OpenCV, albumentations, segmentation models, lmdb, facenet,
# and the OpenAI CLIP git package.
#
# Recommended on Jupyter/KTH if /opt/conda/envs is ephemeral:
#   source /opt/conda/etc/profile.d/conda.sh
#   conda create -p /home/jovyan/conda-envs/DeepfakeBench python=3.9 -y
#   conda activate /home/jovyan/conda-envs/DeepfakeBench
#   bash install_light.sh
#
# If you already have an activated env:
#   bash install_light.sh
#
# CPU-only:
#   INSTALL_CPU_ONLY=1 bash install_light.sh
#
# Override CUDA wheel index:
#   PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash install_light.sh

PYTHON_BIN="${PYTHON:-python}"
PIP=("${PYTHON_BIN}" -m pip)

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${PYTHON_VERSION}" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "Expected Python 3.9-3.12 for the PEFT pipeline; found ${PYTHON_VERSION}." >&2
    exit 1
    ;;
esac

if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Warning: no active conda or virtualenv environment detected." >&2
fi

"${PIP[@]}" install --upgrade pip setuptools wheel

if [[ "${INSTALL_CPU_ONLY:-0}" == "1" ]]; then
  "${PIP[@]}" install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
else
  PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  "${PIP[@]}" install torch torchvision torchaudio \
    --index-url "${PYTORCH_INDEX_URL}"
fi

"${PIP[@]}" install \
  "numpy>=1.23.5,<2" \
  "pandas>=1.5.3,<2.3" \
  "Pillow>=9.5,<11" \
  "PyYAML>=6,<7" \
  "tqdm>=4.66,<5" \
  "scikit-learn>=1.3,<1.6" \
  "open-clip-torch>=2.24,<3" \
  "transformers>=4.35,<5" \
  "safetensors>=0.4,<1"

"${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "numpy",
    "open_clip",
    "pandas",
    "PIL",
    "sklearn",
    "torch",
    "torchvision",
    "tqdm",
    "yaml",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    raise SystemExit("Install completed, but import checks failed:\n" + "\n".join(missing))

print("Light PEFT install completed and import checks passed.")
PY
