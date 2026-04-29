#!/usr/bin/env bash
set -Eeuo pipefail

# System packages needed by dlib/OpenCV on Ubuntu/Debian:
#   sudo apt install build-essential cmake libopenblas-dev liblapack-dev \
#     libx11-dev libgtk-3-dev ffmpeg
#
# Run from an activated conda/venv environment:
#   conda create -n DeepfakeBench python=3.11
#   conda activate DeepfakeBench
#   bash install.sh
#
# For CPU-only PyTorch:
#   INSTALL_CPU_ONLY=1 bash install.sh
#
# To override the CUDA wheel index:
#   PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash install.sh

PYTHON_BIN="${PYTHON:-python}"
PIP=("${PYTHON_BIN}" -m pip)

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "This installer targets the current project code and requires Python "
        "3.10 or newer. Create a fresh env, for example: "
        "conda create -n DeepfakeBench python=3.11"
    )
PY

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
    --extra-index-url "${PYTORCH_INDEX_URL}"
fi

CORE_PACKAGES=(
  "numpy>=1.23.5,<2"
  "pandas>=1.5.3,<2.3"
  "Pillow>=9.5,<11"
  "dlib>=19.24"
  "imageio>=2.31,<3"
  "imgaug==0.4.0"
  "tqdm>=4.66,<5"
  "scipy>=1.10,<1.14"
  "seaborn>=0.12,<0.14"
  "matplotlib>=3.7,<3.10"
  "PyYAML>=6,<7"
  "imutils==0.5.4"
  "opencv-python>=4.8,<5"
  "scikit-image>=0.21,<0.23"
  "scikit-learn>=1.3,<1.5"
  "albumentations==1.3.1"
  "efficientnet-pytorch==0.7.1"
  "timm==0.6.12"
  "segmentation-models-pytorch>=0.3.3,<0.4"
  "torchtoolbox==0.1.8.2"
  "tensorboard>=2.13,<3"
  "loralib>=0.1.2"
  "einops>=0.6,<0.9"
  "transformers>=4.35,<5"
  "filterpy==1.4.5"
  "simplejson>=3.19,<4"
  "kornia>=0.7,<0.9"
  "fvcore>=0.1.5.post20221221"
  "lmdb>=1.4,<2"
  "open-clip-torch>=2.24,<3"
  "optuna>=3.6,<5"
  "pingouin>=0.5.4,<0.6"
  "psutil>=5.9,<6"
  "statsmodels>=0.14,<0.15"
)

"${PIP[@]}" install "${CORE_PACKAGES[@]}"

# facenet-pytorch declares narrow torch pins; keep the PyTorch version selected
# above and install only the package code.
"${PIP[@]}" install --no-deps "facenet-pytorch>=2.5.3"

# Needed by the legacy StA-CLIP detector. Most current project code uses
# open-clip-torch instead.
"${PIP[@]}" install "git+https://github.com/openai/CLIP.git"

"${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "albumentations",
    "cv2",
    "dlib",
    "lmdb",
    "matplotlib",
    "numpy",
    "open_clip",
    "optuna",
    "pandas",
    "pingouin",
    "sklearn",
    "torch",
    "torchvision",
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

print("Install completed and import checks passed.")
PY
