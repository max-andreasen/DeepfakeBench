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
# Existing Python 3.7/3.8 DeepfakeBench environments are supported in legacy
# mode for the original DeepfakeBench pipeline. The newer PEFT/temporal code
# uses Python 3.9+ syntax and newer PyTorch APIs.
#
# For CPU-only PyTorch:
#   INSTALL_CPU_ONLY=1 bash install.sh
#
# To override the CUDA wheel index:
#   PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash install.sh

PYTHON_BIN="${PYTHON:-python}"
PIP=("${PYTHON_BIN}" -m pip)

PYTHON_VERSION="$("${PYTHON_BIN}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "${PYTHON_VERSION}" in
  3.7|3.8)
    LEGACY_PYTHON=1
    echo "Detected Python ${PYTHON_VERSION}; using legacy DeepfakeBench dependency pins." >&2
    echo "Note: the newer PEFT/temporal code in this repo is intended for Python 3.9+." >&2
    ;;
  *)
    LEGACY_PYTHON=0
    ;;
esac

if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Warning: no active conda or virtualenv environment detected." >&2
fi

if [[ "${LEGACY_PYTHON}" == "1" ]]; then
  "${PIP[@]}" install "pip<24.1" "setuptools==59.5.0" "wheel<0.43"
else
  "${PIP[@]}" install --upgrade pip setuptools wheel
fi

if [[ "${INSTALL_CPU_ONLY:-0}" == "1" ]]; then
  if [[ "${LEGACY_PYTHON}" == "1" ]]; then
    "${PIP[@]}" install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
  else
    "${PIP[@]}" install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu
  fi
else
  if [[ "${LEGACY_PYTHON}" == "1" ]]; then
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu113}"
    "${PIP[@]}" install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 \
      --extra-index-url "${PYTORCH_INDEX_URL}"
  else
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    "${PIP[@]}" install torch torchvision torchaudio \
      --extra-index-url "${PYTORCH_INDEX_URL}"
  fi
fi

if [[ "${LEGACY_PYTHON}" == "1" ]]; then
  CORE_PACKAGES=(
    "numpy==1.21.5"
    "pandas==1.4.2"
    "Pillow==9.0.1"
    "dlib==19.24.0"
    "imageio==2.9.0"
    "imgaug==0.4.0"
    "tqdm==4.61.0"
    "scipy==1.7.3"
    "seaborn==0.11.2"
    "matplotlib==3.5.3"
    "PyYAML==6.0"
    "imutils==0.5.4"
    "opencv-python==4.6.0.66"
    "scikit-image==0.19.2"
    "scikit-learn==1.0.2"
    "albumentations==1.1.0"
    "efficientnet-pytorch==0.7.1"
    "timm==0.6.12"
    "segmentation-models-pytorch==0.3.2"
    "torchtoolbox==0.1.8.2"
    "tensorboard==2.10.1"
    "loralib>=0.1.2"
    "einops>=0.4,<0.7"
    "transformers>=4.26,<4.31"
    "filterpy==1.4.5"
    "simplejson>=3.17,<4"
    "kornia>=0.6,<0.7"
    "fvcore>=0.1.5.post20221221"
    "lmdb>=1.4,<2"
    "open-clip-torch<2.21"
    "optuna>=3.0,<3.5"
    "pingouin>=0.5.3,<0.6"
    "psutil>=5.9,<6"
    "statsmodels>=0.13,<0.14"
  )
else
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
fi

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
