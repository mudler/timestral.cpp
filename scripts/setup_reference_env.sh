#!/usr/bin/env bash
# Create the reference virtual environment (Python 3.12, pinned packages).
# Usage: scripts/setup_reference_env.sh
# Environment: TIMESTRAL_WORK (default ./work), TIMESTRAL_REF_VENV (default $TIMESTRAL_WORK/venv-ref),
#              PYTHON (default python3.12).
set -euo pipefail
TIMESTRAL_WORK=${TIMESTRAL_WORK:-$PWD/work}
VENV=${TIMESTRAL_REF_VENV:-$TIMESTRAL_WORK/venv-ref}
PYTHON=${PYTHON:-python3.12}

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || { echo "Python 3.12 is required (numpy 2.5.4 is pinned for it)" >&2; exit 1; }
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install \
  "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff" \
  numpy==2.5.4 safetensors==0.8.0 "huggingface_hub>=0.23" gguf==0.19.0
"$VENV/bin/python" -c "import timestral, torch, numpy, gguf; print(timestral.__version__, torch.__version__)"
