#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
if [[ -n "${WHEELHOUSE:-}" ]]; then
  .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -c 'import torch, torchvision; print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda, "vision:", torchvision.__version__)'
