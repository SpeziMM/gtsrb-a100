#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/python.sh
select_project_python
if [[ "${1:-}" == "--check" ]]; then
  exit 0
elif [[ $# -gt 0 ]]; then
  printf 'Usage: bash scripts/setup.sh [--check]\n' >&2
  exit 1
fi
if [[ -d .venv ]]; then
  if [[ ! -x .venv/bin/python ]] || \
     ! .venv/bin/python -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)'; then
    printf '%s\n' \
      'The existing .venv is missing a supported Python.' \
      'Rename it before recreating the environment: mv .venv .venv-python-old' \
      'Then rerun bash scripts/setup.sh with a supported interpreter.' >&2
    exit 1
  fi
else
  "$PYTHON_BIN" -m venv .venv
fi
if [[ -n "${WHEELHOUSE:-}" ]]; then
  .venv/bin/python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -c 'import torch, torchvision; print("torch:", torch.__version__, "CUDA runtime:", torch.version.cuda, "vision:", torchvision.__version__)'
