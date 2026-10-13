#!/usr/bin/env bash
# Run on an internet-connected Linux x86_64 host with the cluster's Python version.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/python.sh
select_project_python
"$PYTHON_BIN" -m pip download --only-binary=:all: --dest wheelhouse \
  --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
"$PYTHON_BIN" - <<'PY'
import hashlib
from pathlib import Path
root = Path('wheelhouse')
with (root/'SHA256SUMS').open('w') as f:
    for p in sorted(root.glob('*.whl')):
        h = hashlib.sha256()
        with p.open('rb') as src:
            for chunk in iter(lambda: src.read(1024*1024), b''):
                h.update(chunk)
        f.write(f'{h.hexdigest()}  {p.name}\n')
PY
