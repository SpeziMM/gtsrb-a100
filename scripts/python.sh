#!/usr/bin/env bash
# Shared interpreter selection for setup and offline wheel preparation.

select_project_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
      return 1
    fi
  else
    local candidate
    for candidate in python3 python3.12 python3.11 python3.10 python3.13 python; do
      if command -v "$candidate" >/dev/null 2>&1 && \
         "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
    if [[ -z "${PYTHON_BIN:-}" ]]; then
      printf '%s\n' \
        'No supported Python found. This project needs Python 3.10–3.13 (3.12 recommended).' \
        'On the cluster, run: module avail python' \
        'Load an available supported version, then rerun bash scripts/setup.sh.' \
        'Alternatively activate a Python 3.12 Conda environment.' \
        'To select an installed interpreter explicitly: PYTHON_BIN=python3.12 bash scripts/setup.sh' >&2
      return 1
    fi
  fi
  "$PYTHON_BIN" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    sys.exit('Unsupported Python {}.{}; use Python 3.10–3.13 (3.12 recommended).'.format(
        sys.version_info[0], sys.version_info[1]))
print('Using Python {} at {}'.format(sys.version.split()[0], sys.executable))
PY
}
