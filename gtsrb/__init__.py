"""A small, fixed-architecture GTSRB experiment."""

import sys


def require_supported_python():
    # Keep this check parseable on the cluster's Python 3.6. Package initialization
    # runs before Python tries to parse data.py or import the pinned dependencies.
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise SystemExit(
            "This project requires Python 3.10–3.13 (3.12 recommended); "
            "you are using Python {}.{}.\n"
            "Load a newer Python module or activate a modern Conda environment, "
            "then run bash scripts/setup.sh.\n"
            "Use .venv/bin/python -m gtsrb.data --download afterward."
            .format(sys.version_info[0], sys.version_info[1])
        )


require_supported_python()
