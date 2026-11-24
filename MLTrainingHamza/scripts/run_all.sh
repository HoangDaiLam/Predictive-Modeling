#!/usr/bin/env bash
# Train all candidates, then open the themed MLflow UI.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e ".[dev]"
python -m pytest
evmlops train
evmlops ui
