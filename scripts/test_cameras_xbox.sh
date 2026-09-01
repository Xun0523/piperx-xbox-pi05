#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/openpi}"
PYTHON_BIN="${PYTHON_BIN:-$OPENPI_ROOT/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"

exec "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/test_cameras_xbox.py"
