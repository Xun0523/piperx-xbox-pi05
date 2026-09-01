#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/openpi}"
PYTHON_BIN="${PYTHON_BIN:-$OPENPI_ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/check_piper_status.py"
