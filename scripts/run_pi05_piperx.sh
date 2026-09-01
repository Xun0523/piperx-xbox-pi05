#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/openpi}"
PYTHON_BIN="${PYTHON_BIN:-$OPENPI_ROOT/.venv/bin/python}"

SKIP_CAN_PREFLIGHT=0
for ARG in "$@"; do
    if [[ "$ARG" == "-h" || "$ARG" == "--help" ]]; then
        SKIP_CAN_PREFLIGHT=1
    fi
done

if (( ! SKIP_CAN_PREFLIGHT )); then
    if [[ ! -e /sys/class/net/can0/flags ]]; then
        echo "未检测到 can0。请检查 USB-CAN 并先运行 ./scripts/setup_can.sh。" >&2
        exit 4
    fi
    CAN_FLAGS="$(( $(</sys/class/net/can0/flags) ))"
    if (( (CAN_FLAGS & 1) == 0 )); then
        echo "can0 当前为 DOWN。请先运行 ./scripts/setup_can.sh。" >&2
        exit 4
    fi
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/data/openpi_cache}"

exec "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/run_pi05_piperx.py" "$@"
