#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/openpi}"
PYTHON_BIN="${PYTHON_BIN:-$OPENPI_ROOT/.venv/bin/python}"
EXP_NAME="${EXP_NAME:-piperx_pi05_lora_50ep_bs16_v1}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-19999}"
CHECKPOINT="${CHECKPOINT:-$OPENPI_ROOT/checkpoints/pi05_piperx_lora/$EXP_NAME/$CHECKPOINT_STEP}"
POLICY_PORT="${POLICY_PORT:-8000}"
TASK="${TASK:-put the yellow rubber duck onto the white plate}"

if [[ ! -f "$CHECKPOINT/_CHECKPOINT_METADATA" ]]; then
    echo "找不到 checkpoint: $CHECKPOINT" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/data/openpi_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

cd "$OPENPI_ROOT"
exec "$PYTHON_BIN" scripts/serve_policy.py \
    --port="$POLICY_PORT" \
    --default-prompt="$TASK" \
    policy:checkpoint \
    --policy.config=pi05_piperx_lora \
    --policy.dir="$CHECKPOINT"
