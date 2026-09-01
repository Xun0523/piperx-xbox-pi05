#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PROJECT_ROOT/openpi}"
PYTHON_BIN="${PYTHON_BIN:-$OPENPI_ROOT/.venv/bin/python}"
EXP_NAME="${EXP_NAME:-piperx_pi05_lora_50ep_bs16_v1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-1000}"
RESUME="${RESUME:-0}"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-0}"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PROJECT_ROOT/data/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PROJECT_ROOT/data/openpi_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

cd "$OPENPI_ROOT"
if [[ "$SKIP_NORM_STATS" != "1" ]]; then
    "$PYTHON_BIN" scripts/compute_norm_stats.py --config-name pi05_piperx_lora
fi

RESUME_FLAG="--no-resume"
if [[ "$RESUME" == "1" ]]; then
    RESUME_FLAG="--resume"
fi

"$PYTHON_BIN" scripts/train.py pi05_piperx_lora \
    --exp-name="$EXP_NAME" \
    --batch-size="$BATCH_SIZE" \
    --num-train-steps="$NUM_TRAIN_STEPS" \
    --save-interval="$SAVE_INTERVAL" \
    --keep-period="$KEEP_PERIOD" \
    --no-overwrite \
    "$RESUME_FLAG"
