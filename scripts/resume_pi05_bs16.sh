#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESUME=1
export SKIP_NORM_STATS="${SKIP_NORM_STATS:-1}"
exec "$PROJECT_ROOT/scripts/train_pi05_piperx.sh" "$@"
