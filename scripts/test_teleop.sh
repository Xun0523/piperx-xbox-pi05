#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec "$PROJECT_ROOT/scripts/record.sh" \
    --task "hardware teleoperation test" \
    --enable-motion \
    --teleop-only \
    --display \
    --robot-speed-percent 20 \
    --control-hz 50 \
    --joint-speed-deg-s 10 \
    --max-episode-seconds 10
