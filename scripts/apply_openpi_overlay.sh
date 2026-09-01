#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_ROOT="${1:-${OPENPI_ROOT:-$PROJECT_ROOT/openpi}}"
EXPECTED_COMMIT="15a9616a00943ada6c20a0f158e3adb39df2ccac"

if [[ ! -d "$OPENPI_ROOT/.git" ]]; then
    echo "未找到 OpenPI Git 仓库: $OPENPI_ROOT" >&2
    exit 2
fi

ACTUAL_COMMIT="$(git -C "$OPENPI_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "OpenPI commit 不匹配。" >&2
    echo "expected: $EXPECTED_COMMIT" >&2
    echo "actual:   $ACTUAL_COMMIT" >&2
    exit 3
fi

CONFIG_FILE="$OPENPI_ROOT/src/openpi/training/config.py"
if ! grep -q "class LeRobotPiperXDataConfig" "$CONFIG_FILE"; then
    git -C "$OPENPI_ROOT" apply --check "$PROJECT_ROOT/patches/openpi-piperx.patch"
    git -C "$OPENPI_ROOT" apply "$PROJECT_ROOT/patches/openpi-piperx.patch"
fi

install -D -m 0644 \
    "$PROJECT_ROOT/openpi_overlay/src/openpi/policies/piperx_policy.py" \
    "$OPENPI_ROOT/src/openpi/policies/piperx_policy.py"
install -D -m 0644 \
    "$PROJECT_ROOT/openpi_overlay/src/openpi/policies/piperx_policy_test.py" \
    "$OPENPI_ROOT/src/openpi/policies/piperx_policy_test.py"

echo "Piper-X OpenPI overlay 已应用到 $OPENPI_ROOT"
