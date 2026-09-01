#!/usr/bin/env bash
set -euo pipefail

CHECK_SECONDS="${1:-60}"
if [[ ! "$CHECK_SECONDS" =~ ^[0-9]+$ ]] || (( CHECK_SECONDS < 10 || CHECK_SECONDS > 600 )); then
    echo "用法: $0 [10..600 秒]" >&2
    exit 2
fi

CAN_INTERFACE="can0"
EXPECTED_SERIAL="${PIPERX_CAN_SERIAL:-}"

if [[ ! -e "/sys/class/net/$CAN_INTERFACE/flags" ]]; then
    echo "未检测到 $CAN_INTERFACE。" >&2
    exit 1
fi

CAN_FLAGS="$(( $(<"/sys/class/net/$CAN_INTERFACE/flags") ))"
if (( (CAN_FLAGS & 1) == 0 )); then
    echo "$CAN_INTERFACE 为 DOWN；先运行 ./scripts/setup_can.sh。" >&2
    exit 1
fi

USB_INTERFACE_PATH="$(readlink -f "/sys/class/net/$CAN_INTERFACE/device")"
USB_DEVICE_PATH="$(dirname "$USB_INTERFACE_PATH")"
USB_SERIAL="$(<"$USB_DEVICE_PATH/serial")"
if [[ -n "$EXPECTED_SERIAL" && "$USB_SERIAL" != "$EXPECTED_SERIAL" ]]; then
    echo "检测到非预期 USB-CAN: $USB_SERIAL" >&2
    exit 1
fi
echo "USB-CAN serial: $USB_SERIAL"

INITIAL_IFINDEX="$(<"/sys/class/net/$CAN_INTERFACE/ifindex")"
LAST_RX="$(<"/sys/class/net/$CAN_INTERFACE/statistics/rx_packets")"
TASK_START_SECONDS=$SECONDS

echo "开始 $CHECK_SECONDS 秒 CAN 稳定性检查：$USB_DEVICE_PATH"
while (( SECONDS - TASK_START_SECONDS < CHECK_SECONDS )); do
    sleep 1
    if [[ ! -e "/sys/class/net/$CAN_INTERFACE/ifindex" ]]; then
        echo "失败：USB-CAN 在检查期间消失。" >&2
        exit 1
    fi
    CURRENT_IFINDEX="$(<"/sys/class/net/$CAN_INTERFACE/ifindex")"
    CAN_FLAGS="$(( $(<"/sys/class/net/$CAN_INTERFACE/flags") ))"
    CURRENT_RX="$(<"/sys/class/net/$CAN_INTERFACE/statistics/rx_packets")"
    if [[ "$CURRENT_IFINDEX" != "$INITIAL_IFINDEX" ]]; then
        echo "失败：USB-CAN 已重新枚举（ifindex $INITIAL_IFINDEX -> $CURRENT_IFINDEX）。" >&2
        exit 1
    fi
    if (( (CAN_FLAGS & 1) == 0 )); then
        echo "失败：$CAN_INTERFACE 在检查期间变为 DOWN。" >&2
        exit 1
    fi
    if (( CURRENT_RX <= LAST_RX )); then
        echo "失败：一秒内没有收到 Piper CAN 反馈。" >&2
        exit 1
    fi
    LAST_RX=$CURRENT_RX
    ELAPSED_SECONDS=$((SECONDS - TASK_START_SECONDS))
    if (( ELAPSED_SECONDS % 5 == 0 )); then
        echo "  ${ELAPSED_SECONDS}s: UP, RX=$CURRENT_RX"
    fi
done

echo "通过：$CHECK_SECONDS 秒内接口未重连，Piper 反馈持续增长。"
