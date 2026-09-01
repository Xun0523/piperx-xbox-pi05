#!/usr/bin/env bash
set -euo pipefail

CAN_INTERFACE="${1:-can0}"

if [[ "$CAN_INTERFACE" != "can0" ]]; then
    echo "拒绝配置未知接口: $CAN_INTERFACE（本项目当前只允许 can0）" >&2
    exit 2
fi

if [[ ! -e "/sys/class/net/$CAN_INTERFACE" ]]; then
    echo "未检测到 $CAN_INTERFACE；请重新插拔 USB-CAN 后再试。" >&2
    exit 1
fi

# Keep these as separate operations. This follows Piper SDK's activation flow
# and lets us distinguish USB control-transfer failures from link-up failures.
sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true

if ! sudo ip link set "$CAN_INTERFACE" type can bitrate 1000000; then
    echo >&2
    echo "设置 CAN 位时序失败。若看到 'Broken pipe'，说明 gs_usb 与 USB-CAN 的" >&2
    echo "USB 控制通信失败；请将 USB-CAN 改接主板后置直连 USB 口，避免与相机共用 Hub。" >&2
    exit 1
fi

if ! sudo ip link set "$CAN_INTERFACE" up; then
    echo >&2
    echo "CAN 参数已写入，但接口启动失败。请重新插拔 USB-CAN 后重试。" >&2
    exit 1
fi

DETAILS="$(ip -details -statistics link show "$CAN_INTERFACE")"
printf '%s\n' "$DETAILS"
if ! grep -Eq '<([^>]*,)?UP(,|>)' <<<"$DETAILS"; then
    echo "$CAN_INTERFACE 未进入 UP 状态，拒绝继续。" >&2
    exit 1
fi

RX_BEFORE="$(<"/sys/class/net/$CAN_INTERFACE/statistics/rx_packets")"
sleep 2
RX_AFTER="$(<"/sys/class/net/$CAN_INTERFACE/statistics/rx_packets")"
if (( RX_AFTER <= RX_BEFORE )); then
    echo >&2
    echo "$CAN_INTERFACE 已启用为 1 Mbps，但两秒内没有收到任何 CAN 帧。" >&2
    echo "请检查 Piper 电源、物理急停，以及 USB-CAN 到机械臂的 CAN 插头/线缆。" >&2
    exit 3
fi

echo "$CAN_INTERFACE 已启用：1 Mbps；两秒内收到 $((RX_AFTER - RX_BEFORE)) 帧。"
