#!/usr/bin/env python3
"""Read Piper status without enabling motors or sending motion targets."""

from __future__ import annotations

import time

from piper_sdk import C_PiperInterface_V2


def main() -> int:
    piper = C_PiperInterface_V2("can0", True, True)
    try:
        piper.ConnectPort()
        time.sleep(0.5)
        wrapper = piper.GetArmStatus()
        status = wrapper.arm_status
        print(f"status_hz={wrapper.Hz:.1f}")
        print(f"ctrl_mode={status.ctrl_mode}")
        print(f"arm_status={status.arm_status}")
        print(f"mode_feed={status.mode_feed}")
        print(f"teach_status={status.teach_status}")
        print(f"motion_status={status.motion_status}")
        print(f"err_code={status.err_code}")
        print(f"motor_enable={piper.GetArmEnableStatus()}")
        return 0
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    raise SystemExit(main())
