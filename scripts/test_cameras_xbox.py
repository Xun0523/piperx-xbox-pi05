#!/usr/bin/env python3
"""Preview both cameras and Xbox inputs without opening CAN or moving Piper."""

from __future__ import annotations

import cv2
import numpy as np

from record_piperx_lerobot import DEFAULT_BASE_CAMERA
from record_piperx_lerobot import DEFAULT_WRIST_CAMERA
from record_piperx_lerobot import LatestFrameCamera
from record_piperx_lerobot import XboxController


def main() -> int:
    base_camera = LatestFrameCamera(DEFAULT_BASE_CAMERA, 640, 480, 30, "第三视角")
    wrist_camera = LatestFrameCamera(DEFAULT_WRIST_CAMERA, 640, 480, 30, "腕部")
    xbox: XboxController | None = None
    try:
        base_camera.start()
        wrist_camera.start()
        base_camera.wait_ready()
        wrist_camera.wait_ready()
        xbox = XboxController()
        print("相机和 Xbox 预览已启动；移动摇杆/扳机观察终端数值，START 或窗口 q 退出。")

        while True:
            axes, hat, buttons, rising = xbox.sample()
            base_rgb, base_ts = base_camera.get()
            wrist_rgb, wrist_ts = wrist_camera.get()
            skew_ms = abs(base_ts - wrist_ts) * 1000.0

            base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
            wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                base_bgr,
                "THIRD PERSON",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                wrist_bgr,
                "WRIST",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            preview = np.concatenate([base_bgr, wrist_bgr], axis=1)
            cv2.putText(
                preview,
                f"camera skew: {skew_ms:.0f} ms",
                (16, 464),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("PiperX input test: third-person | wrist", preview)
            print(
                f"\raxes={np.round(axes, 2).tolist()} hat={hat} "
                f"buttons={[i for i, down in enumerate(buttons) if down]} skew={skew_ms:.0f}ms",
                end="",
                flush=True,
            )

            if XboxController.BUTTON_START in rising:
                break
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        print("\n相机和 Xbox 预览已退出。")
        return 0
    finally:
        if xbox is not None:
            xbox.close()
        base_camera.stop()
        wrist_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
