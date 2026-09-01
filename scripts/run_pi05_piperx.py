#!/usr/bin/env python3
"""Run a PI0.5 checkpoint on PiperX with camera and motion safety gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import select
import signal
import sys
import threading
import time

import cv2
import numpy as np
from openpi_client import websocket_client_policy

from record_piperx_lerobot import (
    DEFAULT_BASE_CAMERA,
    DEFAULT_WRIST_CAMERA,
    GRIPPER_MAX_M,
    GRIPPER_MIN_M,
    JOINT_LOWER,
    JOINT_UPPER,
    LatestFrameCamera,
    Piper,
    clamp_target,
)


WINDOW_NAME = "PiperX PI0.5: third-person | wrist"


class StopRequested(Exception):
    """Raised when the operator requests a controlled emergency stop."""


class PolicyHoldRequested(Exception):
    """Raised when policy-side failure should fall back to enabled position hold."""


class StationaryDetector:
    """Stop policy inference after sustained commanded and observed stillness."""

    def __init__(
        self,
        initial_target: np.ndarray,
        *,
        required_frames: int,
        minimum_runtime_s: float,
        movement_joint_rad: float,
        movement_gripper_m: float,
        stationary_joint_rad: float,
        stationary_gripper_m: float,
    ):
        self.initial_target = np.asarray(initial_target, dtype=np.float32).copy()
        self.required_frames = required_frames
        self.minimum_runtime_s = minimum_runtime_s
        self.movement_joint_rad = movement_joint_rad
        self.movement_gripper_m = movement_gripper_m
        self.stationary_joint_rad = stationary_joint_rad
        self.stationary_gripper_m = stationary_gripper_m
        self.previous_target: np.ndarray | None = None
        self.previous_state: np.ndarray | None = None
        self.movement_seen = False
        self.stationary_frames = 0

    def update(
        self,
        target: np.ndarray,
        state: np.ndarray,
        elapsed_s: float,
    ) -> bool:
        target = np.asarray(target, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)

        initial_joint_motion = float(np.max(np.abs(target[:6] - self.initial_target[:6])))
        initial_gripper_motion = float(abs(target[6] - self.initial_target[6]))
        if (
            initial_joint_motion >= self.movement_joint_rad
            or initial_gripper_motion >= self.movement_gripper_m
        ):
            self.movement_seen = True

        if self.previous_target is None or self.previous_state is None:
            self.previous_target = target.copy()
            self.previous_state = state.copy()
            return False

        target_joint_delta = float(np.max(np.abs(target[:6] - self.previous_target[:6])))
        target_gripper_delta = float(abs(target[6] - self.previous_target[6]))
        state_joint_delta = float(np.max(np.abs(state[:6] - self.previous_state[:6])))
        state_gripper_delta = float(abs(state[6] - self.previous_state[6]))
        stationary = (
            self.movement_seen
            and elapsed_s >= self.minimum_runtime_s
            and target_joint_delta <= self.stationary_joint_rad
            and target_gripper_delta <= self.stationary_gripper_m
            and state_joint_delta <= self.stationary_joint_rad
            and state_gripper_delta <= self.stationary_gripper_m
        )
        self.stationary_frames = self.stationary_frames + 1 if stationary else 0
        self.previous_target = target.copy()
        self.previous_state = state.copy()
        return self.stationary_frames >= self.required_frames


@dataclass(frozen=True)
class CameraSample:
    third_person: np.ndarray
    wrist: np.ndarray
    third_person_ts: float
    wrist_ts: float

    @property
    def skew_s(self) -> float:
        return abs(self.third_person_ts - self.wrist_ts)

    @property
    def brightness(self) -> tuple[float, float]:
        return float(self.third_person.mean()), float(self.wrist.mean())


def parse_action_chunk(actions: np.ndarray, *, expected_horizon: int = 16) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (expected_horizon, 7):
        raise RuntimeError(f"策略动作形状应为 ({expected_horizon}, 7)，实际为 {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise RuntimeError("策略输出含 NaN 或 Inf")
    return actions


def sanitize_action_chunk(
    actions: np.ndarray,
    *,
    expected_horizon: int = 16,
    max_joint_overshoot_rad: float = np.deg2rad(1.0),
    max_gripper_overshoot_m: float = 0.002,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Clamp small boundary overshoots and reject materially unsafe predictions."""
    actions = parse_action_chunk(actions, expected_horizon=expected_horizon)

    joint_below = np.maximum(JOINT_LOWER[None, :] - actions[:, :6], 0.0)
    joint_above = np.maximum(actions[:, :6] - JOINT_UPPER[None, :], 0.0)
    gripper_below = np.maximum(GRIPPER_MIN_M - actions[:, 6], 0.0)
    gripper_above = np.maximum(actions[:, 6] - GRIPPER_MAX_M, 0.0)

    severe_joint = (joint_below > max_joint_overshoot_rad) | (joint_above > max_joint_overshoot_rad)
    severe_gripper = (gripper_below > max_gripper_overshoot_m) | (
        gripper_above > max_gripper_overshoot_m
    )
    if np.any(severe_joint) or np.any(severe_gripper):
        details: list[str] = []
        for step, joint in np.argwhere(severe_joint)[:8]:
            value_deg = float(np.rad2deg(actions[step, joint]))
            if joint_below[step, joint] > 0:
                overshoot_deg = float(np.rad2deg(joint_below[step, joint]))
                boundary = "下限"
            else:
                overshoot_deg = float(np.rad2deg(joint_above[step, joint]))
                boundary = "上限"
            details.append(
                f"step={step} J{joint + 1}={value_deg:.2f}deg，超{boundary}{overshoot_deg:.2f}deg"
            )
        for step in np.flatnonzero(severe_gripper)[:8]:
            value_mm = float(actions[step, 6] * 1000.0)
            if gripper_below[step] > 0:
                overshoot_mm = float(gripper_below[step] * 1000.0)
                boundary = "下限"
            else:
                overshoot_mm = float(gripper_above[step] * 1000.0)
                boundary = "上限"
            details.append(
                f"step={step} gripper={value_mm:.2f}mm，超{boundary}{overshoot_mm:.2f}mm"
            )
        raise RuntimeError("策略动作显著超出 Piper 硬限位：" + "; ".join(details))

    clipped = actions.copy()
    clipped[:, :6] = np.clip(clipped[:, :6], JOINT_LOWER[None, :], JOINT_UPPER[None, :])
    clipped[:, 6] = np.clip(clipped[:, 6], GRIPPER_MIN_M, GRIPPER_MAX_M)

    corrections: list[str] = []
    changed = np.abs(clipped - actions) > 1e-7
    for step, dimension in np.argwhere(changed)[:8]:
        if dimension < 6:
            raw = float(np.rad2deg(actions[step, dimension]))
            fixed = float(np.rad2deg(clipped[step, dimension]))
            corrections.append(f"step={step} J{dimension + 1} {raw:.2f}->{fixed:.2f}deg")
        else:
            raw = float(actions[step, dimension] * 1000.0)
            fixed = float(clipped[step, dimension] * 1000.0)
            corrections.append(f"step={step} gripper {raw:.2f}->{fixed:.2f}mm")
    if int(changed.sum()) > len(corrections):
        corrections.append(f"另有 {int(changed.sum()) - len(corrections)} 处")
    return clipped, tuple(corrections)


def limit_target_step(
    previous: np.ndarray,
    desired: np.ndarray,
    *,
    max_joint_step_rad: float,
    max_gripper_step_m: float,
) -> tuple[np.ndarray, bool]:
    previous = np.asarray(previous, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    limited_delta = desired - previous
    limited_delta[:6] = np.clip(limited_delta[:6], -max_joint_step_rad, max_joint_step_rad)
    limited_delta[6] = np.clip(limited_delta[6], -max_gripper_step_m, max_gripper_step_m)
    limited = clamp_target(previous + limited_delta)
    was_limited = not np.allclose(limited, desired, rtol=0.0, atol=1e-7)
    return limited, was_limited


class PositionHeartbeat:
    """Refresh Piper's position command at 50 Hz while inference is blocking."""

    def __init__(
        self,
        piper: Piper,
        initial_target: np.ndarray,
        *,
        control_hz: int,
        tracking_error_deg: float,
        tracking_error_gripper_m: float,
        stop_event: threading.Event,
    ):
        self._piper = piper
        self._target = np.asarray(initial_target, dtype=np.float32).copy()
        self._period = 1.0 / control_hz
        self._tracking_error_rad = np.deg2rad(tracking_error_deg)
        self._tracking_error_gripper_m = tracking_error_gripper_m
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="piper-heartbeat", daemon=True)
        self._thread.start()

    def set_target(self, target: np.ndarray) -> None:
        with self._lock:
            self._target = np.asarray(target, dtype=np.float32).copy()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"机械臂控制心跳失败: {self._error}") from self._error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _trip(self, exc: Exception) -> None:
        if self._error is None:
            self._error = exc
        self._stop_event.set()
        try:
            self._piper.emergency_stop()
        except Exception as stop_exc:
            if self._error is None:
                self._error = stop_exc

    def _run(self) -> None:
        next_tick = time.monotonic()
        last_status = 0.0
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_tick:
                    self._stop_event.wait(next_tick - now)
                    continue
                next_tick = max(next_tick + self._period, now)

                with self._lock:
                    target = self._target.copy()

                state = self._piper.read_state()
                # Give position mode a short settling period immediately after enable.
                if now - self._started_at > 0.5:
                    joint_error = float(np.max(np.abs(target[:6] - state[:6])))
                    gripper_error = float(abs(target[6] - state[6]))
                    if joint_error > self._tracking_error_rad:
                        raise RuntimeError(f"关节跟踪误差过大: {np.rad2deg(joint_error):.1f} deg")
                    if gripper_error > self._tracking_error_gripper_m:
                        raise RuntimeError(f"夹爪跟踪误差过大: {gripper_error * 1000:.1f} mm")

                self._piper.send(target)
                if now - last_status >= 0.5:
                    _ctrl, _arm, _move, _teach, enabled = self._piper.status()
                    if not all(enabled):
                        raise RuntimeError("Piper 六轴使能在运行中丢失")
                    last_status = now
        except Exception as exc:
            self._trip(exc)


def read_camera_sample(
    third_person_camera: LatestFrameCamera,
    wrist_camera: LatestFrameCamera,
) -> CameraSample:
    third_person, third_person_ts = third_person_camera.get()
    wrist, wrist_ts = wrist_camera.get()
    return CameraSample(third_person, wrist, third_person_ts, wrist_ts)


def camera_sample_is_safe(sample: CameraSample, *, max_skew_s: float) -> tuple[bool, str]:
    brightness = sample.brightness
    if sample.skew_s > max_skew_s:
        return False, f"双相机时间偏差 {sample.skew_s * 1000:.0f} ms 超限"
    if min(brightness) < 2.0:
        return False, f"相机画面近乎全黑（亮度 {brightness[0]:.1f}/{brightness[1]:.1f}）"
    return True, "OK"


def make_observation(sample: CameraSample, state: np.ndarray, task: str) -> dict:
    return {
        "observation/image": sample.third_person,
        "observation/wrist_image": sample.wrist,
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": task,
    }


def draw_preview(sample: CameraSample, state: np.ndarray, status: str) -> int:
    third_person = cv2.cvtColor(sample.third_person, cv2.COLOR_RGB2BGR)
    wrist = cv2.cvtColor(sample.wrist, cv2.COLOR_RGB2BGR)
    for frame, label in ((third_person, "THIRD PERSON"), (wrist, "WRIST")):
        cv2.putText(
            frame,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    preview = np.concatenate([third_person, wrist], axis=1)
    state_text = "joint(deg)=" + np.array2string(
        np.rad2deg(state[:6]), precision=1, separator=",", suppress_small=True
    )
    cv2.putText(
        preview,
        status,
        (16, preview.shape[0] - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        state_text,
        (16, preview.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WINDOW_NAME, preview)
    return cv2.waitKey(1) & 0xFF


def wait_for_enable(
    piper: Piper,
    third_person_camera: LatestFrameCamera,
    wrist_camera: LatestFrameCamera,
    *,
    max_camera_skew_s: float,
) -> tuple[CameraSample, np.ndarray]:
    if not sys.stdin.isatty():
        raise RuntimeError("真机模式必须从交互式本地终端启动")

    print(
        "\n[只读预检] 机械臂尚未使能，也不会发送动作。\n"
        "1. 调整第三视角相机，使机械臂、鸭子和盘子完整可见；\n"
        "2. 将机械臂和任务物体恢复到与示教数据一致的初始状态；\n"
        "3. 若使用绿色示教按钮拖动机械臂，确认完成后退出拖动示教；\n"
        "4. 工作区无人、物理急停可触达后，在本终端输入 ENABLE 并回车。\n"
        "预检期间可按预览窗口 q/Esc 或终端 Ctrl+C 退出。"
    )
    print("ENABLE> ", end="", flush=True)
    while True:
        state = piper.read_state()
        sample = read_camera_sample(third_person_camera, wrist_camera)
        safe, reason = camera_sample_is_safe(sample, max_skew_s=max_camera_skew_s)
        ctrl_mode, _arm_status, _move_mode, teach_status, motor_enable = piper.status()
        teach_active = ctrl_mode == 0x02 and teach_status == 0x01
        motor_count = sum(bool(value) for value in motor_enable)
        mode_text = "TEACH" if teach_active else "READ-ONLY"
        key = draw_preview(
            sample,
            state,
            f"PREFLIGHT {mode_text} | commands=0 | motors={motor_count}/6 | camera={reason}",
        )
        if key in (27, ord("q")):
            raise StopRequested("用户从预览窗口退出")

        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            confirmation = sys.stdin.readline().strip()
            if confirmation != "ENABLE":
                print("输入不匹配，机械臂保持未使能。请输入 ENABLE：", end="", flush=True)
                continue
            if not safe:
                print(f"相机预检未通过：{reason}。调整后重新输入 ENABLE：", end="", flush=True)
                continue
            if not np.all(np.isfinite(state)):
                print("关节反馈含 NaN/Inf，拒绝使能。", flush=True)
                continue
            if teach_active:
                print(
                    "Piper 仍处于拖动示教模式；托住机械臂并按绿色按钮退出后，重新输入 ENABLE：",
                    end="",
                    flush=True,
                )
                continue
            if any(motor_enable):
                print("检测到遗留电机使能，正在先执行软件急停并失能……", flush=True)
                piper.emergency_stop()
                if any(piper.status()[4]):
                    raise RuntimeError("ENABLE 前无法确认六轴全部失能")
                state = piper.read_state()
            return sample, state

        # Camera and feedback remain live in the OpenCV window.  Do not print
        # per-frame status here: it makes the interactive ENABLE prompt scroll.
        time.sleep(0.01)


def infer_actions(
    policy: websocket_client_policy.WebsocketClientPolicy,
    observation: dict,
    *,
    heartbeat: PositionHeartbeat | None,
    timeout_s: float,
    max_joint_overshoot_rad: float,
    max_gripper_overshoot_m: float,
) -> tuple[np.ndarray, float, tuple[str, ...]]:
    started = time.monotonic()
    if heartbeat is None:
        result = policy.infer(observation)
    else:
        result_box: list[dict] = []
        error_box: list[Exception] = []
        done = threading.Event()

        def inference_worker() -> None:
            try:
                result_box.append(policy.infer(observation))
            except Exception as exc:
                error_box.append(exc)
            finally:
                done.set()

        threading.Thread(target=inference_worker, name="policy-inference", daemon=True).start()
        deadline = time.monotonic() + timeout_s
        while not done.wait(0.05):
            heartbeat.raise_if_failed()
            if time.monotonic() >= deadline:
                raise PolicyHoldRequested(f"策略推理超过 {timeout_s:g}s")
        heartbeat.raise_if_failed()
        if error_box:
            raise PolicyHoldRequested(f"策略推理失败: {error_box[0]}") from error_box[0]
        result = result_box[0]

    infer_ms = (time.monotonic() - started) * 1000.0
    try:
        actions, corrections = sanitize_action_chunk(
            result.get("actions"),
            max_joint_overshoot_rad=max_joint_overshoot_rad,
            max_gripper_overshoot_m=max_gripper_overshoot_m,
        )
    except Exception as exc:
        if heartbeat is not None:
            raise PolicyHoldRequested(str(exc)) from exc
        raise
    return actions, infer_ms, corrections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--third-person-camera", default=DEFAULT_BASE_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--policy-hz", type=float, default=15.0)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--robot-speed-percent", type=int, default=10)
    parser.add_argument("--max-joint-step-deg", type=float, default=0.8)
    parser.add_argument("--max-gripper-step-mm", type=float, default=2.0)
    parser.add_argument("--max-joint-overshoot-deg", type=float, default=1.0)
    parser.add_argument("--max-gripper-overshoot-mm", type=float, default=2.0)
    parser.add_argument("--max-initial-jump-deg", type=float, default=5.0)
    parser.add_argument("--max-initial-gripper-jump-mm", type=float, default=10.0)
    parser.add_argument("--tracking-error-deg", type=float, default=8.0)
    parser.add_argument("--tracking-error-gripper-mm", type=float, default=15.0)
    parser.add_argument("--inference-timeout-s", type=float, default=3.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=120.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=60.0)
    parser.add_argument("--stationary-frames", type=int, default=30)
    parser.add_argument("--stationary-min-runtime-s", type=float, default=15.0)
    parser.add_argument("--movement-joint-deg", type=float, default=2.0)
    parser.add_argument("--movement-gripper-mm", type=float, default=5.0)
    parser.add_argument("--stationary-joint-deg", type=float, default=0.1)
    parser.add_argument("--stationary-gripper-mm", type=float, default=0.2)
    parser.add_argument(
        "--task",
        default="put the yellow rubber duck onto the white plate",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="允许在输入 ENABLE 后使能真机；不传时只做一次策略干运行",
    )
    args = parser.parse_args()
    if args.can_name != "can0":
        parser.error("当前安全配置只允许 can0")
    if not 1 <= args.execute_steps <= 16:
        parser.error("--execute-steps 必须在 1..16 内")
    if not 0 < args.policy_hz <= 15:
        parser.error("--policy-hz 必须在 (0, 15] 内")
    if not 15 <= args.control_hz <= 100:
        parser.error("--control-hz 必须在 15..100 内")
    if not 1 <= args.robot_speed_percent <= 20:
        parser.error("--robot-speed-percent 必须在 1..20 内")
    if not 0 < args.max_joint_step_deg <= 2.0:
        parser.error("--max-joint-step-deg 必须在 (0, 2] 内")
    if not 0 <= args.max_joint_overshoot_deg <= 2.0:
        parser.error("--max-joint-overshoot-deg 必须在 [0, 2] 内")
    if not 0 <= args.max_gripper_overshoot_mm <= 5.0:
        parser.error("--max-gripper-overshoot-mm 必须在 [0, 5] 内")
    if not 0 < args.max_runtime_seconds <= 300:
        parser.error("--max-runtime-seconds 必须在 (0, 300] 内")
    if not 5 <= args.stationary_frames <= 150:
        parser.error("--stationary-frames 必须在 [5, 150] 内")
    if not 1 <= args.stationary_min_runtime_s <= 60:
        parser.error("--stationary-min-runtime-s 必须在 [1, 60] 内")
    if not 0.5 <= args.movement_joint_deg <= 10:
        parser.error("--movement-joint-deg 必须在 [0.5, 10] 内")
    if not 1 <= args.movement_gripper_mm <= 20:
        parser.error("--movement-gripper-mm 必须在 [1, 20] 内")
    if not 0.01 <= args.stationary_joint_deg <= 0.5:
        parser.error("--stationary-joint-deg 必须在 [0.01, 0.5] 内")
    if not 0.01 <= args.stationary_gripper_mm <= 1:
        parser.error("--stationary-gripper-mm 必须在 [0.01, 1] 内")
    return args


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    previous_signal_handlers: dict[int, object] = {}

    def request_stop(signum, _frame):
        stop_event.set()
        raise StopRequested(f"收到信号 {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_signal_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    third_person_camera = LatestFrameCamera(
        args.third_person_camera, args.width, args.height, args.camera_fps, "第三视角"
    )
    wrist_camera = LatestFrameCamera(
        args.wrist_camera, args.width, args.height, args.camera_fps, "腕部"
    )
    piper: Piper | None = None
    heartbeat: PositionHeartbeat | None = None
    motion_started = False

    try:
        print(f"连接策略服务器 ws://{args.host}:{args.port} ……")
        policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        print(f"策略服务器 metadata: {policy.get_server_metadata()}")

        third_person_camera.start()
        wrist_camera.start()
        third_person_camera.wait_ready()
        wrist_camera.wait_ready()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        piper = Piper(args.can_name)
        piper.wait_feedback()

        # A crashed previous controller may have left Piper enabled.  Do not
        # allow stale CAN position mode to survive into this process.  Drag
        # teaching is the only enabled preflight mode that is intentionally
        # preserved so the operator can restore the initial pose.
        ctrl_mode, _arm_status, _move_mode, teach_status, motor_enable = piper.status()
        teach_active = ctrl_mode == 0x02 and teach_status == 0x01
        if any(motor_enable) and not teach_active:
            print("[PREFLIGHT] 检测到上一个进程遗留的电机使能，立即软件急停并失能。")
            piper.emergency_stop()
            if any(piper.status()[4]):
                raise RuntimeError("启动预检时无法确认六轴全部失能")

        # Trigger JAX compilation before ENABLE.  This inference is discarded,
        # and no Piper command is sent while it runs.
        warmup_sample = read_camera_sample(third_person_camera, wrist_camera)
        warmup_state = piper.read_state()
        draw_preview(warmup_sample, warmup_state, "WARMUP | commands=0 | no robot motion")
        print("[WARMUP] 正在预热 PI0.5；机械臂未接收动作，首次可能约需 10 秒……")
        warmup_started = time.monotonic()
        warmup_result = policy.infer(make_observation(warmup_sample, warmup_state, args.task))
        parse_action_chunk(warmup_result.get("actions"))
        print(f"[WARMUP] 完成，用时 {(time.monotonic() - warmup_started) * 1000.0:.0f}ms。")

        sample, state = wait_for_enable(
            piper,
            third_person_camera,
            wrist_camera,
            max_camera_skew_s=args.max_camera_skew_ms / 1000.0,
        )

        # First inference happens while Piper is still disabled.
        actions, infer_ms, corrections = infer_actions(
            policy,
            make_observation(sample, state, args.task),
            heartbeat=None,
            timeout_s=args.inference_timeout_s,
            max_joint_overshoot_rad=np.deg2rad(args.max_joint_overshoot_deg),
            max_gripper_overshoot_m=args.max_gripper_overshoot_mm / 1000.0,
        )
        if corrections:
            print("[WARN] 首个动作小幅越界，已饱和到硬限位：" + "; ".join(corrections))
        initial_joint_jump = np.rad2deg(np.abs(actions[0, :6] - state[:6]))
        initial_gripper_jump_mm = abs(float(actions[0, 6] - state[6])) * 1000.0
        print(
            "\n首个动作预检："
            f"infer={infer_ms:.0f}ms, joint_jump(deg)={initial_joint_jump.round(2).tolist()}, "
            f"gripper_jump={initial_gripper_jump_mm:.1f}mm"
        )
        if float(np.max(initial_joint_jump)) > args.max_initial_jump_deg:
            raise RuntimeError("首个策略动作与当前关节姿态差值过大，拒绝使能")
        if initial_gripper_jump_mm > args.max_initial_gripper_jump_mm:
            raise RuntimeError("首个策略动作与当前夹爪位置差值过大，拒绝使能")

        if not args.enable_motion:
            print("干运行通过：未传 --enable-motion，机械臂始终未使能。")
            return 0

        initial_target = clamp_target(state)
        piper.enable(initial_target, args.robot_speed_percent)
        motion_started = True
        heartbeat = PositionHeartbeat(
            piper,
            initial_target,
            control_hz=args.control_hz,
            tracking_error_deg=args.tracking_error_deg,
            tracking_error_gripper_m=args.tracking_error_gripper_mm / 1000.0,
            stop_event=stop_event,
        )
        heartbeat.start()
        print(
            f"[RUN] 已使能：模型 horizon=16，每轮执行前 {args.execute_steps} 步，"
            f"动作频率={args.policy_hz:g}Hz。只有 Ctrl+C 会退出并失能。"
        )
        stationary_detector = StationaryDetector(
            initial_target,
            required_frames=args.stationary_frames,
            minimum_runtime_s=args.stationary_min_runtime_s,
            movement_joint_rad=np.deg2rad(args.movement_joint_deg),
            movement_gripper_m=args.movement_gripper_mm / 1000.0,
            stationary_joint_rad=np.deg2rad(args.stationary_joint_deg),
            stationary_gripper_m=args.stationary_gripper_mm / 1000.0,
        )
        print(
            f"[AUTO-HOLD] 有效运动后，目标与实测状态连续 {args.stationary_frames} 帧静止"
            "即停止模型推理，但保持 Piper 使能和 50Hz 位置保持。"
        )

        action_period = 1.0 / args.policy_hz
        run_started = time.monotonic()
        cycle_index = 0
        target = initial_target
        hold_reason: str | None = None
        while not stop_event.is_set():
            if time.monotonic() - run_started >= args.max_runtime_seconds:
                hold_reason = f"达到最大推理时长 {args.max_runtime_seconds:g}s"
                break

            limited_count = 0
            for action_index, desired in enumerate(actions[: args.execute_steps]):
                tick = time.monotonic()
                heartbeat.raise_if_failed()
                target, was_limited = limit_target_step(
                    target,
                    desired,
                    max_joint_step_rad=np.deg2rad(args.max_joint_step_deg),
                    max_gripper_step_m=args.max_gripper_step_mm / 1000.0,
                )
                limited_count += int(was_limited)
                heartbeat.set_target(target)

                current_state = piper.read_state()
                if stationary_detector.update(
                    target,
                    current_state,
                    time.monotonic() - run_started,
                ):
                    hold_reason = f"连续 {args.stationary_frames} 帧静止"
                    break
                try:
                    current_sample = read_camera_sample(third_person_camera, wrist_camera)
                    safe, reason = camera_sample_is_safe(
                        current_sample, max_skew_s=args.max_camera_skew_ms / 1000.0
                    )
                except Exception as exc:
                    hold_reason = f"相机读取失败: {exc}"
                    break
                if not safe:
                    hold_reason = reason
                    break
                key = draw_preview(
                    current_sample,
                    current_state,
                    f"RUN cycle={cycle_index} action={action_index + 1}/{args.execute_steps}",
                )
                if key in (27, ord("q")):
                    hold_reason = "用户从预览窗口请求停止模型推理"
                    break

                remaining = action_period - (time.monotonic() - tick)
                if remaining > 0 and stop_event.wait(remaining):
                    raise StopRequested("收到停止请求")

            if hold_reason is not None:
                break

            current_state = piper.read_state()
            try:
                current_sample = read_camera_sample(third_person_camera, wrist_camera)
                safe, reason = camera_sample_is_safe(
                    current_sample, max_skew_s=args.max_camera_skew_ms / 1000.0
                )
            except Exception as exc:
                hold_reason = f"相机读取失败: {exc}"
                break
            if not safe:
                hold_reason = reason
                break
            try:
                actions, infer_ms, corrections = infer_actions(
                    policy,
                    make_observation(current_sample, current_state, args.task),
                    heartbeat=heartbeat,
                    timeout_s=args.inference_timeout_s,
                    max_joint_overshoot_rad=np.deg2rad(args.max_joint_overshoot_deg),
                    max_gripper_overshoot_m=args.max_gripper_overshoot_mm / 1000.0,
                )
            except PolicyHoldRequested as exc:
                hold_reason = str(exc)
                break
            if corrections:
                print("\n[WARN] 动作小幅越界，已饱和到硬限位：" + "; ".join(corrections))
            print(
                f"\r[RUN] cycle={cycle_index:03d} infer={infer_ms:.0f}ms "
                f"limited={limited_count}/{args.execute_steps} "
                f"joint(deg)={np.rad2deg(current_state[:6]).round(1).tolist()} "
                f"grip={current_state[6] * 1000:.1f}mm    ",
                end="",
                flush=True,
            )
            cycle_index += 1

        heartbeat.raise_if_failed()
        if hold_reason is not None and not stop_event.is_set():
            print(
                f"\n[HOLD] {hold_reason}；模型推理已停止。"
                "Piper 仍保持使能并以 50Hz 保持最后目标。"
                "确认机械臂安全后，在本终端按 Ctrl+C 才会退出并失能。"
            )
            while not stop_event.is_set():
                heartbeat.raise_if_failed()
                current_state = piper.read_state()
                try:
                    current_sample = read_camera_sample(third_person_camera, wrist_camera)
                    draw_preview(
                        current_sample,
                        current_state,
                        "HOLD | inference stopped | Piper enabled | Ctrl+C to exit",
                    )
                except Exception:
                    # Camera preview is not part of the 50Hz position-hold loop.
                    pass
                stop_event.wait(0.05)
        return 0
    except (KeyboardInterrupt, StopRequested) as exc:
        print(f"\n[STOP] {exc}")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        if heartbeat is not None:
            heartbeat.stop()
        if piper is not None:
            if motion_started:
                try:
                    piper.emergency_stop()
                except Exception as exc:
                    print(f"\n[ERROR] 急停失败: {exc}", file=sys.stderr)
            piper.close()
        third_person_camera.stop()
        wrist_camera.stop()
        cv2.destroyAllWindows()
        for signum, previous in previous_signal_handlers.items():
            signal.signal(signum, previous)


if __name__ == "__main__":
    raise SystemExit(main())
