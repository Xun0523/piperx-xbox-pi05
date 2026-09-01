#!/usr/bin/env python3
"""Safely teleoperate PiperX with an Xbox controller and record a LeRobot dataset.

The collector records the *absolute command actually sent to the robot*, not raw
gamepad axes. Joint values use radians and gripper values use meters.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np

# Pygame should not require an audio device or a visible window.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from piper_sdk import C_PiperInterface_V2  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_HOME = ROOT / "data" / "lerobot"
DEFAULT_REPO_ID = "local/piperx_pi05"
DEFAULT_BASE_CAMERA = os.environ.get("PIPERX_THIRD_PERSON_CAMERA", "/dev/video0")
DEFAULT_WRIST_CAMERA = os.environ.get("PIPERX_WRIST_CAMERA", "/dev/video2")

# Piper SDK hard limits. Using the exact limits is important at the home pose,
# where J2/J3 can legitimately be 0 degrees: clamping to an inset margin would
# create an unexpected command as soon as the arm is enabled.
JOINT_LOWER = np.array([-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439], dtype=np.float32)
JOINT_UPPER = np.array([2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439], dtype=np.float32)
GRIPPER_MIN_M = 0.0
GRIPPER_MAX_M = 0.07


def rescale_deadzone(value: float, deadzone: float) -> float:
    """Map an axis outside the deadzone smoothly back to [-1, 1]."""
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return float(np.sign(value) * (magnitude - deadzone) / (1.0 - deadzone))


def xbox_delta(
    axes: list[float],
    hat: tuple[int, int],
    *,
    dt: float,
    joint_speed_rad_s: float,
    gripper_speed_m_s: float,
    deadzone: float,
) -> np.ndarray:
    """Convert Linux SDL Xbox axes into [j1..j6, gripper] increments."""
    if len(axes) < 6:
        raise ValueError(f"Xbox 控制器应至少有 6 个轴，实际为 {len(axes)}")

    left_x = rescale_deadzone(axes[0], deadzone)
    left_y = rescale_deadzone(axes[1], deadzone)
    right_x = rescale_deadzone(axes[3], deadzone)
    right_y = rescale_deadzone(axes[4], deadzone)
    left_trigger = np.clip((axes[2] + 1.0) * 0.5, 0.0, 1.0)
    right_trigger = np.clip((axes[5] + 1.0) * 0.5, 0.0, 1.0)

    delta = np.zeros(7, dtype=np.float32)
    delta[0] = -left_x * joint_speed_rad_s * dt
    delta[1] = -left_y * joint_speed_rad_s * dt
    delta[2] = right_y * joint_speed_rad_s * dt
    delta[3] = float(hat[0]) * joint_speed_rad_s * dt
    delta[4] = -float(hat[1]) * joint_speed_rad_s * dt
    delta[5] = right_x * joint_speed_rad_s * dt
    delta[6] = (right_trigger - left_trigger) * gripper_speed_m_s * dt
    return delta


def clamp_target(target: np.ndarray) -> np.ndarray:
    result = np.asarray(target, dtype=np.float32).copy()
    result[:6] = np.clip(result[:6], JOINT_LOWER, JOINT_UPPER)
    result[6] = np.clip(result[6], GRIPPER_MIN_M, GRIPPER_MAX_M)
    return result


class LatestFrameCamera:
    def __init__(self, path: str, width: int, height: int, fps: int, name: str):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self._capture: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._timestamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        capture = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开{self.name}相机: {self.path}")
        self._capture = capture
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._capture is not None
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError(f"{self.name}相机读取失败")
                timestamp = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._timestamp = timestamp
        except Exception as exc:  # surfaced by get()
            self._error = exc

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._error is not None:
                raise self._error
            with self._lock:
                if self._frame is not None:
                    return
            time.sleep(0.02)
        raise TimeoutError(f"等待{self.name}相机首帧超时")

    def get(self, max_age_s: float = 0.25) -> tuple[np.ndarray, float]:
        if self._error is not None:
            raise self._error
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"{self.name}相机尚无图像")
            frame = self._frame.copy()
            timestamp = self._timestamp
        age = time.monotonic() - timestamp
        if age > max_age_s:
            raise RuntimeError(f"{self.name}相机帧过旧: {age:.3f}s")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._capture is not None:
            self._capture.release()


class XboxController:
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LB = 4
    BUTTON_START = 7

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        pygame.event.pump()
        if pygame.joystick.get_count() != 1:
            raise RuntimeError(f"需要且只允许 1 个 Xbox 手柄，检测到 {pygame.joystick.get_count()} 个")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        if self.joystick.get_numaxes() < 6:
            raise RuntimeError("检测到的控制器没有完整的 Xbox 6 轴输入")
        self._previous = [False] * self.joystick.get_numbuttons()

    def sample(self) -> tuple[list[float], tuple[int, int], list[bool], set[int]]:
        pygame.event.pump()
        axes = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() else (0, 0)
        buttons = [bool(self.joystick.get_button(i)) for i in range(self.joystick.get_numbuttons())]
        rising = {i for i, value in enumerate(buttons) if value and not self._previous[i]}
        self._previous = buttons
        return axes, hat, buttons, rising

    def close(self) -> None:
        self.joystick.quit()
        pygame.quit()


class Piper:
    def __init__(self, can_name: str):
        self.interface = C_PiperInterface_V2(
            can_name,
            True,
            True,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
        self.interface.ConnectPort()
        self.motion_enabled = False
        self._ever_enabled = False
        self.speed_percent = 10
        self._last_feedback_timestamp = -1.0
        self._last_feedback_seen = time.monotonic()

    def wait_feedback(self, timeout_s: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.read_state(require_fresh=False)
            if self.interface.GetArmJointMsgs().Hz > 1.0:
                return state
            time.sleep(0.05)
        raise TimeoutError("未收到 Piper 关节反馈；检查机械臂电源、急停和 can0")

    def read_state(self, require_fresh: bool = True) -> np.ndarray:
        joint_wrapper = self.interface.GetArmJointMsgs()
        gripper_wrapper = self.interface.GetArmGripperMsgs()
        joint = joint_wrapper.joint_state
        raw_joints = np.array(
            [joint.joint_1, joint.joint_2, joint.joint_3, joint.joint_4, joint.joint_5, joint.joint_6],
            dtype=np.float64,
        )
        state = np.empty(7, dtype=np.float32)
        state[:6] = np.deg2rad(raw_joints * 0.001)
        state[6] = float(gripper_wrapper.gripper_state.grippers_angle) * 1e-6

        feedback_timestamp = float(joint_wrapper.time_stamp)
        if feedback_timestamp != self._last_feedback_timestamp:
            self._last_feedback_timestamp = feedback_timestamp
            self._last_feedback_seen = time.monotonic()
        if require_fresh and time.monotonic() - self._last_feedback_seen > 0.25:
            raise RuntimeError("Piper 关节反馈超过 250ms 未更新，停止发送动作")
        return state

    def enable(self, initial_target: np.ndarray, speed_percent: int) -> None:
        self.speed_percent = speed_percent
        # B leaves Piper in software emergency stop. Recover it before enabling
        # CAN position control; this command does not contain a motion target.
        self.interface.MotionCtrl_1(0x02, 0, 0)
        self.interface.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
        time.sleep(0.15)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.interface.EnablePiper():
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("Piper 电机使能超时")
        self.motion_enabled = True
        self._ever_enabled = True
        self.send(initial_target)
        time.sleep(0.15)
        status = self.status()
        if status[0] != 0x01 or status[1] != 0x00 or not all(status[4]):
            self._disable_and_wait()
            raise RuntimeError(
                "Piper 未进入可控状态: "
                f"ctrl=0x{status[0]:02X} arm=0x{status[1]:02X} "
                f"mode=0x{status[2]:02X} enabled={status[4]}"
            )

    def send(self, target: np.ndarray) -> None:
        joints_mdeg = np.rint(np.rad2deg(target[:6]) * 1000.0).astype(int)
        gripper_um = int(round(float(target[6]) * 1e6))
        # Piper's official joint-control demo refreshes mode 0x151 on every
        # control cycle before sending 0x155..0x157 joint targets.
        self.interface.ModeCtrl(0x01, 0x01, self.speed_percent, 0x00)
        self.interface.JointCtrl(*joints_mdeg.tolist())
        self.interface.GripperCtrl(gripper_um, 2000, 0x01, 0)

    def status(self) -> tuple[int, int, int, int, list[bool]]:
        status = self.interface.GetArmStatus().arm_status
        return (
            int(status.ctrl_mode),
            int(status.arm_status),
            int(status.mode_feed),
            int(status.teach_status),
            self.interface.GetArmEnableStatus(),
        )

    def emergency_stop(self) -> None:
        try:
            self.interface.MotionCtrl_1(0x01, 0, 0)
        finally:
            self._disable_and_wait()

    def _disable_and_wait(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not any(self.interface.GetArmEnableStatus()):
                self.motion_enabled = False
                return
            self.interface.DisablePiper()
            time.sleep(0.05)
        self.motion_enabled = any(self.interface.GetArmEnableStatus())

    def close(self) -> None:
        try:
            if self._ever_enabled and any(self.interface.GetArmEnableStatus()):
                self._disable_and_wait()
        finally:
            self.interface.DisconnectPort()


def dataset_features(height: int, width: int) -> dict:
    image_feature = {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
    }
    names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
    return {
        "observation.images.third_person": image_feature,
        "observation.images.wrist": image_feature.copy(),
        "observation.state": {"dtype": "float32", "shape": (7,), "names": names},
        "action": {"dtype": "float32", "shape": (7,), "names": names},
        "observation.deadman": {"dtype": "float32", "shape": (1,), "names": ["deadman"]},
        "observation.camera_skew_s": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["camera_skew_s"],
        },
    }


def reset_episode_buffer(dataset: LeRobotDataset) -> None:
    """Reset recording state, including freshly loaded resume datasets.

    LeRobotDataset.create() creates an episode buffer, while loading an
    existing dataset for --resume deliberately leaves episode_buffer as None.
    Its clear_episode_buffer() method does not handle that None state.
    """
    if dataset.episode_buffer is None:
        dataset.episode_buffer = dataset.create_episode_buffer()
    else:
        dataset.clear_episode_buffer()


def open_dataset(args: argparse.Namespace) -> LeRobotDataset:
    dataset_path = args.data_home / args.repo_id
    if dataset_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"数据集已存在: {dataset_path}\n"
                "如需追加 episode，请加 --resume；如需新数据集，请换 --repo-id。"
            )
        dataset = LeRobotDataset(args.repo_id, root=dataset_path, video_backend="pyav")
        dataset.start_image_writer(num_threads=4)
        reset_episode_buffer(dataset)
        return dataset
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_path,
        robot_type="agilex_piperx",
        fps=args.fps,
        features=dataset_features(args.height, args.width),
        use_videos=True,
        video_backend="pyav",
        image_writer_threads=4,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_DATA_HOME)
    parser.add_argument("--task", required=True, help="固定、简短的英文任务指令")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--can-name", default="can0")
    parser.add_argument("--base-camera", default=DEFAULT_BASE_CAMERA)
    parser.add_argument("--wrist-camera", default=DEFAULT_WRIST_CAMERA)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--max-episode-seconds", type=float, default=60.0)
    parser.add_argument("--joint-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--gripper-speed-m-s", type=float, default=0.025)
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--robot-speed-percent", type=int, default=20)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--teleop-only",
        action="store_true",
        help="只测试相机、手柄和机械臂遥操作，不创建或写入数据集",
    )
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.fps <= args.camera_fps:
        parser.error("--fps 必须在 1 和 --camera-fps 之间")
    if not args.fps <= args.control_hz <= 200:
        parser.error("--control-hz 必须不低于 --fps 且不高于 200")
    if not 1 <= args.robot_speed_percent <= 30:
        parser.error("桌面采集限制 --robot-speed-percent 为 1..30")
    if not 0 < args.joint_speed_deg_s <= 20:
        parser.error("桌面采集限制 --joint-speed-deg-s 为 (0, 20]")
    if not 0 < args.max_episode_seconds <= 300:
        parser.error("--max-episode-seconds 必须在 (0, 300] 内")
    if args.teleop_only and not args.enable_motion:
        parser.error("--teleop-only 必须和 --enable-motion 一起使用")
    return args


def main() -> int:
    args = parse_args()
    if args.can_name != "can0":
        raise ValueError("当前安全配置只允许 can0")

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    base_camera = LatestFrameCamera(
        args.base_camera, args.width, args.height, args.camera_fps, "第三视角"
    )
    wrist_camera = LatestFrameCamera(
        args.wrist_camera, args.width, args.height, args.camera_fps, "腕部"
    )
    xbox: XboxController | None = None
    piper: Piper | None = None
    dataset: LeRobotDataset | None = None
    emergency = False

    try:
        base_camera.start()
        wrist_camera.start()
        base_camera.wait_ready()
        wrist_camera.wait_ready()
        # Initialise OpenCV's GUI before LeRobot starts its asynchronous image
        # writer threads.  Some Linux Qt/X11 combinations otherwise create the
        # preview window but never repaint its (black) client area.
        if args.display:
            initial_base_rgb, _ = base_camera.get()
            initial_wrist_rgb, _ = wrist_camera.get()
            initial_preview = np.concatenate(
                [
                    cv2.cvtColor(initial_base_rgb, cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(initial_wrist_rgb, cv2.COLOR_RGB2BGR),
                ],
                axis=1,
            )
            cv2.namedWindow("PiperX: third-person | wrist", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("PiperX: third-person | wrist", initial_preview)
            cv2.waitKey(30)
        xbox = XboxController()
        piper = Piper(args.can_name)
        state = piper.wait_feedback()
        target = clamp_target(state)

        if args.enable_motion:
            ctrl_mode, _arm_status, _move_mode, teach_status, _motor_enable = piper.status()
            if ctrl_mode == 0x02 and teach_status == 0x01:
                raise RuntimeError(
                    "Piper 正在拖动示教录制（ctrl=2, teach=1）。"
                    "请托住机械臂并再次按绿色示教按钮退出，确认绿灯熄灭后重试。"
                )
            print(
                "\n即将恢复软件急停并使能 CAN 控制。"
                "确认急停可触达、工作区无人、低速空间安全。"
            )
            confirmation = input("请输入 ENABLE 后回车: ").strip()
            if confirmation != "ENABLE":
                print("未确认，保持只读并退出。")
                return 2
            if not args.teleop_only:
                dataset = open_dataset(args)
                print(f"数据目录: {dataset.root}")
            piper.enable(target, args.robot_speed_percent)
        else:
            print("\n当前为只读预检模式，不会使能或发送机械臂动作。")

        print(f"任务指令: {args.task}")
        print(
            f"速度映射: 摇杆满推={args.joint_speed_deg_s:g}°/s | "
            f"Piper 速度上限={args.robot_speed_percent}% | 控制={args.control_hz} Hz"
        )
        if args.teleop_only:
            print("遥操作测试: LB按住才运动 | B急停 | START退出；不会记录数据")
        elif dataset is None:
            print("只读控制: START退出；手柄、相机和机械臂反馈将持续显示")
        else:
            print("控制: LB按住才运动 | A开始episode | Y保存 | X丢弃 | B急停 | START退出")

        period = 1.0 / args.control_hz
        record_period = 1.0 / args.fps
        next_tick = time.monotonic()
        next_record_tick = next_tick
        last_tick = next_tick
        last_status = 0.0
        recording = False
        episode_frames = 0
        episode_started = 0.0
        deadman_was_down = False
        black_frame_streak = 0

        while not stop_requested:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            dt = min(max(tick - last_tick, 0.0), 2.0 * period)
            last_tick = tick
            next_tick = max(next_tick + period, tick)

            axes, hat, buttons, rising = xbox.sample()
            state = piper.read_state()
            deadman = buttons[XboxController.BUTTON_LB]

            if XboxController.BUTTON_B in rising:
                if args.enable_motion:
                    print("\nB 急停触发")
                    piper.emergency_stop()
                    emergency = True
                else:
                    print("\n只读预检退出（未发送急停或失能指令）")
                break
            if XboxController.BUTTON_START in rising and not recording:
                break
            if XboxController.BUTTON_A in rising and not recording:
                if dataset is None:
                    print("\n当前测试模式不会开始或写入 episode。")
                else:
                    reset_episode_buffer(dataset)
                    recording = True
                    episode_frames = 0
                    episode_started = tick
                    next_record_tick = tick
                    print(f"\n[REC] episode {dataset.meta.total_episodes} 开始")
            if XboxController.BUTTON_X in rising and recording:
                assert dataset is not None
                reset_episode_buffer(dataset)
                recording = False
                print("\n[DROP] 当前 episode 已丢弃")
            if XboxController.BUTTON_Y in rising and recording:
                assert dataset is not None
                if episode_frames < max(5, args.fps):
                    print("\nepisode 少于 1 秒，拒绝保存；继续录制或按 X 丢弃")
                else:
                    print("\n正在编码并保存两路视频……")
                    dataset.save_episode()
                    recording = False
                    print(f"[SAVE] 已保存，当前共 {dataset.meta.total_episodes} 个 episodes")

            if deadman and not deadman_was_down:
                target = clamp_target(state)
            if deadman:
                target = clamp_target(
                    target
                    + xbox_delta(
                        axes,
                        hat,
                        dt=dt,
                        joint_speed_rad_s=np.deg2rad(args.joint_speed_deg_s),
                        gripper_speed_m_s=args.gripper_speed_m_s,
                        deadzone=args.deadzone,
                    )
                )
            elif deadman_was_down:
                # On deadman release, discard any unexecuted command error and
                # hold the measured pose instead of continuing toward an old target.
                target = clamp_target(state)
            if args.enable_motion:
                # Piper expects its CAN/MOVE J mode and position target to be
                # refreshed continuously. LB gates target changes, not the
                # position-hold heartbeat.
                piper.send(target)
            deadman_was_down = deadman

            base_rgb, base_ts = base_camera.get()
            wrist_rgb, wrist_ts = wrist_camera.get()
            camera_skew = abs(base_ts - wrist_ts)
            if camera_skew > 0.12:
                raise RuntimeError(f"两相机时间偏差过大: {camera_skew:.3f}s")
            base_brightness = float(base_rgb.mean())
            wrist_brightness = float(wrist_rgb.mean())
            if min(base_brightness, wrist_brightness) < 2.0:
                black_frame_streak += 1
            else:
                black_frame_streak = 0
            if black_frame_streak >= args.control_hz:
                raise RuntimeError(
                    "相机连续 1 秒返回近乎全黑画面；"
                    "当前 episode 将不会保存，请检查 USB 和镜头"
                )

            if recording and tick >= next_record_tick:
                assert dataset is not None
                dataset.add_frame(
                    {
                        # The asynchronous writer owns these copies; preview
                        # rendering keeps its own arrays in the main thread.
                        "observation.images.third_person": base_rgb.copy(),
                        "observation.images.wrist": wrist_rgb.copy(),
                        "observation.state": state.astype(np.float32),
                        "action": target.astype(np.float32),
                        "observation.deadman": np.array([float(deadman)], dtype=np.float32),
                        "observation.camera_skew_s": np.array([camera_skew], dtype=np.float32),
                        "task": args.task,
                    }
                )
                episode_frames += 1
                next_record_tick = max(next_record_tick + record_period, tick)
                if tick - episode_started >= args.max_episode_seconds:
                    print("\n达到 episode 时长上限，自动保存……")
                    dataset.save_episode()
                    recording = False

            if args.display:
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
                status_text = "LB: MOVE" if deadman and args.enable_motion else "HOLD"
                cv2.putText(
                    preview,
                    status_text,
                    (16, args.height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if deadman else (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("PiperX: third-person | wrist", preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    stop_requested = True

            if tick - last_status >= 0.5:
                mode = "REC" if recording else "READY"
                motion = "MOVE" if deadman and args.enable_motion else "HOLD"
                ctrl_mode, arm_status, move_mode, teach_status, motor_enable = piper.status()
                target_error_deg = np.rad2deg(target[:6] - state[:6])
                if args.enable_motion and not all(motor_enable):
                    piper.emergency_stop()
                    emergency = True
                    raise RuntimeError("Piper 六轴使能在运行中丢失，已急停并退出")
                print(
                    f"\r[{mode}/{motion}] frames={episode_frames:04d} "
                    f"axis={np.round(axes[:6], 2).tolist()} "
                    f"cmd_err(deg)={target_error_deg.round(1).tolist()} "
                    f"ctrl={ctrl_mode}/arm={arm_status}/move={move_mode}/teach={teach_status} "
                    f"enabled={int(all(motor_enable))} grip={state[6] * 1000:.1f}mm "
                    f"brightness={base_brightness:.0f}/{wrist_brightness:.0f} "
                    f"skew={camera_skew * 1000:.0f}ms",
                    end="",
                    flush=True,
                )
                last_status = tick

        if recording and dataset is not None:
            reset_episode_buffer(dataset)
            print("\n退出时当前未完成 episode 已丢弃。")
        return 130 if emergency else 0
    finally:
        if dataset is not None:
            dataset.stop_image_writer()
        if piper is not None:
            piper.close()
        if xbox is not None:
            xbox.close()
        base_camera.stop()
        wrist_camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        raise
