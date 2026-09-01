import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pi05_piperx.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_pi05_piperx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sanitize_action_chunk_accepts_safe_horizon():
    actions = np.zeros((16, 7), dtype=np.float32)
    sanitized, corrections = MODULE.sanitize_action_chunk(actions)
    assert sanitized.shape == (16, 7)
    assert not corrections


def test_sanitize_action_chunk_rejects_wrong_shape():
    actions = np.zeros((10, 7), dtype=np.float32)
    try:
        MODULE.sanitize_action_chunk(actions)
    except RuntimeError as exc:
        assert "形状" in str(exc)
    else:
        raise AssertionError("wrong action horizon should be rejected")


def test_sanitize_action_chunk_clamps_small_gripper_overshoot():
    actions = np.zeros((16, 7), dtype=np.float32)
    actions[0, 6] = -0.0002
    sanitized, corrections = MODULE.sanitize_action_chunk(actions)
    assert sanitized[0, 6] == 0.0
    assert "gripper" in corrections[0]


def test_sanitize_action_chunk_rejects_large_gripper_overshoot_with_details():
    actions = np.zeros((16, 7), dtype=np.float32)
    actions[0, 6] = -0.01
    try:
        MODULE.sanitize_action_chunk(actions)
    except RuntimeError as exc:
        message = str(exc)
        assert "step=0" in message
        assert "gripper=-10.00mm" in message
        assert "10.00mm" in message
    else:
        raise AssertionError("large gripper overshoot should be rejected")


def test_sanitize_action_chunk_clamps_small_joint_overshoot():
    actions = np.zeros((16, 7), dtype=np.float32)
    actions[0, 1] = -np.deg2rad(0.5)
    sanitized, corrections = MODULE.sanitize_action_chunk(actions)
    assert sanitized[0, 1] == MODULE.JOINT_LOWER[1]
    assert "J2" in corrections[0]


def test_limit_target_step_caps_joint_and_gripper_speed():
    previous = np.zeros(7, dtype=np.float32)
    desired = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 0.05], dtype=np.float32)
    target, limited = MODULE.limit_target_step(
        previous,
        desired,
        max_joint_step_rad=np.deg2rad(0.8),
        max_gripper_step_m=0.002,
    )
    assert limited
    np.testing.assert_allclose(target[:2], np.deg2rad([0.8, 0.8]), rtol=1e-6)
    # J3 has an upper hard limit of zero, so its negative step is allowed.
    assert np.isclose(target[2], -np.deg2rad(0.8))
    assert np.isclose(target[6], 0.002)


def make_stationary_detector(required_frames=3):
    return MODULE.StationaryDetector(
        np.zeros(7, dtype=np.float32),
        required_frames=required_frames,
        minimum_runtime_s=1.0,
        movement_joint_rad=np.deg2rad(2.0),
        movement_gripper_m=0.005,
        stationary_joint_rad=np.deg2rad(0.1),
        stationary_gripper_m=0.0002,
    )


def test_stationary_detector_does_not_stop_before_real_motion():
    detector = make_stationary_detector()
    pose = np.zeros(7, dtype=np.float32)
    assert not any(detector.update(pose, pose, step * 0.1) for step in range(50))
    assert not detector.movement_seen


def test_stationary_detector_requires_consecutive_still_frames():
    detector = make_stationary_detector(required_frames=3)
    moving = np.zeros(7, dtype=np.float32)
    moving[0] = np.deg2rad(3.0)
    detector.update(moving, moving, 1.0)
    assert detector.movement_seen
    assert not detector.update(moving, moving, 1.1)
    assert not detector.update(moving, moving, 1.2)
    assert detector.update(moving, moving, 1.3)


def test_stationary_detector_resets_count_when_observed_pose_moves():
    detector = make_stationary_detector(required_frames=3)
    moving = np.zeros(7, dtype=np.float32)
    moving[0] = np.deg2rad(3.0)
    detector.update(moving, moving, 1.0)
    detector.update(moving, moving, 1.1)
    changed_state = moving.copy()
    changed_state[0] += np.deg2rad(0.5)
    assert not detector.update(moving, changed_state, 1.2)
    assert detector.stationary_frames == 0
