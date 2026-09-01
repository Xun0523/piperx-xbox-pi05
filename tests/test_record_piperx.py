import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_piperx_lerobot.py"
SPEC = importlib.util.spec_from_file_location("record_piperx_lerobot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stationary_xbox_has_zero_delta():
    delta = MODULE.xbox_delta(
        [0.0, 0.0, -1.0, 0.0, 0.0, -1.0],
        (0, 0),
        dt=0.1,
        joint_speed_rad_s=1.0,
        gripper_speed_m_s=0.1,
        deadzone=0.12,
    )
    np.testing.assert_allclose(delta, np.zeros(7))


def test_xbox_mapping_and_trigger_direction():
    delta = MODULE.xbox_delta(
        [1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        (1, -1),
        dt=0.1,
        joint_speed_rad_s=1.0,
        gripper_speed_m_s=0.1,
        deadzone=0.12,
    )
    np.testing.assert_allclose(delta[:6], [-0.1, 0.1, -0.1, 0.1, 0.1, 0.1])
    assert delta[6] < 0


def test_target_is_clamped_to_robot_limits():
    result = MODULE.clamp_target(np.full(7, 999.0, dtype=np.float32))
    np.testing.assert_allclose(result[:6], MODULE.JOINT_UPPER, rtol=1e-6)
    assert np.isclose(result[6], MODULE.GRIPPER_MAX_M)


def test_reset_episode_buffer_handles_resumed_dataset():
    class FakeDataset:
        episode_buffer = None
        cleared = False

        def create_episode_buffer(self):
            return {"episode_index": 2, "size": 0}

        def clear_episode_buffer(self):
            self.cleared = True

    dataset = FakeDataset()
    MODULE.reset_episode_buffer(dataset)

    assert dataset.episode_buffer == {"episode_index": 2, "size": 0}
    assert not dataset.cleared

    MODULE.reset_episode_buffer(dataset)
    assert dataset.cleared
