# PiperX + Xbox + Dual-Camera π0.5

An end-to-end pipeline for collecting LeRobot demonstrations, offline LoRA fine-tuning OpenPI π0.5, and running safety-gated real-robot inference with one AgileX Piper-X arm, an Xbox controller, and two RGB cameras.

No leader arm or dedicated teaching pendant is required. The Xbox controller is used to collect demonstrations; during deployment, the fine-tuned policy controls the robot directly.

> ⚠️ This repository is a research prototype, not a safety-certified industrial controller. Before operating the real robot, keep the physical emergency stop within reach, clear the workspace, and complete all low-speed preflight checks. A learned policy can always produce unexpected actions.

## Features

- Joint-space Xbox teleoperation: LB deadman switch, sticks/D-pad for six joints, and triggers for the gripper
- Synchronized third-person and wrist-camera recording
- LeRobot v2.1 datasets containing absolute joint/gripper targets, robot state, video, task text, and collection diagnostics
- OpenPI/JAX π0.5 LoRA integration with a 7-D Piper-X action adapter, dual-camera mapping, and PyAV decoding
- Single-GPU RTX 4090 training scripts with a default batch size of 16 and 20,000 optimization steps
- Real-robot inference with an `ENABLE` gate, first-action validation, hard-limit checks, per-step rate limiting, tracking-error monitoring, and action-chunk execution
- Position HOLD when inference finishes or recoverable failures occur; Ctrl+C exits HOLD and disables the robot

## Validated Setup

| Component | Configuration |
|---|---|
| Robot | AgileX Piper-X, 6 DoF + gripper, USB-CAN on `can0` |
| Teleoperation | Xbox Series Controller over USB or Bluetooth |
| Cameras | One third-person RGB camera + one wrist RGB camera, 640×480 at 30 Hz |
| Dataset | 50 episodes, 26,567 frames, 15 Hz, LeRobot v2.1 |
| Model | OpenPI π0.5 base with LoRA-style partial-parameter fine-tuning |
| Training | RTX 4090 24 GB, batch size 16, 20,000 steps |
| Inference | Horizon 16, execute the first 10 actions per chunk by default, 15 Hz |

The complete experiment record is available in Chinese at [`docs/pi05_piperx_training_setup_zh.md`](docs/pi05_piperx_training_setup_zh.md).

## Repository Layout

```text
.
├── openpi_overlay/       # Piper-X policy adapter copied into the pinned OpenPI commit
├── patches/              # OpenPI training config and LeRobot PyAV patch
├── scripts/              # CAN, teleoperation, recording, training, and deployment tools
├── tests/                # Unit tests that do not connect to the real robot
└── docs/                 # Experiment configuration and reproduction notes
```

The full OpenPI, Piper SDK, and Gamepad_PiPER source trees are intentionally not vendored. See [`NOTICE`](NOTICE) for pinned upstream revisions and licenses.

## 1. Installation

The validated environment uses Ubuntu 22.04, Python 3.11, a SocketCAN-compatible USB-CAN adapter, an NVIDIA GPU driver, and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Xun0523/piperx-xbox-pi05.git
cd piperx-xbox-pi05

git clone https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac

cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install piper_sdk==0.6.2 pygame==2.6.1 av==17.0.0
cd ..

./scripts/apply_openpi_overlay.sh
```

The patch is validated only against the pinned OpenPI commit above. Do not apply it blindly to an arbitrary newer revision.

## 2. Camera and CAN Configuration

Prefer stable `/dev/v4l/by-id/` camera paths instead of `/dev/videoN`, which may change after a reboot:

```bash
ls -l /dev/v4l/by-id/

export PIPERX_THIRD_PERSON_CAMERA=/dev/v4l/by-id/<third-person-camera>-video-index0
export PIPERX_WRIST_CAMERA=/dev/v4l/by-id/<wrist-camera>-video-index0
```

Bring up and validate CAN:

```bash
./scripts/setup_can.sh
./scripts/check_can_stability.sh 60
./scripts/check_piper_status.sh
```

To require a specific USB-CAN adapter, set its serial number:

```bash
export PIPERX_CAN_SERIAL=<serial-from-check_can_stability>
```

## 3. Input and Real-Robot Teleoperation Tests

First verify both cameras and the Xbox controller without opening CAN or moving the robot:

```bash
./scripts/test_cameras_xbox.sh
```

Then run the low-speed real-robot teleoperation test:

```bash
./scripts/test_teleop.sh
```

The robot is not enabled until the operator types `ENABLE` in the terminal. Xbox axis indices can vary across Linux controller drivers, so verify every axis and direction individually at low speed before collecting data.

## 4. Record a LeRobot Dataset

```bash
./scripts/record.sh \
  --task "put the yellow rubber duck onto the white plate" \
  --enable-motion \
  --display \
  --robot-speed-percent 20 \
  --joint-speed-deg-s 10
```

Controller mapping:

| Input | Function |
|---|---|
| Hold LB | Allow target updates; releasing LB holds the measured pose |
| Left stick | J1 / J2 |
| Right stick | J6 / J3 |
| D-pad | J4 / J5 |
| LT / RT | Close / open the gripper |
| A | Start an episode |
| Y | Save the current episode |
| X | Discard and re-record the current episode |
| B | Software emergency stop and exit |
| START | Exit while no episode is being recorded |

By default, data is written to `data/lerobot/local/piperx_pi05`. Add `--resume` to append episodes to an existing dataset.

## 5. Offline π0.5 LoRA Fine-Tuning

```bash
./scripts/train_pi05_piperx.sh
```

Common overrides:

```bash
BATCH_SIZE=8 NUM_TRAIN_STEPS=20000 EXP_NAME=my_run \
  ./scripts/train_pi05_piperx.sh
```

Resume an experiment with the same name:

```bash
EXP_NAME=my_run ./scripts/resume_pi05_bs16.sh
```

This is offline fine-tuning, not online parameter updates during robot execution. The training workflow retains checkpoint candidates every 1,000 steps. Treat `latest` as the recovery checkpoint and select `best` separately through held-out real-robot evaluation.

## 6. Load a Checkpoint and Run the Robot

Start the policy server in terminal 1:

```bash
CHECKPOINT=/absolute/path/to/checkpoint ./scripts/serve_pi05_piperx.sh
```

Run a dry test in terminal 2. This connects to the policy and cameras but never enables the robot:

```bash
./scripts/run_pi05_piperx.sh
```

After the dry run and camera-position checks pass, enable real-robot execution:

```bash
./scripts/run_pi05_piperx.sh --enable-motion --execute-steps 10
```

The client first warms up the model, then continuously displays both camera streams and robot state. After the operator types `ENABLE`, the first policy action is still checked while the motors remain disabled. Motion begins only if that check passes.

When the policy remains stationary, reaches its runtime limit, loses a camera, or encounters a recoverable inference error, model inference stops while the robot continues holding its last target. This avoids an uncontrolled drop. After confirming the scene is safe, press Ctrl+C to execute the software emergency stop and disable the motors. Lower-level safety faults can trigger an immediate emergency stop automatically.

## Tests

```bash
cd openpi
.venv/bin/python -m pytest ../tests \
  ../openpi_overlay/src/openpi/policies/piperx_policy_test.py
```

These tests do not send CAN commands, but they do not replace supervised low-speed validation on the real robot.

## Dataset and Model Release

The initial code release does not include the base model, training checkpoints, or the complete real-robot dataset. These artifacts are large and may contain images of the experiment environment. They should be released separately through Hugging Face Dataset and Model repositories after license and privacy review.

## Acknowledgements

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk)
- [kehuanjack/Gamepad_PiPER](https://github.com/kehuanjack/Gamepad_PiPER)

## License

This project is licensed under Apache-2.0. Third-party components remain subject to their respective licenses; see [`NOTICE`](NOTICE).
