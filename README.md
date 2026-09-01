# PiperX + Xbox + 双相机 + π0.5

使用一台 AgileX Piper-X、一个 Xbox 手柄和两路 RGB 相机，完成 LeRobot 数据采集、OpenPI π0.5 LoRA 离线微调以及带安全门控的真机推理。

本项目不需要额外的 leader arm 或专用示教器。Xbox 只用于采集示范；部署阶段由微调后的策略直接控制机械臂。

> ⚠️ 本仓库是研究原型，不是经过安全认证的工业控制器。运行真机前必须确保物理急停可触达、工作区无人，并先完成低速预检。任何学习策略都可能输出不可预期动作。

## 功能

- Xbox 关节空间遥操作：LB 死手开关，摇杆/十字键控制 6 个关节，扳机控制夹爪
- 第三视角 + 腕部相机同步采集
- LeRobot v2.1 数据集：绝对关节/夹爪目标、状态、视频、任务文本和采集诊断量
- OpenPI/JAX π0.5 LoRA：7 维 Piper-X 动作适配、双相机映射、PyAV 解码
- RTX 4090 单卡训练脚本，默认 batch size 16、20,000 steps
- 真机推理：ENABLE 门控、首次动作预检、硬限位检查、逐步限速、跟踪误差监控和动作 chunk 执行
- 策略结束或异常时进入位置 HOLD；只有 Ctrl+C 才执行急停并失能

## 已验证配置

| 项目 | 配置 |
|---|---|
| 机械臂 | AgileX Piper-X，6 DoF + gripper，USB-CAN `can0` |
| 示教输入 | Xbox Series Controller，USB/蓝牙 |
| 相机 | 1×第三视角 + 1×腕部 RGB，640×480@30 Hz |
| 数据 | 50 episodes，26,567 frames，15 Hz，LeRobot v2.1 |
| 模型 | OpenPI π0.5 base，LoRA-style 部分参数微调 |
| 训练 | RTX 4090 24 GB，batch 16，20,000 steps |
| 推理 | horizon 16，每轮默认执行前 10 步，15 Hz |

完整实验设置见 [`docs/pi05_piperx_training_setup_zh.md`](docs/pi05_piperx_training_setup_zh.md)。

## 仓库结构

```text
.
├── openpi_overlay/       # Piper-X policy adapter，复制到指定 OpenPI commit
├── patches/              # OpenPI 训练配置与 LeRobot PyAV 适配补丁
├── scripts/              # CAN、遥操作、采集、训练和部署脚本
├── tests/                # 不连接真机的单元测试
└── docs/                 # 实验配置和复现说明
```

OpenPI、Piper SDK 和 Gamepad_PiPER 不直接复制进本仓库。固定的上游版本和许可证见 [`NOTICE`](NOTICE)。

## 1. 安装

要求 Ubuntu 22.04、Python 3.11、支持 SocketCAN 的 USB-CAN、NVIDIA 驱动和 `uv`。

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

补丁只针对上面的 OpenPI commit 验证。不要直接应用到任意新版本。

## 2. 配置相机与 CAN

优先使用不会随重启变化的 `/dev/v4l/by-id/` 路径：

```bash
ls -l /dev/v4l/by-id/

export PIPERX_THIRD_PERSON_CAMERA=/dev/v4l/by-id/<third-person-camera>-video-index0
export PIPERX_WRIST_CAMERA=/dev/v4l/by-id/<wrist-camera>-video-index0
```

启用并检查 CAN：

```bash
./scripts/setup_can.sh
./scripts/check_can_stability.sh 60
./scripts/check_piper_status.sh
```

如需锁定指定 USB-CAN，可设置：

```bash
export PIPERX_CAN_SERIAL=<serial-from-check_can_stability>
```

## 3. 输入与真机遥操作测试

先在不连接 CAN、不运动机械臂的模式下检查手柄和相机：

```bash
./scripts/test_cameras_xbox.sh
```

再运行真机低速遥操作测试：

```bash
./scripts/test_teleop.sh
```

必须在终端输入 `ENABLE` 后机械臂才会使能。Xbox 在不同内核驱动下的轴编号可能不同，第一次运行必须逐轴低速确认方向。

## 4. 采集 LeRobot 数据

```bash
./scripts/record.sh \
  --task "put the yellow rubber duck onto the white plate" \
  --enable-motion \
  --display \
  --robot-speed-percent 20 \
  --joint-speed-deg-s 10
```

控制映射：

| 输入 | 功能 |
|---|---|
| LB（按住） | 允许更新动作目标；松开后保持实测姿态 |
| 左摇杆 | J1 / J2 |
| 右摇杆 | J6 / J3 |
| 十字键 | J4 / J5 |
| LT / RT | 关闭 / 打开夹爪 |
| A | 开始 episode |
| Y | 保存 episode |
| X | 丢弃并重录 |
| B | 软件急停并退出 |
| START | 未录制时退出 |

数据默认写入 `data/lerobot/local/piperx_pi05`。已有数据集继续采集时增加 `--resume`。

## 5. π0.5 LoRA 离线微调

```bash
./scripts/train_pi05_piperx.sh
```

常用覆盖参数：

```bash
BATCH_SIZE=8 NUM_TRAIN_STEPS=20000 EXP_NAME=my_run \
  ./scripts/train_pi05_piperx.sh
```

恢复同名实验：

```bash
EXP_NAME=my_run ./scripts/resume_pi05_bs16.sh
```

这一步是离线微调，不是边执行边更新参数的在线微调。训练脚本默认保留每 1,000 steps 的候选 checkpoint，`latest` 与最终 `best` 应通过独立真机评测区分。

## 6. 加载 checkpoint 并运行真机

终端一启动策略服务器：

```bash
CHECKPOINT=/absolute/path/to/checkpoint ./scripts/serve_pi05_piperx.sh
```

终端二先做干运行，不使能机械臂：

```bash
./scripts/run_pi05_piperx.sh
```

干运行及相机位置确认通过后，再运行真机：

```bash
./scripts/run_pi05_piperx.sh --enable-motion --execute-steps 10
```

流程会先预热模型，然后持续显示两路相机和机械臂状态。输入 `ENABLE` 后仍会在机械臂未使能时执行首个动作预检；通过后才开始运动。

策略连续静止、超时、相机异常或推理异常时，程序停止模型推理但保持最后目标和电机使能，防止机械臂突然下落。确认安全后按 Ctrl+C，程序才会执行软件急停并失能。

## 测试

```bash
cd openpi
.venv/bin/python -m pytest ../tests \
  ../openpi_overlay/src/openpi/policies/piperx_policy_test.py
```

这些测试不发送 CAN 动作，但不能替代真实机械臂的低速安全验证。

## 数据集与模型

第一版仓库不包含基础模型、训练 checkpoint 和完整真机数据集。它们体积较大，也可能包含实验环境图像；后续应通过 Hugging Face Dataset/Model 仓库独立发布，并附数据许可和隐私检查。

## 致谢

- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- [huggingface/lerobot](https://github.com/huggingface/lerobot)
- [agilexrobotics/piper_sdk](https://github.com/agilexrobotics/piper_sdk)
- [kehuanjack/Gamepad_PiPER](https://github.com/kehuanjack/Gamepad_PiPER)

## License

本项目采用 Apache-2.0。第三方项目仍受各自许可证约束，详见 [`NOTICE`](NOTICE)。
