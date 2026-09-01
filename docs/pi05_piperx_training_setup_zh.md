# PiperX 上的 π0.5 LoRA 微调实验设置

## 实验资料卡

- **实验名称**：PiperX π0.5 LoRA Fine-tuning（50 Episodes，Batch Size 16）
- **实验配置名**：`pi05_piperx_lora`
- **实验目录名**：`piperx_pi05_lora_50ep_bs16_v1`
- **状态**：训练完成，最终 checkpoint 已提交
- **配置核验来源**：OpenPI 本地代码、LeRobot 数据集元信息、训练命令及训练日志
- **OpenPI commit**：`15a9616a00943ada6c20a0f158e3adb39df2ccac`
- **完成日期**：2026-08-14

## 1. 基础模型

本实验采用 Physical Intelligence 开源的 **π0.5 base model** 作为预训练基础模型：

```text
pi05_base
```

基础权重在本机的缓存位置为：

```text
${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_base
```

模型配置如下：

| 配置项 | 设置 |
|---|---:|
| 模型类型 | π0.5 |
| 视觉语言主干 | PaliGemma / Gemma 2B |
| Action Expert | Gemma 300M |
| 模型内部动作维度 | 32 |
| 动作预测长度（action horizon） | 16 |
| 最大非图像 token 数 | 200 |
| π0.5 离散状态输入 | 开启 |

PiperX 原始状态和动作均为 7 维：6 个机械臂关节加 1 个夹爪维度。进入模型前，状态和动作由 7 维补零至 π0.5 使用的 32 维；模型推理输出再裁剪回 7 维。

## 2. 微调方法

本实验不是全量微调，而是采用 **LoRA-style parameter-efficient fine-tuning**：

| 模块 | LoRA 配置 |
|---|---:|
| Gemma 2B 主干 Attention/FFN | rank 16，alpha 16 |
| Gemma 300M Action Expert Attention/FFN | rank 32，alpha 32 |

对应的 OpenPI 模型变体为：

```text
paligemma_variant = gemma_2b_lora
action_expert_variant = gemma_300m_lora
```

冻结规则会冻结两个 LLM 主干中的非 LoRA 权重。LoRA 参数以及不属于 LLM 路径的视觉、动作投影和时间条件模块仍可参与训练。因此，本实验准确地说是 **LoRA-style 部分参数微调**，而不是全参数微调，也不是严格意义上只更新 LoRA 矩阵的 adapter-only 微调。

冻结权重在训练状态中转换为 BF16，可训练参数主要保持 FP32。实验关闭了 EMA。

## 3. 训练框架

训练使用 OpenPI 官方的 JAX 路径，而不是 LeRobot 自带策略训练器或 Hugging Face Trainer。

```text
LeRobot v2.1 dataset
        ↓ PyAV 视频解码、PyTorch DataLoader
OpenPI 数据变换与归一化
        ↓ JAX arrays
π0.5 / Flax NNX
        ↓ Optax AdamW
JAX + XLA on RTX 4090
        ↓
Orbax checkpoint
```

主要软件版本如下：

| 组件 | 版本 | 用途 |
|---|---:|---|
| OpenPI | commit `15a9616` | π0.5 模型与训练代码 |
| JAX | 0.5.3 | 数值计算与自动微分 |
| JAXlib | 0.5.3 | CUDA/XLA 后端 |
| Flax NNX | 0.10.2 | 模型及参数管理 |
| Optax | 0.2.4 | 优化器和学习率调度 |
| Orbax Checkpoint | 0.11.13 | checkpoint 保存与恢复 |
| PyTorch | 2.7.1 | LeRobot DataLoader |
| LeRobot | 0.1.0（本地包版本） | 数据集读取 |
| PyAV | 本地环境版本 | AV1 视频解码 |

训练入口为：

```text
openpi/scripts/train.py
```

使用的配置为：

```text
pi05_piperx_lora
```

## 4. 训练数据集

实验使用 Xbox 手柄遥操作 PiperX 采集的单任务真机示教数据，存储为 LeRobot v2.1 格式：

```text
${HF_LEROBOT_HOME}/local/piperx_pi05
```

数据集统计如下：

| 数据项 | 数值 |
|---|---:|
| 数据集数量 | 1 |
| 示教轨迹数 | 50 episodes |
| 总帧数 | 26,567 |
| 采样频率 | 15 Hz |
| 总时长 | 约 29 分 31 秒 |
| 任务数 | 1 |
| 相机数 | 2 |
| 视频文件数 | 100 |
| 数据集体积 | 约 208 MB |
| 原始状态维度 | 7 |
| 原始动作维度 | 7 |

语言任务为：

```text
put the yellow rubber duck onto the white plate
```

视觉输入包括：

1. 第三视角相机：`observation.images.third_person`，480×640 RGB；
2. 腕部相机：`observation.images.wrist`，480×640 RGB。

输入模型前，两路图像被缩放至 224×224。π0.5 接口要求的第三路右腕相机使用全零图像补齐，并设置 mask 为 false，因此不会被视为有效视觉输入。

数据集中的关节动作是绝对目标。训练时，前 6 个关节转换为相对于当前状态的 delta action，夹爪维度保持绝对动作：

```text
joint_delta = target_joint_position - current_joint_position
gripper_action = absolute_gripper_target
```

状态和动作使用 q01/q99 分位数归一化到近似 `[-1, 1]`，而不是均值/标准差归一化。

需要注意：当前 50 条 episode 全部位于训练 split，尚未预先划分独立 validation 集。因此当前候选 checkpoint 只能通过离线回放和真机实验选优，不能声称是基于无数据泄漏验证集得到的 validation-best。

## 5. 训练超参数与学习率

正式训练采用以下超参数：

| 超参数 | 设置 |
|---|---:|
| Physical batch size | 16 |
| Gradient accumulation | 无 |
| Effective batch size | 16 |
| 总优化步数 | 20,000 |
| 随机种子 | 42 |
| DataLoader workers | 2 |
| 优化器 | AdamW |
| Adam β1 | 0.9 |
| Adam β2 | 0.95 |
| Adam epsilon | 1e-8 |
| Weight decay | 1e-10 |
| 全局梯度裁剪 | 1.0 |
| EMA | 关闭 |
| FSDP | 关闭，单 GPU |
| WandB | 关闭 |
| 日志间隔 | 100 steps |
| XLA 显存池比例 | 0.95 |

学习率采用带 warmup 的 cosine decay：

| 学习率参数 | 设置 |
|---|---:|
| Warmup steps | 1,000 |
| Peak learning rate | 2.5e-5 |
| Cosine decay steps | 30,000 |
| Target decay learning rate | 2.5e-6 |

由于本实验在 20,000 step 结束，学习率调度不会完整运行到设定的 30,000-step 衰减终点。

训练损失为 π0.5 的 flow-matching mean squared error。模型在随机 flow timestep 上预测动作噪声路径的速度场，优化预测速度与目标速度之间的均方误差。

训练图像增强包括：

- 第三视角：95% 随机裁剪、缩放回原尺寸、±5° 随机旋转及颜色抖动；
- 腕部相机：颜色抖动；
- 颜色抖动参数：brightness 0.3、contrast 0.4、saturation 0.5。

## 6. 训练步数与数据预算

正式实验训练 20,000 个 optimizer steps，每个 step 使用 16 个样本：

```text
20,000 steps × 16 samples/step = 320,000 sample draws
```

以数据集的 26,567 帧为一个近似 epoch：

```text
320,000 / 26,567 ≈ 12.05 epochs
```

因此，正式 batch-16 实验约等价于 12 个数据遍历周期。这里的 epoch 是按帧级样本抽取次数换算；每个样本的监督目标实际包含长度为 16 的动作块，即约 1.07 秒的未来动作。

此前还完成了一个 batch-size 1 的基线实验：

```text
20,000 × 1 = 20,000 sample draws ≈ 0.75 epochs
```

batch-size 1 基线主要用于验证完整训练链路，正式 batch-size 16 实验的数据计算预算是它的 16 倍。

## 7. 硬件与运行时间

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| 显存 | 24,564 MiB |
| NVIDIA driver | 535.288.01 |
| 物理 batch size | 16 |
| 稳态训练速度 | 约 2.4 秒/step |
| batch 16 XLA 峰值估算 | 约 17.49 GiB |

batch size 32 已完成完整测试，但在反向传播阶段发生 OOM；因此正式实验选择 batch size 16。

训练曾在约 18.2k step 时因 Codex 托管会话结束而中止。最近一次完整 checkpoint 为 15k，随后通过 Linux 终端中的 tmux 从 15k 恢复到 20k。最终模型仍对应连续有效的 20,000 个优化步骤；中断前未写入 checkpoint 的计算不进入最终模型状态。

## 8. 检查点与实验输出

实验输出目录：

```text
${OPENPI_ROOT}/checkpoints/pi05_piperx_lora/piperx_pi05_lora_50ep_bs16_v1
```

当前保留的候选 checkpoint：

```text
15000
16000
17000
18000
19000
19999
```

其中：

- `19999` 是 latest checkpoint；
- best checkpoint 尚未确定；
- 后续将通过离线推理、动作安全检查和真机任务成功率选出 best；
- 在 best 确定之前不删除候选 checkpoint。

恢复训练段总耗时约 3 小时 25 分钟。最终 `19999` checkpoint 于 2026-08-14 18:30:32 完整提交，末段训练 loss 约为 0.0022–0.0034，未观察到数值发散。

## 9. 复现实验命令

正式实验等价训练命令如下：

```bash
cd /path/to/piperx-xbox-pi05/openpi

HF_HOME=/path/to/piperx-xbox-pi05/.hf-cache \
HF_DATASETS_CACHE=/path/to/piperx-xbox-pi05/.hf-cache/datasets \
HF_LEROBOT_HOME=/path/to/piperx-xbox-pi05/data/lerobot \
OPENPI_DATA_HOME=/path/to/piperx-xbox-pi05/data/openpi_cache \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
.venv/bin/python \
scripts/train.py pi05_piperx_lora \
--exp-name=piperx_pi05_lora_50ep_bs16_v1 \
--batch-size=16 \
--num-train-steps=20000 \
--save-interval=1000 \
--keep-period=1000 \
--no-overwrite \
--no-resume
```

若从已有 checkpoint 恢复，应将最后一项替换为：

```text
--resume
```

并保持 `--no-overwrite`，避免删除已有 checkpoint。

## 10. 当前实验限制

本轮实验的主要限制是没有在训练前划分独立验证集。下一轮建议：

```text
45 episodes for training
5 episodes for validation
```

归一化统计只使用训练集计算，并以 validation flow-matching loss 或离线动作误差自动选择 best checkpoint；latest checkpoint 则独立保留用于恢复训练。
