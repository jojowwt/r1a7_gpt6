# Unitree R1-A7 GPT-6 Control

[中文](#中文) | [English](#english)

## 中文

本项目把 [RoboICL](https://github.com/Mosi-AI/RoboICL) 的 GPT-6 Astra 视觉上下文学习策略接到 Unitree R1-A7 + Dex1 真机控制栈，作为独立项目发布。策略使用头部及左右腕部相机与机器人状态生成双臂末端动作；本地控制器负责 IK、动作连续性检查和 DDS 下发。仓库只包含推理执行所需代码，不包含 VR 遥操作、示教采集工具、RoboICL 仿真训练环境或模型 checkpoint。已有示教数据需在仓库外准备为 RoboICL reference bundle。

### Demo

<!-- 将 demo.mp4 放入 docs/demo/；也可放置 GIF 并取消下一行注释。 -->

**Demo video will be added here.**

<!-- ![R1-A7 GPT-6 demo](docs/demo/demo.gif) -->

[Demo 视频目录](docs/demo/README.md)

### 运行前提

- Ubuntu/Linux 主机、Unitree R1-A7 + Dex1，机器人与主机的 Unitree DDS 网络已连通。
- 已部署并启动四路图像 `image_server`，包含 head、left wrist、right wrist 等所需相机画面。
- 推理入口只订阅相机画面，不启动相机采集或示教录制流程。
- 已完成头部相机标定。标定文件属于设备相关数据，不随仓库发布；默认读取 `robot_teleop/config/camera_calibration.yaml`。
- Python 3.10+；Placo、Unitree SDK2 Python、图像服务和 DDS 所需系统依赖。
- 调用 GPT-6 Astra 时需配置服务端认可的 `ASTRA_API_KEY`。

### 安装

```bash
git clone <YOUR_GITHUB_URL>/r1a7_gpt6.git
cd r1a7_gpt6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r third_party/RoboICL/setup/requirements-policy.txt
python -m pip install -e robot_teleop
```

`unitree-sdk2py` 的安装方式依赖 Unitree SDK2 部署环境，请按对应版本文档安装。Qt 控制台需要 `PySide6`；若不安装，可运行时选择 `--ui cv` 或 `--ui none`。

### 使用

先检查命令行参数：

```bash
./scripts/run_r1a7_gpt6.sh --help
```

先进行 dry-run。dry-run 不请求模型，但仍会初始化真机通信并发布控制指令；执行前确保机器人处于可控调试状态、工作空间清空且急停可用：

```bash
./scripts/run_r1a7_gpt6.sh \
  --task bread_to_plate \
  --net enp5s0 \
  --dry-run \
  --ui cv
```

确认配置、图像、IK 和运行状态后，再配置 API key 并选择任务。真实模型运行示例：

```bash
export ASTRA_API_KEY='YOUR_API_KEY'
./scripts/run_r1a7_gpt6.sh \
  --task bread_to_plate \
  --net enp5s0 \
  --ui qt \
  --reference /path/to/train_reference_bundle.json
```

`--reference` 可省略以进行 zero-shot rollout。one-shot bundle 必须与当前任务指令和本地数据完整性校验要求匹配。API key 只从环境变量读取，不要写入配置、脚本或提交到 Git。

### 真机安全

该程序会接管 R1-A7 控制并通过 DDS 下发机械臂及夹爪命令。每次真机运行前，操作员必须确认机器人处于正确的调试/控制模式，周围无人且无障碍物，急停可立即触达，并核对网络接口、机器人型号、URDF、相机流和标定。首次部署只允许操作员现场监督下做小范围验证。不要将 dry-run 误认为无运动测试：它绕过模型 API，但机器人控制循环仍会运行。程序不会替操作员判断外部碰撞或人员安全。

### 目录

| 路径 | 内容 |
|---|---|
| `robot_teleop/run_roboicl_live.py` | GPT-6 Astra 真机 rollout 入口 |
| `robot_teleop/src/robot_teleop/` | R1-A7 环境、相机、IK、DDS publisher 与运行监视器 |
| `robot_teleop/config/` | 默认控制参数；设备地址和标定需本地配置 |
| `assets/r1_a7/` | R1-A7 URDF 与网格模型 |
| `third_party/RoboICL/` | 运行策略所需 RoboICL 源码与配置子集 |
| `docs/demo/` | 项目演示视频占位 |

### 来源与许可证

本项目保留了 GPT-6 真机推理所需的 R1-A7 DDS 控制、IK、相机接入和运行监视代码；VR 遥操作及示教数据录制功能不属于本仓库发布内容。项目结构参考了 [Ticotico410/r1a7_gpt6_control](https://github.com/Ticotico410/r1a7_gpt6_control)，RoboICL GPT-6 Astra 策略和必要的 XPolicyLab 图像工具位于 `third_party/RoboICL/`。各组件来源、版本及许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。顶层 MIT 许可只适用于本项目自有代码，不覆盖第三方文件、Unitree 机器人模型或商标。

## English

This standalone project connects the GPT-6 Astra visual in-context policy from [RoboICL](https://github.com/Mosi-AI/RoboICL) to the Unitree R1-A7 + Dex1 hardware control stack. The policy receives head and wrist camera images plus robot state and proposes dual-arm end-effector actions. Local code performs IK, action-continuity checks, and DDS publishing. This repository contains only the inference execution stack; it excludes VR teleoperation, demonstration capture, RoboICL's simulator training environment, and model checkpoints. Prepare demonstration reference bundles outside this repository.

### Demo

<!-- Add demo.mp4 under docs/demo/. A GIF preview can be added below. -->

**Demo video will be added here.**

<!-- ![R1-A7 GPT-6 demo](docs/demo/demo.gif) -->

See [the demo directory](docs/demo/README.md).

### Requirements

- An Ubuntu/Linux host and Unitree R1-A7 + Dex1 with a working Unitree DDS network.
- A running four-camera `image_server` with the required head and wrist views.
- The inference runner subscribes to camera frames; it does not capture demonstrations.
- A device-specific head-camera calibration at `robot_teleop/config/camera_calibration.yaml` (not included).
- Python 3.10+ and the system dependencies required by Placo, Unitree SDK2 Python, image transport, and DDS.
- `ASTRA_API_KEY` for live GPT-6 Astra requests.

### Install

```bash
git clone <YOUR_GITHUB_URL>/r1a7_gpt6.git
cd r1a7_gpt6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r third_party/RoboICL/setup/requirements-policy.txt
python -m pip install -e robot_teleop
```

Install `unitree-sdk2py` using the instructions for your Unitree SDK2 deployment. The Qt console requires `PySide6`; use `--ui cv` or `--ui none` without it.

### Run

Inspect the command-line options:

```bash
./scripts/run_r1a7_gpt6.sh --help
```

Start with dry-run. It skips model requests but still initializes hardware communication and publishes robot commands. Ensure the robot is in its supported debug mode, the workspace is clear, and the emergency stop is reachable:

```bash
./scripts/run_r1a7_gpt6.sh \
  --task bread_to_plate \
  --net enp5s0 \
  --dry-run \
  --ui cv
```

After checking the configuration, cameras, IK, and runtime state, set the API key and launch a real model rollout:

```bash
export ASTRA_API_KEY='YOUR_API_KEY'
./scripts/run_r1a7_gpt6.sh \
  --task bread_to_plate \
  --net enp5s0 \
  --ui qt \
  --reference /path/to/train_reference_bundle.json
```

Omit `--reference` for zero-shot rollout. A one-shot bundle must match the task instruction and pass the local data-integrity checks. Keep credentials in environment variables; never commit them to config or scripts.

### Hardware safety

This program takes control of the R1-A7 and publishes arm and gripper commands over DDS. Before each run, the operator must confirm the supported robot mode, an empty workspace, immediate access to the emergency stop, and the network interface, robot model, URDF, camera streams, and calibration. Supervise initial hardware tests in person and begin with small movements. Dry-run bypasses model requests only; the robot control loop still runs. The software does not guarantee clearance from people or external obstacles.

### Provenance and licenses

The repository retains only the R1-A7 control, IK, camera input, and monitoring code needed for inference. VR teleoperation and demonstration capture are excluded. Project structure was informed by [Ticotico410/r1a7_gpt6_control](https://github.com/Ticotico410/r1a7_gpt6_control); no files from that repository are redistributed. RoboICL and the required XPolicyLab image utilities are retained under `third_party/RoboICL/` with their upstream notices. The top-level MIT license applies only to original project code and does not relicense third-party files, Unitree robot assets, or trademarks.
