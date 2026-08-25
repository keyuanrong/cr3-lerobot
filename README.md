# CR3 具身智能项目

本项目基于 LeRobot，为 Dobot CR3 机械臂提供从真机采集、数据审核与转换、策略训练、远程 Pi0 推理到 MuJoCo 仿真的完整研发流程。项目重点是将双相机观测、机械臂状态与夹爪动作组织为可复现的数据和训练闭环。

本仓库保留 LeRobot 的核心目录、通用数据集、策略、训练、评估与机器人抽象；项目自研内容集中在 CR3 真机适配、项目脚本和 MuJoCo 工具链。代码遵循根目录 `LICENSE` 中的 Apache License 2.0 许可条款，使用或分发时还应遵守 LeRobot 及其依赖的相应许可证。

## 核心功能

- Dobot CR3 真机通信、夹爪控制与 leader-follower 跟随。
- 双相机与 RealSense 拖拽数据采集，保存关节状态、目标动作、夹爪状态和任务文本。
- 原始轨迹审核、阶段切分和 LeRobot 数据集转换。
- 基于 LeRobot 的策略训练，以及 Pi0 和 LoRA 实验支持。
- 本地验证集远程 Pi0 回放、离线指标汇总与结果保存。
- CR3 MuJoCo 场景、关节映射、末端遥操作、采集、回放和策略 rollout。

## 技术栈

Python、PyTorch、LeRobot、MuJoCo、Dobot CR3 通信接口、RealSense、Hugging Face 数据集与模型工具链。

## 目录结构

- `src/lerobot/`：LeRobot 上游核心实现。
- `src/lerobot/robots/dobot_cr3/`：CR3 真机、夹爪与 leader-follower 适配。
- `scripts/collection/`：真机数据采集。
- `scripts/data_processing/`：轨迹审核、切分与恢复。
- `scripts/datasets/`：原始数据到 LeRobot 数据集的转换。
- `scripts/training/`：训练辅助与指标绘图。
- `scripts/evaluation/`：远程 Pi0 回放与离线评估。
- `scripts/inference/`：远程 Pi0 服务端与真机客户端。
- `sim/cr3_mujoco/`：CR3 MuJoCo 仿真工具链。
- `data/`：原始轨迹和审核、切分清单。
- `lerobot_data/`：转换后的 LeRobot 数据集。
- `tests/`：上游和项目自研流程的回归测试。

## 运行说明

先按现有文档确认目标、硬件、相机和运行环境，再选择对应流程。在仓库根目录安装当前源码及 feetech 可选依赖：

```bash
python -m pip install -e '.[feetech]'
```

连接串口设备前可使用已有的端口发现命令：

```bash
lerobot-find-port
```

训练入口为 `lerobot-train`；CR3 数据采集、转换、远程 Pi0 回放和 MuJoCo 流程的具体入口与参数见 `docs/project-map.md`、`docs/development-workflow.md` 和 `AGENT_GUIDE.md`。`data/` 和 `lerobot_data/` 是采集与转换流程在本地创建的运行时目录，分别保存原始轨迹和转换后的 LeRobot 数据集。原始数据、转换数据、checkpoint、logs 和 outputs 默认不提交到 GitHub；训练输出和评估结果应与对应实验配置和数据版本一并记录，且不要覆盖原始采集数据。执行任何真机命令前，必须先核对设备连接、串口或网络端口、相机索引、工作空间和关节限位；控制相关修改应先在回放或 MuJoCo 中验证。
