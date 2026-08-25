# 项目地图

## 定位与边界

本仓库以 [LeRobot](../src/lerobot/) 为基础，增加 Dobot CR3 真机适配、拖拽采集数据工具、Qwen 轨迹审核与切分、Pi0 远程推理，以及 CR3 MuJoCo 仿真工具链。

`src/lerobot/` 主体、根目录 `pyproject.toml`、`Makefile`、`examples/`、上游测试目录结构和 `docs/source/` 属于 LeRobot 上游基础。项目自研内容主要集中在 `src/lerobot/robots/dobot_cr3/`、根目录 `scripts/`、`sim/cr3_mujoco/`、`data/episode_lists/`、项目根目录的训练文档及为 CR3 流程新增的测试。

当前工作目录没有 `.git` 元数据，无法可靠区分历史上的每一处改动或查看提交历史；以上“上游/自研”按目录职责和当前代码结构划分。

## 主要模块

| 模块 | 职责 | 关键入口 |
| --- | --- | --- |
| LeRobot 核心 | Dataset、Policy、Processor、Train、Eval、Env、Robot 通用抽象 | `src/lerobot/datasets/`, `policies/`, `processor/`, `scripts/lerobot_train.py` |
| CR3 真机适配 | Dobot Dashboard/Move TCP 通信、夹爪、观测、leader-follower 跟随 | `src/lerobot/robots/dobot_cr3/dobot_cr3.py`, `leader_follower_copy.py` |
| 真机采集 | 双相机、RealSense、leader-follower 控制、CSV/图像落盘 | `scripts/collection/record_drag_dataset.py` |
| 数据审核与切分 | 规则/Qwen 审核、阶段边界精分、人工回放、训练清单生成 | `scripts/data_processing/*filter*`, `scripts/data_processing/*segment*`, `scripts/data_processing/*refine*`, `scripts/datasets/build_pi0_unified_goal_manifests.py` |
| 数据转换 | 原始 `data.csv`、RGB 图像、任务文本转换为 LeRobot 数据集 | `scripts/datasets/convert_drag_to_lerobot.py`, `scripts/datasets/convert_pi0_unified_goal_datasets.sh` |
| 训练 | `lerobot-train`、Pi0/LoRA、checkpoint、loss 图 | `src/lerobot/scripts/lerobot_train.py`, `scripts/training/plot_training_metrics.py` |
| 评估 | 本地验证集回放、离线指标汇总与结果目录落盘 | `scripts/evaluation/eval_remote_pi0_replay.py` |
| 真机远程 Pi0 | 本地相机与 CR3 客户端、TCP 模型服务、动作队列/夹爪过滤 | `scripts/inference/run_remote_pi0_policy.py`, `scripts/inference/pi0_remote_policy_server.py` |
| MuJoCo CR3 | 场景、关节映射、末端遥操作、仿真采集/回放/策略 rollout | `sim/cr3_mujoco/` |
| 回归测试 | 上游单测与自研数据处理/切分/采样器测试 | `tests/` |

## 真机数据流

### 采集到 Dataset

```text
leader arm joints
  -> LeaderFollowerCopyController.step()
  -> follower CR3 ServoJ target
  -> scripts/collection/record_drag_dataset.py
       + front RGB / wrist RealSense RGB
       + joint state, follower target, gripper, task
  -> data/cr3_real_drag_raw/<task>/drag_episode_*/
       data.csv + image files
  -> 审核/切分 JSONL 清单
  -> scripts/datasets/convert_drag_to_lerobot.py
  -> lerobot_data/local/<repo_id>/
       data + videos + meta
```

`record_drag_dataset.py` 写入的是以帧为单位的原始记录。`convert_drag_to_lerobot.py` 将 `front_rgb` 和 `wrist_rgb` 映射为 `observation.images.front_rgb`、`observation.images.wrist_rgb`，将 6 个关节和夹爪写为 state/action；夹爪语义必须显式指定，目前 Pi0 数据使用 `close_high`，即 `0=open`、`1=close`。

### Dataset 到训练

```text
LeRobotDataset / MultiLeRobotDataset
  -> dataset factory (delta timestamps + image transforms)
  -> DataLoader
  -> policy preprocessor (normalization / tokenizer / device)
  -> Pi0 / LoRA policy forward + loss
  -> optimizer + scheduler
  -> checkpoint + train log
  -> scripts/training/plot_training_metrics.py -> loss 曲线
```

目前项目还支持统一总目标的四数据集混合训练：完整轨迹、总目标抓取事件、总目标放置事件、原子阶段辅助。`FixedRatioMixtureSampler` 在训练阶段按固定配比抽样，转换阶段仍然是一份清单转换成一个普通 LeRobot 数据集。具体命令见 [pi0_unified_goal_weighted_training.md](./pi0_unified_goal_weighted_training.md)。

## Policy 推理到真机 action

```text
front/wrist camera frames + CR3 joint state + task text
  -> scripts/inference/run_remote_pi0_policy.py
  -> JPEG 或原始帧 TCP payload
  -> scripts/inference/pi0_remote_policy_server.py
       payload decode -> observation tensor -> preprocessor
       -> policy.predict_action_chunk()/select_action()
       -> postprocessor -> action chunk
  -> client ActionQueue / latency compensation / max joint delta
  -> clamp_absolute_action + send_action_to_robot
  -> Dobot CR3 ServoJ or JointMovJ + gripper command
```

远程协议用 `pickle` 传输，仅适用于受信任的内网或 SSH 隧道。服务端 `load_policy()` 对 LoRA 先加载 base policy，再加载 adapter；全量 checkpoint 则直接加载 checkpoint 本体。

## MuJoCo observation/action 流

```text
MuJoCo XML + meshes
  -> MjModel / MjData
  -> teleop_cr3_eef.py
       IK、工作空间、夹爪、GraspAssist、相机渲染
  -> record_vla_teleop_dataset.py
       state_vector + action_vector + front/wrist render
  -> LeRobotDataset
  -> replay_vla_dataset.py / rollout_smolvla_in_mujoco.py
  -> joint_mapping.policy_action_to_mujoco_ctrl()
  -> MuJoCo actuator ctrl
```

`sim/cr3_mujoco/joint_mapping.py` 是仿真与真实展示关节角之间的核心边界：它包含关节符号、零点偏移以及夹爪到两个 MuJoCo actuator 的映射。这里的改动必须配合仿真回放和控制范围检查。

## 真机接口流

`DobotCR3.connect()` 建立 Dashboard 与 Move TCP 客户端，按配置启用机器人，并可初始化 Modbus 夹爪。`get_observation()` 读取 TCP pose、关节、运行状态和夹爪反馈。`LeaderFollowerCopyController.connect()` 负责 leader/follower 连接、启用、初始对齐和 `StartDrag`；`step()` 将 leader 相对运动缩放、连续化、限位、平滑后发送给 follower `ServoJ`。

与真实运动直接相关的关键实现：

- `src/lerobot/robots/dobot_cr3/dobot_cr3.py`
- `src/lerobot/robots/dobot_cr3/leader_follower_copy.py`
- `src/lerobot/robots/dobot_cr3/config_dobot_cr3.py`
- `scripts/collection/record_drag_dataset.py`
- `scripts/inference/run_remote_pi0_policy.py`

## 配置与测试

- 包、CLI 与可选依赖：`pyproject.toml`、`uv.lock`、`requirements*.txt`。
- 训练配置：`src/lerobot/configs/train.py`；策略配置在 `src/lerobot/configs/` 与各 policy 目录。
- 上游广覆盖测试在 `tests/datasets/`、`tests/policies/`、`tests/processor/`、`tests/training/` 等。
- 项目自研流程回归测试包括 `tests/test_refine_drag_phase_boundaries.py`、`tests/test_recover_relaxed_phase_boundaries.py`、`tests/test_two_stage_visual_recovery.py`、`tests/test_build_pi0_unified_goal_manifests.py`、`tests/datasets/test_weighted_mixture.py`。
