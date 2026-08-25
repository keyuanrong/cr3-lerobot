# Pi0 统一总目标加权训练

本流程使用固定的四数据集配比，解决“长轨迹帧数更多，普通按帧随机抽样会让完整任务压过夹取和放置关键帧”的问题。

四类数据集及训练权重固定如下：

| 数据集 | 内容 | 总目标标签 | 每 1000 个训练样本 |
| --- | --- | --- | --- |
| `complete_goal` | 完整成功轨迹 | 对应 red/green/yellow/full 总目标 | 400 |
| `goal_grasp_event` | 张开、对准、闭合、抬起附近 | 对应总目标 | 250 |
| `goal_place_event` | 到黑框、释放、离开附近 | 对应总目标 | 250 |
| `atomic_assist` | 非 full 的四个原子阶段 | 原子阶段文本 | 100 |

`complete_goal`、`goal_grasp_event` 和 `goal_place_event` 都使用相同的总目标文本。因此模型会多次看到“总目标 + 物块进入两指间”应闭合、“总目标 + 已抓住并抬起”应搬运、“总目标 + 黑框”应张开，而不是只学到向黑框移动。

## 1. 转换四个 LeRobot 数据集

在本地仓库根目录执行：

```bash
cd /home/kyr/lerobot/lerobot
bash scripts/datasets/convert_pi0_unified_goal_datasets.sh
```

输出位于：

```text
lerobot_data/local/cr3_pi0_unified_goal_v1_complete_goal
lerobot_data/local/cr3_pi0_unified_goal_v1_goal_grasp_event
lerobot_data/local/cr3_pi0_unified_goal_v1_goal_place_event
lerobot_data/local/cr3_pi0_unified_goal_v1_atomic_assist
```

如果之前转换中断且确认要重新编码四个数据集，才使用：

```bash
bash scripts/datasets/convert_pi0_unified_goal_datasets.sh --force
```

## 2. 验证四个数据集

```bash
/home/kyr/miniconda3/envs/lerobot/bin/python - <<'PY'
from lerobot.datasets import LeRobotDataset

root = "lerobot_data"
for repo_id in [
    "local/cr3_pi0_unified_goal_v1_complete_goal",
    "local/cr3_pi0_unified_goal_v1_goal_grasp_event",
    "local/cr3_pi0_unified_goal_v1_goal_place_event",
    "local/cr3_pi0_unified_goal_v1_atomic_assist",
]:
    ds = LeRobotDataset(repo_id, root=f"{root}/{repo_id}")
    print(repo_id, "episodes=", ds.num_episodes, "frames=", ds.num_frames)
PY
```

## 3. 在服务器训练 LoRA

先将本地 `lerobot_data/local/` 下这四个目录上传到服务器同一个根目录，例如 `/root/autodl-tmp/lerobot_data/local/`。训练时 `mixture.root` 必须是它们共同的上级目录 `/root/autodl-tmp/lerobot_data`。

```bash
cd /root/lerobot

RUN=cr3_pi0_lora_r64_unified_goal_50k
OUT=/root/autodl-tmp/outputs/train/$RUN
LOG=/root/autodl-tmp/train_logs/$RUN.log
mkdir -p /root/autodl-tmp/train_logs

/root/miniconda3/bin/lerobot-train \
  --dataset.repo_id=local/cr3_pi0_unified_goal_v1_complete_goal \
  --dataset.root=/root/autodl-tmp/lerobot_data/local/cr3_pi0_unified_goal_v1_complete_goal \
  --mixture.repo_ids='["local/cr3_pi0_unified_goal_v1_complete_goal","local/cr3_pi0_unified_goal_v1_goal_grasp_event","local/cr3_pi0_unified_goal_v1_goal_place_event","local/cr3_pi0_unified_goal_v1_atomic_assist"]' \
  --mixture.root=/root/autodl-tmp/lerobot_data \
  --mixture.weights='[0.4,0.25,0.25,0.1]' \
  --mixture.block_size=1000 \
  --policy.path=/root/autodl-tmp/hf_models/pi0_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name="$RUN" \
  --batch_size=1 \
  --num_workers=4 \
  --steps=50000 \
  --log_freq=200 \
  --save_freq=50000 \
  --checkpoint_steps='[35000,40000,45000,50000]' \
  --full_checkpoint_steps='[40000]' \
  --optimizer.lr=1e-5 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.num_warmup_steps=1000 \
  --scheduler.num_decay_steps=50000 \
  --scheduler.peak_lr=1e-5 \
  --scheduler.decay_lr=1e-6 \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=128 \
  --early_stopping.enable=false \
  --use_policy_training_preset=false \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

mkdir -p "$OUT/analysis"
/root/miniconda3/bin/python -m scripts.training.plot_training_metrics \
  --log-file "$LOG" \
  --output-dir "$OUT/analysis"
```

混合采样模式目前只支持单 GPU 进程。`--batch_size=1` 时，每 1000 个 optimizer step 精确消耗 400/250/250/100 个来源样本；更大的 batch 则按每 1000 个样本保持同一比例。

## 4. 验证集

训练数据已排除 `data/episode_lists/pi0_unified_goal_v1/source_validation.txt` 中的 80 条源轨迹。训练过程中 loss 只反映训练样本拟合；需要另行转换验证清单并做 teacher-forced validation，才能比较 checkpoint 的离线泛化。最终是否能完成闭环抓取，仍需在真机上测试。
