# README 已用素材索引

根目录 `README.md` 仅展示仓库当前已有的真实素材。路径均为仓库内相对路径，不包含实验室 IP、串口或其他私密配置。

| README 区域 | 素材 | 路径 | 说明 |
| --- | --- | --- | --- |
| 顶部 Hero | CR3 工作台全景 | `assets/images/quanmian.jpeg` | CR3、LMG-90、相机与工作台的整体视图 |
| 数据采集与清洗 | 红色任务示教 | `assets/gifs/red.gif` | 红色物块任务原始示教片段 |
| 数据采集与清洗 | 绿色任务示教 | `assets/gifs/green.gif` | 绿色物块任务原始示教片段 |
| 数据采集与清洗 | 黄色任务示教 | `assets/gifs/yellow.gif` | 黄色物块任务原始示教片段 |
| 数据采集与清洗 | 完整任务示教 | `assets/gifs/full.gif` | 三物块完整任务原始示教片段 |
| 数据分析 | Episode 12 动作曲线 | `docs/assets/episode_12_actions.png` | 六关节与夹爪真实动作标签曲线 |
| 策略训练 | 训练曲线 | `assets/plots/loss_curve.png` | 训练 loss 与学习率曲线，仅用于训练诊断 |
| MuJoCo 仿真 | 场景一 | `assets/images/mujoco1.png` | CR3 + LMG-90 仿真截图 |
| MuJoCo 仿真 | 场景二 | `assets/images/mujoco2.png` | CR3 + LMG-90 仿真截图 |
| 真机 Rollout | 红色任务 | `assets/gifs/redreal.gif` | 当前策略的探索性真机执行过程 |
| 真机 Rollout | 绿色任务 | `assets/gifs/greenreal.gif` | 当前策略的探索性真机执行过程 |
| 真机 Rollout | 黄色任务 | `assets/gifs/yellowreal.gif` | 当前策略的探索性真机执行过程 |

## 使用约束

- GIF 与图片标题必须和实际任务、实际结果一致。
- 训练 loss 不能单独用作真机成功率或抓取可靠性的证据。
- 真机素材展示的是当前探索性执行行为；在建立固定评测协议前，不标注成功率结论。
