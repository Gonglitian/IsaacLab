# G1 Reach 任务记录

## 调研笔记
- 阅读 `source/isaaclab_tasks/isaaclab_tasks/direct/locomotion/locomotion_env.py`，借鉴人形机器人在直接控制流程中的姿态/速度观测、力矩缩放与奖励设计方式。
- 参考 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py`，确认 Unitree G1 的关节命名、初始姿态以及常用仿真设置（episode 长度、场景布局等）。
- 查阅 `source/isaaclab/isaaclab/markers/config/__init__.py` 与 `source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/quadcopter_env.py`，了解可复用的可视化标记配置与调试渲染回调。
- 对照 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` 的重置与随机化策略，在本任务中引入类似的底座姿态/速度与关节噪声采样逻辑，以提升训练多样性与稳定性。

## 计划
1. 用 Unitree G1 的 29 自由度模型替换示例 cartpole，包括场景、动作/观测空间以及目标可视化。
2. 编写新的 RL 环境：控制整个机器人、随机采样 3D 目标、生成包含姿态/速度/相对目标向量的观测，并设计满足“走到目标并以右手触碰且保持平衡”要求的奖励和终止逻辑。
3. 对齐 RSL-RL 与 skrl 的 PPO 训练配置，更新项目笔记并记录训练/分布式启动方式。

## 进度追踪
- [x] 完成本任务相关的调研。
- [x] 实现环境与配置的全面重构（机器人、Reward、目标可视化、reset 随机化等）。
- [x] 更新 RSL-RL / skrl 训练配置与中文笔记。

## 实现说明
- 以 Unitree G1 取代 Cartpole 的占位实现，增加 2048 并行环境、红色球形目标可视化、自定义奖励权重以及全关节动作/观测定义。
- 新的 `G1ReachEnv` 采用平滑关节位置控制，重置阶段随机化底座位置/朝向/速度与关节扰动，并在奖励中结合手部距离、躯干行走距离、朝向、直立度、动作惩罚与成功奖励，满足“抵达并触碰”的任务目标。
- RSL-RL 与 skrl 的 PPO 配置扩大了网络规模与训练步数，改为 `g1_reach_direct` 实验命名，便于与旧示例区分和追踪日志。
- 参数调优（优化版）：参考 G1 速度任务与本地调试结果，对奖励/随机化做了一次迭代——alive reward 降到 0.2，迫使 agent 靠近目标才能得分；hand/body/facing 分别增至 8.0/1.0/2.0，成为主驱动力；upright 保持 1.0，既保证平衡又允许身体有动作；action/joint 惩罚调整为 -0.005/-0.001；success bonus 仍为 15。训练前期把 `target_radius_range` 缩小到 (0.5, 1.5)，`reach_threshold` 放宽至 0.2，方便先学会触手；后期再逐步拉回原来的距离/阈值。同时借鉴 locomotion 任务常用约束：  
  - **姿态/躯干惩罚**：新增 `rew_scale_flat_orientation=-0.5`，利用 roll/pitch 绝对值抑制大幅倾倒。  
  - **关节回中**：`rew_scale_joint_center=-0.05`，以默认姿态为参考，避免髋/臂长期偏离。  
  - **动作平滑**：`rew_scale_action_smooth=-0.002`，惩罚连续步的动作跳变，得到更平滑的控制。  
  - **细化姿态分组**：曾尝试加入 hip/arms/fingers/torso 的 `joint_dev` 惩罚与踝关节 limit penalty，但评估发现这些约束过强会阻碍 reach 行为，因此当前版本把这些系数降为 0，只保留 `flat_orientation` 与 `joint_center` 的轻度约束（分别为 -0.1 / -0.01），并把主任务信号强化为 `rew_scale_hand_target=16`、`success_bonus=24`。

## 训练 / 推理命令
- **单机单卡训练（RSL-RL）**  
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-G1-Reach-Direct-v0 \
      --num_envs 2048 \
      --max_iterations 2500 \
      --headless
  ```
- **单机多卡 / 分布式训练（torch.distributed）**  
  ```bash
  CUDA_VISIBLE_DEVICES=3,4 ./isaaclab.sh -p -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-G1-Reach-Direct-v0 \
      --num_envs 60000 \
      --max_iterations 10000 \
      --headless \
      agent.save_interval=100 \
      --distributed
  ```
  
  多节点时，把 `--standalone --nproc_per_node=2` 替换为  
  `--nnodes=<节点数> --nproc_per_node=<每节点GPU数> --node_rank=<编号> --rdzv_backend=c10d --rdzv_endpoint=<主节点IP:端口>`。
- **继续训练（Resume）**  
  从指定 checkpoint 继续训练：
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-G1-Reach-Direct-v0 \
      --num_envs 2048 \
      --max_iterations 5000 \
      --headless \
      --resume \
      --load_run <run-folder-name> \
      --checkpoint <checkpoint-file>
  ```
  - `--resume`: 启用继续训练模式
  - `--load_run`: 指定运行文件夹名称（如 `2025-11-10_12-53-53`）
  - `--checkpoint`: 指定模型文件（如 `model_1500.pt`，可省略路径前缀）
  
  也可以直接指定完整的 checkpoint 路径：
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-G1-Reach-Direct-v0 \
      --num_envs 2048 \
      --max_iterations 5000 \
      --headless \
      --resume \
      --checkpoint logs/rsl_rl/g1_reach_direct/2025-11-10_12-53-53/model_1500.pt
  ```
- **RSL-RL 推理 & 录制视频（Play）**  
  1. 训练时需开启摄像机（`--enable_cameras`），并在环境中配置想要的视角/Follow 方式。  
  2. 推理/录制采用 `scripts/reinforcement_learning/rsl_rl/play.py`，常用命令：  
     ```bash
     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
         --task Isaac-G1-Reach-Direct-v0 \
         --num_envs 1 \
         --headless \
         --enable_cameras \
         --video \
         --video_length 500 \
         --checkpoint <ckpt-path>
     ```
     这会加载指定 checkpoint，使用训练时保存的摄像机设置并在 `logs/rsl_rl/g1_reach_direct/<run>/videos/play/` 下生成 MP4。若想实时查看，可把 `--headless` 去掉并在 GUI 里调节相机。*** End Patch
