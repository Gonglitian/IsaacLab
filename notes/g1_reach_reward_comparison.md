# G1 Reach 与 G1 Locomotion Velocity（Rough/Flat）奖励函数设计对比调研

本文系统梳理并对比了直接控制的 G1 Reach 任务与基于 Manager 的 G1 速度任务（Rough/Flat）的奖励函数设计。重点关注相同点、差异点以及这些设计对学习行为的影响，并给出可供互相借鉴的建议。

- 范围与文件
  - Reach（Direct）：`source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env.py`（奖励实现，见 478 行起的 `compute_rewards`）与 `source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env_cfg.py`（权重/阈值，见 105–124 行）。
  - Velocity（Manager, Rough）：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`（奖励项与权重，见 19–100、132–144 行）。
  - Velocity（Manager, Flat）：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`（在 Rough 基础上调整的奖励与命令范围，见 27–41 行）。
  - 参考奖励原子函数：
    - 速度跟踪/足部项等（Locomotion 专用）：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
    - 通用 L2/L1 奖励项：`source/isaaclab/isaaclab/envs/mdp/rewards.py`

---

## 奖励设计目标与高层思路
- Reach：核心目标是“朝目标行走+右手接近/触碰+保持稳定站立”。奖励由“目标相关（手、躯干、朝向、成功 Bonus）+ 姿态稳定（直立/平衡）+ 行走形状（朝目标的速度跟踪、步态塑形）+ 代价（动作变化率、关节/动力学正则、关节回中与限位、脚滑）+ 失败惩罚”组成。
- Velocity（Rough/Flat）：核心目标是“跟踪给定的底座速度命令（线速度/角速度）并保持稳定”。奖励由“命令跟踪（线/角速度）+ 姿态稳定（扁平姿态、竖直速度抑制）+ 步态塑形（足空中时间）+ 代价（动作变化率、关节加速度/力矩、脚滑、关节回中与限位）+ 失败惩罚”组成。

两者都采用“密集型 shaping + 失败强惩罚”的结构，但任务驱动信号不同：Reach 强化“到达与触手”，Velocity 强化“速度跟踪”。

---

## Reach（Direct）奖励项与实现要点
- 实现位置：`g1_reach_env.py:478` 起的 `compute_rewards` 定义与其调用；权重在 `g1_reach_env_cfg.py:105` 起。
- 主要正向项
  - 手到目标距离：`rew_scale_hand_target * exp(-2.5 * hand_dist)`（靠近即大幅增益）。
  - 躯干至目标水平距离：`rew_scale_body_target * exp(-0.5 * body_dist)`（引导靠近目标点）。
  - 朝向目标：`rew_scale_facing * clamp(facing, ≥0)`（仅奖励朝向正确的分量）。
  - 直立度：`rew_scale_upright * clamp(upright, ≥0)`（世界z与机体z点积）。
  - 目标速度跟踪（朝目标方向投影）：`rew_scale_target_velocity * exp(-error^2 / std^2)`，其中期望速度=clamp(目标水平距离, 上限)，实际速度=沿目标方向的根部速度分量。
  - 足空中时间（双足交替单支撑，带阈值截断与“在移动才计分”的遮罩）：由接触传感器累计；实现见 `g1_reach_env.py:434–467`。
  - 存活奖励：`rew_scale_alive * (1 - terminated)`；成功奖励：`success_bonus * 1[hand_dist<th, facing>th]`。
- 主要代价项
  - 动作变化率 L2：`rew_scale_action_rate * sum((a_t - a_{t-1})^2)`，与 velocity 任务的 `action_rate_l2` 保持一致。
  - 关节速度 L2：`rew_scale_joint_vel * sum(joint_vel^2)`。
  - 扁平姿态惩罚：`rew_scale_flat_orientation * sum(projected_gravity_xy^2)`，改为与 velocity 任务相同的“flat orientation L2”度量。
  - 竖直速度 L2：`rew_scale_lin_vel_z * root_lin_vel_z^2`，抑制蹦跳。
  - 动力学正则：`rew_scale_dof_acc * sum(joint_acc^2)` 与 `rew_scale_dof_torque * sum(torque^2)`，聚焦髋/膝/踝链条。
  - 关节回中（全部）与分组回中（hip/arms/fingers/torso）：按与默认位姿的偏差、并归一化到各关节可动范围后求和；分组项还可分别加权。
  - 踝关节限位：对软限位留出 0.02 rad 余量后统计越界量。
  - 脚滑：接触时足端速度范数惩罚（阈值由 `feet_slide_contact_threshold` 控制）。
  - 终止惩罚：`rew_scale_termination * terminated`（与速度任务一致，权重通常很大）。
- 典型权重（`g1_reach_env_cfg.py:105–124`）
  - 任务主信号：hand 16.0、body 1.0、facing 2.0、upright 5.0、target_velocity 2.0、success 24.0。
  - 步态塑形：feet_air_time 0.25、feet_slide -0.1。
  - 稳定/代价：action_rate -0.005、joint_vel -0.001、flat_orientation -0.1、lin_vel_z -0.2、joint_center -0.01、分组回中（hip/arms -0.1、fingers -0.05、torso -0.1）、dof_acc -1.25e-7、dof_torque -1.5e-7、ankle_limits -1.0。
  - 存活/失败：alive 0.5、termination -200.0。

---

## Velocity（Manager, Rough）奖励项与实现要点
- 定义入口：`rough_env_cfg.py:19–100`（`G1Rewards`），以及 `__post_init__` 中的权重修订（`rough_env_cfg.py:132–144`）。原子函数来自 `.../velocity/mdp/rewards.py` 与 `isaaclab/.../mdp/rewards.py`。
- 命令跟踪（指数核）
  - 线速度 XY（机体偏航对齐系）：`track_lin_vel_xy_yaw_frame_exp`，权重 1.0（`rough_env_cfg.py:24–28`）。
  - 角速度 Z（世界系）：`track_ang_vel_z_world_exp`，权重 2.0（`rough_env_cfg.py:29–31`）。
- 步态塑形/接触
  - 足空中时间（双足正向版本）：`feet_air_time_positive_biped`，权重 0.25，阈值 0.4（`rough_env_cfg.py:32–40`）。
  - 脚滑惩罚：`feet_slide`，权重 -0.1（`rough_env_cfg.py:41–48`）。
- 姿态/稳定
  - 扁平姿态 L2：`flat_orientation_l2`，权重 -1.0（`rough_env_cfg.py:135`）。
  - 竖直线速度 L2：`lin_vel_z_l2` 置权重 0.0（`rough_env_cfg.py:133`）。
- 动作/动力学代价
  - 动作变化率 L2：`action_rate_l2`，权重 -0.005（`rough_env_cfg.py:136`）。
  - 关节加速度 L2（只在髋/膝）：`dof_acc_l2`，权重 -1.25e-7（`rough_env_cfg.py:137–140`）。
  - 关节力矩 L2（髋/膝/踝）：`dof_torques_l2`，权重 -1.5e-7（`rough_env_cfg.py:141–144`）。
- 关节约束/回中
  - 限位（踝）：`joint_pos_limits`，权重 -1.0（`rough_env_cfg.py:50–55`）。
  - 分组回中：hip/arms/fingers/torso 的 `joint_deviation_l1`（`rough_env_cfg.py:56–100`）。
- 失败惩罚
  - `is_terminated`，权重 -200.0（`rough_env_cfg.py:23`）。

---

## Velocity（Manager, Flat）在 Rough 基础上的改动
- 奖励改动（`flat_env_cfg.py:27–37`）
  - `track_ang_vel_z_exp` 权重从 2.0 降到 1.0。
  - `lin_vel_z_l2` 从 0.0 调整为 -0.2（对竖直速度加入抑制）。
  - `feet_air_time` 权重提升至 0.75，阈值 0.4。
  - `dof_acc_l2` 调整为 -1.0e-7；`dof_torques_l2` 提升至 -2.0e-6 且仅限髋/膝（去掉踝）。
- 地形/观测简化：改平地、移除高度扫描与课程学习（`flat_env_cfg.py:18–25`）。
- 命令范围：放宽横向速度范围至 [-0.5, 0.5]（`flat_env_cfg.py:39–41`）。

---

## 关键相同点
- 强化“存活/失败”结构：两类任务都使用大权重的终止惩罚（-200），Reach 另有小幅存活正奖励以稳住学习初期。
- 步态与接触塑形：均使用基于接触传感器的足空中时间正向项与脚滑惩罚（实现机制一致，阈值/遮罩细节略有不同）。
- 姿态与安全约束：都惩罚不理想的姿态（扁平/直立相关）与关节约束（限位与回中）；Reach 在分组回中时继续使用基于关节范围的归一化以避免尺度偏置。
- 动作与动力学正则：都包含“动作变化率”惩罚、关节速度 L2、竖直速度 L2（Flat 与 Reach 启用）以及针对髋/膝/踝链的关节加速度与力矩正则。
- 指数核的追踪项：Reach 用于“朝目标速度”塑形；Velocity 用于“命令速度”追踪（线/角），思路一致。

---

## 关键不同点
- 任务驱动信号
  - Reach：以“手/躯干到目标、朝向、成功 Bonus”为主驱动，速度塑形是辅助（沿目标方向）。
  - Velocity：以“跟踪底座速度命令”为主驱动，不含“手部/目标”相关奖励与成功 Bonus。
- 关节回中度量
  - Reach：对关节偏差先按各自可动范围归一化再求和，更“公平”；并提供整机回中与分组回中两层次。
  - Velocity：直接 `joint_deviation_l1`（原始角度偏差求和），未按关节范围归一，简单直接。
- 足空中时间门控
  - Reach：使用“是否需要移动”的遮罩（由到目标的期望速度推导）来抑制静止时的空中奖励。
  - Velocity：用“命令幅度阈值”门控（命令小于阈值则不计奖励）。
- 竖直速度/动力学权重
  - Reach：默认启用 `lin_vel_z`、`dof_acc`、`dof_torques` 惩罚，权重与 Rough 任务一致（torques）或参考 Flat（lin_vel_z）。
  - Velocity：Rough 任务将 `lin_vel_z` 权重设为 0，而 Flat 任务启用 -0.2；加速度/力矩权重也因场景而异。
- 其他
  - Reach 专有：成功判据（手距与朝向共同满足）与 `success_bonus`。
  - Velocity 专有：`lin_vel_z_l2`、扁平姿态 L2 的较强权重，以及更明确的命令追踪项（线/角）。

---

## 设计影响与调参建议
- Reach 可借鉴
  - 适度加入 `dof_torques_l2` 或 `dof_acc_l2` 以抑制“甩臂/摆动”造成的高能耗或不稳定加速度，尤其在接近目标的细致动作阶段。
  - 如需更强的“站稳”偏好，可把扁平姿态从 L1（|roll|+|pitch|）改为 L2 投影版本或提升其权重；注意避免过度限制导致无法迈步/触 reach。
  - 若出现“脚步过小/不抬脚”，可适当提高 `feet_air_time` 或降低遮罩阈值；若出现“滑步”，可降低 `feet_slide_contact_threshold` 或增大其惩罚权重。
- Velocity（Rough/Flat）可借鉴
  - 引入“面向目标/方向”的 shaping（如以导航方向定义 facing）用于导航型速度任务，或轻量级“体-目标距离” shaping 以鼓励向目标区域移动。
  - 在平地任务（Flat）中，适当引入动作幅值 L2（不是变化率）有助于进一步抑制抖动。
  - 分组回中可按关节范围做归一化，避免不同关节尺度差异导致的偏置。

---

## 快速对照（摘要）
- Reach（Direct）
  - 主驱动：手/体到目标、朝向、成功 Bonus；含“朝目标速度”指数核。
  - 稳定/正则：直立、flat orientation L2、竖直速度 L2、动作变化率、关节速度、关节/踝约束、脚滑、髋-踝动力学正则；失败大惩罚。
- Velocity（Rough/Flat）
  - 主驱动：线/角速度命令指数核（Rough 更强调角速度，Flat 降权）。
  - 稳定/正则：扁平姿态 L2、竖直速度 L2（Flat 启用）、动作变化率、关节加速度/力矩、足空中时间、脚滑、关节回中与限位；失败大惩罚。

---

## 关键代码位置（便于快速跳转）
- Reach 实现与权重
  - 计算：`source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env.py:478`
  - 权重：`source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env_cfg.py:105`
- Velocity（Rough/Flat）配置
  - Rough 奖励：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py:19`
  - Rough 权重修订：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py:132`
  - Flat 调整：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py:27`
- 奖励原子函数
  - Locomotion 专用：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
  - 通用项（L2/L1/动作/接触等）：`source/isaaclab/isaaclab/envs/mdp/rewards.py`

> 注：以上路径与行号基于当前仓库版本，后续若有变更，请以实际文件为准。
