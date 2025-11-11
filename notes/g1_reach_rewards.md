# G1 Reach 环境奖励函数调研

主要文件：`source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env.py`（行 200–565）与 `source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env_cfg.py`（行 84–126）。

## 指标与信号构建
- **目标几何（行 200–215）**  
  - `hand_dist`：右手质心与目标球之间的欧氏距离。  
  - `body_dist`：机器人根部到目标的水平距离（仅 xy），用于“走近” shaping。  
  - `facing`：机体朝前向量与目标方向单位向量的点积。  
  - `upright`：世界 z 轴在机体坐标中的投影 z 分量。
- **手部姿态（行 200–230）**  
  - 获取右手四元数，利用局部 `hand_forward_local`（默认沿掌心正向）变换到世界系，得到 `hand_forward`。  
  - `hand_pose_metric = clamp(hand_forward · hand_pose_desired_dir, ≥0) * exp(-hand_dist / hand_pose_target_radius)`，只有当手靠近目标且掌心对齐配置向量（默认世界 +Z）时才获得奖励。
- **姿态/速度指标（行 200–250）**  
  - `flat_orientation_metric = |projected_gravity_xy|^2`，匹配 locomotion 任务的 flat-orientation L2 惩罚。  
  - `lin_vel_z_metric = root_lin_vel_z^2` 抑制跳跃。  
  - `target_velocity_error`：目标速度（`clamp(body_dist, max=1.0)`）与沿目标方向的实际根部投影速度差。  
  - `action_rate = ||a_t - a_{t-1}||^2`。
- **足部与接触（行 206–290, 430–467）**  
  - `move_mask`：只有当期望速度超过 `feet_air_time_command_threshold` 时才计入步态奖励。  
  - `_compute_foot_metrics` 结合接触传感器历史，返回单足支撑时间（截断到 `feet_air_time_threshold`）与接触状态下的足端滑移速度积分。
- **关节/姿态偏差（行 211–238）**  
  - `joint_dev_total`、`hip/arms/fingers/torso_dev`：相对于默认关节姿态的 L1 偏差，并按可动范围归一化。  
  - `ankle_limit_metric`：踝关节在软限位 ±0.02 rad 内的越界量。  
  - `dof_acc_metric` 与 `dof_torque_metric`：髋/膝/踝关节的加速度与力矩 L2 和；若仿真未提供对应数据则为 0。
- **成功/终止**  
  - `success = (hand_dist < reach_threshold) ∧ (facing > success_heading_threshold)`（行 209）。  
  - 终止来自 `self.reset_terminated` 与 `_get_dones`（高度或翻倒）。

## 奖励项一览
权重均在 `G1ReachEnvCfg` 中配置，符号参见下表：

| 项目 | 权重字段 | 计算方式（未乘权重） | 意义 |
| --- | --- | --- | --- |
| `reach_reward` | `rew_scale_hand_target` | `exp(-2.5 * hand_dist)` | 右手靠近目标 |
| `travel_reward` | `rew_scale_body_target` | `exp(-0.5 * body_dist)` | 根部靠近目标 |
| `facing_reward` | `rew_scale_facing` | `max(facing, 0)` | 朝向目标 |
| `upright_reward` | `rew_scale_upright` | `max(upright, 0)` | 保持直立 |
| `alive_reward` | `rew_scale_alive` | `1 - terminated` | 存活激励 |
| `success_reward` | `success_bonus` | `1[success]` | 达成条件奖励 |
| `target_velocity_reward` | `rew_scale_target_velocity` | `exp(-error^2 / (std^2 + 1e-6))` | 鼓励朝目标方向行走速度 |
| `feet_air_time_reward` | `rew_scale_feet_air_time` | `feet_air_time_metric` | 正常步态、交替单支撑 |
| `termination_penalty` | `rew_scale_termination` | `terminated` | 失败惩罚 |
| `feet_slide_penalty` | `rew_scale_feet_slide` | `feet_slide_metric` | 抑制接触滑步 |
| `action_penalty` | `rew_scale_action_rate` | `∑ actions^2` | 动作幅值抑制 |
| `action_smooth_penalty` | `rew_scale_action_smooth` | `∑ (a_t - a_{t-1})^2` | 动作变化率平滑 |
| `joint_vel_penalty` | `rew_scale_joint_vel` | `∑ joint_vel^2` | 关节速度正则 |
| `flat_orientation_penalty` | `rew_scale_flat_orientation` | `flat_orientation_metric` | 控制 roll/pitch |
| `joint_center_penalty` | `rew_scale_joint_center` | `joint_dev_total` | 整体关节回中 |
| `hip/arms/fingers/torso_penalty` | 各自权重 | 对应 `joint_dev_*` | 针对特定部位的约束 |
| `ankle_penalty` | `rew_scale_ankle_limits` | `ankle_limit_metric` | 防止踝越界 |
| `lin_vel_z_penalty` | `rew_scale_lin_vel_z` | `lin_vel_z_metric` | 抑制竖直速度 |
| `joint_acc_penalty` | `rew_scale_dof_acc` | `dof_acc_metric` | 抑制髋/膝加速度 |
| `joint_torque_penalty` | `rew_scale_dof_torque` | `dof_torque_metric` | 抑制髋/膝/踝力矩 |
| `hand_pose_reward` | `rew_scale_hand_pose` | `clamp(hand_forward·dir, ≥0) * exp(-dist / radius)` | 约束右手到目标时的姿态 |

> 注：旧版 `rew_scale_action_smooth`（动作幅值惩罚）已移除，现与 locomotion reward 一致，仅保留动作差分 L2。

## 计算流程概述
1. **提取观测与几何关系**：`_get_rewards` 首先读取根部/手的位置、姿态与速度，并对目标向量进行标准化（行 200–215）。  
2. **构造奖励信号**：将上述指标传入 `compute_rewards` 脚本函数（行 252–317）。  
3. **日志记录**：在 `self.extras["log"]` 中记录手距、体距、朝向、直立度、成功率以及分项奖励/惩罚（行 300–338）。  
4. **动作记忆**：`self.prev_actions.copy_(self.actions)`，供下一步计算 `action_rate`（行 342）。  
5. **终止条件**：`_get_dones` 使用根部高度与 upright dot 检测跌倒，同时结合 episode 超时（行 348–362）。

## 关联配置
- 所有权重、阈值（如 `reach_threshold`, `feet_air_time_threshold`, `target_velocity_tracking_std`）位于 `G1ReachEnvCfg`，可直接通过任务配置或 CLI 覆盖。
- 关节分组在构造函数中根据名称正则表达式生成（行 52–74），需与 Unitree G1 关节命名保持一致。

通过上述结构，Reach 任务的奖励已经与 manager-based locomotion reward 库中的设计对齐：同样的 flat-orientation、动作差分、竖直速度与动力学正则，同时保留了触达目标所需的 Task-specific shaping。
