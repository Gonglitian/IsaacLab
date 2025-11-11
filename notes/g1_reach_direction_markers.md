# G1 Reach 方向可视化标记说明

## 功能概述

为 G1 Reach 任务添加了方向箭头标记，用于可视化与奖励函数相关的方向向量，方便 debug 和理解奖励函数的计算。

## 可视化的方向向量

实现了5个方向箭头，分别用不同颜色标识：

### 1. 🔴 **Heading Vector** (红色箭头)
- **位置**: 机器人基座上方 0.5m
- **含义**: 机器人的朝向向量 (heading_vec)
- **计算方式**: `quat_apply(root_quat, [1,0,0])`
- **相关奖励**: `rew_scale_facing` - 奖励机器人朝向目标

### 2. 🟢 **Direction to Target** (绿色箭头)
- **位置**: 机器人基座上方 0.7m
- **含义**: 从机器人基座指向目标的单位方向向量
- **计算方式**: `normalize(target_pos - root_pos)`
- **相关奖励**: `rew_scale_facing`, `rew_scale_target_velocity` - 引导机器人移动向目标

### 3. 🔵 **Up Vector** (蓝色箭头)
- **位置**: 机器人基座上方 0.3m
- **含义**: 机器人的向上方向 (up_vec)
- **计算方式**: `quat_apply(root_quat, [0,0,1])`
- **相关奖励**: `rew_scale_upright` - 奖励机器人保持直立

### 4. 🟡 **Hand Forward** (黄色箭头)
- **位置**: 机器人右手腕位置
- **含义**: 手腕的朝向向量
- **计算方式**: `quat_apply(hand_quat, [0,0,1])`
- **相关奖励**: `rew_scale_hand_pose` - 奖励手部姿态对齐期望方向

### 5. 🟣 **To Target Hand** (品红色箭头)
- **位置**: 机器人右手腕位置
- **含义**: 从手腕指向目标的方向
- **计算方式**: `normalize(target_pos - hand_pos)`
- **相关奖励**: `rew_scale_hand_target` - 奖励手部接近目标

## 使用方法

### 1. 配置文件控制

在 `g1_reach_env_cfg.py` 中，可以通过以下参数控制是否启用方向标记：

```python
# 启用方向标记 (默认为 True)
enable_direction_markers: bool = True

# 禁用方向标记
enable_direction_markers: bool = False
```

### 2. 运行时启用可视化

运行训练或测试脚本时，需要：

1. 确保使用了GUI模式（不是headless模式）
2. 在环境配置中启用debug visualization

```bash
# 训练示例
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 64

# 播放示例 (自动启用debug_vis)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 4 \
    --checkpoint <path_to_checkpoint>
```

3. 在模拟器界面中按 `d` 键切换debug可视化

## Debug 技巧

### 检查 facing 奖励是否正确
- **红色箭头** (heading) 和 **绿色箭头** (dir_to_target) 应该尽量对齐
- 当两个箭头方向一致时，`facing` 值应该接近 1.0
- TensorBoard中查看 `log/facing_dot` 的值

### 检查 hand_pose 奖励是否正确
- **黄色箭头** (hand_forward) 应该朝向目标或期望方向
- `hand_pose_desired_dir` 配置决定了期望的手部朝向（默认 [0,0,1] 即向上）
- TensorBoard中查看 `reward/hand_pose` 的值

### 检查 upright 奖励是否正确
- **蓝色箭头** (up_vec) 应该竖直向上
- 如果机器人倾倒，蓝色箭头会偏离垂直方向
- TensorBoard中查看 `log/upright` 的值（应该接近 1.0）

### 检查 reach 目标是否合理
- **品红色箭头** (to_target_hand) 指向目标位置
- 手部应该沿着品红色箭头方向移动
- TensorBoard中查看 `log/hand_target_dist` 的距离

## 技术实现细节

### 箭头转换函数

`_direction_to_arrow_transform(direction, arrow_length)` 函数负责将方向向量转换为箭头的四元数和缩放：

- 输入：归一化的方向向量 (N, 3)
- 输出：四元数 (N, 4) 和缩放 (N, 3)
- 方法：使用 yaw 和 pitch 角度计算旋转

### 箭头配置

所有箭头都基于 `arrow_x.usd` 模型，使用不同的颜色材质：
- 红色：RGB(1.0, 0.0, 0.0)
- 绿色：RGB(0.0, 1.0, 0.0)
- 蓝色：RGB(0.0, 0.0, 1.0)
- 黄色：RGB(1.0, 1.0, 0.0)
- 品红：RGB(1.0, 0.0, 1.0)

## 性能影响

- 方向标记的更新在 `_debug_vis_callback` 中进行，仅在启用debug可视化时生效
- 禁用 `enable_direction_markers` 可以完全跳过标记的初始化和更新
- 对训练性能几乎无影响（仅在渲染时计算）

## 未来扩展

可以根据需要添加更多方向标记：
- 速度方向
- 期望速度方向
- 脚步方向
- COM (质心) 方向

只需在配置文件中添加新的 marker 配置，然后在环境中初始化和更新即可。

