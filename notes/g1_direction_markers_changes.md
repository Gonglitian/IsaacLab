# G1 Reach 方向标记功能 - 更改总结

## 📝 更改概述

为 G1 Reach 任务添加了方向箭头可视化功能，用于 debug 和理解奖励函数中的方向计算。

## 🔧 修改的文件

### 1. `source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env_cfg.py`

**新增内容：**

- 导入箭头marker配置：
  ```python
  from isaaclab.markers.config import (
      RED_ARROW_X_MARKER_CFG,
      GREEN_ARROW_X_MARKER_CFG,
      BLUE_ARROW_X_MARKER_CFG,
  )
  from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
  ```

- 添加配置参数：
  - `enable_direction_markers: bool = True` - 控制是否启用方向标记
  - `heading_marker_cfg` - 红色箭头配置（机器人朝向）
  - `dir_to_target_marker_cfg` - 绿色箭头配置（指向目标）
  - `up_vec_marker_cfg` - 蓝色箭头配置（向上方向）
  - `hand_forward_marker_cfg` - 黄色箭头配置（手腕朝向）
  - `to_target_hand_marker_cfg` - 品红箭头配置（手到目标）

### 2. `source/isaaclab_tasks/isaaclab_tasks/direct/g1_reach/g1_reach_env.py`

**新增/修改内容：**

- **在 `__init__` 中初始化5个方向标记**：
  ```python
  if self.cfg.enable_direction_markers:
      self.heading_markers = VisualizationMarkers(self.cfg.heading_marker_cfg)
      self.dir_to_target_markers = VisualizationMarkers(self.cfg.dir_to_target_marker_cfg)
      # ... 其他标记
  ```

- **更新 `_set_debug_vis_impl` 方法**：
  - 添加所有方向标记的可见性控制

- **增强 `_debug_vis_callback` 方法**：
  - 计算5个方向向量
  - 调用 `_direction_to_arrow_transform` 转换为箭头四元数
  - 在适当位置可视化每个箭头

- **新增辅助方法 `_direction_to_arrow_transform`**：
  - 将方向向量转换为箭头的四元数和缩放
  - 使用 yaw/pitch 角度计算旋转
  - 支持自定义箭头长度

## 📋 新增文件

### 1. `notes/g1_reach_direction_markers.md`
详细的功能说明文档，包括：
- 5个方向向量的含义和用途
- 与奖励函数的对应关系
- 使用方法和 debug 技巧
- 技术实现细节

### 2. `scripts/tools/test_g1_direction_markers.py`
测试脚本，用于验证方向标记功能：
- 创建 G1 Reach 环境
- 启用方向标记可视化
- 运行随机策略并显示关键指标

## 🎨 方向标记说明

| 颜色 | 名称 | 显示位置 | 含义 | 相关奖励 |
|------|------|----------|------|----------|
| 🔴 红色 | Heading | 基座上方0.5m | 机器人朝向 | `rew_scale_facing` |
| 🟢 绿色 | Dir to Target | 基座上方0.7m | 指向目标方向 | `rew_scale_facing`, `rew_scale_target_velocity` |
| 🔵 蓝色 | Up Vector | 基座上方0.3m | 机器人向上 | `rew_scale_upright` |
| 🟡 黄色 | Hand Forward | 手腕位置 | 手腕朝向 | `rew_scale_hand_pose` |
| 🟣 品红 | To Target Hand | 手腕位置 | 手到目标 | `rew_scale_hand_target` |

## 🚀 使用方法

### 方法1: 使用测试脚本

```bash
# 基本测试 (4个环境)
./isaaclab.sh -p scripts/tools/test_g1_direction_markers.py

# 更多环境
./isaaclab.sh -p scripts/tools/test_g1_direction_markers.py --num_envs 16

# 禁用方向标记
./isaaclab.sh -p scripts/tools/test_g1_direction_markers.py --disable_markers
```

### 方法2: 在训练中使用

```bash
# 训练时可视化 (使用较少环境)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 64
```

然后在模拟器中按 `d` 键启用debug可视化。

### 方法3: 播放已训练的策略

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-G1-Reach-Direct-v0 \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/g1_reach_direct/<run_id>/model_<step>.pt
```

播放模式会自动启用debug可视化。

### 禁用方向标记

如果不需要方向可视化（例如大规模训练），可以在配置中设置：

```python
cfg.enable_direction_markers = False
```

或者创建一个新的配置类：

```python
@configclass
class G1ReachEnvCfgNoMarkers(G1ReachEnvCfg):
    enable_direction_markers: bool = False
```

## 🔍 Debug 示例

### 检查 facing 奖励

观察**红色箭头**（heading）和**绿色箭头**（dir_to_target）：
- 两箭头对齐 → facing 接近 1.0 → 正向奖励
- 两箭头垂直 → facing 接近 0.0 → 无奖励
- 两箭头相反 → facing 接近 -1.0 → 可能有惩罚

在 TensorBoard 中查看 `log/facing_dot` 验证。

### 检查 upright 奖励

观察**蓝色箭头**（up_vec）：
- 竖直向上 → upright 接近 1.0 → 正向奖励
- 倾斜 → upright < 1.0 → 奖励减少
- 倾倒 → upright < 0.4 → 触发终止

在 TensorBoard 中查看 `log/upright` 验证。

### 检查 hand_pose 奖励

观察**黄色箭头**（hand_forward）：
- 应该朝向配置的期望方向（默认向上 [0,0,1]）
- 越接近期望方向，奖励越高
- 只在手接近目标时激活（使用指数衰减）

在 TensorBoard 中查看 `reward/hand_pose` 验证。

## ⚙️ 技术细节

### 箭头转换算法

`_direction_to_arrow_transform` 方法将3D方向向量转换为箭头的四元数：

1. 计算水平面投影长度
2. 计算 yaw 角度（绕Z轴）
3. 计算 pitch 角度（绕Y轴）
4. 从欧拉角生成四元数

### 性能考虑

- 方向标记仅在 `_debug_vis_callback` 中更新
- 只在启用debug可视化时生效
- 对训练性能影响极小
- 可以通过 `enable_direction_markers=False` 完全禁用

## 📊 验证清单

- [x] 配置文件添加marker配置
- [x] 环境初始化5个marker
- [x] 实现方向向量到箭头的转换
- [x] 更新debug可视化回调
- [x] 添加开关控制
- [x] 创建测试脚本
- [x] 编写文档

## 🎯 未来改进

可能的扩展：
- 添加脚部速度箭头
- 添加期望速度箭头
- 添加COM (质心) 箭头
- 支持动态调整箭头长度
- 添加箭头颜色渐变（根据奖励值）

## 📞 问题反馈

如果遇到问题：
1. 检查 `enable_direction_markers` 是否为 True
2. 确认不是在 headless 模式下运行
3. 按 `d` 键切换debug可视化
4. 查看终端输出的错误信息

