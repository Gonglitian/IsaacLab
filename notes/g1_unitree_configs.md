# Unitree G1 系列配置差异调研

调研文件：`source/isaaclab_assets/isaaclab_assets/robots/unitree.py`（参考行号约 272–610）。

| 配置 | USD/碰撞 | 传感/重力/基座 | 主要执行器与用途 |
| --- | --- | --- | --- |
| `G1_CFG` | `g1.usd` 全碰撞 | 启用接触传感，重力开启，移动基座 | 腿/脚/臂均为隐式关节力，适合完整 humanoid 场景 |
| `G1_MINIMAL_CFG` | `g1_minimal.usd` 精简碰撞 | 继承 `G1_CFG` 其他设置 | 通过删除多余碰撞体加速仿真 |
| `G1_29DOF_CFG` | `g1.usd`（ISAAC Nucleus） | 默认关闭接触传感，可设定固定或移动基座 | 腿脚使用直流电机，腰/臂/手使用隐式执行器，聚焦 29DOF locomanipulation |
| `G1_INSPIRE_FTP_CFG` | `g1_29dof_inspire_hand.usd` + Inspire 手 | 启用接触传感，重力关闭，基座固定 | 在 `G1_29DOF_CFG` 基础上替换手部，臂/手执行器参数为抓取调优 |

## `G1_CFG`（行 272–360）
- 场景：`{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd`，Solver 迭代 8/4，自碰撞关闭，接触传感开启。
- 初始状态：腰高约 0.74 m，给定膝/踝/手臂/手指默认姿态（例如膝 0.42 rad、踝 -0.23 rad）。
- 执行器划分：
  - **legs**：隐式执行器，髋/膝/躯干统一 150–200 N·m 刚度、5 N·m·s 阻尼，并设置 armature 0.01。
  - **feet**：隐式，踝关节 20/2 刚度/阻尼，effort 20。
  - **arms**：隐式，手臂/手指统一 40/10 刚度/阻尼，手指 armature 降至 0.001。  
适用于一般 humanoid 任务，保持完整碰撞与传感信息。

## `G1_MINIMAL_CFG`（行 380–386）
- 通过 `G1_CFG.copy()` 构建，仅将 USD 路径切换到 `g1_minimal.usd`。
- 作用：删除多数碰撞网格，加快 rough-terrain 或大规模并行场景的仿真；动力学/控制参数保持一致。

## `G1_29DOF_CFG`（行 388–549）
- 资源路径：改为 `{ISAAC_NUCLEUS_DIR}` 版本的 `g1.usd`；默认关闭接触传感器（可按需开启）。
- 重要属性：
  - `fix_root_link` 可配置（默认 False），允许在 locomanipulation 中切换浮动或固定基座。
  - 初始姿态更直立（膝 0.30 rad），并在 w-x-y-z 表示中提供 90° 绕 x 的旋转。
- 执行器拆分更细：
  - **legs/feet**：直流电机（`DCMotorCfg`），显式给出 effort/velocity/stiffness/damping/armature 与饱和力矩。
  - **waist**：隐式执行器，但采用高刚度（5000）保持躯干稳定。
  - **arms**：隐式执行器，肩/肘/腕共享 3000 刚度、10 阻尼。
  - **hands**：隐式执行器，面向 index/middle/thumb 三指，刚度 20、阻尼 2。
- 面向“locomotion + manipulation”任务，`prim_path` 预设为 `/World/envs/env_.*/Robot` 以便多环境克隆。

## `G1_INSPIRE_FTP_CFG`（行 566–610）
- 基于 `G1_29DOF_CFG` 复制并替换资源为 `g1_29dof_inspire_hand.usd`（包含 Inspire 五指手）。
- 环境设置：接触传感开启、重力禁用（用于固定基座抓取）、`fix_root_link=True`，初始位置 1 m，所有关节零姿态。
- 执行器调优：
  - **arms**：仍为隐式执行器，但阻尼从 10 提升到 100，提升受力稳定性。
  - **hands**：覆盖 index/middle/thumb/ring/pinky，effort 降至 30，刚度 10、阻尼 0.2，提升手指柔顺性。
- 用途：台式/抓取实验，根链接固定 + 关闭重力避免整机摔倒，通过 Inspire 手实现高自由度抓取。

## 选型建议
- **全身仿真且关注原始碰撞/传感**：使用 `G1_CFG`。
- **大规模并行或自定义地形（碰撞简化）**：选择 `G1_MINIMAL_CFG`。
- **需要精细手臂/手爪控制，并在 manager-based locomanipulation 任务中使用**：`G1_29DOF_CFG`。
- **固定基座 + Inspire 手抓取任务**：`G1_INSPIRE_FTP_CFG`，可直接用于 teleop/抓取示例。
