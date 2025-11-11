# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class G1ReachEnvCfg(DirectRLEnvCfg):
    """Configuration for the Unitree G1 reach task."""

    # env
    decimation = 4
    episode_length_s = 5.0
    action_space = 43
    observation_space = 148
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        use_fabric=True,
        physx=PhysxCfg(
        gpu_max_rigid_contact_count=2**23,         # e.g., 8,388,608
        gpu_max_rigid_patch_count=2**20,           # e.g., 1,048,576
        gpu_found_lost_pairs_capacity=2**22,       # e.g., 4,194,304
        gpu_total_aggregate_pairs_capacity=2**22   # 可一并提升
    )
    )

    # robot(s)
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # marker for the spatial target
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/G1ReachTarget",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # controllable elements / reference names
    right_hand_body_name = "right_wrist_yaw_link"
    torso_body_name = "torso_link"

    # action scaling
    action_scale = 0.35  # [rad] relative to default pose
    action_smoothing = 0.15

    # phased reward parameters (阶段性奖励参数)
    facing_threshold_for_movement: float = 0.7
    """转向阈值：facing > 此值时才启用移动相关奖励 (cos(45°) ≈ 0.707)"""

    # observation normalization scales
    lin_vel_scale = 1.0
    ang_vel_scale = 0.25
    dof_vel_scale = 0.1

    # reward / termination parameters
    reach_threshold = 0.2  # [m]
    success_heading_threshold = 0.866  # cos(30°)
    upright_dot_threshold = 0.4
    termination_height = 0.65  # [m]
    target_radius_range = (1, 1.5)  # [m]
    target_height_range = (0.8, 1)  # [m]
    base_xy_range = (-0.25, 0.25)  # random XY shift per reset [m]
    base_yaw_range = (-0.5 * math.pi, 0.5 * math.pi)  # random heading offset [rad]
    
    # reset randomization
    base_lin_vel_range = (-0.2, 0.2)
    base_ang_vel_range = (-0.2, 0.2)
    joint_pos_noise = 0.1
    joint_vel_noise = 0.2

    # target-directed locomotion shaping
    target_velocity_max_cmd = 1.0
    target_velocity_tracking_std = 0.5
    feet_air_time_threshold = 0.4
    feet_air_time_command_threshold = 0.1
    feet_slide_contact_threshold = 1.0
    hand_pose_desired_dir = (0.0, 0.0, 1.0)
    hand_pose_target_radius = 0.35

    # positive task rewards / bonuses
    rew_scale_alive = 0.05
    rew_scale_hand_target = 2
    rew_scale_body_target = 2
    rew_scale_facing = 0.5
    rew_scale_upright = 0.5
    rew_scale_target_velocity = 5
    rew_scale_feet_air_time = 6
    rew_scale_hand_pose = 2.0
    rew_scale_torso_upright = -0.2
    success_bonus = 10

    # contact-related penalties
    rew_scale_feet_slide = -0.1
    rew_scale_ankle_limits = -1.5
    rew_scale_termination = -200.0

    # action / velocity penalties
    rew_scale_action_rate = -0.01
    rew_scale_action_smooth = -0.03
    rew_scale_joint_vel = -0.0004
    rew_scale_lin_vel_z = -0.2
    rew_scale_dof_acc = -1.25e-7
    rew_scale_dof_torque = -3.0e-7

    # pose alignment & joint deviation penalties
    rew_scale_flat_orientation = -0.1
    rew_scale_joint_center = -0.05
    rew_scale_joint_hip = -0.12
    rew_scale_joint_arms = -0.15
    rew_scale_joint_fingers = -0.08
    rew_scale_joint_torso = -0.12
    
    # left hand stability penalty (左手稳定性惩罚)
    rew_scale_left_hand = -0.25
    """惩罚左手偏离默认位置"""

    def __post_init__(self):
        # keep render interval aligned with control frequency
        self.sim.render_interval = self.decimation
        # ensure USD prims provide contact data for the added sensor
        if self.robot_cfg.spawn is not None:
            self.robot_cfg.spawn.activate_contact_sensors = True
