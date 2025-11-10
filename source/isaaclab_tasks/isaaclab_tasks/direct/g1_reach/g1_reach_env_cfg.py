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
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class G1ReachEnvCfg(DirectRLEnvCfg):
    """Configuration for the Unitree G1 reach task."""

    # env
    decimation = 4
    episode_length_s = 12.0
    action_space = 43
    observation_space = 148
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        use_fabric=True,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=True,
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

    # controllable elements / reference names
    right_hand_body_name = "right_wrist_yaw_link"

    # action scaling
    action_scale = 0.35  # [rad] relative to default pose
    action_smoothing = 0.15

    # observation normalization scales
    lin_vel_scale = 1.5
    ang_vel_scale = 0.5
    dof_vel_scale = 0.05

    # reward / termination parameters
    reach_threshold = 0.12  # [m]
    success_heading_threshold = 0.75
    upright_dot_threshold = 0.4
    termination_height = 0.65  # [m]
    target_radius_range = (0.8, 2.6)  # [m]
    target_height_range = (1.0, 1.8)  # [m]
    base_xy_range = (-0.4, 0.4)  # random XY shift per reset [m]
    base_yaw_range = (-math.pi, math.pi)  # random heading offset [rad]
    base_lin_vel_range = (-0.4, 0.4)
    base_ang_vel_range = (-0.4, 0.4)
    joint_pos_noise = 0.15
    joint_vel_noise = 0.3

    rew_scale_alive = 1.0
    rew_scale_hand_target = 4.0
    rew_scale_body_target = 0.6
    rew_scale_facing = 1.5
    rew_scale_upright = 2.0
    rew_scale_action_rate = -0.01
    rew_scale_joint_vel = -0.002
    success_bonus = 12.0

    def __post_init__(self):
        # keep render interval aligned with control frequency
        self.sim.render_interval = self.decimation
