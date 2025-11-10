# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils

from .g1_reach_env_cfg import G1ReachEnvCfg


class G1ReachEnv(DirectRLEnv):
    cfg: G1ReachEnvCfg

    def __init__(self, cfg: G1ReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # robot handles
        self._dof_ids, self._dof_names = self.robot.find_joints(".*")
        self.num_dof = len(self._dof_ids)

        if self.num_dof != self.single_action_space.shape[0]:
            raise RuntimeError(
                f"Configured action space ({self.single_action_space.shape[0]}) "
                f"does not match articulation DOF count ({self.num_dof})."
            )

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        joint_limits = self.robot.data.soft_joint_pos_limits[0]
        self.joint_lower_limits = joint_limits[:, 0]
        self.joint_upper_limits = joint_limits[:, 1]
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        self.root_state = self.robot.data.root_state_w
        self.lin_vel = self.robot.data.root_lin_vel_w
        self.ang_vel = self.robot.data.root_ang_vel_w
        self.root_quat = self.robot.data.root_quat_w

        self.prev_actions = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float32, device=self.device)
        self.smooth_joint_targets = self.default_joint_pos.clone()

        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.goal_markers = VisualizationMarkers(self.cfg.target_marker_cfg)

        hand_body_ids, _ = self.robot.find_bodies(self.cfg.right_hand_body_name)
        if len(hand_body_ids) == 0:
            raise ValueError(f"Unable to find body named '{self.cfg.right_hand_body_name}' on the G1 articulation.")
        self._right_hand_body_id = hand_body_ids[0]

        # helper tensors
        self._env_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self._env_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
        self._hand_forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

    #
    # Implementation of DirectRLEnv abstract methods.
    #

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2700.0, color=(0.9, 0.9, 0.9))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        desired = self.default_joint_pos + self.actions * self.cfg.action_scale
        desired = torch.clamp(desired, self.joint_lower_limits, self.joint_upper_limits)
        self.smooth_joint_targets = (
            self.cfg.action_smoothing * self.smooth_joint_targets + (1.0 - self.cfg.action_smoothing) * desired
        )
        self.robot.set_joint_position_target(self.smooth_joint_targets, joint_ids=self._dof_ids)
        self.prev_actions[:] = self.actions

    def _get_observations(self) -> dict:
        root_pos = self.root_state[:, :3]
        root_quat = self.root_quat
        hand_pos = self.robot.data.body_pos_w[:, self._right_hand_body_id]
        hand_quat = self.robot.data.body_quat_w[:, self._right_hand_body_id]

        to_target_root = self.target_pos - root_pos
        to_target_hand = self.target_pos - hand_pos

        heading_vec = math_utils.quat_apply(root_quat, self._env_x)
        heading_vec = math_utils.normalize(heading_vec)
        dir_to_target = math_utils.normalize(to_target_root)

        up_vec = math_utils.quat_apply(root_quat, self._env_z)
        up_proj = up_vec[:, 2]

        facing = torch.sum(heading_vec * dir_to_target, dim=-1)
        hand_forward = math_utils.quat_apply(hand_quat, self._hand_forward_local)
        hand_facing = torch.sum(math_utils.normalize(hand_forward) * math_utils.normalize(to_target_hand), dim=-1)

        joint_pos_normalized = torch_utils_scale(self.joint_pos, self.joint_lower_limits, self.joint_upper_limits)

        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
        euler = torch.stack((roll, pitch, yaw), dim=-1)

        obs = torch.cat(
            (
                root_pos[:, 2:3],
                self.lin_vel * self.cfg.lin_vel_scale,
                self.ang_vel * self.cfg.ang_vel_scale,
                euler,
                up_proj.unsqueeze(-1),
                facing.unsqueeze(-1),
                joint_pos_normalized,
                self.joint_vel * self.cfg.dof_vel_scale,
                self.prev_actions,
                to_target_root,
                to_target_hand,
                hand_facing.unsqueeze(-1),
            ),
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self.root_state[:, :3]
        root_quat = self.root_quat
        hand_pos = self.robot.data.body_pos_w[:, self._right_hand_body_id]

        to_target_root = self.target_pos - root_pos
        to_target_hand = self.target_pos - hand_pos

        hand_dist = torch.linalg.norm(to_target_hand, dim=-1)
        body_dist = torch.linalg.norm(to_target_root[:, :2], dim=-1)

        dir_to_target = math_utils.normalize(to_target_root)
        heading_vec = math_utils.normalize(math_utils.quat_apply(root_quat, self._env_x))
        facing = torch.sum(heading_vec * dir_to_target, dim=-1)

        up_vec = math_utils.quat_apply(root_quat, self._env_z)
        upright = up_vec[:, 2]

        success = (hand_dist < self.cfg.reach_threshold) & (facing > self.cfg.success_heading_threshold)

        reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_hand_target,
            self.cfg.rew_scale_body_target,
            self.cfg.rew_scale_facing,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_joint_vel,
            self.cfg.success_bonus,
            hand_dist,
            body_dist,
            facing,
            upright,
            self.actions,
            self.joint_vel,
            success,
            self.reset_terminated,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["hand_target_dist"] = hand_dist.mean().item()
        self.extras["log"]["body_target_dist"] = body_dist.mean().item()
        self.extras["log"]["facing_dot"] = facing.mean().item()
        self.extras["log"]["upright"] = upright.mean().item()
        self.extras["log"]["success_rate"] = success.float().mean().item()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.root_state[:, :3]
        root_quat = self.root_quat

        up_vec = math_utils.quat_apply(root_quat, self._env_z)
        upright = up_vec[:, 2]
        fell = (root_pos[:, 2] < self.cfg.termination_height) | (upright < self.cfg.upright_dot_threshold)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self.joint_vel[env_ids])
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # randomize base XY offsets similar to locomotion velocity envs
        if self.cfg.base_xy_range is not None:
            low, high = self.cfg.base_xy_range
            shift_xy = torch.rand((len(env_ids), 2), device=self.device) * (high - low) + low
            root_state[:, 0] += shift_xy[:, 0]
            root_state[:, 1] += shift_xy[:, 1]

        if self.cfg.base_yaw_range is not None:
            yaw_low, yaw_high = self.cfg.base_yaw_range
            yaw_delta = torch.rand((len(env_ids),), device=self.device) * (yaw_high - yaw_low) + yaw_low
            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw_delta), torch.zeros_like(yaw_delta), yaw_delta
            )
            root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

        if self.cfg.base_lin_vel_range is not None:
            lin_low, lin_high = self.cfg.base_lin_vel_range
            lin_vel = torch.rand((len(env_ids), 3), device=self.device) * (lin_high - lin_low) + lin_low
            root_state[:, 7:10] = lin_vel

        if self.cfg.base_ang_vel_range is not None:
            ang_low, ang_high = self.cfg.base_ang_vel_range
            ang_vel = torch.rand((len(env_ids), 3), device=self.device) * (ang_high - ang_low) + ang_low
            root_state[:, 10:] = ang_vel

        if self.cfg.joint_pos_noise > 0.0:
            noise = (torch.rand_like(joint_pos) * 2.0 - 1.0) * self.cfg.joint_pos_noise
            joint_pos = torch.clamp(joint_pos + noise, self.joint_lower_limits, self.joint_upper_limits)

        if self.cfg.joint_vel_noise > 0.0:
            joint_vel_noise = (torch.rand_like(joint_vel) * 2.0 - 1.0) * self.cfg.joint_vel_noise
            joint_vel = joint_vel + joint_vel_noise

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.smooth_joint_targets[env_ids] = joint_pos
        self.prev_actions[env_ids] = 0.0

        self._sample_targets(env_ids)

    #
    # Debug visualization helpers.
    #

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.goal_markers.set_visibility(debug_vis)

    def _debug_vis_callback(self, _event):
        self.goal_markers.visualize(self.target_pos)

    #
    # Helpers.
    #

    def _sample_targets(self, env_ids: Sequence[int]):
        num = len(env_ids)
        device = self.device
        radius = torch.rand((num,), device=device)
        radius = radius * (self.cfg.target_radius_range[1] - self.cfg.target_radius_range[0]) + self.cfg.target_radius_range[0]
        yaw = torch.rand((num,), device=device) * (2 * math.pi) - math.pi
        x = radius * torch.cos(yaw)
        y = radius * torch.sin(yaw)
        z = torch.rand((num,), device=device)
        z = z * (self.cfg.target_height_range[1] - self.cfg.target_height_range[0]) + self.cfg.target_height_range[0]
        targets = torch.stack((x, y, z), dim=-1) + self.scene.env_origins[env_ids]
        self.target_pos[env_ids] = targets
        self.goal_markers.visualize(self.target_pos)


def torch_utils_scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale joints into [-1, 1] range."""
    offset = (lower + upper) * 0.5
    extent = (upper - lower) * 0.5
    return torch.clamp((x - offset) / torch.clamp(extent, min=1e-6), min=-1.0, max=1.0)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_hand_target: float,
    rew_scale_body_target: float,
    rew_scale_facing: float,
    rew_scale_upright: float,
    rew_scale_action_rate: float,
    rew_scale_joint_vel: float,
    success_bonus: float,
    hand_dist: torch.Tensor,
    body_dist: torch.Tensor,
    facing: torch.Tensor,
    upright: torch.Tensor,
    actions: torch.Tensor,
    joint_vel: torch.Tensor,
    success: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    reach_reward = rew_scale_hand_target * torch.exp(-2.5 * hand_dist)
    travel_reward = rew_scale_body_target * torch.exp(-0.5 * body_dist)
    facing_reward = rew_scale_facing * torch.clamp(facing, min=0.0)
    upright_reward = rew_scale_upright * torch.clamp(upright, min=0.0)
    alive_reward = rew_scale_alive * (1.0 - reset_terminated.float())
    action_penalty = rew_scale_action_rate * torch.sum(actions**2, dim=-1)
    joint_vel_penalty = rew_scale_joint_vel * torch.sum(joint_vel**2, dim=-1)
    success_reward = success_bonus * success.float()

    total = (
        reach_reward
        + travel_reward
        + facing_reward
        + upright_reward
        + alive_reward
        + success_reward
        + action_penalty
        + joint_vel_penalty
    )
    return total
