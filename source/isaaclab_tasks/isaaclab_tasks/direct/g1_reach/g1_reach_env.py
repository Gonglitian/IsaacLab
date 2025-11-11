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
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils

from .g1_reach_env_cfg import G1ReachEnvCfg


class G1ReachEnv(DirectRLEnv):
    cfg: G1ReachEnvCfg

    def __init__(self, cfg: G1ReachEnvCfg, render_mode: str | None = None, **kwargs):
        self._contact_sensor: ContactSensor | None = None
        self._feet_sensor_body_ids: list[int] = []
        self._feet_body_ids: list[int] = []
        super().__init__(cfg, render_mode, **kwargs)

        # robot handles
        self._dof_ids, self._dof_names = self.robot.find_joints(".*")
        self.num_dof = len(self._dof_ids)

        if self.num_dof != self.single_action_space.shape[0]:
            raise RuntimeError(
                f"Configured action space ({self.single_action_space.shape[0]}) "
                f"does not match articulation DOF count ({self.num_dof})."
            )

        joint_limits = self.robot.data.soft_joint_pos_limits[0]
        self.joint_lower_limits = joint_limits[:, 0]
        self.joint_upper_limits = joint_limits[:, 1]
        self.joint_range = torch.clamp(self.joint_upper_limits - self.joint_lower_limits, min=1e-3)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_pos_ref = self.default_joint_pos[0].clone()

        self.prev_actions = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float32, device=self.device)

        def build_group(patterns: list[str]) -> torch.Tensor:
            ids = self._collect_joint_indices(patterns)
            if len(ids) == 0:
                return torch.empty(0, dtype=torch.long, device=self.device)
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        self._joint_groups = {
            "hip": build_group([".*_hip_yaw_joint", ".*_hip_roll_joint"]),
            "arms": build_group(
                [
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ]
            ),
            "fingers": build_group([".*_hand_.*", ".*_thumb_.*"]),
            "torso": build_group(["waist_.*_joint"]),
        }
        self._ankle_joint_ids = build_group([".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
        self._dof_acc_joint_ids = build_group([".*_hip_.*", ".*_knee_joint"])
        self._dof_torque_joint_ids = build_group([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        desired_hand_dir = torch.tensor(self.cfg.hand_pose_desired_dir, dtype=torch.float32, device=self.device)
        desired_hand_dir = desired_hand_dir / torch.linalg.norm(desired_hand_dir).clamp_min(1.0e-6)
        self._hand_pose_desired_dir = desired_hand_dir
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
        self._contact_sensor = self.scene.sensors.get("contact_sensor")
        if self.cfg.contact_sensor is not None and self._contact_sensor is None:
            raise RuntimeError("Contact sensor failed to initialize despite being requested in the config.")
        if self._contact_sensor is not None:
            feet_sensor_ids, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link")
            self._feet_sensor_body_ids = feet_sensor_ids
        feet_body_ids, _ = self.robot.find_bodies(".*_ankle_roll_link")
        self._feet_body_ids = feet_body_ids
        self._has_foot_feedback = (
            self._contact_sensor is not None and len(self._feet_sensor_body_ids) > 0 and len(self._feet_body_ids) > 0
        )

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
        if self.cfg.contact_sensor is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
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

    def _get_observations(self) -> dict:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        hand_pos = self.robot.data.body_pos_w[:, self._right_hand_body_id]
        hand_quat = self.robot.data.body_quat_w[:, self._right_hand_body_id]
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

        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        joint_pos_normalized = torch_utils_scale(joint_pos, self.joint_lower_limits, self.joint_upper_limits)

        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
        euler = torch.stack((roll, pitch, yaw), dim=-1)

        obs = torch.cat(
            (
                root_pos[:, 2:3],
                self.robot.data.root_lin_vel_w * self.cfg.lin_vel_scale,
                self.robot.data.root_ang_vel_w * self.cfg.ang_vel_scale,
                euler,
                up_proj.unsqueeze(-1),
                facing.unsqueeze(-1),
                joint_pos_normalized,
                joint_vel * self.cfg.dof_vel_scale,
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
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        hand_pos = self.robot.data.body_pos_w[:, self._right_hand_body_id]
        hand_quat = self.robot.data.body_quat_w[:, self._right_hand_body_id]

        to_target_root = self.target_pos - root_pos
        to_target_hand = self.target_pos - hand_pos

        hand_dist = torch.linalg.norm(to_target_hand, dim=-1)
        body_dist = torch.linalg.norm(to_target_root[:, :2], dim=-1)
        hand_forward = math_utils.quat_apply(hand_quat, self._hand_forward_local)
        hand_forward = math_utils.normalize(hand_forward)
        hand_pose_alignment = torch.sum(hand_forward * self._hand_pose_desired_dir, dim=-1)
        hand_pose_mask = torch.exp(-hand_dist / (self.cfg.hand_pose_target_radius + 1.0e-6))
        hand_pose_metric = torch.clamp(hand_pose_alignment, min=0.0) * hand_pose_mask

        dir_to_target = math_utils.normalize(to_target_root)
        dir_to_target_xy = to_target_root[:, :2]
        dir_to_target_xy_norm = torch.linalg.norm(dir_to_target_xy, dim=-1, keepdim=True).clamp_min(1e-4)
        dir_to_target_xy_unit = dir_to_target_xy / dir_to_target_xy_norm
        heading_vec = math_utils.normalize(math_utils.quat_apply(root_quat, self._env_x))
        facing = torch.sum(heading_vec * dir_to_target, dim=-1)

        up_vec = math_utils.quat_apply(root_quat, self._env_z)
        upright = up_vec[:, 2]
        flat_orientation_metric = torch.sum(torch.square(up_vec[:, :2]), dim=-1)

        desired_speed = torch.clamp(body_dist, max=self.cfg.target_velocity_max_cmd)
        actual_speed_along_dir = torch.sum(root_lin_vel[:, :2] * dir_to_target_xy_unit, dim=-1)
        target_velocity_error = desired_speed - actual_speed_along_dir
        move_mask = desired_speed > self.cfg.feet_air_time_command_threshold
        feet_air_time_metric, feet_slide_metric = self._compute_foot_metrics(move_mask)

        success = (hand_dist < self.cfg.reach_threshold) & (facing > self.cfg.success_heading_threshold)

        joint_pos = self.robot.data.joint_pos

        def group_dev(indices: torch.Tensor):
            if indices.numel() == 0:
                return torch.zeros(self.num_envs, device=self.device)
            return torch.sum(
                torch.abs(joint_pos[:, indices] - self.default_joint_pos_ref[indices]) / self.joint_range[indices],
                dim=-1,
            )

        joint_dev_total = torch.sum(
            torch.abs(joint_pos - self.default_joint_pos_ref) / self.joint_range,
            dim=-1,
        )
        hip_dev = group_dev(self._joint_groups["hip"])
        arms_dev = group_dev(self._joint_groups["arms"])
        fingers_dev = group_dev(self._joint_groups["fingers"])
        torso_dev = group_dev(self._joint_groups["torso"])

        if self._ankle_joint_ids.numel() > 0:
            ankle_pos = joint_pos[:, self._ankle_joint_ids]
            ankle_low = self.joint_lower_limits[self._ankle_joint_ids] + 0.02
            ankle_high = self.joint_upper_limits[self._ankle_joint_ids] - 0.02
            ankle_violation = torch.clamp(ankle_pos - ankle_high, min=0.0) + torch.clamp(ankle_low - ankle_pos, min=0.0)
            ankle_limit_metric = ankle_violation.sum(dim=-1)
        else:
            ankle_limit_metric = torch.zeros(self.num_envs, device=self.device)

        action_delta = torch.sum((self.actions - self.prev_actions) ** 2, dim=-1)
        action_magnitude = torch.sum(self.actions**2, dim=-1)
        lin_vel_z_metric = torch.square(root_lin_vel[:, 2])
        joint_acc_data = getattr(self.robot.data, "joint_acc", None)
        if joint_acc_data is not None and self._dof_acc_joint_ids.numel() > 0:
            dof_acc_metric = torch.sum(torch.square(joint_acc_data[:, self._dof_acc_joint_ids]), dim=-1)
        else:
            dof_acc_metric = torch.zeros(self.num_envs, device=self.device)
        joint_torque_data = getattr(self.robot.data, "applied_torque", None)
        if joint_torque_data is not None and self._dof_torque_joint_ids.numel() > 0:
            dof_torque_metric = torch.sum(torch.square(joint_torque_data[:, self._dof_torque_joint_ids]), dim=-1)
        else:
            dof_torque_metric = torch.zeros(self.num_envs, device=self.device)

        reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_hand_target,
            self.cfg.rew_scale_body_target,
            self.cfg.rew_scale_facing,
            self.cfg.rew_scale_upright,
            self.cfg.rew_scale_target_velocity,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_action_smooth,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_flat_orientation,
            self.cfg.rew_scale_joint_center,
            self.cfg.rew_scale_joint_hip,
            self.cfg.rew_scale_joint_arms,
            self.cfg.rew_scale_joint_fingers,
            self.cfg.rew_scale_joint_torso,
            self.cfg.rew_scale_ankle_limits,
            self.cfg.rew_scale_feet_air_time,
            self.cfg.rew_scale_feet_slide,
            self.cfg.rew_scale_lin_vel_z,
            self.cfg.rew_scale_dof_acc,
            self.cfg.rew_scale_dof_torque,
            self.cfg.rew_scale_hand_pose,
            self.cfg.rew_scale_termination,
            self.cfg.success_bonus,
            hand_dist,
            body_dist,
            facing,
            upright,
            flat_orientation_metric,
            target_velocity_error,
            self.cfg.target_velocity_tracking_std,
            hand_pose_metric,
            self.robot.data.joint_vel,
            hip_dev,
            arms_dev,
            fingers_dev,
            torso_dev,
            ankle_limit_metric,
            feet_air_time_metric,
            feet_slide_metric,
            action_magnitude,
            action_delta,
            lin_vel_z_metric,
            dof_acc_metric,
            dof_torque_metric,
            success,
            self.reset_terminated,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        # Log key task signals
        self.extras["log"]["hand_target_dist"] = hand_dist.mean().item()
        self.extras["log"]["body_target_dist"] = body_dist.mean().item()
        self.extras["log"]["facing_dot"] = facing.mean().item()
        self.extras["log"]["upright"] = upright.mean().item()
        self.extras["log"]["success_rate"] = success.float().mean().item()

        # Log per-term reward components for TensorBoard
        reach_reward = self.cfg.rew_scale_hand_target * torch.exp(-2.5 * hand_dist)
        travel_reward = self.cfg.rew_scale_body_target * torch.exp(-0.5 * body_dist)
        facing_reward = self.cfg.rew_scale_facing * torch.clamp(facing, min=0.0)
        upright_reward = self.cfg.rew_scale_upright * torch.clamp(upright, min=0.0)
        alive_reward = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
        action_penalty = self.cfg.rew_scale_action_rate * action_magnitude
        action_smooth_penalty = self.cfg.rew_scale_action_smooth * action_delta
        joint_vel_penalty = self.cfg.rew_scale_joint_vel * torch.sum(self.robot.data.joint_vel**2, dim=-1)
        success_reward = self.cfg.success_bonus * success.float()
        target_velocity_reward = self.cfg.rew_scale_target_velocity * torch.exp(
            -torch.square(target_velocity_error) / (self.cfg.target_velocity_tracking_std**2 + 1.0e-6)
        )
        feet_air_time_reward = self.cfg.rew_scale_feet_air_time * feet_air_time_metric
        feet_slide_penalty = self.cfg.rew_scale_feet_slide * feet_slide_metric
        hand_pose_reward = self.cfg.rew_scale_hand_pose * hand_pose_metric
        termination_penalty = self.cfg.rew_scale_termination * self.reset_terminated.float()

        self.extras["log"]["reward/reach_hand"] = reach_reward.mean().item()
        self.extras["log"]["reward/travel_body"] = travel_reward.mean().item()
        self.extras["log"]["reward/facing"] = facing_reward.mean().item()
        self.extras["log"]["reward/upright"] = upright_reward.mean().item()
        self.extras["log"]["reward/alive"] = alive_reward.mean().item()
        self.extras["log"]["reward/target_velocity"] = target_velocity_reward.mean().item()
        self.extras["log"]["reward/feet_air_time"] = feet_air_time_reward.mean().item()
        self.extras["log"]["reward/success_bonus"] = success_reward.mean().item()
        self.extras["log"]["reward/hand_pose"] = hand_pose_reward.mean().item()
        orientation_penalty = self.cfg.rew_scale_flat_orientation * flat_orientation_metric
        joint_center_penalty = self.cfg.rew_scale_joint_center * joint_dev_total
        hip_penalty = self.cfg.rew_scale_joint_hip * hip_dev
        arms_penalty = self.cfg.rew_scale_joint_arms * arms_dev
        fingers_penalty = self.cfg.rew_scale_joint_fingers * fingers_dev
        torso_penalty = self.cfg.rew_scale_joint_torso * torso_dev
        ankle_penalty = self.cfg.rew_scale_ankle_limits * ankle_limit_metric
        lin_vel_z_penalty = self.cfg.rew_scale_lin_vel_z * lin_vel_z_metric
        dof_acc_penalty = self.cfg.rew_scale_dof_acc * dof_acc_metric
        dof_torque_penalty = self.cfg.rew_scale_dof_torque * dof_torque_metric
        self.extras["log"]["penalty/action_rate"] = action_penalty.mean().item()
        self.extras["log"]["penalty/action_smooth"] = action_smooth_penalty.mean().item()
        self.extras["log"]["penalty/joint_vel"] = joint_vel_penalty.mean().item()
        self.extras["log"]["penalty/orientation"] = orientation_penalty.mean().item()
        self.extras["log"]["penalty/joint_center"] = joint_center_penalty.mean().item()
        self.extras["log"]["penalty/hip_dev"] = hip_penalty.mean().item()
        self.extras["log"]["penalty/arms_dev"] = arms_penalty.mean().item()
        self.extras["log"]["penalty/fingers_dev"] = fingers_penalty.mean().item()
        self.extras["log"]["penalty/torso_dev"] = torso_penalty.mean().item()
        self.extras["log"]["penalty/ankle_limit"] = ankle_penalty.mean().item()
        self.extras["log"]["penalty/lin_vel_z"] = lin_vel_z_penalty.mean().item()
        self.extras["log"]["penalty/dof_acc"] = dof_acc_penalty.mean().item()
        self.extras["log"]["penalty/dof_torque"] = dof_torque_penalty.mean().item()
        self.extras["log"]["penalty/feet_slide"] = feet_slide_penalty.mean().item()
        self.extras["log"]["penalty/termination"] = termination_penalty.mean().item()

        self.prev_actions.copy_(self.actions)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        up_vec = math_utils.quat_apply(root_quat, self._env_z)
        upright = up_vec[:, 2]
        fell = (root_pos[:, 2] < self.cfg.termination_height) | (upright < self.cfg.upright_dot_threshold)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fell, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])
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

    def _collect_joint_indices(self, patterns: list[str]) -> list[int]:
        indices = set()
        for expr in patterns:
            joint_ids, _ = self.robot.find_joints(expr)
            indices.update(joint_ids)
        return sorted(indices)

    def _compute_foot_metrics(self, move_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._has_foot_feedback:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros

        assert self._contact_sensor is not None
        net_contact_forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_sensor_body_ids, :]
        force_magnitude = torch.norm(net_contact_forces, dim=-1)
        contacts = torch.max(force_magnitude, dim=1)[0] > self.cfg.feet_slide_contact_threshold

        if len(self._feet_body_ids) == 0:
            zeros = torch.zeros(self.num_envs, device=self.device)
            return zeros, zeros

        feet_vel = self.robot.data.body_lin_vel_w[:, self._feet_body_ids, :2]
        feet_slide_metric = torch.sum(feet_vel.norm(dim=-1) * contacts.float(), dim=-1)

        air_time = self._contact_sensor.data.current_air_time[:, self._feet_sensor_body_ids]
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_sensor_body_ids]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        stance_time = torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time))
        feet_air_time_metric = torch.min(stance_time, dim=1)[0]
        feet_air_time_metric = torch.clamp(feet_air_time_metric, max=self.cfg.feet_air_time_threshold)
        feet_air_time_metric = feet_air_time_metric * move_mask.float()

        return feet_air_time_metric, feet_slide_metric


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
    rew_scale_target_velocity: float,
    rew_scale_action_rate: float,
    rew_scale_action_smooth: float,
    rew_scale_joint_vel: float,
    rew_scale_flat_orientation: float,
    rew_scale_joint_center: float,
    rew_scale_joint_hip: float,
    rew_scale_joint_arms: float,
    rew_scale_joint_fingers: float,
    rew_scale_joint_torso: float,
    rew_scale_ankle_limits: float,
    rew_scale_feet_air_time: float,
    rew_scale_feet_slide: float,
    rew_scale_lin_vel_z: float,
    rew_scale_dof_acc: float,
    rew_scale_dof_torque: float,
    rew_scale_hand_pose: float,
    rew_scale_termination: float,
    success_bonus: float,
    hand_dist: torch.Tensor,
    body_dist: torch.Tensor,
    facing: torch.Tensor,
    upright: torch.Tensor,
    orientation_metric: torch.Tensor,
    target_velocity_error: torch.Tensor,
    target_velocity_tracking_std: float,
    hand_pose_metric: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_dev_hip: torch.Tensor,
    joint_dev_arms: torch.Tensor,
    joint_dev_fingers: torch.Tensor,
    joint_dev_torso: torch.Tensor,
    ankle_limits: torch.Tensor,
    feet_air_time_metric: torch.Tensor,
    feet_slide_metric: torch.Tensor,
    action_magnitude: torch.Tensor,
    action_delta: torch.Tensor,
    lin_vel_z_metric: torch.Tensor,
    joint_acc_metric: torch.Tensor,
    joint_torque_metric: torch.Tensor,
    success: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    reach_reward = rew_scale_hand_target * torch.exp(-2.5 * hand_dist)
    travel_reward = rew_scale_body_target * torch.exp(-0.5 * body_dist)
    facing_reward = rew_scale_facing * torch.clamp(facing, min=0.0)
    upright_reward = rew_scale_upright * torch.clamp(upright, min=0.0)
    alive_reward = rew_scale_alive * (1.0 - reset_terminated.float())
    std_sq = target_velocity_tracking_std * target_velocity_tracking_std + 1.0e-6
    target_velocity_reward = rew_scale_target_velocity * torch.exp(-torch.square(target_velocity_error) / std_sq)
    action_penalty = rew_scale_action_rate * action_magnitude
    action_smooth_penalty = rew_scale_action_smooth * action_delta
    joint_vel_penalty = rew_scale_joint_vel * torch.sum(joint_vel**2, dim=-1)
    orientation_penalty = rew_scale_flat_orientation * orientation_metric
    joint_center_penalty = rew_scale_joint_center * (
        joint_dev_hip + joint_dev_arms + joint_dev_fingers + joint_dev_torso
    )
    hip_penalty = rew_scale_joint_hip * joint_dev_hip
    arms_penalty = rew_scale_joint_arms * joint_dev_arms
    fingers_penalty = rew_scale_joint_fingers * joint_dev_fingers
    torso_penalty = rew_scale_joint_torso * joint_dev_torso
    ankle_penalty = rew_scale_ankle_limits * ankle_limits
    lin_vel_z_penalty = rew_scale_lin_vel_z * lin_vel_z_metric
    joint_acc_penalty = rew_scale_dof_acc * joint_acc_metric
    joint_torque_penalty = rew_scale_dof_torque * joint_torque_metric
    feet_air_time_reward = rew_scale_feet_air_time * feet_air_time_metric
    feet_slide_penalty = rew_scale_feet_slide * feet_slide_metric
    hand_pose_reward = rew_scale_hand_pose * hand_pose_metric
    success_reward = success_bonus * success.float()
    termination_penalty = rew_scale_termination * reset_terminated.float()

    total = (
        reach_reward
        + travel_reward
        + facing_reward
        + upright_reward
        + alive_reward
        + success_reward
        + hand_pose_reward
        + target_velocity_reward
        + feet_air_time_reward
        + termination_penalty
        + action_penalty
        + action_smooth_penalty
        + joint_vel_penalty
        + orientation_penalty
        + joint_center_penalty
        + hip_penalty
        + arms_penalty
        + fingers_penalty
        + torso_penalty
        + ankle_penalty
        + feet_slide_penalty
        + lin_vel_z_penalty
        + joint_acc_penalty
        + joint_torque_penalty
    )
    return total
