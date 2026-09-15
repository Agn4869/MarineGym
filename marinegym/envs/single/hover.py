import torch
import torch.distributions as D

from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
from marinegym.utils.isaacsim_compat import prim_utils
from marinegym.views import ArticulationView, RigidPrimView
from marinegym.utils.torch import euler_to_quaternion, quat_axis, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec

from marinegym.robots.drone import UnderwaterVehicle

from ..utils import attach_payload

class Hover(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_velocity_weight = cfg.task.get("reward_velocity_weight", 0.0)
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_position_weight = float(cfg.task.get("reward_position_weight", 0.8))
        self.reward_heading_weight = float(cfg.task.get("reward_heading_weight", 0.2))
        self.reward_near_target_weight = float(cfg.task.get("reward_near_target_weight", 0.0))
        self.near_target_radius = float(cfg.task.get("near_target_radius", 0.2))
        self.reward_position_precision_weight = float(
            cfg.task.get("reward_position_precision_weight", 0.0)
        )
        self.reward_success_weight = float(cfg.task.get("reward_success_weight", 0.0))
        self.success_position_radius = float(cfg.task.get("success_position_radius", 0.15))
        self.success_velocity_threshold = float(
            cfg.task.get("success_velocity_threshold", 0.2)
        )
        self.action_smoothing = float(cfg.task.get("action_smoothing", 1.0))
        self.action_smoothing = max(0.0, min(1.0, self.action_smoothing))
        self.time_encoding = cfg.task.time_encoding
        self.mode = cfg.mode
        self.disturbances = cfg.task.get("disturbances", {})
        self.enable_payload = self.disturbances[self.mode]['payload']['enable_payload']
        self.enable_flow = self.disturbances[self.mode]['flow']['enable_flow']
        self.max_flow_velocity = self.disturbances[self.mode]['flow']['max_flow_velocity']
        self.flow_velocity_gaussian_noise = self.disturbances[self.mode]['flow']['flow_velocity_gaussian_noise']
        self.curriculum_cfg = cfg.task.get("curriculum", {})
        # IsaacEnv.__init__() calls _set_specs(), so this must be available
        # before entering the base-class initializer.
        self.control_mode = cfg.task.get("control_mode", "direct")

        super().__init__(cfg, headless)
        self.episode_count = torch.zeros(self.num_envs, device=self.device)

        self.drone.initialize()
        # Keep a filtered action per environment.  This is intentionally part
        # of the environment dynamics rather than a policy-only postprocess,
        # so training and deterministic replay see the same actuator behavior.
        self.prev_actions = torch.zeros(
            self.num_envs, 1, self.drone.num_rotors, device=self.device
        )
        if self.control_mode == "s_surface":
            from marinegym.controllers import ControllerBase
            controller_name = cfg.task.drone_model.controller
            controller_cls = ControllerBase.REGISTRY[controller_name]
            # The articulation backend may expose the diagonal inertia as
            # either ``(3,)`` or a singleton-batched tensor.  Normalize it
            # here before converting values for the controller configuration.
            inertia = self.drone.INERTIA_0.detach().reshape(-1)
            uav_params = {
                "name": self.drone.name,
                "mass": float(self.drone.MASS_0.detach().reshape(-1)[0].item()),
                "inertia": {"xx": float(inertia[0]), "yy": float(inertia[1]), "zz": float(inertia[2])},
                "rotor_configuration": self.drone.params["rotor_configuration"],
            }
            surface_cfg = cfg.task.get("s_surface", {})
            self.controller = controller_cls(
                9.81,
                uav_params,
                surface_lambda=float(surface_cfg.get("surface_lambda", 1.5)),
                reaching_gain=float(surface_cfg.get("reaching_gain", 2.0)),
                boundary_layer=float(surface_cfg.get("boundary_layer", 0.15)),
            ).to(self.device)
            # Read the actual BlueROV thruster poses from USD.  The first four
            # thrusters are horizontal and the last two are vertical; using a
            # synthetic quadrotor mixer here would silently flip/lose axes.
            if hasattr(self.controller, "set_thruster_geometry"):
                rotor_pos, rotor_rot = self.drone.rotors_view.get_world_poses()
                base_pos, base_rot = self.drone.get_world_poses()
                base_rotors = base_rot.unsqueeze(-2).expand_as(rotor_rot)
                local_pos = quat_rotate_inverse(
                    base_rotors, rotor_pos - base_pos.unsqueeze(-2)
                )
                local_axes = quat_rotate_inverse(
                    base_rotors, quat_axis(rotor_rot, axis=0)
                )
                self.controller.set_thruster_geometry(
                    local_pos[0, 0], local_axes[0, 0]
                )
        else:
            self.controller = None
        if self.enable_payload:
            payload_cfg = self.disturbances[self.mode]['payload']
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -2.5, 1.5], device=self.device),
            torch.tensor([2.5, 2.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )
        self.target_yaw_dist = D.Uniform(
            torch.tensor(0.0, device=self.device),
            torch.tensor(2.0 * torch.pi, device=self.device),
        )

        self.target_pos = torch.tensor([[0.0, 0.0, 2.]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8

    def _design_scene(self):
        import marinegym.utils.kit as kit_utils
        from marinegym.utils.isaacsim_compat import prim_utils, stage_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = UnderwaterVehicle.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        
        from marinegym.robots.robot import ASSET_PATH
        stage_utils.add_reference_to_stage(usd_path= ASSET_PATH + "/usd/worlds/EmptyMarine.usd",prim_path="/World/defaultGroundPlane")

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.fixed_usd_path,
            translation=(0.0, 0.0, 2.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(),
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        if self.enable_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        action_leaf = (
            # Keep the singleton agent axis consistent with the direct rotor
            # action spec (``(num_envs, 1, action_dim)``), which PPO uses to
            # infer ``n_agents`` and ``action_dim``.
            BoundedTensorSpec(-1, 1, (1, 4), device=self.device)
            if self.control_mode == "s_surface"
            else self.drone.action_spec.unsqueeze(0)
        )
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": action_leaf,
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        if self.enable_flow:
            self.drone.set_flow_velocities(env_ids, self.max_flow_velocity, self.flow_velocity_gaussian_noise)
        self.drone._reset_idx(env_ids, self.training)

        curriculum_enabled = bool(self.curriculum_cfg.get("enable", False))
        if curriculum_enabled:
            warmup = max(float(self.curriculum_cfg.get("warmup_episodes", 200)), 1.0)
            alpha = (self.episode_count[env_ids] / warmup).clamp(0.0, 1.0)
            xy_start = float(self.curriculum_cfg.get("start_xy_radius", 0.5))
            xy_end = float(self.curriculum_cfg.get("end_xy_radius", 2.5))
            z_start = float(self.curriculum_cfg.get("start_z_half_range", 0.25))
            z_end = float(self.curriculum_cfg.get("end_z_half_range", 0.5))
            xy_radius = xy_start + (xy_end - xy_start) * alpha
            z_half_range = z_start + (z_end - z_start) * alpha
            offset = torch.empty((len(env_ids), 1, 3), device=self.device)
            offset[..., :2] = (torch.rand((len(env_ids), 1, 2), device=self.device) * 2.0 - 1.0) * xy_radius[:, None, None]
            offset[..., 2] = (torch.rand((len(env_ids), 1), device=self.device) * 2.0 - 1.0) * z_half_range[:, None]
            pos = self.target_pos[:, None, :] + offset
        else:
            pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.enable_payload:
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids, joint_indices=joint_indices)

            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = torch.zeros(*env_ids.shape, 1, 3, device=self.device)
        target_rpy[..., 2] = self.target_yaw_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        self.prev_actions[env_ids] = 0.0

        self.stats[env_ids] = 0.
        self.episode_count[env_ids] += 1

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        if self.action_smoothing < 1.0:
            # ``action_smoothing`` is the fraction of the new command applied
            # this step; the remainder comes from the previous filtered cmd.
            actions = (
                (1.0 - self.action_smoothing) * self.prev_actions
                + self.action_smoothing * actions
            )
            self.prev_actions.copy_(actions)
        if self.control_mode == "s_surface":
            root_state = self.drone.get_state()[..., :13]
            # Agent actions are commonly stored as ``(num_envs, 4)`` while
            # the articulation state carries an extra singleton agent axis
            # ``(num_envs, 1, 13)``.  Add that axis before broadcasting the
            # references so multi-environment runs behave like the 1-env case.
            while actions.ndim < root_state.ndim:
                actions = actions.unsqueeze(-2)
            while actions.ndim > root_state.ndim:
                # PPO may retain one or more singleton agent axes while the
                # articulation view is flattened to (num_envs, 13).
                actions = actions.squeeze(-2)
            target_vel, target_yaw = self.controller.process_rl_actions(actions)
            actions = self.controller.compute(
                root_state,
                target_pos=self.target_pos,
                target_vel=target_vel,
                target_yaw=target_yaw,
            )
        self.effort = torch.abs(self.drone.apply_action(actions))

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]
        self.rheading = self.target_heading - self.drone_state[..., 13:16]

        obs = [self.rpos, self.drone_state[..., 3:], self.rheading,]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        heading_alignment = torch.sum(self.drone.heading * self.target_heading, dim=-1)

        heading_error = torch.norm(self.rheading, dim=-1)
        # Keep position and heading objectives separate.  This prevents a
        # nearly-correct attitude from masking a persistent position error.
        position_reward = 1.0 / (
            1.0 + torch.square(self.reward_distance_scale * pos_error)
        )
        heading_reward = 1.0 / (1.0 + torch.square(heading_error))
        reward_pose = 0.5 * (
            self.reward_position_weight * position_reward
            + self.reward_heading_weight * heading_reward
        )
        reward_near_target = self.reward_near_target_weight * torch.exp(
            -torch.square(pos_error / max(self.near_target_radius, 1e-3))
        )
        # Apply an additional bounded precision penalty only near the target.
        # Clamping at 0.5 m keeps this term from dominating the coarse reaching
        # objective while giving PPO a useful gradient for the final correction.
        precision_error = torch.clamp(pos_error, max=0.5)
        reward_position_precision = -self.reward_position_precision_weight * torch.square(
            precision_error
        )
        # uprightness
        uprightness = torch.square((self.drone.up[..., 2] + 1) / 2)
        reward_up = uprightness

        # spin reward
        spinnage = torch.square(self.drone.vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)
        linear_speed = torch.norm(self.drone.vel[..., :3], dim=-1)
        reward_velocity = -self.reward_velocity_weight * torch.tanh(linear_speed)
        success = (
            (pos_error < self.success_position_radius)
            & (linear_speed < self.success_velocity_threshold)
            & (uprightness > 0.98)
        )
        reward_success = self.reward_success_weight * success.to(pos_error.dtype)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
            + reward_velocity
            + reward_near_target
            + reward_position_precision
            + reward_success
        )

        distance = torch.norm(torch.cat([self.rpos, self.rheading], dim=-1), dim=-1)
        misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["pos_error"].lerp_(pos_error, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(heading_alignment, (1-self.alpha))
        self.stats["uprightness"].lerp_(self.drone_state[..., 18], (1-self.alpha))
        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
