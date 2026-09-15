"""Nonlinear S-surface position controller for MarineGym UUVs.

The controller keeps the existing Lee attitude/thrust mixer, but replaces the
linear position/velocity feedback term with a boundary-layer sliding surface::

    s = (v - v_ref) + lambda * (p - p_ref)

The smooth ``tanh`` reaching term avoids the high-frequency chattering of a
hard sign function and remains differentiable for debugging and analysis.
"""

from __future__ import annotations

import torch
import math

from .lee_position_controller import LeePositionController
from marinegym.utils.torch import (
    normalize,
    quat_rotate_inverse,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)


class SSurfaceController(LeePositionController):
    """Sliding-surface controller with the same rotor mixer as Lee control."""

    def __init__(
        self,
        g: float,
        uav_params: dict,
        surface_lambda: float = 1.5,
        reaching_gain: float = 2.0,
        boundary_layer: float = 0.15,
    ) -> None:
        # Some MarineGym vehicle YAML files were originally used only with
        # direct rotor commands and therefore omit the arm geometry required
        # by the Lee mixer.  Keep the asset YAML unchanged, but provide a
        # documented six-thruster fallback for the BlueROV-style vehicle.
        rotor_cfg = uav_params.get("rotor_configuration", {})
        if "arm_lengths" not in rotor_cfg or "rotor_angles" not in rotor_cfg:
            num_rotors = int(rotor_cfg.get("num_rotors", 6))
            if num_rotors != 6:
                raise ValueError(
                    "S-surface control needs rotor_angles and arm_lengths "
                    f"for {num_rotors} rotors"
                )
            rotor_cfg = dict(rotor_cfg)
            rotor_cfg["arm_lengths"] = [0.2] * 6
            rotor_cfg["rotor_angles"] = [
                math.pi / 4,
                3 * math.pi / 4,
                5 * math.pi / 4,
                7 * math.pi / 4,
                0.0,
                math.pi,
            ]
            uav_params = dict(uav_params)
            uav_params["rotor_configuration"] = rotor_cfg
        super().__init__(g, uav_params)
        self.surface_lambda = float(surface_lambda)
        self.reaching_gain = float(reaching_gain)
        self.boundary_layer = float(boundary_layer)
        self.allocation_pinv = None

        # T200's normalized command is converted to force by the actuator
        # model.  Use the same polynomial at the configured maximum RPM to
        # convert the wrench allocator's Newton output back to [-1, 1].
        rotor_cfg = uav_params["rotor_configuration"]
        kf_scale = torch.as_tensor(rotor_cfg["force_constants"]).float() / 4.4e-7
        max_rpm = torch.as_tensor(rotor_cfg["max_rotation_velocities"]).float()
        rpm_force = 4.7368e-7 * max_rpm.square() - 1.9275e-4 * max_rpm + 8.4452e-2
        self.max_force = (kf_scale * 9.81 * rpm_force).clamp_min(1.0)

    def set_thruster_geometry(self, positions: torch.Tensor, axes: torch.Tensor) -> None:
        """Configure the body-frame force/torque allocation from USD poses."""
        positions = positions.to(self.max_force)
        axes = normalize(axes.to(self.max_force))
        # Each column maps one scalar thruster force to [Fx,Fy,Fz,Tx,Ty,Tz].
        wrench_matrix = torch.cat([axes, torch.linalg.cross(positions, axes)], dim=-1).T
        self.allocation_pinv = torch.linalg.pinv(wrench_matrix)

    def _compute(self, root_state, target_pos, target_vel, target_acc, target_yaw, body_rate):
        if self.allocation_pinv is None:
            return super()._compute(root_state, target_pos, target_vel, target_acc, target_yaw, body_rate)

        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        pos_error = pos - target_pos
        vel_error = vel - target_vel
        surface = vel_error + self.surface_lambda * pos_error
        reaching = self.reaching_gain * torch.tanh(surface / self.boundary_layer)
        desired_acc = -self.surface_lambda * vel_error - reaching + target_acc

        # Underwater vehicles are force-controlled by six independent
        # thrusters, not by a quadrotor tilt/thrust pair.  Keep the force in
        # world coordinates, then rotate it to the body frame for allocation.
        force_body = quat_rotate_inverse(rot, self.mass * desired_acc)

        rpy = quaternion_to_euler(rot)
        yaw_error = torch.atan2(torch.sin(rpy[..., 2] - target_yaw), torch.cos(rpy[..., 2] - target_yaw))
        attitude_error = torch.stack(
            [rpy[..., 0], rpy[..., 1], yaw_error.squeeze(-1)], dim=-1
        )
        torque_body = -attitude_error * self.attitute_gain - ang_vel * self.ang_rate_gain

        wrench = torch.cat([force_body, torque_body], dim=-1)
        thrust = (self.allocation_pinv.to(wrench) @ wrench.T).T
        return torch.clamp(thrust / self.max_force.to(thrust) , -1.0, 1.0)
