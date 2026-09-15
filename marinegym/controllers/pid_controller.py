"""PID position controller and wrench allocator for underwater vehicles.

Unlike :class:`LeePositionController`, this controller does not assume a
quadrotor/aircraft mixer.  It computes a body-frame wrench and maps that
wrench to the actual thruster geometry read from the BlueROV USD asset.

The control loop is deliberately small and explicit::

    position error -> PID force
    attitude error -> PID torque
    [force, torque] -> thruster allocation -> normalized T200 commands

This is intended as a reliable low-level controller.  A future RL policy can
provide ``target_vel`` and ``target_yaw`` without learning individual
thruster commands.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .controller import ControllerBase
from marinegym.utils.torch import normalize, quat_rotate_inverse, quaternion_to_euler


class PIDController(ControllerBase):
    """Cascaded position/attitude PID controller for a multi-thruster UUV.

    ``root_state`` uses the MarineGym convention ``[pos(3), quat(4),
    linear_vel(3), angular_vel(3)]``.  The controller returns six normalized
    commands in ``[-1, 1]`` after allocating a body-frame wrench.
    """

    def __init__(
        self,
        g: float,
        uav_params: dict,
        dt: float = 0.02,
        position_gain: Sequence[float] = (1.2, 1.2, 1.8),
        integral_gain: Sequence[float] = (0.08, 0.08, 0.12),
        velocity_gain: Sequence[float] = (1.8, 1.8, 2.2),
        attitude_gain: Sequence[float] = (0.8, 0.8, 0.5),
        angular_rate_gain: Sequence[float] = (0.15, 0.15, 0.12),
        integral_limit: float = 1.0,
        water_density: float = 997.0,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.num_rotors = int(uav_params["rotor_configuration"]["num_rotors"])
        self.integral_limit = float(integral_limit)

        self.register_buffer("mass", torch.tensor(float(uav_params["mass"])))
        inertia = uav_params.get("inertia", {})
        self.register_buffer(
            "inertia",
            torch.tensor(
                [float(inertia.get(k, 1.0)) for k in ("xx", "yy", "zz")]
            ),
        )
        self.register_buffer("gravity", torch.tensor(float(g)))
        self.register_buffer("volume", torch.tensor(float(uav_params.get("volume", 0.0))))
        self.register_buffer("coBM", torch.tensor(float(uav_params.get("coBM", 0.0))))
        self.register_buffer("water_density", torch.tensor(float(water_density)))

        self.register_buffer("kp_pos", torch.as_tensor(position_gain, dtype=torch.float32))
        self.register_buffer("ki_pos", torch.as_tensor(integral_gain, dtype=torch.float32))
        self.register_buffer("kd_pos", torch.as_tensor(velocity_gain, dtype=torch.float32))
        self.register_buffer("kp_att", torch.as_tensor(attitude_gain, dtype=torch.float32))
        self.register_buffer("kd_att", torch.as_tensor(angular_rate_gain, dtype=torch.float32))

        rotor_cfg = uav_params["rotor_configuration"]
        force_constants = torch.as_tensor(rotor_cfg["force_constants"], dtype=torch.float32)
        max_rpm = torch.as_tensor(rotor_cfg["max_rotation_velocities"], dtype=torch.float32)
        # Match the T200 actuator's RPM-to-force polynomial.  This is the
        # positive-force magnitude used to normalize the allocator output.
        rpm_force = (
            4.7368e-7 * max_rpm.square()
            - 1.9275e-4 * max_rpm
            + 8.4452e-2
        )
        max_force = (force_constants / 4.4e-7 * 9.81 * rpm_force).abs().clamp_min(1.0)
        self.register_buffer("max_force", max_force)

        self.allocation_pinv: Optional[torch.Tensor] = None
        self.wrench_matrix: Optional[torch.Tensor] = None
        self._position_integral: Optional[torch.Tensor] = None

    def set_thruster_geometry(self, positions: torch.Tensor, axes: torch.Tensor) -> None:
        """Set body-frame thruster positions and force axes from USD poses.

        ``positions`` and ``axes`` have shape ``(num_rotors, 3)``.  Each
        allocation matrix column maps one signed scalar thruster force to
        ``[Fx, Fy, Fz, Mx, My, Mz]``.
        """
        positions = torch.as_tensor(positions, dtype=self.max_force.dtype, device=self.max_force.device)
        axes = torch.as_tensor(axes, dtype=self.max_force.dtype, device=self.max_force.device)
        if positions.shape != (self.num_rotors, 3) or axes.shape != (self.num_rotors, 3):
            raise ValueError(
                f"Expected geometry ({self.num_rotors}, 3), got "
                f"positions={tuple(positions.shape)}, axes={tuple(axes.shape)}"
            )
        axes = normalize(axes)
        moments = torch.linalg.cross(positions, axes, dim=-1)
        self.wrench_matrix = torch.cat((axes, moments), dim=-1).T.contiguous()
        self.allocation_pinv = torch.linalg.pinv(self.wrench_matrix)

    def reset(self, env_ids: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> None:
        """Reset the integral state for all or selected environments."""
        if self._position_integral is None:
            return
        if env_ids is None:
            self._position_integral.zero_()
        else:
            self._position_integral[env_ids.reshape(-1)] = 0.0

    def _ensure_integral(self, n: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._position_integral is None or self._position_integral.shape != (n, 3):
            self._position_integral = torch.zeros(n, 3, device=device, dtype=dtype)

    def process_rl_actions(self, actions: torch.Tensor):
        """Interpret four policy outputs as ``[vx, vy, vz, yaw]`` references."""
        if actions.shape[-1] != 4:
            raise ValueError(f"PID high-level action must have 4 values, got {actions.shape[-1]}")
        target_vel, target_yaw = actions.split((3, 1), dim=-1)
        return target_vel, target_yaw * torch.pi

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        body_rate: bool = False,
    ) -> torch.Tensor:
        if self.allocation_pinv is None:
            raise RuntimeError("Call set_thruster_geometry() before PIDController.compute().")

        batch_shape = root_state.shape[:-1]
        if root_state.shape[-1] != 13:
            raise ValueError(f"root_state must end in 13 values, got {root_state.shape}")
        flat = root_state.reshape(-1, 13)
        n = flat.shape[0]
        device, dtype = flat.device, flat.dtype
        self._ensure_integral(n, device, dtype)

        def expand_arg(value: Optional[torch.Tensor], width: int, default: float = 0.0) -> torch.Tensor:
            if value is None:
                return torch.full((n, width), default, device=device, dtype=dtype)
            return value.to(device=device, dtype=dtype).expand(*batch_shape, width).reshape(n, width)

        pos, quat, linear_vel, angular_vel = torch.split(flat, (3, 4, 3, 3), dim=-1)
        target_pos_f = expand_arg(target_pos, 3)
        target_vel_f = expand_arg(target_vel, 3)
        target_yaw_f = expand_arg(target_yaw, 1)

        pos_error = target_pos_f - pos
        vel_error = target_vel_f - linear_vel
        self._position_integral.add_(pos_error * self.dt).clamp_(-self.integral_limit, self.integral_limit)
        force_world = self.mass.to(dtype=dtype) * (
            self.kp_pos.to(dtype=dtype) * pos_error
            + self.ki_pos.to(dtype=dtype) * self._position_integral
            + self.kd_pos.to(dtype=dtype) * vel_error
        )
        # Compensate only the net weight.  Buoyancy itself is applied by the
        # vehicle model, so adding full ``m*g`` here would double-count it.
        net_weight = self.mass.to(dtype=dtype) * self.gravity.to(dtype=dtype)
        net_weight = net_weight - self.water_density.to(dtype=dtype) * self.volume.to(dtype=dtype) * self.gravity.to(dtype=dtype)
        force_world[..., 2] += net_weight

        rpy = quaternion_to_euler(quat)
        yaw_error = torch.atan2(
            torch.sin(target_yaw_f[:, 0] - rpy[:, 2]),
            torch.cos(target_yaw_f[:, 0] - rpy[:, 2]),
        )
        attitude_error = torch.stack((-rpy[:, 0], -rpy[:, 1], yaw_error), dim=-1)
        if body_rate:
            angular_vel_body = angular_vel
        else:
            angular_vel_body = quat_rotate_inverse(quat, angular_vel)
        torque_body = self.kp_att.to(dtype=dtype) * attitude_error - self.kd_att.to(dtype=dtype) * angular_vel_body
        force_body = quat_rotate_inverse(quat, force_world)
        wrench = torch.cat((force_body, torque_body), dim=-1)

        pinv = self.allocation_pinv.to(device=device, dtype=dtype)
        thrust = wrench @ pinv.T
        commands = thrust / self.max_force.to(device=device, dtype=dtype)
        commands = torch.clamp(commands, -1.0, 1.0)
        return commands.reshape(*batch_shape, self.num_rotors)

