import torch
import torch.nn as nn


class M600(nn.Module):
    def __init__(self, rotor_config, dt: float):
        super().__init__()
        self.force_constants = nn.Parameter(torch.as_tensor(rotor_config["force_constants"]))
        self.moment_constants = nn.Parameter(torch.as_tensor(rotor_config["moment_constants"]))
        self.max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"]).float()
        self.num_rotors = len(self.force_constants)

        self.dt = dt
        self.time_up = 0.15
        self.time_down = 0.15
        self.noise_scale = 0.002

        self.throttle = nn.Parameter(torch.zeros(self.num_rotors))
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"]).float())

        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors))
        
        self.rpm = nn.Parameter(torch.zeros(self.num_rotors))
        self.time_constants = nn.Parameter(torch.as_tensor(rotor_config["time_constants"]))

        self.f = torch.square
        self.f_inv = torch.sqrt

        self.requires_grad_(False)

    def forward(self, cmds: torch.Tensor, params=None):
        if params is None:
            force_constants = self.force_constants
            throttle = self.throttle
            directions = self.directions
            tau_up = self.tau_up
            tau_down = self.tau_down
            rpm_state = self.rpm
            time_constants = self.time_constants
        else:
            force_constants = params["force_constants"]
            throttle = params["throttle"]
            directions = params["directions"]
            tau_up = params["tau_up"]
            tau_down = params["tau_down"]
            rpm_state = params["rpm"]
            time_constants = params["time_constants"]

        target_throttle = torch.clamp(cmds, -1, 1)

        tau = torch.where(target_throttle > throttle, tau_up, tau_down)
        tau = torch.clamp(tau, 0, 1)
        throttle.add_(tau * (target_throttle - throttle))
        
        target_rpm = torch.where(throttle > 0.075, 5.6599e+02 * throttle + 3.4521e+01,
        torch.where(throttle < -0.075, 5.4944e+02 * throttle - 4.3350e+01, torch.zeros_like(throttle)
        ))
        alpha = torch.exp(-self.dt / time_constants)
        
        noise = torch.randn_like(rpm_state) * self.noise_scale * 0.
        rpm = alpha * rpm_state + (1 - alpha) * target_rpm
        rpm_state.copy_(torch.clamp(rpm + noise, -600, 600))
        
        thrusts = force_constants * torch.abs(rpm_state) * rpm_state
        moments = thrusts * -directions * 0

        return thrusts, moments
