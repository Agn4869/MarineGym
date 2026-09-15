import torch
import torch.nn as nn


class T200(nn.Module):
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

        # self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        # self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
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
        
        # 修改目标转速的计算逻辑
        target_rpm = torch.where(throttle > 0.075, 3.6599e+03 * throttle + 3.4521e+02,
        torch.where(throttle < -0.075, 3.4944e+03 * throttle - 4.3350e+02, torch.zeros_like(throttle)
        ))
        alpha = torch.exp(-self.dt / time_constants)
        
        noise = torch.randn_like(rpm_state) * self.noise_scale * 0.
        rpm = alpha * rpm_state + (1 - alpha) * target_rpm
        rpm_state.copy_(torch.clamp(rpm + noise, -3900, 3900))
        
        thrusts = force_constants /4.4e-7 * 9.81 * torch.where(rpm_state>0, 4.7368e-07 * self.f(rpm_state) - 1.9275e-04 * rpm_state + 8.4452e-02, -3.8442e-07 * self.f(rpm_state) - 1.6186e-04 * rpm_state - 3.9139e-02)# rpm2force
        
        moments = thrusts * -directions * 0

        return thrusts, moments
