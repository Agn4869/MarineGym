"""Check the world-frame sign of S-surface velocity references.

Runs short one-environment probes for +/-x, +/-y and +/-z references and
prints the measured world-frame displacement and velocity.  This is a sanity
check for frame/sign conventions, not a controller-tuning benchmark.
"""

from __future__ import annotations

import torch

from smoke_test_task import load_task_config
from marinegym import init_simulation_app
from marinegym.utils.torch import quat_axis, quat_rotate_inverse


def main() -> None:
    cfg = load_task_config("Hover", 1, True)
    app = init_simulation_app(cfg)
    env = None
    try:
        from marinegym.envs import IsaacEnv

        env = IsaacEnv.REGISTRY["Hover"](cfg, headless=True)
        rotor_pos, rotor_rot = env.drone.rotors_view.get_world_poses()
        base_pos, base_rot = env.drone.get_world_poses()
        base_rotors = base_rot.unsqueeze(-2).expand_as(rotor_rot)
        local_pos = quat_rotate_inverse(base_rotors, rotor_pos - base_pos.unsqueeze(-2))
        local_axis = quat_rotate_inverse(base_rotors, quat_axis(rotor_rot, axis=0))
        print(f"[THRUSTERS] positions={local_pos[0, 0].tolist()}", flush=True)
        print(f"[THRUSTERS] axes={local_axis[0, 0].tolist()}", flush=True)
        probes = (("+x", 0, 1.0), ("-x", 0, -1.0),
                  ("+y", 1, 1.0), ("-y", 1, -1.0),
                  ("+z", 2, 1.0), ("-z", 2, -1.0))
        for label, axis, value in probes:
            td = env.reset()
            # Remove reset randomization from this sign test: hold the vehicle
            # at the origin of its local environment with identity attitude,
            # and make the position target coincide with that pose.
            pose = torch.tensor([[[0.0, 0.0, 2.0]]], device=env.device)
            quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=env.device)
            env.drone.set_world_poses(pose, quat, torch.tensor([0], device=env.device))
            env.drone.set_velocities(
                torch.zeros_like(env.drone.get_velocities()), torch.tensor([0], device=env.device)
            )
            env.target_pos = pose[:, 0, :].clone()
            env.target_heading.zero_()
            p0 = env.drone.get_state()[..., :3].detach().clone()
            action = torch.zeros((1, 4), device=env.device)
            action[:, axis] = value
            for _ in range(40):
                td.set(("agents", "action"), action)
                td = env.step(td)
            state = env.drone.get_state()
            delta = (state[..., :3] - p0)[..., 0, :].squeeze(0)
            # get_state layout is position(3), quaternion(4), linear velocity(3).
            velocity = state[..., 7:10][..., 0, :].squeeze(0)
            print(f"[DIRECTION] {label}: delta={delta.tolist()} velocity={velocity.tolist()}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
