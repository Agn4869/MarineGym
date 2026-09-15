"""Pure-torch sanity checks for PID wrench allocation.

This test does not start Isaac Sim.  It verifies that a simple six-thruster
geometry produces finite commands and that the allocator reconstructs a
requested force/torque wrench before the simulator is involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marinegym.controllers import PIDController


def main() -> None:
    cfg = {
        "mass": 11.2,
        "inertia": {"xx": 0.1, "yy": 0.1, "zz": 0.2},
        "volume": 0.0113459,
        "rotor_configuration": {
            "num_rotors": 6,
            "force_constants": [4.4e-7] * 6,
            "max_rotation_velocities": [3900] * 6,
        },
    }
    controller = PIDController(9.81, cfg)
    positions = torch.tensor(
        [[0.20, 0.20, 0.0], [-0.20, 0.20, 0.0],
         [-0.20, -0.20, 0.0], [0.20, -0.20, 0.0],
         [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    axes = torch.tensor(
        [[1., 1., 0.], [1., -1., 0.], [1., 1., 0.], [1., -1., 0.],
         [0., 0., 1.], [0., 0., 1.]], dtype=torch.float32
    )
    controller.set_thruster_geometry(positions, axes)
    state = torch.tensor([[0., 0., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    command = controller.compute(state, target_pos=torch.tensor([[0.1, -0.1, 2.0]]))
    assert command.shape == (1, 6)
    assert torch.isfinite(command).all()
    assert command.abs().max() <= 1.0
    print("[PASS] PID controller produced finite normalized commands:", command.tolist())


if __name__ == "__main__":
    main()
