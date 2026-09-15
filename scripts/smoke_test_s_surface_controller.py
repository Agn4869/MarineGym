"""Pure-Torch sanity check for S-surface command symmetry.

This test does not start Isaac Sim.  It verifies that at a level pose, +axis
and -axis velocity references produce finite, different rotor command vectors.
Physical world-frame direction still requires the simulator probe.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from marinegym.controllers import SSurfaceController


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with open(root / "marinegym/robots/assets/usd/BlueROV/BlueROV.yaml", "r") as f:
        params = yaml.safe_load(f)
    params["mass"] = 1.0
    params["inertia"] = {"xx": 1.0, "yy": 1.0, "zz": 1.0}
    controller = SSurfaceController(9.81, params)
    state = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 2.0]])
    outputs = {}
    for label, velocity in (("+x", (1.0, 0.0, 0.0)), ("-x", (-1.0, 0.0, 0.0)),
                            ("+y", (0.0, 1.0, 0.0)), ("-y", (0.0, -1.0, 0.0)),
                            ("+z", (0.0, 0.0, 1.0)), ("-z", (0.0, 0.0, -1.0))):
        command = controller.compute(
            state, target_pos=target,
            target_vel=torch.tensor([velocity]), target_yaw=torch.zeros(1, 1)
        )
        outputs[label] = command.squeeze(0)
        assert torch.isfinite(command).all(), f"non-finite output for {label}"
        print(f"[COMMAND] {label}: {command.squeeze(0).tolist()}", flush=True)
    assert not torch.allclose(outputs["+x"], outputs["-x"])
    assert not torch.allclose(outputs["+y"], outputs["-y"])
    assert not torch.allclose(outputs["+z"], outputs["-z"])
    print("[PASS] S-surface command outputs are finite and sign-sensitive.", flush=True)


if __name__ == "__main__":
    main()
