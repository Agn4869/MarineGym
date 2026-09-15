"""Run the fixed-target BlueROV PID hover controller.

Examples (from the repository root)::

    python scripts/run_pid_hover.py --headless --steps 500
    python scripts/run_pid_hover.py --no-headless --steps 1000

The four high-level action values are held at zero, so the PID receives a
zero velocity and zero-yaw reference while the environment target remains at
``(0, 0, 2)``.  This is a controller smoke test, not PPO training.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from marinegym import init_simulation_app
from smoke_test_task import load_task_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic BlueROV PID hover.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="pid_hover_trajectory.csv")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")

    cfg = load_task_config("Hover", args.num_envs, args.headless)
    cfg.task.control_mode = "pid"
    cfg.task.env.max_episode_length = max(int(cfg.task.env.max_episode_length), args.steps + 1)
    simulation_app = init_simulation_app(cfg)
    env = None
    rows = []
    try:
        from marinegym.envs.isaac_env import IsaacEnv

        env = IsaacEnv.REGISTRY["Hover"](cfg, headless=args.headless)
        td = env.reset()
        action = torch.zeros((args.num_envs, 1, 4), device=env.device)
        for step in range(args.steps):
            td.set(("agents", "action"), action)
            td = env.step(td)
            state = env.drone.get_state()[0, 0].detach().cpu()
            target = env.target_pos[0].detach().cpu()
            error = torch.linalg.vector_norm(target - state[:3]).item()
            rows.append([step, *state[:3].tolist(), *target.tolist(), error])
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
                print(
                    f"[PID] step={step + 1:4d} pos="
                    f"({state[0]:+.3f}, {state[1]:+.3f}, {state[2]:+.3f}) "
                    f"pos_error={error:.3f}",
                    flush=True,
                )

        output = Path(args.output).expanduser().resolve()
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "x", "y", "z", "target_x", "target_y", "target_z", "pos_error"])
            writer.writerows(rows)
        print(f"[PASS] Wrote {len(rows)} samples to {output}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
