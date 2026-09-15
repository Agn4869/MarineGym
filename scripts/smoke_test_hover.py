"""Create one Hover environment and exercise its reset/step path.

This is intentionally smaller than ``scripts/train.py``. It isolates the
Isaac Sim scene, robot, and TorchRL environment layers from PPO and WandB.

Run from the repository root:

    python scripts/smoke_test_hover.py
    python scripts/smoke_test_hover.py --steps 10
    python scripts/smoke_test_hover.py --no-headless --steps 300
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marinegym import init_simulation_app


def load_hover_config(num_envs: int, headless: bool):
    """Load the same config fragments used by the original Hydra setup."""
    from omegaconf import OmegaConf

    config_dir = REPO_ROOT / "cfg"
    task_cfg = OmegaConf.merge(
        OmegaConf.load(config_dir / "base" / "env_base.yaml"),
        OmegaConf.load(config_dir / "base" / "sim_base.yaml"),
        OmegaConf.load(config_dir / "task" / "randomization.yaml"),
        OmegaConf.load(config_dir / "task" / "disturbances.yaml"),
        OmegaConf.load(config_dir / "task" / "Hover.yaml"),
    )
    task_cfg.env.num_envs = num_envs

    cfg = OmegaConf.create(
        {
            "task": task_cfg,
            "sim": "${task.sim}",
            "env": "${task.env}",
            "headless": headless,
            "enable_livestream": False,
            "mode": "train",
            "seed": 0,
            "viewer": {
                "resolution": [1280, 720],
                "eye": [8.0, 0.0, 6.0],
                "lookat": [0.0, 0.0, 3.0],
            },
        }
    )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one MarineGym Hover environment.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without a GUI. Use --no-headless to open a window.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of cloned environments.")
    parser.add_argument("--steps", type=int, default=0, help="Number of random-action steps after reset.")
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.steps < 0:
        parser.error("--steps cannot be negative")

    cfg = load_hover_config(args.num_envs, args.headless)
    simulation_app = init_simulation_app(cfg)
    env = None

    try:
        # Isaac Sim modules must be imported after SimulationApp starts.
        from marinegym.envs import Hover

        print(f"[INFO] Creating Hover with {cfg.env.num_envs} environment(s).", flush=True)
        env = Hover(cfg, headless=args.headless)
        tensordict = env.reset()
        print(f"[PASS] Hover reset completed. Keys: {list(tensordict.keys(True, True))}", flush=True)

        for step in range(args.steps):
            tensordict = env.rand_step(tensordict)
            if step == 0 or step + 1 == args.steps:
                print(f"[INFO] Completed random step {step + 1}/{args.steps}.", flush=True)

        if args.steps:
            print(f"[PASS] Hover completed {args.steps} random-action step(s).", flush=True)
    except BaseException as exc:
        # Kit can raise SystemExit for some startup failures, which otherwise
        # looks like a successful shell exit with no Python traceback.
        print(f"[FAIL] {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
