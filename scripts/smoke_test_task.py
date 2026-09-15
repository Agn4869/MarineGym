"""Smoke-test a registered MarineGym task without starting PPO or WandB.

Examples::

    python scripts/smoke_test_task.py --task Hover --steps 10
    python scripts/smoke_test_task.py --task Track --num-envs 4 --steps 10
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


def load_task_config(task_name: str, num_envs: int, headless: bool):
    """Compose the task fragments used by the original Hydra configuration."""
    from omegaconf import OmegaConf

    config_dir = REPO_ROOT / "cfg"
    task_path = config_dir / "task" / f"{task_name}.yaml"
    if not task_path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {task_path}")

    task_cfg = OmegaConf.merge(
        OmegaConf.load(config_dir / "base" / "env_base.yaml"),
        OmegaConf.load(config_dir / "base" / "sim_base.yaml"),
        OmegaConf.load(config_dir / "task" / "randomization.yaml"),
        OmegaConf.load(config_dir / "task" / "disturbances.yaml"),
        OmegaConf.load(task_path),
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
    parser = argparse.ArgumentParser(description="Smoke-test a MarineGym task.")
    parser.add_argument("--task", choices=("Hover", "Track"), default="Hover")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without a GUI. Use --no-headless to open a window.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.steps < 0:
        parser.error("--steps cannot be negative")

    cfg = load_task_config(args.task, args.num_envs, args.headless)
    simulation_app = init_simulation_app(cfg)
    env = None
    try:
        from marinegym.envs import IsaacEnv

        env_class = IsaacEnv.REGISTRY[args.task]
        print(
            f"[INFO] Creating {args.task} with {cfg.env.num_envs} environment(s).",
            flush=True,
        )
        env = env_class(cfg, headless=args.headless)
        tensordict = env.reset()
        print(
            f"[PASS] {args.task} reset completed. "
            f"Keys: {list(tensordict.keys(True, True))}",
            flush=True,
        )

        for step in range(args.steps):
            tensordict = env.rand_step(tensordict)
            if step == 0 or step + 1 == args.steps:
                print(
                    f"[INFO] Completed random step {step + 1}/{args.steps}.",
                    flush=True,
                )

        if args.steps:
            print(
                f"[PASS] {args.task} completed {args.steps} random-action step(s).",
                flush=True,
            )
    except BaseException as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
