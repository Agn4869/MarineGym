"""Record one deterministic Hover rollout for plotting.

Run from ``scripts`` with the same environment variables as ``evaluate.py``.
"""

from __future__ import annotations

import csv
import os

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marinegym import init_simulation_app
from marinegym.learning import ALGOS


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    base_env = env = None
    try:
        from marinegym.envs.isaac_env import IsaacEnv

        checkpoint = os.path.abspath(os.path.expanduser(str(cfg.algo.checkpoint_path)))
        env_class = IsaacEnv.REGISTRY[cfg.task.name]
        base_env = env_class(cfg, headless=cfg.headless)
        env = TransformedEnv(base_env, Compose(InitTracker())).eval()
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )
        print(f"[INFO] Recording checkpoint: {checkpoint}", flush=True)
        rows = []

        def callback(_env, *_args):
            state = base_env.drone.get_state().detach().cpu()[0, 0]
            pos = state[:3].tolist()
            target = base_env.target_pos.detach().cpu()[0].tolist()
            rows.append([len(rows), *pos, *target, float(torch.linalg.vector_norm(base_env.rpos[0, 0]).item())])
            return len(rows)

        with set_exploration_type(ExplorationType.MODE):
            td = env.reset()
            for _ in range(base_env.max_episode_length):
                td = policy(td)
                td = env.step(td)
                callback(env)
                if bool(td.get("done", torch.tensor(False)).any()):
                    break

        output_path = os.path.abspath(str(cfg.get("trajectory_output", "hover_trajectory.csv")))
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "target_x", "target_y", "target_z", "pos_error"])
            writer.writerows(rows)
        print(f"[PASS] Wrote {len(rows)} trajectory samples to {output_path}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
