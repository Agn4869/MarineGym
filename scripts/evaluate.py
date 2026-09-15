"""Load and evaluate a MarineGym policy checkpoint.

Run from the ``scripts`` directory so Hydra can resolve ``../cfg``::

    PYTHONPATH=.. python evaluate.py task=Hover algo=ppo \
        algo.checkpoint_path=/path/to/checkpoint_final.pt \
        task.env.num_envs=1 eval_episodes=1 headless=true

Use ``headless=false`` to inspect the policy in the Isaac Sim window.
"""

from __future__ import annotations

import os

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from marinegym import init_simulation_app
from marinegym.learning import ALGOS


def build_environment_and_policy(cfg):
    """Create the evaluation environment and restore the requested policy."""
    from marinegym.envs.isaac_env import IsaacEnv

    checkpoint_path = cfg.algo.checkpoint_path
    if not checkpoint_path:
        raise ValueError(
            "A checkpoint is required. Pass "
            "algo.checkpoint_path=/absolute/path/to/checkpoint.pt"
        )

    checkpoint_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    cfg.algo.checkpoint_path = checkpoint_path

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
    return base_env, env, policy, checkpoint_path


@torch.no_grad()
def evaluate_model(env, policy, num_episodes: int, seed: int = 0):
    """Run deterministic rollouts and return their statistics TensorDicts."""
    if num_episodes < 1:
        raise ValueError("eval_episodes must be at least 1")

    env.eval()
    env.set_seed(seed)
    results = []
    with set_exploration_type(ExplorationType.MODE):
        for episode in range(num_episodes):
            trajectory = env.rollout(
                max_steps=env.base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
            )
            results.append(trajectory[("next", "stats")].cpu())
            print(
                f"[INFO] Completed evaluation episode {episode + 1}/{num_episodes}.",
                flush=True,
            )
    return results


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)
    env = None
    try:
        _, env, policy, checkpoint_path = build_environment_and_policy(cfg)
        print(f"[PASS] Loaded checkpoint: {checkpoint_path}", flush=True)
        results = evaluate_model(
            env,
            policy,
            num_episodes=int(cfg.eval_episodes),
            seed=int(cfg.seed),
        )
        print(f"[PASS] Evaluated {len(results)} episode(s).", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
