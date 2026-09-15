![Visualization of MarineGym](docs/overview.png)

---

# MarineGym

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-5.0-orange.svg)](https://developer.nvidia.com/isaac/sim)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3.11/)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://marinegym.netlify.app/)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmarine-gym.com&label=website&up_message=online&down_message=offline)](https://marine-gym.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!NOTE]
> This branch ports the runnable MarineGym path to Isaac Sim 5.0 and Python 3.11.
> The upstream MarineGym release targets Isaac Sim 4.1 and Python 3.10.

## Isaac Sim 5.0 Porting Status

This work is based on the original [MarineGym](https://github.com/Marine-RL/MarineGym) project.

- [x] Start MarineGym with Isaac Sim 5.0
- [x] Update deprecated Isaac Sim and TorchRL APIs used by the runnable path
- [x] Run the Hover environment and 16-way parallel smoke test
- [x] Restore PPO training and evaluation for Hover and Track
- [x] Validate checkpoint saving and loading
- [ ] Validate the optional S-surface Hover controller against the BlueROV thruster frame
- [ ] Port remaining/unregistered environments (the repository contains no `landing.py`)

The first compatibility milestone can be tested independently from the MarineGym environments:

```bash
# Headless startup test (recommended first)
python scripts/smoke_test_simulation_app.py

# Windowed startup test
python scripts/smoke_test_simulation_app.py --no-headless
```

A successful run prints `[PASS] Isaac Sim bootstrap completed ...` and closes Isaac Sim cleanly.

### Reproducible smoke tests

From the repository root, after activating the Isaac Sim 5.0 Python environment:

```bash
python scripts/smoke_test_task.py --task Hover --num-envs 16 --steps 10
python scripts/smoke_test_task.py --task Track --num-envs 16 --steps 10
python scripts/smoke_test_camera.py --headless
```

For servers where PyTorch shared libraries are not on the default loader path:

```bash
TORCH_LIB_PATH=$(python -c 'import os,torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$TORCH_LIB_PATH:${LD_LIBRARY_PATH:-}"
```

*MarineGym* is a large-scale parallel framework designed for reinforcement learning research on unmanned underwater vehicles (UUVs). It is built upon [OmniDrones](https://github.com/btx0424/OmniDrones) and [Isaac Sim](https://developer.nvidia.com/isaac/sim), offering the following features:

- Efficiency: Achieve a simulation speed of up to 10<sup>7</sup> steps per second.
- Fidelity: Accurately replicate the physical environment, including physical laws, kinematics, and dynamics.
- Flexibility:  Ensure compatibility with existing RL frameworks and offer user-friendly APIs to facilitate seamless integration and usage.
- Evaluation: Assesses and contrasts various RL strategies through multiple tasks.

> [!TIP]
>
> 🚀 **Collaborate with us on Underwater Embodied AI!**
>
> We are actively seeking research partners in the field of Underwater Embodied Intelligence and Reinforcement Learning. If you are interested in leveraging MarineGym for your project, please contact us at:
>
> 📮 **Email**: zjuoyh@163.com

## Installation

To install MarineGym, we recommend reading one of the following guides:
- [Installation from Source](https://marinegym.netlify.app/installation_from_source) (recommended for development)
- [Docker Environment](https://marinegym.netlify.app/docker_environment) (recommended for training purposes; no visualization interface)

If you encounter any issues, you can find solutions to common problems in the [FAQ](https://marinegym.netlify.app/faq) or feel free to open an issue.

For training and evaluation commands, please take a look at the [Quick Start](https://marinegym.netlify.app/quick_start).

## Usage
For installation details, please refer to our [Setup Guide](https://marinegym.netlify.app/installation_from_source/).

The currently registered and verified environments in this branch are `Hover` and `Track`.
`Track` contains the circle/helical/lemniscate trajectory helpers. `Landing.yaml` exists in
the upstream configuration, but this checkout does not contain a corresponding `landing.py`
implementation, so Landing is not reported as migrated yet.

### Vision and sonar status

The sensor layer includes a camera wrapper with RGB and depth annotators, depth
normalization, image export helpers, and MobileNetV3-based vision encoders. The
camera path is verified independently on Isaac Sim 5.0 with:

```bash
python scripts/smoke_test_camera.py --headless
```

This returns RGB tensors with shape `(N, 3, H, W)` and depth tensors with shape
`(N, 1, H, W)`. No registered task currently consumes these images, so a vision
RL task still needs an observation adapter that attaches cameras to each cloned
robot and feeds the image keys into `MixedEncoder`.

Sonar is not implemented in this checkout. The only related references are
future-work comments for scene-query/LiDAR support; there is no sonar sensor,
sonar observation spec, or sonar task to port. A future sonar migration should
therefore be treated as a new sensor implementation (ray casting or a dedicated
underwater acoustic model), followed by a task and sensor smoke test—not as a
simple Isaac Sim API rename.

The training script is located in the `scripts` folder, named `train.py`.

### S-surface Hover (experimental, opt-in)

The default `Hover` configuration uses direct six-thruster actions so that the
baseline PPO path stays simple and reproducible. An experimental
sliding-surface control path is also included. When enabled, the policy
outputs four normalized references `(v_x, v_y, v_z, yaw)`;
the controller forms `s = (v - v_ref) + lambda * (p - p_ref)` and applies a
smooth `tanh` reaching term before the existing rotor mixer. The path is
engineering-complete enough for smoke tests and short PPO runs, but its
world-frame sign convention has not yet been validated against the BlueROV
USD thruster axes. Use the two diagnostic scripts before treating it as a
research result:

```bash
python scripts/smoke_test_s_surface_controller.py
python scripts/smoke_test_s_surface_direction.py
```

The first checks finite, sign-sensitive rotor commands. The second runs a
controlled simulator probe and prints measured world-frame displacement; a
real vehicle/asset geometry check is still required before tuning gains.

The default direct-control Hover task now uses a small initial-position
curriculum: episodes start within roughly 0.5 m of the target and expand to
the original 2.5 m range over 200 episodes. Set `curriculum.enable: false` in
`cfg/task/Hover.yaml` to restore the original random initialization.


To start the training process, run:

```bash
PYTHONPATH=.. python -u scripts/train.py task=Hover algo=ppo headless=true \
    enable_livestream=false wandb.mode=offline \
    task.env.num_envs=16 total_frames=1024 max_iters=2 save_interval=1
```
where `task` currently supports `Hover` and `Track` in this branch. Use the new
`scripts/evaluate.py` to load a saved checkpoint:

```bash
PYTHONPATH=.. python -u scripts/evaluate.py task=Hover algo=ppo headless=true \
    task.env.num_envs=1 eval_episodes=1 \
    algo.checkpoint_path=/absolute/path/to/checkpoint_final.pt
```


## Citation

If you build on this work, please cite our paper:

```bibtex
@inproceedings{chu2025marinegym,
  title={MarineGym: A high-performance reinforcement learning platform for underwater robotics},
  author={Chu, Shuguang and Huang, Zebin and Li, Yutong and Lin, Mingwei and Li, Dejun and Carlucho, Ignacio and Petillot, Yvan R and Yang, Canjun},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={17146--17153},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

The architecture and certain implementation ideas build upon concepts introduced in [OmniDrones](https://github.com/btx0424/OmniDrones).
