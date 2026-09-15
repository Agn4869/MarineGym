"""MarineGym package bootstrap helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "cfg")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a value from a mapping-like or attribute-based configuration."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_experience(cfg: Any) -> str | None:
    """Resolve an optional Kit experience without requiring a legacy EXP_PATH."""
    configured_experience = _cfg_get(cfg, "experience") or os.getenv("MARINEGYM_EXPERIENCE")
    if configured_experience:
        experience_path = Path(os.path.expandvars(str(configured_experience))).expanduser()
        if not experience_path.is_file():
            raise FileNotFoundError(f"Isaac Sim experience file does not exist: {experience_path}")
        return str(experience_path)

    # Isaac Sim 4.1 installations expose this legacy experience through EXP_PATH.
    # Isaac Sim 5.0 pip installations should use SimulationApp's default experience.
    exp_path = os.getenv("EXP_PATH")
    if exp_path:
        legacy_experience = Path(exp_path) / "omni.isaac.sim.python.kit"
        if legacy_experience.is_file():
            return str(legacy_experience)

    return None


def init_simulation_app(cfg: Any):
    """Start Isaac Sim using a configuration compatible with 4.1 and 5.0."""
    # Importing SimulationApp here keeps lightweight tools able to import
    # MarineGym without requiring Isaac Sim to be installed locally.
    from isaacsim import SimulationApp

    enable_livestream = bool(_cfg_get(cfg, "enable_livestream", False))
    config = {
        "headless": bool(_cfg_get(cfg, "headless", False)),
        "enable_livestream": enable_livestream,
        "anti_aliasing": int(_cfg_get(cfg, "anti_aliasing", 1)),
        "width": int(_cfg_get(cfg, "width", 1280)),
        "height": int(_cfg_get(cfg, "height", 720)),
        "window_width": int(_cfg_get(cfg, "window_width", 1920)),
        "window_height": int(_cfg_get(cfg, "window_height", 1080)),
        "renderer": _cfg_get(cfg, "renderer", "RayTracedLighting"),
        "display_options": int(_cfg_get(cfg, "display_options", 3286)),
    }

    experience = _resolve_experience(cfg)
    if experience is None:
        simulation_app = SimulationApp(config)
    else:
        simulation_app = SimulationApp(config, experience=experience)

    if enable_livestream:
        try:
            from isaacsim.core.utils.extensions import enable_extension
        except ModuleNotFoundError:
            # Backward compatibility for Isaac Sim 4.1.
            from omni.isaac.core.utils.extensions import enable_extension

        simulation_app.set_setting("/app/window/drawMouse", True)
        simulation_app.set_setting("/app/livestream/proto", "ws")
        simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        simulation_app.set_setting("/ngx/enabled", False)
        enable_extension(_cfg_get(cfg, "livestream_extension", "omni.services.streamclient.webrtc"))

    return simulation_app


def _install_tensordict_compatibility() -> None:
    """Install the legacy helpers only when TensorDict is available."""
    try:
        import torch
        from tensordict import TensorDict
    except ModuleNotFoundError:
        return

    def _get_shapes(self: TensorDict):
        return {
            key: value.shape if isinstance(value, torch.Tensor) else value.shapes
            for key, value in self.items()
        }

    def _get_devices(self: TensorDict):
        return {
            key: value.device if isinstance(value, torch.Tensor) else value.devices
            for key, value in self.items()
        }

    if not hasattr(TensorDict, "shapes"):
        TensorDict.shapes = property(_get_shapes)
    if not hasattr(TensorDict, "devices"):
        TensorDict.devices = property(_get_devices)


_install_tensordict_compatibility()
