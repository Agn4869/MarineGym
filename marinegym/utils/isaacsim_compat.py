"""Compatibility imports for Isaac Sim 4.1 and 5.0.

Version-specific module paths live here so simulation code does not need
scattered import fallbacks. Legacy imports can be removed after the 5.0 port
is fully validated.
"""

from __future__ import annotations


try:
    from isaacsim.core.cloner import GridCloner
except ModuleNotFoundError:
    from omni.isaac.cloner import GridCloner

try:
    from isaacsim.core.api.simulation_context import SimulationContext
except ModuleNotFoundError:
    from omni.isaac.core.simulation_context import SimulationContext

try:
    from isaacsim.core.utils import prims as prim_utils
    from isaacsim.core.utils import stage as stage_utils
    from isaacsim.core.utils import torch as torch_utils
except ModuleNotFoundError:
    from omni.isaac.core.utils import prims as prim_utils
    from omni.isaac.core.utils import stage as stage_utils
    from omni.isaac.core.utils import torch as torch_utils

try:
    from isaacsim.core.utils.extensions import enable_extension
except ModuleNotFoundError:
    from omni.isaac.core.utils.extensions import enable_extension

try:
    from isaacsim.core.utils.viewports import set_camera_view
except ModuleNotFoundError:
    from omni.isaac.core.utils.viewports import set_camera_view

try:
    from isaacsim.util.debug_draw import _debug_draw
except ModuleNotFoundError:
    from omni.isaac.debug_draw import _debug_draw


def get_physics_sim_view(simulation_context=None):
    """Return the initialized PhysX tensor view across Isaac Sim versions."""
    if simulation_context is None:
        simulation_context = SimulationContext.instance()
    if simulation_context is None:
        return None

    # Isaac Sim 5.0 exposes this as a public property backed by
    # SimulationManager. Isaac Sim 4.1 stored it directly on the context.
    if hasattr(type(simulation_context), "physics_sim_view"):
        return simulation_context.physics_sim_view
    return getattr(simulation_context, "_physics_sim_view", None)


__all__ = [
    "GridCloner",
    "SimulationContext",
    "_debug_draw",
    "enable_extension",
    "get_physics_sim_view",
    "prim_utils",
    "set_camera_view",
    "stage_utils",
    "torch_utils",
]
