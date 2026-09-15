"""Smoke-test MarineGym's RGB/depth camera sensor on Isaac Sim 5.0.

This validates the sensor layer independently from a task. It creates two
USD cameras, attaches Replicator RGB/depth annotators, renders a few frames,
and checks the returned TensorDict shapes.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test RGB/depth cameras.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    if args.width < 8 or args.height < 8:
        parser.error("camera resolution must be at least 8x8")

    cfg = {
        "headless": args.headless,
        "enable_livestream": False,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
    }
    simulation_app = init_simulation_app(cfg)
    camera = None
    sim = None
    try:
        from marinegym.utils.isaacsim_compat import SimulationContext, stage_utils
        from marinegym.sensors import Camera, PinholeCameraCfg
        import omni.replicator.core as rep

        stage_utils.create_new_stage()
        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=0.016,
            rendering_dt=0.016,
            backend="torch",
            sim_params={"use_gpu_pipeline": True, "use_gpu": True},
            physics_prim_path="/physicsScene",
            device="cuda:0",
        )
        sim.initialize_physics()

        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            resolution=(args.width, args.height),
            data_types=["rgb", "distance_to_image_plane"],
        )
        camera = Camera(camera_cfg)
        camera.spawn(
            ["/World/Camera_0", "/World/Camera_1"],
            translations=[(0.0, -4.0, 2.0), (0.0, 4.0, 2.0)],
            targets=[(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
        )
        camera.initialize(prim_paths_expr="/World/Camera_*")
        for _ in range(4):
            sim.render()
            rep.orchestrator.step(rt_subframes=1)
        images = camera.get_images()
        print(f"[INFO] Camera batch shape: {images.batch_size}", flush=True)
        print(f"[INFO] Camera keys: {list(images.keys(True, True))}", flush=True)
        for key in images.keys():
            value = images[key]
            print(f"[INFO] {key}: shape={tuple(value.shape)} dtype={value.dtype}", flush=True)
        expected_rgb = (2, 3, args.height, args.width)
        if tuple(images["rgb"].shape) != expected_rgb:
            raise RuntimeError(
                f"Unexpected RGB shape: {tuple(images['rgb'].shape)} != {expected_rgb}"
            )
        print("[PASS] RGB/depth camera smoke test completed.", flush=True)
    except BaseException as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
