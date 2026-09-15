"""Start Isaac Sim, update a few frames, and close it cleanly.

Run from the repository root:

    python scripts/smoke_test_simulation_app.py
    python scripts/smoke_test_simulation_app.py --no-headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marinegym import init_simulation_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the MarineGym Isaac Sim bootstrap.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without a GUI. Use --no-headless to open a window.",
    )
    parser.add_argument("--frames", type=int, default=10, help="Number of application frames to update.")
    args = parser.parse_args()

    simulation_app = init_simulation_app(
        {
            "headless": args.headless,
            "enable_livestream": False,
        }
    )

    try:
        # Isaac Sim modules must be imported only after SimulationApp starts.
        # Loading this module checks the renamed Isaac Sim 5.0 namespaces used
        # by the first MarineGym environment port.
        from marinegym.utils import isaacsim_compat

        print(
            "[PASS] Isaac Sim compatibility imports loaded "
            f"({isaacsim_compat.SimulationContext.__module__})."
        )
        for _ in range(args.frames):
            simulation_app.update()
        print(f"[PASS] Isaac Sim bootstrap completed {args.frames} frame updates.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
