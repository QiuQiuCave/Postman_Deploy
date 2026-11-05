"""Run or view the continuous three-bottle Loco + IK workflow."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["R2V2_RESOURCE_ROOT"] = str(ROOT)


def replay(speed: float) -> None:
    """View the saved full demonstration in MuJoCo without running controllers."""
    import mujoco
    import mujoco.viewer
    import numpy as np
    from r2v2_deploy.tasks.three_bottle.resources import resource, verify_runtime

    verify_runtime()
    model = mujoco.MjModel.from_xml_path(str(resource("scene.xml")))
    data = mujoco.MjData(model)
    with np.load(resource("demonstration_50hz.npz")) as saved:
        times, positions = saved["time_s"], saved["physical_qpos"]
    with mujoco.viewer.launch_passive(model, data) as view:
        start = time.monotonic()
        while view.is_running():
            elapsed = (time.monotonic() - start) * speed
            index = min(int(elapsed * 50), len(times) - 1)
            data.qpos[:] = positions[index]
            mujoco.mj_forward(model, data)
            view.sync()
            if index == len(times) - 1:
                break
            time.sleep(1 / 60)


def run(output: Path, headless: bool, duration: float) -> dict:
    import mujoco.viewer
    from r2v2_deploy.tasks.three_bottle.resources import verify_runtime
    from r2v2_deploy.tasks.three_bottle.recording import save_run
    from r2v2_deploy.tasks.three_bottle.workflow import Workflow

    runtime = verify_runtime()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    controller = Workflow()
    controller.outdir = output
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    previous = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    context = (
        nullcontext(None)
        if headless
        else mujoco.viewer.launch_passive(controller.model, controller.data)
    )
    try:
        with context as view:
            while (
                not controller.done
                and not controller.failure
                and not interrupted
                and controller.data.time < duration
            ):
                if view is not None and not view.is_running():
                    interrupted = True
                    break
                controller.step()
                if view is not None and controller.steps % 20 == 0:
                    view.sync()
        return save_run(controller, output, runtime, interrupted)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="Run physics without opening a viewer"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / datetime.now().strftime("three_bottle_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--replay", action="store_true", help="View the included 901.80 s demonstration"
    )
    parser.add_argument(
        "--speed", type=float, default=4.0, help="Playback speed for --replay"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=920.0,
        help="Maximum physical seconds; shorter values allow a quick startup check",
    )
    args = parser.parse_args()
    if args.speed <= 0 or args.duration <= 0:
        parser.error("speed and duration must be positive")
    if args.replay:
        replay(args.speed)
        return 0
    result = run(args.output.resolve(), args.headless, args.duration)
    print(json.dumps(result, indent=2))
    return (
        0 if result["success"] or args.duration < 901.8 and not result["failure"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
