"""Run the fixed-wrist binary hand test; no RL model or hardware connection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from datetime import datetime, timezone
import json
import os
import queue
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true", help="Record two close-up views (1280x720)")
    parser.add_argument("--interactive", action="store_true", help="L/R toggle left/right; O open both; C close both")
    parser.add_argument("--duration", type=float, help="Seconds; default 32 for demo, unlimited for interactive")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "deploy_mujoco/config/r2v2_hands.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.interactive and args.headless:
        parser.error("--interactive requires a viewer (omit --headless)")
    if args.duration is not None and (not 0 < args.duration < float("inf")):
        parser.error("duration must be finite and positive")
    if not 1 <= args.fps <= 120:
        parser.error("fps must be between 1 and 120")
    if args.headless and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from common.r2v2_hand_test import DEMO_DURATION, HandExperiment
    from r2v2_description.model import SIDES, hand_names, load_config

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output or PROJECT_ROOT / "artifacts/r2v2_hands" / stamp).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    exp = HandExperiment(load_config(args.config), automatic=not args.interactive)
    duration = args.duration if args.duration is not None else (float("inf") if args.interactive else DEMO_DURATION)
    print(f"Fixed-wrist model: {exp.model.nq} joints, {exp.model.nu} actuators, {exp.model.neq} couplings")
    print("CLOSED only means preset pose reached. No object/grasp-success detection in this test.")
    print(f"Output: {output}")
    commands = queue.SimpleQueue()
    viewer = renderer = writer = None
    frame_index = 0
    start_wall = time.monotonic()
    snapshots = {0.0: "open", 15.5: "closed", 20.3: "reversal"}
    saved = set()

    def key_callback(keycode):
        if keycode in (ord("L"), ord("R"), ord("O"), ord("C")):
            commands.put(chr(keycode))

    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(exp.model, exp.data, key_callback=key_callback)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.42]
            viewer.cam.distance = 0.95
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -30
            viewer.opt.geomgroup[3] = 0
            print("Viewer keys: L/R = toggle a hand; O = open both; C = close both (use --interactive)")
        if args.video:
            renderer = mujoco.Renderer(exp.model, height=720, width=640)
            writer = imageio.get_writer(str(output / "hands.mp4"), fps=args.fps, codec="libx264", quality=8)
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            cameras = []
            for side in SIDES:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.015, 0.15 if side == "left" else -0.15, 0.45]
                camera.distance = 0.40
                camera.azimuth = 90 if side == "left" else -90
                camera.elevation = -30
                cameras.append(camera)
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, 20) if Path(font_path).exists() else ImageFont.load_default()

        while exp.data.time < duration - 1e-9:
            if viewer is not None and not viewer.is_running():
                break
            while not commands.empty():
                key = commands.get_nowait()
                if args.interactive:
                    left, right = [exp.control.controllers[s].command for s in SIDES]
                    if key == "L": left = 1 - left
                    elif key == "R": right = 1 - right
                    elif key == "O": left = right = 0
                    elif key == "C": left = right = 1
                    exp.command(left, right)
            exp.step()
            if renderer is not None and exp.data.time + 1e-9 >= frame_index / args.fps:
                panels = []
                for side, camera in zip(SIDES, cameras):
                    options.geomgroup[1] = side == "left"
                    options.geomgroup[2] = side == "right"
                    renderer.update_scene(exp.data, camera=camera, scene_option=options)
                    panels.append(renderer.render())
                frame = Image.fromarray(np.concatenate(panels, axis=1))
                draw = ImageDraw.Draw(frame)
                states = exp.statuses()
                for index, side in enumerate(SIDES):
                    x = 640 * index
                    draw.rectangle((x, 0, x + 640, 85), fill=(20, 28, 36))
                    text = f"{side.upper()}   cmd={exp.control.controllers[side].command}   {states[side]}"
                    draw.text((x + 16, 12), text, font=font, fill="white")
                    draw.text((x + 16, 45), f"t={exp.data.time:05.2f}s | FIXED WRIST", font=font, fill=(150, 210, 240))
                draw.rectangle((0, 680, 1280, 720), fill=(20, 28, 36))
                draw.text((16, 689), "EMPTY-HAND MOTION TEST  |  CLOSED = preset pose, not grasp success", font=font, fill="white")
                array = np.asarray(frame)
                writer.append_data(array)
                for when, name in snapshots.items():
                    if name not in saved and exp.data.time >= when:
                        imageio.imwrite(output / f"{name}.png", array)
                        saved.add(name)
                frame_index += 1
            if viewer is not None:
                if exp.steps % exp.decimation == 0:
                    viewer.sync()
                remaining = exp.data.time - (time.monotonic() - start_wall)
                if remaining > 0:
                    time.sleep(min(remaining, exp.model.opt.timestep))
            if not exp.finite or np.any(exp.data.warning.number):
                raise RuntimeError("Non-finite state or MuJoCo warning; stopping simulation")
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        if viewer is not None: viewer.close()
        report = exp.report()
        report["config"] = exp.cfg
        report["config_path"] = str(args.config.resolve())
        report["video_frames"] = frame_index
        report["wall_time_s"] = time.monotonic() - start_wall
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        columns = ["time"]
        for side in SIDES:
            for field in ("measured_q", "reference_q", "measured_dq", "reference_dq"):
                columns.extend(f"{name}/{field}" for name in hand_names(side))
        np.savez_compressed(output / "trajectory.npz", values=np.asarray(exp.samples), columns=np.array(columns))
    failed = [name for name, passed in report["checks"].items() if not passed]
    print(json.dumps({"passed": report["passed"], "failed_checks": failed,
                      "max_tracking_error_rad": {s: max(report["peaks"][s]["tracking_error_rad"]) for s in SIDES},
                      "max_mimic_error_rad": report["max_mimic_error_rad"],
                      "max_penetration_m": report["max_penetration_m"], "output": str(output)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
