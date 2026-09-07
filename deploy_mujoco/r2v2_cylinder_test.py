"""Try a free cylinder with the existing binary controller; optionally record."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from datetime import datetime, timezone
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--radius", type=float, default=0.020)
    parser.add_argument("--height", type=float, default=0.120)
    parser.add_argument("--mass", type=float, default=0.100)
    parser.add_argument("--x", type=float, default=0.015)
    parser.add_argument("--palm-offset", type=float, default=0.035)
    parser.add_argument("--no-grasp", action="store_true", help="Negative control: leave the hand open")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from common.r2v2_cylinder_test import CylinderExperiment, CylinderParameters
    from common.r2v2_grasp_recording import GraspRecorder

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_cylinder" / stamp).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    params = CylinderParameters(side=args.side, radius=args.radius, height=args.height,
                                mass=args.mass, x=args.x, palm_offset=args.palm_offset,
                                grasp=not args.no_grasp)
    exp = CylinderExperiment(params)
    recorder = GraspRecorder(exp)
    print(f"{args.side} hand: cylinder diameter={2*args.radius*1000:.0f} mm, mass={args.mass*1000:.0f} g")
    print("Support withdraws at 5-7 s; unsupported hold at 7-12 s; open at 12 s.")
    print(f"Output: {out}")
    renderer = writer = None
    frame_index = 0
    saved = set()
    try:
        if args.video:
            renderer = mujoco.Renderer(exp.model, height=720, width=640)
            writer = imageio.get_writer(str(out / "cylinder.mp4"), fps=30, codec="libx264", quality=8)
            opt = mujoco.MjvOption()
            opt.geomgroup[1] = args.side == "left"
            opt.geomgroup[2] = args.side == "right"
            opt.geomgroup[3] = 0
            cameras = []
            # Palm-side close-up exposes thumb motion; wide view shows support
            # withdrawal and the landing after release without moving cameras.
            for distance, elevation, azimuth, z in ((0.50, -25, 50, 0.45), (0.78, -25, -55, 0.30)):
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [params.x - 0.015, params.center[1], z]
                cam.distance = distance
                cam.elevation = elevation
                cam.azimuth = azimuth if args.side == "left" else -azimuth
                cameras.append(cam)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
        while exp.data.time < params.duration - 1e-8:
            exp.step()
            if exp.steps % exp.decimation == 0:
                recorder.capture()
            if not exp.finite or np.any(exp.data.warning.number):
                raise RuntimeError("Nonfinite simulation or MuJoCo warning")
            if renderer is not None and exp.data.time >= frame_index / 30:
                panels = []
                for cam in cameras:
                    renderer.update_scene(exp.data, camera=cam, scene_option=opt)
                    panels.append(renderer.render())
                frame = Image.fromarray(np.concatenate(panels, axis=1))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 96), fill=(20, 28, 36))
                draw.text((16, 10), f"{args.side.upper()} HAND | t={exp.data.time:05.2f}s | {exp.phase()}", font=font, fill="white")
                thumb_ref = np.rad2deg(exp.cfg["hands"][args.side]["open"][0])
                draw.text((16, 39), f"Cylinder: diameter {params.radius*2000:.0f} mm / height {params.height*1000:.0f} mm / {params.mass*1000:.0f} g | Thumb initial ref: {thumb_ref:.0f} deg", font=font, fill=(160, 210, 240))
                forces = "  ".join(f"{k}={v:.1f}N" for k, v in exp.current_contacts.items() if v > 0.05)
                draw.text((16, 67), "Contact normal forces: " + (forces or "none"), font=font, fill="white")
                draw.rectangle((0, 674, 1280, 720), fill=(20, 28, 36))
                z = exp.data.xpos[exp.cylinder_body, 2]
                text = f"cmd={exp.control.controllers[args.side].command} | object z={z:.3f}m | gravity ON | free object, no weld/adhesion"
                draw.text((16, 687), text, font=font, fill="white")
                array = np.asarray(frame)
                writer.append_data(array)
                for when, name in ((0.0, "initial"), (4.5, "closing"), (9.5, "hold"), (14.5, "release")):
                    if exp.data.time >= when and name not in saved:
                        imageio.imwrite(out / f"{name}.png", array)
                        saved.add(name)
                frame_index += 1
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        report = exp.report()
        report["hand_configuration"] = exp.cfg
        report["mujoco_version"] = mujoco.__version__
        report["video_frames"] = frame_index
        (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "trace.json").write_text(json.dumps(exp.samples) + "\n")
        recorder.save(out, report)
    print(json.dumps(report, indent=2))
    if args.no_grasp:
        return 0 if (not report["grasp_passed"] and report["hold_min_height_m"] < params.z - 0.08) else 1
    return 0 if report["grasp_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
