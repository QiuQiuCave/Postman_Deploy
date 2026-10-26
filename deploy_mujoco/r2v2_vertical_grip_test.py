"""Run and optionally render the independent vertical-handle grip experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from common.r2v2_vertical_grip_test import GripTestParameters, VerticalGripExperiment
from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _write_json


def cameras():
    result = []
    for az, el, dist, look in ((155, -25, 1.35, (0, 0, .5)), (-55, -15, .65, (-.025, .24, .51))):
        c = mujoco.MjvCamera()
        c.azimuth, c.elevation, c.distance = az, el, dist
        c.lookat[:] = look
        result.append(c)
    return result


def frame(exp, renderer, views, options, font):
    from PIL import Image, ImageDraw
    exp.measure()
    panels = []
    for camera in views:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        panels.append(renderer.render())
    picture = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(picture)
    m = exp.metrics
    draw.rectangle((0, 0, 1280, 102), fill=(20, 28, 36))
    draw.text((12, 7), "VERTICAL-HANDLE GRASP | DRIVEN WRISTS ONLY / NO WHOLE-BODY POLICY", font=font, fill="white")
    command = exp.hands.controllers["left"].command
    draw.text((12, 30), f"t={m['time_s']:.2f}s  {exp.phase} | hand command={command} | pitch={exp.params.pitch_deg:+.0f} deg | 0.40 kg free crate", font=font, fill=(255, 205, 90))
    loads = [m["hands"][s]["vertical_N"] for s in ("left", "right")]
    draw.text((12, 53), f"Bottom clearance {m['clearance_m']*1000:+.1f} mm | tilt {m['tilt_deg']:.2f} deg | L/R load {loads[0]:+.2f}/{loads[1]:+.2f} N", font=font, fill="white")
    slip = max(h["slip_m"] for h in m["hands"].values())
    opposed = "/".join(str(m["hands"][s]["opposed"]) for s in ("left", "right"))
    draw.text((12, 76), f"Shaft thumb+fingers L/R: {opposed} | slip {slip*1000:.2f} mm (baseline at lift start)", font=font, fill="white")
    draw.rectangle((0, 680, 1280, 720), fill=(20, 28, 36))
    draw.text((12, 690), "Real contact + gravity | no crate weld / adhesion / external force | bottle hand profile, thumb 75 deg", font=font, fill="white")
    return np.asarray(picture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-grasp", action="store_true")
    parser.add_argument("--z-offset", type=float, default=0.)
    parser.add_argument("--outward-offset", type=float, default=0.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--pitch", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    exp = VerticalGripExperiment(GripTestParameters(grasp=not args.no_grasp,
        wrist_z_offset_m=args.z_offset, wrist_outward_offset_m=args.outward_offset, yaw_deg=args.yaw, pitch_deg=args.pitch))
    (args.output/"scene.xml").write_text(exp.xml)
    renderer = writer = None
    try:
        if args.video:
            import imageio.v2 as imageio
            from PIL import ImageFont
            renderer = mujoco.Renderer(exp.model, width=640, height=720)
            writer = imageio.get_writer(str(args.output/"grasp.mp4"), fps=30, codec="libx264", quality=8)
            views = cameras()
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
        frames, last = 0, None
        exp.record()
        while True:
            changed = last != exp.phase
            if changed:
                print(f"t={exp.data.time:.3f} {exp.phase}", flush=True)
            if renderer is not None and (exp.data.time >= frames/30-1e-8 or exp.done):
                array = frame(exp, renderer, views, options, font)
                writer.append_data(array)
                frames += 1
                if changed or exp.done:
                    imageio.imwrite(args.output/f"{exp.phase.lower()}.png", array)
            last = exp.phase
            if exp.done:
                break
            exp.step()
    except BaseException as exc:
        exp.failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        report = exp.report()
        _write_json(args.output/"report.json", report)
        _write_json(args.output/"trace.json", exp.samples)
        _write_json(args.output/"transitions.json", exp.transitions)
        if exp.states:
            times, qpos, qvel, mocap_pos, mocap_quat = zip(*exp.states)
            np.savez_compressed(args.output/"trajectory.npz", time_s=times, qpos=qpos, qvel=qvel,
                                mocap_pos=mocap_pos, mocap_quat=mocap_quat)
    print(json.dumps(_json_safe({k: v for k, v in report.items() if k not in
        ("layout", "hand_configuration", "initial_metrics", "crate_parameters")}), indent=2), flush=True)
    return 0 if exp.failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
