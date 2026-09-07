"""Gate new-asset free-standing and empty-hand Reach before any grasp trial."""

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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from common.r2v2_reach_sim import ReachCompatibilityExperiment, load_reach_config, require_parity, sha256

    cfg = load_reach_config(args.config)
    evidence = require_parity(args.parity_report, cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_reach" / stamp).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    exp = ReachCompatibilityExperiment(cfg)
    print(f"New full model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}", flush=True)
    print(f"Foot-calibrated initial base height: {exp.initial_base_height:.9f} m", flush=True)
    print(f"Output: {out}", flush=True)
    renderer = writer = None
    frames = 0
    last_phase = None
    try:
        if args.video:
            renderer = mujoco.Renderer(exp.model, height=720, width=640)
            writer = imageio.get_writer(str(out / "compatibility.mp4"), fps=30, codec="libx264", quality=8)
            opt = mujoco.MjvOption()
            opt.geomgroup[3] = 0
            cameras = []
            for distance, elevation, azimuth, target in ((3.15, -10, 35, (0.08, 0.0, 0.92)),
                                                       (1.65, -10, 90, (0.2, 0.0, 1.13))):
                cam = mujoco.MjvCamera()
                cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
                cam.lookat[:] = target
                cameras.append(cam)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
        while not exp.done:
            exp.step()
            if exp.phase != last_phase:
                print(f"t={exp.data.time:.3f}: {exp.phase} {exp.failure or ''}", flush=True)
                last_phase = exp.phase
            if renderer is not None and (exp.data.time >= frames / 30 or exp.done):
                panels = []
                for cam in cameras:
                    renderer.update_scene(exp.scratch, camera=cam, scene_option=opt)
                    panels.append(renderer.render())
                frame = Image.fromarray(np.concatenate(panels, axis=1))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 112), fill=(20, 28, 36))
                draw.text((14, 10), f"NEW FULL R2V2 | t={exp.data.time:05.2f}s | {exp.phase} | EMPTY-HAND GATE", font=font, fill="white")
                for i, side in enumerate(("left", "right")):
                    e = exp.errors(side)
                    draw.text((14, 40 + 28*i),
                              f"{side.upper()} WRIST: error {e['wrist_position_m']*1000:.1f} mm / {e['orientation_deg']:.1f} deg | speed {e['wrist_linear_speed_mps']:.3f} m/s",
                              font=font, fill=(160, 210, 240))
                draw.rectangle((0, 660, 1280, 720), fill=(20, 28, 36))
                draw.text((14, 665), "Free base | native collision/inertia | policy 50 Hz | hands 100 Hz | physics 1 kHz", font=font, fill="white")
                draw.text((14, 691), exp.failure or "No table/cylinder/grasp task until compatibility passes", font=font,
                          fill=(255, 140, 100) if exp.failure else "white")
                array = np.asarray(frame)
                writer.append_data(array)
                if frames == 0:
                    imageio.imwrite(out / "initial.png", array)
                if exp.done:
                    imageio.imwrite(out / "final.png", array)
                frames += 1
    except Exception as exc:
        if not exp.done:
            exp.fail(f"Runtime error: {type(exc).__name__}: {exc}")
        raise
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        report = exp.report()
        report["parity_report"] = str(args.parity_report.resolve())
        report["parity_report_sha256"] = sha256(args.parity_report)
        report["checkpoint_sha256"] = evidence["checkpoint_sha256"]
        report["video_frames"] = frames
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "trace.json").write_text(json.dumps(exp.samples) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "standing_passed", "phase", "failure", "duration_s", "final_errors")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
