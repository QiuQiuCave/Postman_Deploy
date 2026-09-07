"""Record the full-body tabletop demo, explicitly bypassing precision acceptance.

Policy/checkpoint numerical parity remains mandatory. This entry point never
changes grasp success criteria, object poses, contact forces, or robot controls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from datetime import datetime, timezone
import json
import math
import os
import re


def _json_safe(value):
    """Keep failure evidence valid JSON, even after non-finite simulation state."""
    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


def _phase_filename(index, phase):
    return f"{index:02d}_{re.sub(r'[^a-zA-Z0-9_-]+', '_', phase).strip('_') or 'phase'}.png"


def _make_cameras(exp):
    import mujoco
    import numpy as np

    position = np.asarray(exp.object_state["position_m"], dtype=float)
    cameras = []
    # A fixed world camera makes genuine object translation visible. The close
    # view covers the initial cylinder and the 8 cm lift / 10 cm inward motion.
    for distance, elevation, azimuth, target in (
        (3.40, 0, 90, (0.13, 0.0, 0.92)),
        (0.73, -22, 50, position + np.array([-0.035, -0.03, 0.035])),
    ):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _add_place_marker(scene, destination_transform, table_height):
    """Add a flat translucent destination disc to rendering, never to physics."""
    import mujoco
    import numpy as np

    if scene.ngeom >= scene.maxgeom:
        return False
    position = np.asarray(destination_transform, dtype=float)[:3, 3].copy()
    position[2] = float(table_height) + 0.0005
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.025, 0.0003, 0.0], dtype=float),
        position,
        np.eye(3, dtype=float).ravel(),
        np.array([0.1, 0.9, 0.3, 0.4], dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1
    return True


def _render_frame(exp, renderer, cameras, options, font):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_place_marker(renderer.scene, exp.destination_transform, exp.table_height)
        panels.append(renderer.render())
    frame = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, 1280, 136), fill=(20, 28, 36))
    draw.text((14, 8), f"DEMO / precision gate bypassed | t={exp.data.time:05.2f}s | {exp.phase}",
              font=font, fill=(255, 207, 95))
    for index, side in enumerate(("left", "right")):
        error = exp.errors(side)
        draw.text((14, 36 + index * 27),
                  f"{side.upper()} WRIST: {float(error['wrist_position_m'])*1000:.1f} mm / "
                  f"{float(error['orientation_deg']):.1f} deg | speed "
                  f"{float(error['wrist_linear_speed_mps']):.3f} m/s",
                  font=font, fill=(160, 210, 240))
    contacts = exp.contact_state
    object_state = exp.object_state
    clearance = float(object_state["bottom_height_m"]) - float(exp.table_height)
    draw.text((14, 90),
              f"ACTUAL contacts: opposing fingers={'YES' if contacts['opposed'] else 'NO'} | "
              f"table={'YES' if contacts['table_contact'] else 'NO'} | bottom-table={clearance*1000:+.1f} mm",
              font=font, fill="white")
    draw.text((14, 114), "FULL BODY + TABLE", font=font, fill=(190, 200, 210))
    draw.text((654, 114), "HAND / OBJECT | green disc = PLACE TARGET", font=font, fill=(190, 200, 210))
    draw.rectangle((0, 635, 1280, 720), fill=(20, 28, 36))
    forces = contacts["fingers_normal_force_N"]
    force_text = "  ".join(f"{name}={float(force):.1f}N" for name, force in forces.items())
    draw.text((14, 639), "Contact normal forces: " + (force_text or "none"), font=font, fill="white")
    draw.text((14, 665),
              f"Cylinder tilt={float(object_state['tilt_deg']):.1f} deg | "
              "gravity ON | free object, no weld/adhesion | policy 50 Hz / hand 100 Hz / physics 1 kHz",
              font=font, fill="white")
    if exp.failure:
        status = "FAILED: " + str(exp.failure)
        color = (255, 145, 120)
    elif exp.done:
        completed = bool(exp.report().get("demo_completed", False))
        status = "DEMO COMPLETED (not precision acceptance)" if completed else "DEMO STOPPED / completion not verified"
        color = (155, 225, 175) if completed else (255, 207, 95)
    else:
        status = "DEMO IN PROGRESS | reach precision is displayed, not an advance gate"
        color = (255, 207, 95)
    draw.text((14, 692), status[:126], font=font, fill=color)
    return np.asarray(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true", help="Record 30 fps MP4 in addition to phase screenshots")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    from PIL import ImageFont
    from common.r2v2_reach_sim import require_parity, sha256
    from common.r2v2_tabletop_demo import TabletopDemoExperiment, load_demo_config

    cfg = load_demo_config(args.config)
    evidence = require_parity(args.parity_report, cfg["reach"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_tabletop_demo" / stamp).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("DEMO ONLY: reach precision acceptance is bypassed; numerical parity remains required.", flush=True)
    print(f"Output: {out}", flush=True)
    exp = renderer = writer = None
    frames = 0
    frame_times, screenshots = [], []
    runtime_error = None
    cleanup_errors = []
    last_phase = None
    report = {"demo_completed": False, "phase": "INITIALIZATION"}
    try:
        exp = TabletopDemoExperiment(cfg)
        print(f"New full model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}", flush=True)
        print(f"Table height: {exp.table_height:.4f} m", flush=True)
        renderer = mujoco.Renderer(exp.model, height=720, width=640)
        if args.video:
            writer = imageio.get_writer(str(out / "demo.mp4"), fps=30, codec="libx264", quality=8)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        cameras = _make_cameras(exp)
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
        while True:
            phase_changed = exp.phase != last_phase
            video_due = writer is not None and (float(exp.data.time) >= frames / 30 or exp.done)
            if phase_changed:
                print(f"t={exp.data.time:.3f}: {exp.phase} {exp.failure or ''}", flush=True)
            if phase_changed or video_due or exp.done:
                array = _render_frame(exp, renderer, cameras, options, font)
                if phase_changed:
                    name = _phase_filename(len(screenshots), exp.phase)
                    imageio.imwrite(out / name, array)
                    screenshots.append({"time_s": float(exp.data.time), "phase": exp.phase, "file": name})
                    if last_phase is None:
                        imageio.imwrite(out / "initial.png", array)
                if video_due:
                    writer.append_data(array)
                    frames += 1
                    frame_times.append(float(exp.data.time))
                if exp.done:
                    imageio.imwrite(out / "final.png", array)
            last_phase = exp.phase
            if exp.done:
                break
            exp.step()
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        print(f"Demo terminated: {runtime_error}", file=sys.stderr, flush=True)
        if exp is not None and not exp.done and callable(getattr(exp, "fail", None)):
            try:
                exp.fail(runtime_error)
            except Exception as fail_error:
                cleanup_errors.append(f"Mark failure: {type(fail_error).__name__}: {fail_error}")
    finally:
        # Close failures must not prevent the physics trace/report being saved.
        for name, resource in (("video writer", writer), ("renderer", renderer)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as close_error:
                    cleanup_errors.append(f"{name}: {type(close_error).__name__}: {close_error}")
        if exp is not None:
            try:
                report = exp.report()
            except Exception as report_error:
                cleanup_errors.append(f"Experiment report: {type(report_error).__name__}: {report_error}")
                report = {"demo_completed": False, "phase": exp.phase, "failure": exp.failure}
        report.update({
            "mode": "demo_precision_gate_bypassed",
            "precision_acceptance_passed": False,
            "configuration": cfg,
            "parity_report": str(args.parity_report.resolve()),
            "parity_report_sha256": sha256(args.parity_report),
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "onnx_sha256": evidence["onnx_sha256"],
            "adapter_sha256": evidence["adapter_sha256"],
            "mujoco_version": mujoco.__version__,
            "video_frames": frames,
            "video_fps": 30 if args.video else None,
            "video_frame_times_s": frame_times,
            "phase_screenshots": screenshots,
            "runtime_error": runtime_error,
            "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "demo_completed", "phase", "failure", "duration_s", "runtime_error", "cleanup_errors"
    )}), indent=2))
    return 0 if report.get("demo_completed") is True and runtime_error is None and not cleanup_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
