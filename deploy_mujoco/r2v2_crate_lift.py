"""Render a two-hand crate-lift fixture test, without a full-body policy.

Both views use fixed world cameras. Rendering never moves the crate, supplies
forces, or changes contact parameters; the experiment alone owns the physics.
Dynamic wrists follow support fixtures through wrist-only welds. The crate
itself has no weld and receives lifting forces only through actual contacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
import traceback

from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json


def _make_cameras(exp):
    import mujoco
    import numpy as np

    exp.sync()
    crate = exp.model.body("cargo_crate").id
    initial = exp.scratch.xpos[crate].copy()
    cameras = []
    extra_span = max(0., exp.crate_params.width-.28, exp.crate_params.depth-.20)
    extra_height = exp.crate_params.height-.10
    for distance, elevation, azimuth, offset in (
        (1.20+extra_span, -25, 135, (0.0, 0.0, 0.090+extra_height)),
        (0.70+extra_span, -20, -65, (0.0, exp.crate_params.width*.375, 0.095+extra_height)),
    ):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = initial + np.asarray(offset)
        cameras.append(camera)
    return cameras


def _camera_record(cameras):
    return [{"distance_m": camera.distance, "elevation_deg": camera.elevation,
             "azimuth_deg": camera.azimuth, "lookat_world_m": camera.lookat.copy()}
            for camera in cameras]


def _render_frame(exp, renderer, cameras, options, font):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        panels.append(renderer.render())
    image = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(image)
    metrics = exp.current_metrics
    left_load = float(metrics["hands"]["left"]["vertical_force_N"])
    right_load = float(metrics["hands"]["right"]["vertical_force_N"])
    slip = metrics.get("grasp_slip_m")
    slip_text = "n/a (no grip baseline)" if slip is None else f"{float(slip)*1000:.2f} mm"
    wrist_errors = metrics.get("wrist_tracking_error_m")
    if isinstance(wrist_errors, dict):
        wrist_errors = list(wrist_errors.values())
    wrist_text = ("n/a" if wrist_errors is None or np.size(wrist_errors) == 0
                  else f"{float(np.max(wrist_errors))*1000:.2f} mm")
    draw.rectangle((0, 0, 1280, 137), fill=(20, 28, 36))
    box = exp.crate_params
    draw.text((14, 8), "WRIST-FIXTURE TEST / NO FULL-BODY POLICY | "
              f"BOX W{box.width*1000:.0f} D{box.depth*1000:.0f} H{box.height*1000:.0f} mm | "
              f"insert {exp.params.insertion_m*1000:.0f} mm", font=font, fill=(255, 207, 95))
    draw.text((14, 35), f"t={exp.data.time:05.2f}s | {exp.phase} | "
              f"grasp={'ON' if exp.params.grasp_enabled else 'OFF (negative control)'} | "
              f"closed curl={exp.params.closed_curl_rad:.3f} rad | wrist tracking max={wrist_text}",
              font=font, fill="white")
    draw.text((14, 62),
              f"NET whole-hand world-Z load: LEFT {left_load:+.2f} N | "
              f"RIGHT {right_load:+.2f} N | sum {left_load+right_load:+.2f} N",
              font=font, fill=(160, 210, 240))
    draw.text((14, 89),
              f"Crate bottom-table: {float(metrics['clearance_m'])*1000:+.2f} mm | "
              f"tilt: {float(metrics['crate_tilt_deg']):.2f} deg | relative slip: {slip_text}",
              font=font, fill="white")
    draw.text((14, 115), "BOTH HANDS / CRATE / TABLE (fixed camera)", font=font, fill=(190, 200, 210))
    draw.text((654, 115), "HANDLE / LOAD-BEARING FINGERS (fixed camera)", font=font, fill=(190, 200, 210))
    draw.rectangle((0, 644, 1280, 720), fill=(20, 28, 36))
    draw.text((14, 649), "Gravity ON | no CRATE weld / adhesion / auxiliary force | WRISTS: weld-driven support fixtures",
              font=font, fill="white")
    draw.text((14, 674), "NET load includes fingers + thumb + palm; internal clamping forces cancel, not extra lifted weight.",
              font=font, fill=(190, 200, 210))
    if exp.failure:
        status, color = "FAILED: " + str(exp.failure), (255, 145, 120)
    elif exp.done:
        passed = bool(exp.report().get("lift_passed", False))
        status = ("FIXTURE LIFT PASSED / final state: both hands still holding aloft, no release"
                  if passed else "STOPPED / physical lift not verified")
        color = (155, 225, 175) if passed else (255, 207, 95)
    else:
        status, color = "IN PROGRESS / grasp and lift judged from actual physics", (255, 207, 95)
    draw.text((14, 697), status[:130], font=font, fill=color)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Lift YAML configuration (default: deploy_mujoco/config/r2v2_crate_lift.yaml)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true", help="Record 30 fps MP4 in addition to phase screenshots")
    parser.add_argument("--no-grasp", action="store_true", help="Negative control: keep the fingers open")
    parser.add_argument("--closed-curl", type=float, help="Closed four-finger flexion target, radians")
    args = parser.parse_args()
    if args.closed_curl is not None and (not math.isfinite(args.closed_curl) or args.closed_curl < 0):
        parser.error("--closed-curl must be a finite, nonnegative angle in radians")
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    from PIL import ImageFont
    from common.r2v2_crate_lift import CrateLiftExperiment, load_lift_config

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_crate_lift" / stamp).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("WRIST-FIXTURE TEST ONLY: no full-body Reach policy; crate motion must come from contact.", flush=True)
    print(f"Output: {out}", flush=True)
    params = exp = renderer = writer = None
    frames = 0
    frame_times, screenshots, camera_configuration = [], [], []
    runtime_error = runtime_traceback = None
    cleanup_errors = []
    last_phase = None
    report = {"lift_passed": False, "experiment_completed": False, "phase": "INITIALIZATION"}
    try:
        params = load_lift_config(args.config)
        changes = {}
        if args.no_grasp:
            changes["grasp_enabled"] = False
        if args.closed_curl is not None:
            changes["closed_curl_rad"] = args.closed_curl
        params = replace(params, **changes)
        exp = CrateLiftExperiment(params=params)
        print(f"Fixture model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}", flush=True)
        print(f"Table height: {exp.table_height:.4f} m", flush=True)
        renderer = mujoco.Renderer(exp.model, height=720, width=640)
        if args.video:
            writer = imageio.get_writer(str(out / "crate_lift.mp4"), fps=30, codec="libx264", quality=8)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        cameras = _make_cameras(exp)
        camera_configuration = _camera_record(cameras)
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
        runtime_traceback = traceback.format_exc()
        print(f"Fixture test terminated: {runtime_error}", file=sys.stderr, flush=True)
        if exp is not None and not exp.done and callable(getattr(exp, "fail", None)):
            try:
                exp.fail(runtime_error)
            except Exception as fail_error:
                cleanup_errors.append(f"Mark failure: {type(fail_error).__name__}: {fail_error}")
    finally:
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
                report = {"lift_passed": False, "experiment_completed": False,
                          "phase": exp.phase, "failure": exp.failure}
        report.update({
            "renderer_scope": "WRIST-FIXTURE TEST / NO FULL-BODY POLICY",
            "parameters": None if params is None else asdict(params),
            "mujoco_version": mujoco.__version__,
            "negative_control": args.no_grasp if params is None else not params.grasp_enabled,
            "video_frames": frames,
            "video_fps": 30 if args.video else None,
            "video_frame_times_s": frame_times,
            "camera_configuration": camera_configuration,
            "phase_screenshots": screenshots,
            "runtime_error": runtime_error,
            "runtime_traceback": runtime_traceback,
            "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "lift_passed", "experiment_completed", "phase", "failure", "duration_s", "runtime_error", "cleanup_errors"
    )}), indent=2))
    return 0 if (report.get("lift_passed") is True and report.get("experiment_completed") is True
                 and not report.get("failure") and runtime_error is None and not cleanup_errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
