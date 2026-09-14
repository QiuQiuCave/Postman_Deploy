"""Record an unassisted full-body dual-arm Reach crate-lift experiment.

Both cameras are fixed in the world. This entry point only records the
experiment: no IK, wrist fixtures, physical pose changes, or auxiliary forces.
Failure videos and logs remain failures; target markers are rendering only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from datetime import datetime, timezone
import json
import os
import traceback

from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json


def _make_cameras(exp):
    import mujoco

    cameras = []
    for distance, elevation, azimuth, target in (
        # Reserve the central y=164..646 viewport for the entire robot.
        # Looking slightly above its centre shifts the robot below the HUD.
        (4.0, -15, 135, (.20, 0., 1.08)),
        (1.7, -25, 135, (.40, 0., exp.table_height + .10)),
    ):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _camera_record(cameras):
    return [{"distance_m": camera.distance, "elevation_deg": camera.elevation,
             "azimuth_deg": camera.azimuth, "lookat_world_m": camera.lookat.copy()}
            for camera in cameras]


def _add_target_markers(scene, exp):
    """Small desired-wrist spheres in MjvScene only; never physical geoms."""
    import mujoco
    import numpy as np

    goals = getattr(exp, "goal_wrist_transforms", {})
    for side, color in (("left", [0.2, 1., .35, .65]), ("right", [.3, .65, 1., .65])):
        if scene.ngeom >= scene.maxgeom or side not in goals:
            continue
        target = np.asarray(goals[side], dtype=float)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                          np.array([.009, .009, .009]), target[:3, 3].copy(),
                          np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def _render_frame(exp, renderer, cameras, options, font):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()  # Refresh only the experiment's private observation scratch.
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_target_markers(renderer.scene, exp)
        panels.append(renderer.render())
    image = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(image)
    m, box, p = exp.current_metrics, exp.crate_params, exp.params
    left_load = float(m["hands"]["left"]["vertical_force_N"])
    right_load = float(m["hands"]["right"]["vertical_force_N"])
    commands = {side: int(exp.hands.controllers[side].command) for side in ("left", "right")}
    slip = m.get("grasp_slip_m")
    slip_text = "not confirmed" if slip is None else f"{float(slip)*1000:.2f} mm"
    draw.rectangle((0, 0, 1280, 163), fill=(20, 28, 36))
    draw.text((14, 8), "FULL-BODY DUAL-ARM REACH / NO WRIST FIXTURES | "
              f"t={exp.data.time:05.2f}s | {exp.phase}", font=font, fill=(255, 207, 95))
    draw.text((14, 34), f"BOX W{box.width*1000:.0f} D{box.depth*1000:.0f} H{box.height*1000:.0f} mm | "
              f"insertion {p.insertion_m*1000:.0f} mm | curl {p.closed_curl_rad:.2f} rad | "
              f"actual hand command L={commands['left']} / R={commands['right']} (0=open, 1=close)",
              font=font, fill="white")
    for index, side in enumerate(("left", "right")):
        error = m["wrist_errors"][side]
        draw.text((14, 60 + index*25), f"{side.upper()} WRIST: "
                  f"{error['position_m']*1000:.1f} mm / {error['orientation_deg']:.1f} deg | "
                  f"actual speed {error['linear_speed_mps']:.3f} m/s",
                  font=font, fill=(160, 210, 240))
    draw.text((650, 60), f"Crate clearance {m['clearance_m']*1000:+.2f} mm | "
              f"tilt {m['crate_tilt_deg']:.2f} deg", font=font, fill="white")
    draw.text((650, 85), f"NET hand load Z: L {left_load:+.2f} N / R {right_load:+.2f} N",
              font=font, fill="white")
    draw.text((14, 112), f"Base tilt {m['base_tilt_deg']:.2f} deg | "
              f"relative crate/wrist slip {slip_text} | desired wrist markers: green L / blue R",
              font=font, fill="white")
    draw.text((14, 139), "FULL ROBOT / FREE BASE / FIXED WORLD CAMERA", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "BOTH ARMS / HANDLES / CRATE", font=font, fill=(190, 200, 210))
    draw.rectangle((0, 647, 1280, 720), fill=(20, 28, 36))
    draw.text((14, 650), "Policy 50 Hz | hands 100 Hz | physics 1 kHz | no IK / weld / auxiliary force",
              font=font, fill="white")
    draw.text((14, 674), "Native full robot + free crate; success requires actual bilateral contact, clearance and stable hold.",
              font=font, fill=(190, 200, 210))
    if exp.failure:
        status, color = "FAILED: " + str(exp.failure), (255, 145, 120)
    elif exp.done:
        passed = bool(exp.report().get("lift_passed", False))
        status = ("FULL-BODY LIFT PASSED / both hands still holding aloft; no release"
                  if passed else "STOPPED / physical lift not verified")
        color = (155, 225, 175) if passed else (255, 207, 95)
    else:
        status, color = "IN PROGRESS / measured wrist accuracy and physical outcomes are recorded", (255, 207, 95)
    draw.text((14, 698), status[:135], font=font, fill=color)
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Full-body crate Reach YAML configuration")
    parser.add_argument("--parity-report", type=Path, required=True,
                        help="Matching training/deployment numerical parity report")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true", help="Record 30 fps MP4, plus phase screenshots")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    import imageio.v2 as imageio
    import mujoco
    from PIL import ImageFont

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_crate_reach" / stamp).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("FULL-BODY REACH TEST: free robot and free crate; no IK or wrist fixtures.", flush=True)
    print(f"Output: {out}", flush=True)
    cfg = exp = renderer = writer = None
    frames = 0
    frame_times, screenshots, camera_configuration = [], [], []
    runtime_error = runtime_traceback = None
    cleanup_errors = []
    last_phase = None
    report = {"lift_passed": False, "experiment_completed": False, "phase": "INITIALIZATION"}
    try:
        from common.r2v2_crate_reach import CrateReachExperiment, load_crate_reach_config

        cfg = load_crate_reach_config(args.config)
        exp = CrateReachExperiment(config=cfg, parity_report=args.parity_report)
        print(f"Full model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}, "
              f"nmocap={exp.model.nmocap}", flush=True)
        print(f"Table height: {exp.table_height:.4f} m", flush=True)
        renderer = mujoco.Renderer(exp.model, height=720, width=640)
        if args.video:
            writer = imageio.get_writer(str(out / "full_body_crate.mp4"), fps=30, codec="libx264", quality=8)
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
        print(f"Full-body test terminated: {runtime_error}", file=sys.stderr, flush=True)
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
            "renderer_scope": "FULL-BODY DUAL-ARM REACH / NO WRIST FIXTURES",
            "render_configuration": cfg,
            "parity_report": str(args.parity_report.resolve()),
            "mujoco_version": mujoco.__version__,
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
