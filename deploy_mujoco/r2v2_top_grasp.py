"""Record an isolated-hand top-down grasp, transfer, release and retreat.

This is a kinematic wrist-fixture experiment, NOT a full-body Reach-policy run.
The cylinder remains a free rigid body moved only by gravity and contact. The
renderer never changes controls, targets, object state, or acceptance criteria.
The final two video seconds repeat an explicitly labelled frozen image; they
are not additional simulated hold time. Exit code zero means artifact capture
completed normally, NOT that grasping or placing passed acceptance.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json


FPS = 30
PANEL_WIDTH = 640
PANEL_HEIGHT = 560
HEADER_HEIGHT = 160
FRAME_HEIGHT = 800
TERMINAL_HOLD_SECONDS = 2.
VIDEO_NAME = "top_grasp.mp4"


def _metric(value, fmt=".2f", multiplier=1.):
    """Never turn a missing or invalid measurement into a displayed zero."""
    if value is None:
        return "N/A"
    try:
        number = float(value) * multiplier
    except (TypeError, ValueError):
        return "N/A"
    return format(number, fmt) if math.isfinite(number) else "NONFINITE"


def _boolean(value):
    return "N/A" if value is None else "YES" if value else "NO"


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _status(exp, *, terminal_freeze=False):
    if terminal_freeze:
        return "TERMINAL FREEZE | physics stopped; repeated image is NOT additional hold evidence", (255, 207, 95)
    if exp.failure:
        return "STOP: " + str(exp.failure), (255, 145, 120)
    if exp.done:
        return "EXPERIMENT ENDED | completion alone does not mean grasp/place success; see report.json", (255, 207, 95)
    return "FREE OBJECT / REAL CONTACT | finger command 1 does not prove grasp; no object weld or animation", (190, 215, 230)


def _make_cameras(exp):
    """Fixed side and oblique views: apparent object motion is world motion."""
    import mujoco
    import numpy as np

    layout = getattr(exp, "layout", None)
    initial = _field(layout, "cylinder_initial_position")
    if initial is None:
        initial = exp.current_metrics.get("object_position_m")
    if initial is None:
        raise ValueError("Cannot frame top grasp without the actual initial cylinder position")
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("Initial cylinder position must be finite XYZ")
    # Center on the complete planned transfer, if the scene exposes its target.
    destination = _field(layout, "cylinder_place_position")
    if destination is None:
        destination = _field(layout, "cylinder_target_position")
    center = initial.copy()
    if destination is not None:
        destination = np.asarray(destination, dtype=float)
        if destination.shape != (3,) or not np.isfinite(destination).all():
            raise ValueError("Cylinder destination must be finite XYZ")
        center = (initial + destination) / 2.
    # Includes the upper wrist, initial hover and the raised carrying segment.
    center[2] += .08
    cameras = []
    side = getattr(exp, "side", "left")
    for distance, elevation, azimuth in ((.95, 0., 90.), (.76, -35., 135.)):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation = distance, elevation
        camera.azimuth = azimuth if side == "left" else -azimuth
        camera.lookat[:] = center
        cameras.append(camera)
    return cameras


def _camera_record(cameras):
    return [{
        "distance_m": camera.distance, "elevation_deg": camera.elevation,
        "azimuth_deg": camera.azimuth, "lookat_world_m": camera.lookat.copy(),
        "fixed_world_camera": True, "view": view,
    } for camera, view in zip(cameras, ("exact_side", "oblique_closeup"))]


def _render_frame(exp, renderer, cameras, options, font, *, terminal_freeze=False):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (2 * PANEL_WIDTH, FRAME_HEIGHT), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    metrics = exp.current_metrics
    profile = getattr(exp, "profile", {})
    side = getattr(exp, "side", "left")
    controllers = getattr(getattr(exp, "hands", None), "controllers", {})
    command = getattr(controllers.get(side), "command", None)
    draw.text((14, 8), "ISOLATED HAND / TOP GRASP | "
              f"t={exp.data.time:05.2f}s | {exp.phase}", font=font, fill=(255, 207, 95))
    draw.text((14, 34), "KINEMATIC WRIST FIXTURE - NOT FULL-BODY RL | "
              f"{side.upper()} finger command={command if command is not None else 'N/A'}",
              font=font, fill="white")
    draw.text((14, 60), f"Profile: {_field(profile, 'profile_id', 'N/A')} | "
              f"diameter {_metric(_field(profile, 'radius_m'), '.1f', 2000.)} mm / "
              f"height {_metric(_field(profile, 'height_m'), '.1f', 1000.)} mm / "
              f"mass {_metric(_field(profile, 'mass_kg'), '.0f', 1000.)} g", font=font, fill=(160, 210, 240))
    draw.text((14, 86), "ACTUAL contact: opposing fingers "
              f"{_boolean(metrics.get('opposed_contact'))} | table {_boolean(metrics.get('table_contact'))} | "
              f"verified pickup {_boolean(metrics.get('grasp_verified'))}", font=font, fill="white")
    draw.text((14, 112), f"Object bottom/table: {_metric(metrics.get('clearance_m'), '+.1f', 1000.)} mm | "
              f"tilt {_metric(metrics.get('object_tilt_deg'), '.1f')} deg | "
              f"speed {_metric(metrics.get('object_linear_speed_mps'), '.3f')} m/s | "
              f"relative slip {_metric(metrics.get('grasp_slip_m'), '.1f', 1000.)} mm",
              font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE / FIXED WORLD CAMERA", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "OBLIQUE CLOSE-UP / FIXED WORLD CAMERA", font=font, fill=(190, 200, 210))
    draw.text((14, 730), f"Actual upward hand force {_metric(metrics.get('hand_vertical_force_N'), '+.2f')} N | "
              f"table support {_metric(metrics.get('table_vertical_force_N'), '+.2f')} N | "
              "gravity ON; object pose is measured", font=font, fill="white")
    draw.text((14, 754), "Wrist is externally driven for grasp calibration. "
              "Binary finger control and physical object contact remain independent.",
              font=font, fill=(190, 200, 210))
    status, color = _status(exp, terminal_freeze=terminal_freeze)
    draw.text((14, 778), status[:142], font=font, fill=color)
    return np.asarray(canvas)


def _candidate_options(args):
    """JSON is data only; explicit command-line values override candidate data."""
    candidate = {}
    if args.candidate_json is not None:
        candidate = json.loads(args.candidate_json.read_text())
        if not isinstance(candidate, dict):
            raise ValueError("--candidate-json must contain an object of candidate fields")
    for name, value in (("side", args.side), ("depth_m", None if args.depth_mm is None else args.depth_mm / 1000.),
                        ("yaw_deg", args.yaw_deg), ("tilt_deg", args.tilt_deg),
                        ("lateral_m", None if args.lateral_mm is None else args.lateral_mm / 1000.)):
        if value is not None:
            candidate[name] = value
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="baseline_40mm_100g")
    parser.add_argument("--side", choices=("left", "right"))
    parser.add_argument("--depth-mm", type=float)
    parser.add_argument("--yaw-deg", type=float)
    parser.add_argument("--tilt-deg", type=float)
    parser.add_argument("--lateral-mm", type=float)
    parser.add_argument("--candidate-json", type=Path,
                        help="Optional JSON object of TopGraspCandidate fields, in its native units")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("ISOLATED HAND TOP GRASP: wrist fixture, NOT full-body policy; completion != success", flush=True)
    print(f"Output: {out}", flush=True)
    exp = renderer = writer = None
    runtime_error = runtime_traceback = None
    cleanup_errors, frame_times, screenshots, camera_configuration = [], [], [], []
    frame_count = freeze_frame_count = 0
    next_video_time_s, last_phase, final_frame = 0., None, None
    reached_terminal_state = False
    report = {"phase": "INITIALIZATION", "strict_success": None}
    mujoco_version = None
    candidate_options = {}
    try:
        import mujoco
        from common.r2v2_top_grasp import TopGraspExperiment
        from common.r2v2_top_grasp_scene import TopGraspCandidate

        mujoco_version = mujoco.__version__
        candidate_options = _candidate_options(args)
        exp = TopGraspExperiment(profile=args.profile, candidate=TopGraspCandidate(**candidate_options), keep_trace=True)
        next_video_time_s = float(exp.data.time)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / VIDEO_NAME), fps=FPS, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-threads", "2"])
            options = mujoco.MjvOption()
            # Group 3 is collision-only proxy geometry in the R2V2 asset.
            options.geomgroup[3] = 0
            # The unused hand is physically parked, not removed from physics.
            options.geomgroup[1] = getattr(exp, "side", "left") == "left"
            options.geomgroup[2] = getattr(exp, "side", "left") == "right"
            cameras = _make_cameras(exp)
            camera_configuration = _camera_record(cameras)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
        while True:
            phase_changed = exp.phase != last_phase
            video_due = writer is not None and (float(exp.data.time) + 1e-9 >= next_video_time_s or exp.done)
            if phase_changed:
                print(f"t={exp.data.time:.3f}: {exp.phase} {exp.failure or ''}", flush=True)
            if renderer is not None and (phase_changed or video_due or exp.done):
                frame = _render_frame(exp, renderer, cameras, options, font)
                if phase_changed:
                    name = _phase_filename(len(screenshots), exp.phase)
                    imageio.imwrite(out / name, frame)
                    screenshots.append({"time_s": float(exp.data.time), "phase": exp.phase, "file": name})
                    if last_phase is None:
                        imageio.imwrite(out / "initial.png", frame)
                if video_due:
                    writer.append_data(frame)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
                    next_video_time_s += 1. / FPS
                if exp.done:
                    final_frame = _render_frame(exp, renderer, cameras, options, font, terminal_freeze=True)
                    imageio.imwrite(out / "final.png", final_frame)
            last_phase = exp.phase
            if exp.done:
                reached_terminal_state = True
                break
            previous_time = float(exp.data.time)
            exp.step()
            if not exp.done and (not math.isfinite(float(exp.data.time)) or float(exp.data.time) <= previous_time):
                raise RuntimeError("Experiment did not advance a finite simulation clock")
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        runtime_traceback = traceback.format_exc()
        print(runtime_traceback, file=sys.stderr, flush=True)
        if exp is not None and not exp.done and callable(getattr(exp, "fail", None)):
            try:
                exp.fail(runtime_error)
            except Exception as fail_error:
                cleanup_errors.append(f"Mark runtime failure: {type(fail_error).__name__}: {fail_error}")
        if renderer is not None and exp is not None:
            try:
                final_frame = _render_frame(exp, renderer, cameras, options, font, terminal_freeze=True)
                imageio.imwrite(out / "final.png", final_frame)
                if writer is not None:
                    writer.append_data(final_frame)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
            except Exception as render_error:
                cleanup_errors.append(f"Final failure frame: {type(render_error).__name__}: {render_error}")
    finally:
        if writer is not None and final_frame is not None:
            try:
                for _ in range(round(FPS * TERMINAL_HOLD_SECONDS)):
                    writer.append_data(final_frame)
                    frame_count += 1
                    freeze_frame_count += 1
                    frame_times.append(float(exp.data.time))
            except Exception as freeze_error:
                cleanup_errors.append(f"Terminal freeze: {type(freeze_error).__name__}: {freeze_error}")
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
                cleanup_errors.append(f"Report: {type(report_error).__name__}: {report_error}")
                report = {"phase": exp.phase, "failure": exp.failure, "strict_success": None}
        report.update({
            "renderer_scope": "ISOLATED HAND / KINEMATIC WRIST FIXTURE / FREE PHYSICAL CYLINDER",
            "renderer_runtime_completed": bool(reached_terminal_state and runtime_error is None and not cleanup_errors),
            "renderer_exit_code_semantics": "0 = normal terminal state and artifacts, NOT grasp/place success",
            "full_body_policy_used": False, "rendered_object_uses_actual_simulation_pose": True,
            "requested_profile": args.profile, "requested_candidate_options": candidate_options,
            "candidate_json_path": None if args.candidate_json is None else str(args.candidate_json.resolve()),
            "mujoco_version": mujoco_version, "video_file": None if args.no_video else VIDEO_NAME,
            "video_frames": frame_count, "video_fps": None if args.no_video else FPS,
            "video_frame_times_s": frame_times, "terminal_freeze_frames": freeze_frame_count,
            "terminal_freeze_is_not_simulation": True, "camera_configuration": camera_configuration,
            "phase_screenshots": screenshots, "runtime_error": runtime_error,
            "runtime_traceback": runtime_traceback, "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
        _write_json(out / "targets.json", [] if exp is None else exp.targets)
        # Completion report is last: an artifact-write error must not leave a
        # report asserting the complete output set was saved.
        _write_json(out / "report.json", report)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "renderer_runtime_completed", "phase", "failure", "success",
        "grasp_verified", "release_commanded", "duration_s", "runtime_error", "cleanup_errors",
    )}), indent=2))
    return 0 if report["renderer_runtime_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
