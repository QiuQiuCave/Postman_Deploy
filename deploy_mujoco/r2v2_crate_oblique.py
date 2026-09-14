"""Render an oblique hand/crate insertion and lifting fixture experiment.

This is not a whole-body policy rollout. Dynamic hand wrists are driven by
external, wrist-only support fixtures. The free crate receives gravity and
real contacts only: no crate weld, adhesion, pose reset or auxiliary force.
COMPLETE alone is not evidence of a pickup or a stable horizontal lift.
"""

import argparse
from dataclasses import asdict
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
HEADER_HEIGHT = 170
FRAME_HEIGHT = 800
TERMINAL_HOLD_SECONDS = 2.


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tip-up", type=float, default=0., metavar="DEG",
                        help="Insertion pitch; positive values raise the fingertips")
    parser.add_argument("--yaw", type=float, default=0., metavar="DEG",
                        help="Positive yaw deflects the fingertips toward box +X (mirrored left/right)")
    parser.add_argument("--roll", type=float, default=0., metavar="DEG",
                        help="Roll around the finger-extension direction, mirrored left/right")
    parser.add_argument("--insertion-mm", type=float, default=60., metavar="MM")
    parser.add_argument("--curl", type=float, default=.8, metavar="RAD",
                        help="Closed four-finger flexion target")
    parser.add_argument("--seating-mm", type=float, default=17.5, metavar="MM")
    parser.add_argument("--both-hands", action="store_true",
                        help="Optional bilateral comparison; default is LEFT ONLY, right hand parked away")
    parser.add_argument("--top-view", action="store_true",
                        help="Keep the left overview and replace the right handle close-up with a fixed top view")
    parser.add_argument("--output", required=True, type=Path,
                        help="New or empty output directory (always records an MP4)")
    return parser


def _validate_arguments(parser, args):
    for name in ("tip_up", "yaw", "roll", "insertion_mm", "curl", "seating_mm"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    for name in ("tip_up", "yaw", "roll"):
        if abs(getattr(args, name)) > 45:
            parser.error(f"--{name.replace('_', '-')} must be within +/-45 degrees")
    if not 20 <= args.insertion_mm <= 80:
        parser.error("--insertion-mm must be between 20 and 80 mm")
    if not 0 <= args.curl <= 1.4:
        parser.error("--curl must be between 0 and 1.4 rad")
    if not 0 <= args.seating_mm <= 30:
        parser.error("--seating-mm must be between 0 and 30 mm")


def _make_cameras(exp, *, top_view=False):
    """Two world-fixed cameras; neither follows a moving crate or wrist."""
    import mujoco
    import numpy as np

    exp.sync()
    origin = exp.scratch.xpos[exp.model.body("cargo_crate").id].copy()
    box = exp.crate_params
    hole_height = .5 * (box.handle_opening_bottom + box.handle_opening_top)
    right_specification = (
        # Wide enough for both active hands before insertion, including yaw
        # variants. This camera is fixed; it never tracks wrist or box motion.
        (1.35 + max(0., box.width - .26), -90., 90.,
         (.0, .0, box.height * .7))
        if top_view else
        (.71 + max(0., box.width - .26), -15., 65.,
         (.0, box.width * .5, hole_height + .025))
    )
    specifications = (
        (1.12 + max(0., box.width - .26), -26., 135.,
         (.0, .055, box.height * .7)),
        right_specification,
    )
    cameras = []
    for distance, elevation, azimuth, offset in specifications:
        camera = mujoco.MjvCamera()
        camera.distance = distance
        camera.elevation = elevation
        camera.azimuth = azimuth
        camera.lookat[:] = origin + np.asarray(offset)
        cameras.append(camera)
    return cameras


def _camera_record(cameras, *, top_view=False):
    views = ("overview", "top" if top_view else "left_handle")
    return [{"distance_m": camera.distance, "elevation_deg": camera.elevation,
             "azimuth_deg": camera.azimuth, "lookat_world_m": camera.lookat.copy(),
             "fixed_world_camera": True, "view": view}
            for camera, view in zip(cameras, views, strict=True)]


def _status(exp, report=None):
    if exp.failure:
        return "FAILED: " + str(exp.failure), (255, 145, 120)
    if not exp.done:
        return "IN PROGRESS: physical pickup and stable horizontal lift evaluated separately", (255, 210, 110)
    report = exp.report() if report is None else report
    pickup = report.get("pickup_verified") is True
    level_lift = report.get("lift_passed") is True
    if pickup and level_lift:
        return "VERIFIED PICKUP + STABLE HORIZONTAL LIFT | externally supported wrist, not robot Reach", (155, 225, 175)
    if pickup:
        return "PICKUP VERIFIED (>=1 s contact-supported clearance); stable horizontal lift NOT passed", (255, 210, 110)
    return "STOPPED: physical pickup NOT verified; COMPLETE does not mean success", (255, 145, 120)


def _maximum_tracking_error(metrics):
    import numpy as np

    values = metrics.get("wrist_tracking_error_m")
    if isinstance(values, dict):
        values = list(values.values())
    if values is None or np.size(values) == 0:
        return "n/a"
    return f"{float(np.max(values))*1000:.2f} mm"


def _render_frame(exp, renderer, cameras, options, font, *, top_view=False):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        panels.append(renderer.render().copy())
    image = Image.new("RGB", (PANEL_WIDTH * 2, FRAME_HEIGHT), (20, 28, 36))
    image.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(image)
    metrics, candidate = exp.current_metrics, exp.candidate
    left_load = float(metrics["hands"]["left"]["vertical_force_N"])
    right_load = float(metrics["hands"]["right"]["vertical_force_N"])
    box = exp.crate_params
    mode = "BOTH HANDS" if len(candidate.active_sides) == 2 else "LEFT ONLY; right hand parked away"
    draw.text((14, 7), f"OBLIQUE INSERTION / {mode} | NO FULL-BODY POLICY", font=font, fill=(255, 207, 95))
    draw.text((14, 32),
              f"Tip-up {candidate.tip_up_deg:+.1f} deg (positive: fingertips UP) | "
              f"yaw {candidate.yaw_deg:+.1f} deg (+toward box X) | roll {candidate.roll_deg:+.1f} deg | "
              f"insert {candidate.insertion_m*1000:.0f} mm", font=font, fill="white")
    draw.text((14, 57), f"t={exp.data.time:05.2f}s | {exp.phase} | "
              f"curl {candidate.closed_curl_rad:.3f} rad | seat {candidate.seating_m*1000:.1f} mm | "
              f"wrist fixture tracking error {_maximum_tracking_error(metrics)}", font=font, fill="white")
    draw.text((14, 82), f"Lowest box bottom above table: {float(metrics['clearance_m'])*1000:+.2f} mm | "
              f"box tilt {float(metrics['crate_tilt_deg']):.2f} deg | "
              f"BOX W{box.width*100:.0f} D{box.depth*100:.0f} H{box.height*100:.0f} cm", font=font, fill="white")
    draw.text((14, 107), f"NET contact load on crate, WORLD +Z: LEFT {left_load:+.2f} N | "
              f"RIGHT {right_load:+.2f} N | sum {left_load+right_load:+.2f} N | "
              f"weight {float(metrics['crate_weight_N']):.2f} N", font=font, fill=(160, 210, 240))
    draw.text((14, 131), "External wrist fixtures supply motion. Crate: no weld / adhesion / auxiliary applied force.",
              font=font, fill=(220, 215, 190))
    draw.text((14, 151), "OVERVIEW / FIXED WORLD CAMERA", font=font, fill=(190, 200, 210))
    right_label = "TOP / FIXED WORLD CAMERA" if top_view else "LEFT HANDLE / FIXED WORLD CAMERA"
    draw.text((654, 151), right_label, font=font, fill=(190, 200, 210))
    status, color = _status(exp)
    # Long failure reasons are preserved untruncated in report.json.
    draw.text((14, 736), status[:136], font=font, fill=color)
    slip = metrics.get("grasp_slip_m")
    slip_text = "n/a before grip baseline" if slip is None else f"{float(slip)*1000:.2f} mm"
    draw.text((14, 761), f"Grip-relative translation: {slip_text} | "
              f"table net +Z load: {float(metrics['table_vertical_force_N']):+.2f} N | "
              "NET hand load includes palm, thumb and fingers.", font=font, fill=(190, 200, 210))
    return np.asarray(image)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    out.mkdir(parents=True, exist_ok=True)
    exp = renderer = writer = candidate = None
    frames = terminal_hold_frames = 0
    frame_times, screenshots, camera_configuration, cleanup_errors = [], [], [], []
    runtime_error = runtime_traceback = mujoco_version = None
    last_phase = None
    report = {"pickup_verified": False, "lift_passed": False, "experiment_completed": False,
              "phase": "INITIALIZATION"}
    print("WRIST-FIXTURE EXPERIMENT: no robot policy; only actual contact may lift the free crate.", flush=True)
    print(f"Output: {out}", flush=True)
    try:
        import imageio.v2 as imageio
        import mujoco
        from PIL import ImageFont
        from common.r2v2_crate_oblique import ObliqueCrateExperiment, ObliqueParameters

        mujoco_version = mujoco.__version__
        candidate = ObliqueParameters(
            tip_up_deg=args.tip_up, yaw_deg=args.yaw, roll_deg=args.roll,
            insertion_m=args.insertion_mm / 1000., closed_curl_rad=args.curl,
            seating_m=args.seating_mm / 1000.,
            active_sides=("left", "right") if args.both_hands else ("left",),
        )
        exp = ObliqueCrateExperiment(candidate=candidate, keep_trace=True)
        print(f"Fixture model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}", flush=True)
        renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
        writer = imageio.get_writer(str(out / "oblique_crate.mp4"), fps=FPS,
                                   codec="libx264", quality=8, macro_block_size=1,
                                   ffmpeg_params=["-movflags", "+faststart", "-threads", "2"])
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        cameras = _make_cameras(exp, top_view=args.top_view)
        camera_configuration = _camera_record(cameras, top_view=args.top_view)
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
        next_frame_time = 0.
        while True:
            phase_changed = exp.phase != last_phase
            video_due = float(exp.data.time) + 1e-9 >= next_frame_time or exp.done
            if phase_changed:
                print(f"t={exp.data.time:.3f}: {exp.phase} {exp.failure or ''}", flush=True)
            if phase_changed or video_due:
                array = _render_frame(exp, renderer, cameras, options, font, top_view=args.top_view)
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
                    next_frame_time = frames / FPS
                if exp.done:
                    imageio.imwrite(out / "final.png", array)
                    # Freeze the terminal result, never integrate further after a failure.
                    terminal_hold_frames = int(TERMINAL_HOLD_SECONDS * FPS)
                    for _ in range(terminal_hold_frames):
                        writer.append_data(array)
                        frames += 1
                        frame_times.append(float(exp.data.time))
            last_phase = exp.phase
            if exp.done:
                break
            exp.step()
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        runtime_traceback = traceback.format_exc()
        print(f"Fixture rendering terminated: {runtime_error}", file=sys.stderr, flush=True)
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
                report.update({"phase": exp.phase, "failure": exp.failure})
        report.update({
            "renderer_scope": "externally driven wrist fixture, free crate; NO full-body controller",
            "render_candidate": None if candidate is None else asdict(candidate),
            "render_angle_convention": {
                "tip_up_positive": "fingertips rise toward world +Z",
                "yaw_positive": "fingertips deflect toward box +X; mirrored left/right",
                "roll": "about finger extension direction; mirrored left/right",
                "composition": "Rz(sign*yaw) @ Rx(-sign*tip_up) @ R0 @ Rx(sign*roll)",
                "sign": {"left": 1, "right": -1},
            },
            "mujoco_version": mujoco_version,
            "video_file": "oblique_crate.mp4" if frames else None,
            "video_frames": frames,
            "video_fps": FPS,
            "video_duration_s": frames / FPS,
            "terminal_hold_frames": terminal_hold_frames,
            "terminal_hold_is_frozen_not_additional_simulation": True,
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
        "pickup_verified", "lift_passed", "experiment_completed", "phase", "failure", "duration_s",
        "video_duration_s", "runtime_error", "cleanup_errors",
    )}), indent=2))
    if runtime_error is not None or cleanup_errors:
        return 2
    return 0 if (report.get("pickup_verified") is True and report.get("experiment_completed") is True
                 and not report.get("failure")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
