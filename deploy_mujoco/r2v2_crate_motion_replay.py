"""Record a free-base Reach-policy replay of a crate-relative hand trajectory.

Both cameras are fixed in world coordinates. Wrist target markers are visual
decorations only, and the last two video seconds freeze the actual final frame.
The renderer never changes targets, hand commands, physics, or acceptance gates.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_mujoco.r2v2_crate_reach import _add_target_markers
from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json


FPS = 30
PANEL_WIDTH = 640
PANEL_HEIGHT = 560
HEADER_HEIGHT = 160
FRAME_HEIGHT = 800
TERMINAL_HOLD_SECONDS = 2.


def _make_cameras(exp, *, top_view=False):
    """An exact side overview plus a fixed close-up of both handle regions."""
    import mujoco
    import numpy as np

    exp.sync()
    crate_origin = np.asarray(
        exp.scratch.xpos[exp.model.body("cargo_crate").id], dtype=float
    ).copy()
    specifications = (
        (3.05, 0., 90., (.13, 0., .89)),
        (1.26 if top_view else 1.20, -90. if top_view else -30.,
         90. if top_view else 135., crate_origin + np.array([-.015, 0., .095])),
    )
    cameras = []
    for distance, elevation, azimuth, target in specifications:
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _camera_record(cameras, *, top_view=False):
    return [
        {"distance_m": camera.distance, "elevation_deg": camera.elevation,
         "azimuth_deg": camera.azimuth, "lookat_world_m": camera.lookat.copy(),
         "fixed_world_camera": True, "view": view}
        for camera, view in zip(cameras, ("exact_side_full_body", "top" if top_view else "crate_closeup"))
    ]


def _status(exp):
    if exp.failure:
        return "FAILED: " + str(exp.failure), (255, 145, 120)
    if not exp.done:
        return "IN PROGRESS | real tracking, contacts and slip determine phase progression", (255, 210, 110)
    passed = exp.report().get("lift_passed") is True
    if passed:
        return "FULL-BODY LIFT VERIFIED | no wrist support, no release", (155, 225, 175)
    return "STOPPED | successful physical lift NOT verified", (255, 145, 120)


def _render_frame(exp, renderer, cameras, options, font, *, top_view=False):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_target_markers(renderer.scene, exp)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (2 * PANEL_WIDTH, FRAME_HEIGHT), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    m = exp.current_metrics
    source_time = float(getattr(exp, "source_time_s", 0.))
    draw.text((14, 8), "FULL-BODY REACH / CRATE-RELATIVE MOTION REPLAY | "
              f"t={exp.data.time:05.2f}s | source={source_time:05.2f}s | {exp.phase}",
              font=font, fill=(255, 207, 95))
    commands = {side: int(exp.hands.controllers[side].command) for side in ("left", "right")}
    draw.text((14, 34), "SOURCE: downward 20 deg + inward yaw 15 deg | insertion 60 mm | "
              f"binary fingers L={commands['left']} / R={commands['right']}", font=font, fill="white")
    for index, side in enumerate(("left", "right")):
        error = m.get("wrist_errors", {}).get(side, {})
        draw.text((14, 61 + 25 * index),
                  f"{side.upper()} WRIST: {float(error.get('position_m', 0.))*1000:.1f} mm / "
                  f"{float(error.get('orientation_deg', 0.)):.1f} deg | "
                  f"speed {float(error.get('linear_speed_mps', 0.)):.3f} m/s",
                  font=font, fill=(160, 210, 240))
    draw.text((650, 61), f"Crate clearance {float(m.get('clearance_m', 0.))*1000:+.2f} mm | "
              f"tilt {float(m.get('crate_tilt_deg', 0.)):.2f} deg", font=font, fill="white")
    loads = [float(m.get("hands", {}).get(side, {}).get("vertical_force_N", 0.))
             for side in ("left", "right")]
    draw.text((650, 86), f"Actual net hand force Z: L {loads[0]:+.2f} / R {loads[1]:+.2f} N",
              font=font, fill="white")
    slip = m.get("grasp_slip_m")
    slip_text = "not established" if slip is None else f"{float(slip)*1000:.2f} mm"
    draw.text((14, 112), f"Base tilt {float(m.get('base_tilt_deg', 0.)):.2f} deg | "
              f"crate/wrist relative slip {slip_text} | green / blue spheres = desired wrists, not fixtures",
              font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE VIEW / FULL FREE-BASE ROBOT", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "BOTH HANDLES / " + ("FIXED TOP VIEW" if top_view else "FIXED CLOSE-UP"),
              font=font, fill=(190, 200, 210))
    draw.text((14, 729), "Policy 50 Hz | binary hands 100 Hz | physics 1 kHz | no IK / wrist fixtures / auxiliary forces",
              font=font, fill="white")
    draw.text((14, 752), "Recorded crate motion + crate-relative wrists -> world targets; phase gates may pause replay time.",
              font=font, fill=(190, 200, 210))
    status, color = _status(exp)
    draw.text((14, 776), status[:136], font=font, fill=color)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="Recorded motion manifest or bank directory")
    parser.add_argument("--reach-config", type=Path, required=True, help="Pinned wrist-world-v2 Reach YAML")
    parser.add_argument("--parity-report", type=Path, required=True, help="Matching numerical parity report")
    parser.add_argument("--output", type=Path, required=True, help="New or empty artifact directory")
    parser.add_argument("--no-video", action="store_true", help="Run headless physics and save JSON without rendering")
    parser.add_argument("--top-view", action="store_true", help="Replace only the right close-up with a fixed top view")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print("FREE-BASE REACH REPLAY: real model and contacts; no IK or wrist support.", flush=True)
    print(f"Output: {out}", flush=True)
    exp = renderer = writer = None
    runtime_error = runtime_traceback = None
    cleanup_errors, frame_times, screenshots, camera_configuration = [], [], [], []
    frame_count = freeze_frame_count = 0
    next_video_time_s = 0.
    last_phase = None
    final_frame = None
    report = {"lift_passed": False, "experiment_completed": False, "phase": "INITIALIZATION"}
    mujoco_version = None

    try:
        import mujoco
        from common.r2v2_crate_motion_replay import CrateMotionReplayExperiment

        mujoco_version = mujoco.__version__
        exp = CrateMotionReplayExperiment(
            motion_path=args.motion, reach_config=args.reach_config, parity_report=args.parity_report
        )
        print(f"Full model: nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}, "
              f"nmocap={exp.model.nmocap}; table={exp.table_height:.4f} m", flush=True)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / "fullbody_crate_motion_replay.mp4"),
                                        fps=FPS, codec="libx264", quality=8)
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            cameras = _make_cameras(exp, top_view=args.top_view)
            camera_configuration = _camera_record(cameras, top_view=args.top_view)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()

        while True:
            phase_changed = exp.phase != last_phase
            video_due = writer is not None and (float(exp.data.time) + 1e-9 >= next_video_time_s or exp.done)
            if phase_changed:
                print(f"t={exp.data.time:.3f}, source={getattr(exp, 'source_time_s', 0.):.3f}: "
                      f"{exp.phase} {exp.failure or ''}", flush=True)
            if renderer is not None and (phase_changed or video_due or exp.done):
                array = _render_frame(exp, renderer, cameras, options, font, top_view=args.top_view)
                if phase_changed:
                    name = _phase_filename(len(screenshots), exp.phase)
                    imageio.imwrite(out / name, array)
                    screenshots.append({"time_s": float(exp.data.time),
                                        "source_time_s": float(getattr(exp, "source_time_s", 0.)),
                                        "phase": exp.phase, "file": name})
                    if last_phase is None:
                        imageio.imwrite(out / "initial.png", array)
                if video_due:
                    writer.append_data(array)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
                    next_video_time_s += 1. / FPS
                if exp.done:
                    final_frame = array
                    imageio.imwrite(out / "final.png", array)
            last_phase = exp.phase
            if exp.done:
                break
            exp.step()
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        runtime_traceback = traceback.format_exc()
        print(f"Replay terminated: {runtime_error}", file=sys.stderr, flush=True)
        if exp is not None and not exp.done and callable(getattr(exp, "fail", None)):
            try:
                exp.fail(runtime_error)
            except Exception as fail_error:
                cleanup_errors.append(f"Mark failure: {type(fail_error).__name__}: {fail_error}")
        # A finite last simulation state remains useful failure evidence. If a
        # renderer or the simulation state itself failed, retain the prior video.
        if renderer is not None and exp is not None:
            try:
                final_frame = _render_frame(exp, renderer, cameras, options, font, top_view=args.top_view)
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
                cleanup_errors.append(f"Experiment report: {type(report_error).__name__}: {report_error}")
                report = {"lift_passed": False, "experiment_completed": False,
                          "phase": exp.phase, "failure": exp.failure}
        report.update({
            "renderer_scope": "FULL-BODY REACH / RECORDED CRATE-RELATIVE TRAJECTORY / NO WRIST FIXTURES",
            "source_motion_path": str(args.motion.resolve()),
            "reach_configuration_path": str(args.reach_config.resolve()),
            "parity_report": str(args.parity_report.resolve()),
            "mujoco_version": mujoco_version,
            "video_file": None if args.no_video else "fullbody_crate_motion_replay.mp4",
            "video_frames": frame_count,
            "video_fps": None if args.no_video else FPS,
            "video_frame_times_s": frame_times,
            "terminal_freeze_frames": freeze_frame_count,
            "terminal_freeze_is_not_simulation": True,
            "camera_configuration": camera_configuration,
            "phase_screenshots": screenshots,
            "runtime_error": runtime_error,
            "runtime_traceback": runtime_traceback,
            "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
        _write_json(out / "targets.json", [] if exp is None else getattr(exp, "targets", []))
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "lift_passed", "experiment_completed", "phase", "failure", "duration_s", "runtime_error", "cleanup_errors"
    )}), indent=2))
    return 0 if (report.get("lift_passed") is True and report.get("experiment_completed") is True
                 and not report.get("failure") and runtime_error is None and not cleanup_errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
