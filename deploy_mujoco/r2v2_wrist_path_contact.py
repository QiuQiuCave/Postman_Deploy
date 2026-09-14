"""Render the real-contact wrist path, optionally with a grasp/higher-goal FSM.

Omit --grasp-plan to preserve the timed exploratory baseline. Supply a checked
plan to select the separate FSM: only a physically verified pickup permits
higher wrist goals. Neither mode treats a binary close command as success.

The simulator alone moves the crate. Rendering adds only wrist goal markers and
HUD text: it never changes targets, physics, fingers or phase progression. A
terminal image is held for two video seconds without advancing physics. Exit
code zero means that the experiment and artifacts ran normally, not that the
schedule completed, contact was safe or the grasp passed strict evaluation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_mujoco.r2v2_crate_motion_replay import (
    _camera_record, _make_cameras, FPS, PANEL_WIDTH, PANEL_HEIGHT,
    HEADER_HEIGHT, FRAME_HEIGHT, TERMINAL_HOLD_SECONDS,
)
from deploy_mujoco.r2v2_crate_reach import _add_target_markers
from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json


VIDEO_NAME = "wrist_path_contact.mp4"
FSM_VIDEO_NAME = "wrist_path_grasp_fsm.mp4"


def _metric(value, fmt=".2f", multiplier=1.):
    """Missing or nonfinite measurements are not fabricated zeroes."""
    if value is None:
        return "N/A"
    try:
        number = float(value) * multiplier
    except (TypeError, ValueError):
        return "N/A"
    return format(number, fmt) if math.isfinite(number) else "NONFINITE"


def _status(exp, *, terminal_freeze=False, grasp_mode=False):
    grasp_failure = getattr(getattr(exp, "grasp_fsm", None), "failure_reason", None)
    if terminal_freeze:
        if exp.failure:
            return "TERMINAL FREEZE / STOP: " + str(exp.failure), (255, 140, 120)
        if grasp_mode and grasp_failure:
            return "TERMINAL FREEZE / FSM RESULT: " + str(grasp_failure), (255, 140, 120)
        return "TERMINAL FREEZE | physics stopped; this repeated image is not extra stability evidence", (255, 207, 95)
    if exp.failure:
        return "PHYSICS STOP: " + str(exp.failure) + " | grasp success is evaluated separately", (255, 140, 120)
    if grasp_mode and grasp_failure:
        return "FSM RESULT: " + str(grasp_failure) + " | fingers stay closed", (255, 140, 120)
    if exp.done:
        if grasp_mode:
            return "FSM ENDED | clock completion != pickup or higher-hold success; see report.json", (255, 207, 95)
        return "SCHEDULE ENDED | not a grasp-success claim; consult strict evaluation in report.json", (255, 207, 95)
    if grasp_mode:
        return "GRASP FSM | higher wrist goals require verified physical pickup; command 1 alone is not sufficient", (255, 207, 95)
    return "IN PROGRESS | CLOSE is a finger command, not proof of grasp; strict success assessed separately", (255, 207, 95)


def _render_frame(exp, renderer, cameras, options, font, *, terminal_freeze=False, grasp_mode=False):
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
    title = "REAL-CONTACT GRASP / HIGHER-GOAL FSM" if grasp_mode else "TIMED EXPLORATORY REAL-CONTACT TRIAL"
    draw.text((14, 8), title + " | "
              f"t={exp.data.time:05.2f}s | {exp.phase}", font=font, fill=(255, 207, 95))
    subtitle = ("Prop contacts do not abort the attempt. Higher goals require actual pickup; fall / numerical stops remain."
                if grasp_mode else "Box contacts / pose error / grasp failure do NOT stop schedule; hard physical stops remain active.")
    draw.text((14, 34), subtitle, font=font, fill="white")
    for i, side in enumerate(("left", "right")):
        error = m.get("wrist_errors", {}).get(side, {})
        draw.text((14, 60 + 25*i), f"{side.upper()} WRIST: "
                  f"{_metric(error.get('position_m'), '.1f', 1000.)} mm / "
                  f"{_metric(error.get('orientation_deg'), '.1f')} deg | "
                  f"{_metric(error.get('linear_speed_mps'), '.3f')} m/s",
                  font=font, fill=(160, 210, 240))
    draw.text((650, 60), f"Actual box bottom above table: {_metric(m.get('clearance_m'), '+.1f', 1000.)} mm | "
              f"tilt {_metric(m.get('crate_tilt_deg'), '.1f')} deg", font=font, fill="white")
    forces = [m.get("hands", {}).get(side, {}).get("vertical_force_N") for side in ("left", "right")]
    draw.text((650, 85), f"Actual net hand force Z: L {_metric(forces[0], '+.2f')} / "
              f"R {_metric(forces[1], '+.2f')} N", font=font, fill="white")
    commands = [int(exp.hands.controllers[side].command) for side in ("left", "right")]
    draw.text((14, 111), f"Finger command: L={commands[0]} / R={commands[1]} (1 != grasp success) | "
              f"base tilt {_metric(m.get('base_tilt_deg'), '.1f')} deg | real free crate; no object animation",
              font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE / FREE-BASE ROBOT", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "FIXED CLOSE-UP / BOTH HANDS + PHYSICAL CRATE", font=font, fill=(190, 200, 210))
    draw.text((14, 729), "Body 50 Hz | independent fingers 100 Hz | contacts / torque 1 kHz | "
              "green / blue spheres are desired wrists only", font=font, fill="white")
    footer = ("Higher targets: sampled joint/self-contact checks only, NOT collision-safe carrying; failed grasps do not trigger higher goals."
              if grasp_mode else "Timed path exploration is NOT an acceptance test pass; strict tracking / real lift / contact / slip metrics remain separate.")
    draw.text((14, 752), footer,
              font=font, fill=(190, 200, 210))
    status, color = _status(exp, terminal_freeze=terminal_freeze, grasp_mode=grasp_mode)
    draw.text((14, 776), status[:142], font=font, fill=color)
    return np.asarray(canvas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-manifest", type=Path, required=True)
    parser.add_argument("--reach-config", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--grasp-plan", type=Path,
                        help="Optional certified contact-gated grasp/higher-goal plan; omitted preserves the timed baseline")
    args = parser.parse_args(argv)
    grasp_mode = args.grasp_plan is not None
    video_name = FSM_VIDEO_NAME if grasp_mode else VIDEO_NAME
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print(("CONTACT-GATED GRASP FSM" if grasp_mode else "TIMED EXPLORATORY REAL CONTACT")
          + ": schedule completion != grasp success", flush=True)
    print(f"Output: {out}", flush=True)
    exp = renderer = writer = None
    runtime_error = runtime_traceback = None
    cleanup_errors, frame_times, screenshots, camera_configuration = [], [], [], []
    frame_count = freeze_frame_count = 0
    next_video_time_s, last_phase, final_frame = 0., None, None
    reached_terminal_state = False
    report = {"phase": "INITIALIZATION", "strict_success": None}
    mujoco_version = None
    try:
        import mujoco
        mujoco_version = mujoco.__version__
        options_exp = dict(
            path_manifest=args.path_manifest, reach_config=args.reach_config,
            parity_report=args.parity_report,
        )
        if grasp_mode:
            from common.r2v2_wrist_path_grasp_fsm import WristPathGraspFSMExperiment
            exp = WristPathGraspFSMExperiment(**options_exp, grasp_plan=args.grasp_plan)
        else:
            from common.r2v2_wrist_path_contact import WristPathContactExperiment
            exp = WristPathContactExperiment(**options_exp)
        next_video_time_s = float(exp.data.time)
        print(f"Physical model nq={exp.model.nq}, nv={exp.model.nv}, nu={exp.model.nu}, "
              f"nmocap={exp.model.nmocap}; table={exp.table_height:.6f} m", flush=True)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / video_name), fps=FPS, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-threads", "2"])
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            cameras = _make_cameras(exp)
            camera_configuration = _camera_record(cameras)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
        while True:
            phase_changed = exp.phase != last_phase
            video_due = writer is not None and (float(exp.data.time)+1e-9 >= next_video_time_s or exp.done)
            if phase_changed:
                print(f"t={exp.data.time:.3f}: {exp.phase} {exp.failure or ''}", flush=True)
            if renderer is not None and (phase_changed or video_due or exp.done):
                frame = _render_frame(exp, renderer, cameras, options, font, grasp_mode=grasp_mode)
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
                    next_video_time_s += 1./FPS
                if exp.done:
                    final_frame = _render_frame(exp, renderer, cameras, options, font, terminal_freeze=True, grasp_mode=grasp_mode)
                    imageio.imwrite(out / "final.png", final_frame)
            last_phase = exp.phase
            if exp.done:
                reached_terminal_state = True
                break
            exp.step()
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
                final_frame = _render_frame(exp, renderer, cameras, options, font, terminal_freeze=True, grasp_mode=grasp_mode)
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
                for _ in range(round(FPS*TERMINAL_HOLD_SECONDS)):
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
        runtime_completed = bool(reached_terminal_state and runtime_error is None and not cleanup_errors)
        report.update({
            "renderer_scope": ("CONTACT-GATED GRASP / HIGHER-GOAL FSM / REAL OBJECT MOTION" if grasp_mode
                               else "TIMED EXPLORATORY REAL-CONTACT TRIAL / REAL OBJECT MOTION"),
            "grasp_plan_path": str(args.grasp_plan.resolve()) if grasp_mode else None,
            "renderer_runtime_completed": runtime_completed,
            "renderer_exit_code_semantics": "0 = normal terminal state and artifacts, NOT grasp/schedule success",
            "rendered_object_uses_actual_simulation_pose": True,
            "path_manifest_path": str(args.path_manifest.resolve()),
            "reach_configuration_path": str(args.reach_config.resolve()),
            "parity_report_path": str(args.parity_report.resolve()),
            "mujoco_version": mujoco_version, "video_file": None if args.no_video else video_name,
            "video_frames": frame_count, "video_fps": None if args.no_video else FPS,
            "video_frame_times_s": frame_times, "terminal_freeze_frames": freeze_frame_count,
            "terminal_freeze_is_not_simulation": True, "camera_configuration": camera_configuration,
            "phase_screenshots": screenshots, "runtime_error": runtime_error,
            "runtime_traceback": runtime_traceback, "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
        _write_json(out / "targets.json", [] if exp is None else exp.targets)
        # Write the completion report last so an artifact-write failure cannot
        # leave a report claiming that the complete output set was saved.
        _write_json(out / "report.json", report)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "renderer_runtime_completed", "phase", "failure", "strict_success", "lift_passed",
        "duration_s", "runtime_error", "cleanup_errors",
    )}), indent=2))
    return 0 if report["renderer_runtime_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
