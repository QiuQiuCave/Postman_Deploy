"""Record an uncomplicated free-base, dual-wrist world-goal Reach sequence.

The two fixed cameras show the complete robot and an upper-body close-up.
All target markers and wrist axes are render-only. There is no crate, table,
IK, wrist fixture, state replay, or auxiliary control in this recorder.
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
PANEL_WIDTH, PANEL_HEIGHT = 640, 560
HEADER_HEIGHT, FRAME_HEIGHT = 160, 800


def _make_cameras():
    import mujoco

    cameras = []
    for distance, elevation, azimuth, target in (
        (3.05, 0., 90., (.13, 0., .89)),
        (1.75, -16., 140., (.15, 0., 1.14)),
    ):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _camera_record(cameras):
    return [dict(distance_m=c.distance, elevation_deg=c.elevation, azimuth_deg=c.azimuth,
                 lookat_world_m=c.lookat.copy(), fixed_world_camera=True, view=view)
            for c, view in zip(cameras, ("exact_side_full_body", "front_three_quarter_upper_body"))]


def _line(scene, start, end, color, width):
    import mujoco
    import numpy as np

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                      np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width,
                        np.ascontiguousarray(start), np.ascontiguousarray(end))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def _add_wrist_axes(scene, exp):
    """Thick RGB = measured wrist axes; thin L lime / R cyan = desired axes."""
    import numpy as np

    goals = getattr(exp, "goal_wrist_transforms", {})
    for side, color in (("left", (0.35, 1., .35, 1.)), ("right", (.35, .80, 1., 1.))):
        body = exp.model.body(f"{side}_hand_roll_link").id
        origin = exp.scratch.xpos[body]
        rotation = exp.scratch.xmat[body].reshape(3, 3)
        for axis, rgb in enumerate(((1., .15, .15, 1.), (.1, 1., .1, 1.), (.25, .45, 1., 1.))):
            _line(scene, origin, origin + .035*rotation[:, axis], rgb, 3.)
        goal = np.asarray(goals.get(side, np.zeros((4, 4))), dtype=float)
        if goal.shape != (4, 4) or not np.all(np.isfinite(goal)) or side not in goals:
            continue
        for axis in range(3):
            _line(scene, goal[:3, 3], goal[:3, 3] + .06*goal[:3, axis], color, 1.5)


def _render_frame(exp, renderer, cameras, options, font, checkpoint_name):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_target_markers(renderer.scene, exp)
        _add_wrist_axes(renderer.scene, exp)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (2*PANEL_WIDTH, FRAME_HEIGHT), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f"SIMPLE DUAL-ARM REACH | {checkpoint_name} | t={exp.data.time:05.2f}s | {exp.phase}",
              font=font, fill=(255, 207, 95))
    draw.text((14, 35), "TRUE FULL ROBOT / FREE BASE / WORLD WRIST TARGETS / NO TABLE OR CRATE",
              font=font, fill="white")
    for index, side in enumerate(("left", "right")):
        e = exp.errors(side)
        draw.text((14, 62 + 25*index), f"{side.upper()} WRIST: {e.get('wrist_position_m', e['position_m'])*1000:.1f} mm / "
                  f"{e['orientation_deg']:.1f} deg | actual speed "
                  f"{e.get('wrist_linear_speed_mps', e['linear_speed_mps']):.3f} m/s",
                  font=font, fill=(160, 210, 240))
    tilt = float(np.rad2deg(np.arccos(np.clip(exp.scratch.xmat[exp.base].reshape(3, 3)[2, 2], -1, 1))))
    peak = float(exp.peaks.get("base_tilt_deg", tilt))
    draw.text((650, 62), f"Base tilt actual {tilt:.2f} deg / maximum {peak:.2f} deg", font=font, fill="white")
    draw.text((650, 87), "RGB short axes = measured wrist orientation", font=font, fill="white")
    draw.text((14, 112), "Green L / blue R sphere + thin axes = target; no physical fixtures or forces",
              font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE / FULL BODY", font=font, fill=(190, 200, 210))
    draw.text((650, 139), "FRONT 3/4 / BOTH ARMS", font=font, fill=(190, 200, 210))
    draw.text((14, 729), "Policy 50 Hz | independent fingers 100 Hz | physics 1 kHz | real joint limits and collisions",
              font=font, fill="white")
    draw.text((14, 752), "World goals hold for 5-6 s; tracking outcomes are logged separately from hard safety failures.",
              font=font, fill=(190, 200, 210))
    if exp.failure:
        status, color = "FAILED: " + str(exp.failure), (255, 145, 120)
    elif exp.done:
        status, color = "SEQUENCE COMPLETE | see per-case metrics; completion alone does not mean all targets passed", (155, 225, 175)
    else:
        status, color = "IN PROGRESS | zero locomotion command; whole body may move to support both wrist targets", (255, 207, 95)
    draw.text((14, 776), status[:140], font=font, fill=color)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach-config", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty artifact directory")
    parser.add_argument("--cases", help="Optional comma-separated sequence of named cases")
    parser.add_argument("--no-video", action="store_true", help="Only simulate and write JSON")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    case_names = None if args.cases is None else [part.strip() for part in args.cases.split(",")]
    if case_names is not None and (not case_names or any(not name for name in case_names)):
        parser.error("--cases must contain nonempty comma-separated case names")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    exp = renderer = writer = None
    frame_count = freeze_frames = 0
    frame_times, screenshots, camera_configuration, cleanup_errors = [], [], [], []
    runtime_error = runtime_traceback = None
    last_phase, final_frame, mujoco_version = None, None, None
    next_frame = 0.
    report = dict(passed=False, phase="INITIALIZATION")
    try:
        import mujoco
        from common.r2v2_simple_dual_reach import SimpleDualReachExperiment
        from common.r2v2_reach_sim import load_reach_config

        mujoco_version = mujoco.__version__
        cfg = load_reach_config(args.reach_config)
        checkpoint_name = Path(cfg["checkpoint_path"]).name
        exp = SimpleDualReachExperiment(reach_config=args.reach_config,
                                        parity_report=args.parity_report, case_names=case_names)
        print(f"Policy checkpoint: {checkpoint_name}; output: {out}", flush=True)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / "simple_dual_reach.mp4"), fps=FPS, codec="libx264", quality=8)
            cameras, options = _make_cameras(), mujoco.MjvOption()
            options.geomgroup[3] = 0
            camera_configuration = _camera_record(cameras)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
        while True:
            changed = exp.phase != last_phase
            due = writer is not None and (exp.data.time + 1e-9 >= next_frame or exp.done)
            if changed:
                print(f"{exp.data.time:.3f}s {exp.phase}: {exp.failure or ''}", flush=True)
            if renderer is not None and (changed or due or exp.done):
                array = _render_frame(exp, renderer, cameras, options, font, checkpoint_name)
                if changed:
                    name = _phase_filename(len(screenshots), exp.phase)
                    imageio.imwrite(out / name, array)
                    screenshots.append(dict(time_s=float(exp.data.time), phase=exp.phase, file=name))
                    if last_phase is None:
                        imageio.imwrite(out / "initial.png", array)
                if due:
                    writer.append_data(array)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
                    next_frame += 1./FPS
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
        print(f"Recording terminated: {runtime_error}", file=sys.stderr, flush=True)
        if exp is not None and not exp.done:
            try:
                exp.fail(runtime_error)
            except Exception as fail_error:
                cleanup_errors.append(f"Mark failure: {type(fail_error).__name__}: {fail_error}")
        if renderer is not None and exp is not None:
            try:
                final_frame = _render_frame(exp, renderer, cameras, options, font, checkpoint_name)
                imageio.imwrite(out / "final.png", final_frame)
                if writer is not None:
                    writer.append_data(final_frame)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
            except Exception as render_error:
                cleanup_errors.append(f"Failure frame: {type(render_error).__name__}: {render_error}")
    finally:
        if writer is not None and final_frame is not None:
            try:
                for _ in range(2*FPS):
                    writer.append_data(final_frame)
                    frame_count += 1
                    freeze_frames += 1
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
                report = dict(passed=False, phase=exp.phase, failure=exp.failure)
        report.update(renderer_scope="SIMPLE DUAL-ARM WRIST-WORLD REACH / FREE BASE / NO PROPS",
            reach_configuration_path=str(args.reach_config.resolve()), parity_report=str(args.parity_report.resolve()),
            requested_case_names=case_names, mujoco_version=mujoco_version,
            video_file=None if args.no_video else "simple_dual_reach.mp4",
            video_frames=frame_count, video_fps=None if args.no_video else FPS,
            video_frame_times_s=frame_times, terminal_freeze_frames=freeze_frames,
            terminal_freeze_is_not_simulation=True, camera_configuration=camera_configuration,
            phase_screenshots=screenshots, runtime_error=runtime_error,
            runtime_traceback=runtime_traceback, cleanup_errors=cleanup_errors)
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "passed", "phase", "failure", "duration_s", "case_results", "runtime_error", "cleanup_errors"
    )}), indent=2), flush=True)
    return 0 if exp is not None and exp.done and not exp.failure and runtime_error is None and not cleanup_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
