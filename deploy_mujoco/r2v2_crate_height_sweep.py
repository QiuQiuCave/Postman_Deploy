"""Record one fixed-checkpoint crate-height path-screening experiment.

Rendering never alters controls or acceptance criteria. Both cameras are fixed in
world coordinates, and the final two seconds repeat the terminal image rather
than advancing failed physics. The companion comparison tool labels these holds.
"""

import argparse
import json
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


def _box_edges(center, rotation, half_size):
    """Twelve world-space edges for an oriented box; no physics objects."""
    import itertools
    import numpy as np

    center, rotation = np.asarray(center), np.asarray(rotation).reshape(3, 3)
    half_size = np.asarray(half_size)
    signs = tuple(itertools.product((-1., 1.), repeat=3))
    corners = np.asarray(signs) * half_size @ rotation.T + center
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
            if sum(a != b for a, b in zip(signs[i], signs[j])) == 1]


def _virtual_prop_edges(exp):
    """Outline actual prop components, including the open handle apertures.

    These coordinates come from the fixed occupancy model, never from the
    desired crate lift trajectory. Rounded handle beams use their mesh-local
    bounding boxes; all other table/crate solids are already boxes.
    """
    import mujoco
    import numpy as np

    if not getattr(exp, "virtual_props", False):
        return []
    body_colors = {
        exp.model.body("tabletop").id: (1., .73, .32, 1.),
        exp.model.body("cargo_crate").id: (.25, .85, 1., 1.),
    }
    result = []
    for geom_id in range(exp.model.ngeom):
        color = body_colors.get(int(exp.model.geom_bodyid[geom_id]))
        if color is None:
            continue
        center = exp.scratch.geom_xpos[geom_id].copy()
        rotation = exp.scratch.geom_xmat[geom_id].reshape(3, 3)
        geom_type = exp.model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_size = exp.model.geom_size[geom_id]
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(exp.model.geom_dataid[geom_id])
            first = int(exp.model.mesh_vertadr[mesh_id])
            count = int(exp.model.mesh_vertnum[mesh_id])
            vertices = np.asarray(exp.model.mesh_vert[first:first + count], dtype=float)
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            center += rotation @ ((lo + hi) / 2.)
            half_size = (hi - lo) / 2.
        else:
            raise ValueError(f"Unsupported virtual prop geometry: {geom_id} type={geom_type}")
        result.extend((a, b, color) for a, b in _box_edges(center, rotation, half_size))
    return result


def _add_virtual_prop_outlines(scene, exp):
    """Add pixel-width lines to MjvScene only, with no contacts or forces."""
    import mujoco
    import numpy as np

    for start, end, color in _virtual_prop_edges(exp):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Renderer scene capacity exhausted by prop outlines")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                          np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., start, end)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def _render_frame(exp, renderer, cameras, options, font, height_offset):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_virtual_prop_outlines(renderer.scene, exp)
        _add_target_markers(renderer.scene, exp)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (2 * PANEL_WIDTH, FRAME_HEIGHT), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    m = exp.current_metrics
    virtual = bool(getattr(exp, "virtual_props", False))
    x_offset = float(getattr(exp, "delta_x", 0.))
    draw.text((14, 8), ("VIRTUAL PROPS / NO PROP CONTACT / NO GRASP | " if virtual else
                       "EMPTY-HAND PATH SCREEN / NO GRASP | ")
              + f"height {height_offset*100:+.0f} cm | X {x_offset*100:+.0f} cm | "
              f"t={exp.data.time:05.2f}s | {exp.phase}", font=font, fill=(255, 207, 95))
    draw.text((14, 34), "Fixed Reach policy | outside approach -> turn -> preinsert -> "
              "down20/yaw15 wrist path | fingers OPEN", font=font, fill="white")
    for i, side in enumerate(("left", "right")):
        error = m.get("wrist_errors", {}).get(side, {})
        draw.text((14, 60 + 25 * i), f"{side.upper()} WRIST: "
                  f"{float(error.get('position_m', 0.))*1000:.1f} mm / "
                  f"{float(error.get('orientation_deg', 0.)):.1f} deg | "
                  f"speed {float(error.get('linear_speed_mps', 0.)):.3f} m/s",
                  font=font, fill=(160, 210, 240))
    draw.text((650, 60), f"Body tilt {float(m.get('base_tilt_deg', 0.)):.2f} deg | "
              f"crate tilt {float(m.get('crate_tilt_deg', 0.)):.2f} deg",
              font=font, fill="white")
    commands = {s: int(exp.hands.controllers[s].command) for s in ("left", "right")}
    draw.text((650, 85), f"Fingers L={commands['left']} / R={commands['right']} | "
              f"crate clearance {float(m.get('clearance_m', 0.))*1000:+.1f} mm",
              font=font, fill="white")
    draw.text((14, 111), ("Cyan crate / amber table = STATIC LINE PLACEHOLDERS; " if virtual else
                        "Green / blue spheres = desired wrists (render-only); ")
              + f"table top {exp.table_height:.3f} m; same frozen policy / robot initial state.",
              font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE / COMPLETE FREE-BASE ROBOT", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "FIXED CRATE CLOSE-UP / BOTH HANDS", font=font, fill=(190, 200, 210))
    draw.text((14, 729), "Body policy 50 Hz | independent fingers 100 Hz | contact / torque 1 kHz | "
              "no IK / wrist fixture / auxiliary force", font=font, fill="white")
    draw.text((14, 752), f"Recorded-motion clock: {float(getattr(exp, 'source_time_s', 0.)):.2f}s | "
              + ("Prop overlap allowed; robot self-contact / limits / fall safety retained." if virtual else
                 "Timed path + settle; tracking errors measured; hard safety stops enabled."),
              font=font, fill=(190, 200, 210))
    failure = getattr(exp, "failure", None)
    if failure:
        status, color = "FAILED / TERMINAL FREEZE: " + str(failure), (255, 130, 115)
    elif exp.done:
        status, color = "PATH PLAYBACK ENDED / FREEZE: not a grasp; completion does not imply accurate tracking", (160, 230, 175)
    else:
        status = ("IN PROGRESS: empty-hand path only; table / crate collision response and collision stops DISABLED"
                  if virtual else
                  "IN PROGRESS: empty-hand path screening only; real table / crate collisions remain enabled")
        color = (255, 210, 110)
    draw.text((14, 776), status[:142], font=font, fill=color)
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reach-config", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--prepared-state", type=Path,
                        help="Shared, validated robot/controller initial-state artifact")
    parser.add_argument("--height-offset", type=float, required=True,
                        help="Table and crate height delta in metres, e.g. -0.10")
    parser.add_argument("--x-offset", type=float, default=0.,
                        help="Table/crate forward-axis delta in metres; -0.05 is 5 cm closer")
    parser.add_argument("--virtual-props", action="store_true",
                        help="Static line-only table/crate occupancy, no prop collision response or stops")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    exp = renderer = writer = None
    runtime_error = runtime_traceback = None
    cleanup_errors, frame_times, screenshots, camera_configuration = [], [], [], []
    frame_count = freeze_frame_count = 0
    next_video_time_s, last_phase, final_frame = 0., None, None
    report = {"experiment_completed": False, "phase": "INITIALIZATION"}
    mujoco_version = None
    try:
        import mujoco
        from common.r2v2_crate_height_sweep import HeightSweepExperiment

        mujoco_version = mujoco.__version__
        exp = HeightSweepExperiment(
            reach_config=args.reach_config, parity_report=args.parity_report,
            delta_z=args.height_offset, prepared_state=args.prepared_state,
            delta_x=args.x_offset, virtual_props=args.virtual_props,
        )
        next_video_time_s = float(exp.data.time)
        print(f"Height offset {args.height_offset:+.3f} m; X offset {args.x_offset:+.3f} m; "
              f"virtual props={args.virtual_props}; table {exp.table_height:.6f} m; "
              f"model nq={exp.model.nq} nv={exp.model.nv} nu={exp.model.nu} nmocap={exp.model.nmocap}",
              flush=True)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / "videoheight.mp4"), fps=FPS,
                                        codec="libx264", quality=8, macro_block_size=1,
                                        ffmpeg_params=["-threads", "2"])
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0
            cameras = _make_cameras(exp)
            camera_configuration = _camera_record(cameras)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
        while True:
            phase_changed = exp.phase != last_phase
            video_due = writer is not None and (float(exp.data.time) + 1e-9 >= next_video_time_s or exp.done)
            if phase_changed:
                print(f"t={exp.data.time:.3f}: {exp.phase} {getattr(exp, 'failure', None) or ''}", flush=True)
            if renderer is not None and (phase_changed or video_due or exp.done):
                frame = _render_frame(exp, renderer, cameras, options, font, args.height_offset)
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
                    final_frame = frame
                    imageio.imwrite(out / "final.png", frame)
            last_phase = exp.phase
            if exp.done:
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
                cleanup_errors.append(f"Mark failure: {type(fail_error).__name__}: {fail_error}")
        if renderer is not None and exp is not None:
            try:
                final_frame = _render_frame(exp, renderer, cameras, options, font, args.height_offset)
                imageio.imwrite(out / "final.png", final_frame)
                if writer is not None:
                    writer.append_data(final_frame)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
            except Exception as render_error:
                cleanup_errors.append(f"Final frame: {type(render_error).__name__}: {render_error}")
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
        for name, resource in (("writer", writer), ("renderer", renderer)):
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
                report = {"experiment_completed": False, "phase": exp.phase,
                          "failure": getattr(exp, "failure", None)}
        report.update({
            "renderer_scope": "EMPTY-HAND HEIGHT PATH SCREEN / FULL FREE-BASE ROBOT / NO GRASP",
            "height_offset_m": args.height_offset,
            "delta_x_m": args.x_offset,
            "x_offset_m": args.x_offset,
            "virtual_props": args.virtual_props,
            "prop_rendering": "static_render_only_component_edges" if args.virtual_props else "physical_solids",
            "virtual_outline_mesh_beams_use_bounding_boxes": args.virtual_props,
            "reach_configuration_path": str(args.reach_config.resolve()),
            "parity_report": str(args.parity_report.resolve()),
            "prepared_state_path": None if args.prepared_state is None else str(args.prepared_state.resolve()),
            "mujoco_version": mujoco_version,
            "video_file": None if args.no_video else "videoheight.mp4",
            "video_frames": frame_count, "video_fps": None if args.no_video else FPS,
            "video_frame_times_s": frame_times, "terminal_freeze_frames": freeze_frame_count,
            "terminal_freeze_is_not_simulation": True,
            "camera_configuration": camera_configuration, "phase_screenshots": screenshots,
            "runtime_error": runtime_error, "runtime_traceback": runtime_traceback,
            "cleanup_errors": cleanup_errors,
        })
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
        _write_json(out / "targets.json", [] if exp is None else getattr(exp, "targets", []))
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "height_offset_m", "path_passed", "lift_passed", "experiment_completed", "phase", "failure",
        "duration_s", "runtime_error", "cleanup_errors"
    )}), indent=2))
    return 0 if (exp is not None and exp.done and not getattr(exp, "failure", None)
                 and runtime_error is None and not cleanup_errors) else 1


if __name__ == "__main__":
    raise SystemExit(main())
