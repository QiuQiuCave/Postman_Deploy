"""Record a frozen-policy, free-base top-grasp reach/contact experiment.

AIR screens empty-hand targets against render-only prop outlines and is never
grasp evidence. CONTACT records the real robot and free physical object. The
experiment, not this recorder, owns controls, state transitions and acceptance.
Exit zero means capture completed normally, not that any acceptance gate passed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json
from deploy_mujoco.r2v2_top_grasp import _metric, _boolean, _field


FPS = 30
PANEL_WIDTH, PANEL_HEIGHT = 640, 560
HEADER_HEIGHT, FRAME_HEIGHT = 160, 800
TERMINAL_HOLD_SECONDS = 2.
VIDEO_NAME = "top_grasp_fullbody.mp4"


def _status(exp, mode, *, terminal_freeze=False):
    if terminal_freeze:
        return "TERMINAL FREEZE | repeated image is NOT extra physics or hold evidence", (255, 207, 95)
    if exp.failure:
        return "STOP: " + str(exp.failure), (255, 145, 120)
    if exp.done:
        return "EXPERIMENT ENDED | completion alone does not imply acceptance; see report.json", (255, 207, 95)
    if mode == "air":
        return "AIR: NO GRASP EVIDENCE | table/can are line placeholders; tracking remains measured", (190, 215, 230)
    return "CONTACT: FREE ROBOT AND OBJECT | finger command 1 does not prove grasp", (190, 215, 230)


def _initial_object_position(exp):
    import numpy as np

    position = _field(getattr(exp, "layout", None), "cylinder_initial_position")
    if position is None:
        position = exp.current_metrics.get("object_position_m")
    if position is None:
        try:
            geom = exp.model.geom("cylinder_geom").id
            position = exp.scratch.geom_xpos[geom]
        except (KeyError, ValueError, AttributeError) as exc:
            raise ValueError("Cannot frame full-body top grasp without an actual initial object position") from exc
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Initial object position must be finite XYZ")
    return position.copy()


def _make_cameras(exp):
    """Fixed world cameras: a true side full-body view and hand-task close-up."""
    import mujoco
    import numpy as np

    initial = _initial_object_position(exp)
    destination = _field(getattr(exp, "layout", None), "cylinder_place_position")
    center = initial.copy()
    if destination is not None:
        destination = np.asarray(destination, dtype=float)
        if destination.shape != (3,) or not np.isfinite(destination).all():
            raise ValueError("Object destination must be finite XYZ")
        center = (initial + destination) / 2.
    closeup = center + np.array([0., 0., .12])
    full = np.array([center[0] / 2., 0., .87])
    sign = 1 if getattr(exp, "side", "left") == "left" else -1
    cameras = []
    for distance, elevation, azimuth, target in (
        (3.10, 0., sign * 90., full),
        (1.30, -20., sign * 140., closeup),
    ):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _camera_record(cameras):
    return [dict(distance_m=c.distance, elevation_deg=c.elevation,
                 azimuth_deg=c.azimuth, lookat_world_m=c.lookat.copy(),
                 fixed_world_camera=True, view=view)
            for c, view in zip(cameras, ("exact_side_full_body", "fixed_oblique_hand_task"))]


def _cylinder_edges(center, rotation, radius, half_height, segments=24):
    import numpy as np

    center, rotation = np.asarray(center), np.asarray(rotation).reshape(3, 3)
    rings = []
    for height in (-half_height, half_height):
        local = np.array([[radius * math.cos(2. * math.pi * i / segments),
                           radius * math.sin(2. * math.pi * i / segments), height]
                          for i in range(segments)])
        rings.append(local @ rotation.T + center)
    edges = [(ring[i], ring[(i + 1) % segments]) for ring in rings for i in range(segments)]
    edges.extend((rings[0][i], rings[1][i]) for i in range(0, segments, max(1, segments // 8)))
    return edges


def _box_edges(center, rotation, half_size):
    import itertools
    import numpy as np

    signs = tuple(itertools.product((-1., 1.), repeat=3))
    vertices = np.asarray(signs) * half_size @ rotation.T + center
    return [(vertices[i], vertices[j]) for i in range(8) for j in range(i + 1, 8)
            if sum(a != b for a, b in zip(signs[i], signs[j])) == 1]


def _virtual_prop_edges(exp, mode):
    """Read actual scene transforms; never animate an AIR can with hand targets."""
    if mode != "air":
        return []
    import mujoco
    import numpy as np

    specs = getattr(exp, "render_wireframes", None)
    if specs is None:
        layout = getattr(exp, "layout", None)
        center = _field(layout, "table_center")
        half_size = _field(layout, "table_half_size")
        initial = _field(layout, "cylinder_initial_position")
        profile = getattr(exp, "profile", {})
        radius, height = _field(profile, "radius_m"), _field(profile, "height_m")
        if all(value is not None for value in (center, half_size, initial, radius, height)):
            # AIR occupancy is stationary, not attached to measured/desired
            # wrist poses. The physical experiment's layout defines it.
            specs = [dict(kind="box", position=center, size=half_size, color=(1., .73, .32, 1.)),
                     dict(kind="cylinder", position=initial, size=[radius, height / 2.],
                          color=(.25, .85, 1., 1.))]
    if specs is None:
        specs = []
        for names, color in ((("tabletop_geom", "table_geom"), (1., .73, .32, 1.)),
                             (("cylinder_geom",), (.25, .85, 1., 1.))):
            geom_id = None
            for name in names:
                try:
                    geom_id = exp.model.geom(name).id
                    break
                except (KeyError, ValueError, AttributeError):
                    continue
            if geom_id is None:
                raise ValueError(f"AIR renderer needs one of {names}, layout, or explicit render_wireframes")
            geom_type = exp.model.geom_type[geom_id]
            kind = ("box" if geom_type == mujoco.mjtGeom.mjGEOM_BOX else
                    "cylinder" if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER else None)
            if kind is None:
                raise ValueError(f"Unsupported AIR prop geometry {name}: {geom_type}")
            specs.append(dict(kind=kind, position=exp.scratch.geom_xpos[geom_id].copy(),
                              rotation=exp.scratch.geom_xmat[geom_id].copy(),
                              size=exp.model.geom_size[geom_id].copy(), color=color))
    result = []
    for spec in specs:
        rotation = spec.get("rotation")
        if rotation is None:
            quaternion = spec.get("quaternion")
            rotation = np.eye(3)
            if quaternion is not None:
                quaternion = np.asarray(quaternion, dtype=float)
                if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-12:
                    raise ValueError("Wireframe quaternion must be finite nonzero WXYZ")
                mujoco.mju_quat2Mat(rotation.ravel(), quaternion / np.linalg.norm(quaternion))
        center = np.asarray(spec["position"], dtype=float)
        size = np.asarray(spec["size"], dtype=float)
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        if center.shape != (3,) or not all(np.isfinite(x).all() for x in (center, size, rotation)) or (size < 0).any():
            raise ValueError("Wireframe geometry must have finite position, rotation and nonnegative size")
        if spec["kind"] == "box":
            if size.shape != (3,):
                raise ValueError("Box wireframe size is three half-extents")
            edges = _box_edges(center, rotation, size)
        elif spec["kind"] == "cylinder":
            if size.shape not in ((2,), (3,)):
                raise ValueError("Cylinder wireframe size is radius and half-height")
            edges = _cylinder_edges(center, rotation, size[0], size[1])
        else:
            raise ValueError(f"Unsupported wireframe kind: {spec['kind']}")
        result.extend((a, b, spec.get("color", (.25, .85, 1., 1.))) for a, b in edges)
    return result


def _line(scene, start, end, color, width=2.):
    import mujoco
    import numpy as np

    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Renderer scene capacity exhausted by outlines")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                      np.zeros(3), np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width,
                        np.ascontiguousarray(start), np.ascontiguousarray(end))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def _add_render_markers(scene, exp, mode):
    import numpy as np

    for start, end, color in _virtual_prop_edges(exp, mode):
        _line(scene, start, end, color)
    for side, color in (("left", (.35, 1., .35, 1.)), ("right", (.35, .8, 1., 1.))):
        actual = exp.current_metrics.get("wrist_errors", {}).get(side, {}).get("T_world_wrist")
        if actual is not None:
            actual = np.asarray(actual, dtype=float)
            if actual.shape != (4, 4) or not np.isfinite(actual).all():
                raise ValueError("Measured wrist transform must be finite 4 x 4")
            for axis, rgb in enumerate(((1., .15, .15, 1.), (.1, 1., .1, 1.), (.25, .45, 1., 1.))):
                _line(scene, actual[:3, 3], actual[:3, 3] + .028 * actual[:3, axis], rgb, 3.)
        goal = getattr(exp, "goal_wrist_transforms", {}).get(side)
        if goal is None:
            continue
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (4, 4) or not np.isfinite(goal).all():
            raise ValueError("Wrist target transform must be finite 4 x 4")
        for axis in range(3):
            _line(scene, goal[:3, 3], goal[:3, 3] + .045 * goal[:3, axis], color, 1.5)


def _render_frame(exp, renderer, cameras, options, font, mode, *, terminal_freeze=False):
    import numpy as np
    from PIL import Image, ImageDraw

    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _add_render_markers(renderer.scene, exp, mode)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (2 * PANEL_WIDTH, FRAME_HEIGHT), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, HEADER_HEIGHT))
    draw, metrics = ImageDraw.Draw(canvas), exp.current_metrics
    scope = "AIR: NO GRASP EVIDENCE" if mode == "air" else "CONTACT: FREE ROBOT AND OBJECT"
    style = getattr(exp, 'config', {}).get('grasp_style', 'upper')
    title = 'FRONT GRASP' if style == 'front' else 'TOP GRASP'
    phase = ({'TURN_WRIST': 'HOLD_UPRIGHT (NO FLIP)', 'HOVER': 'ALIGN_OUTSIDE'}.get(exp.phase, exp.phase)
             if style == 'front' else exp.phase)
    draw.text((14, 8), f"FULL-BODY {title} | {scope} | t={exp.data.time:05.2f}s | {phase}",
              font=font, fill=(255, 207, 95))
    draw.text((14, 34), "Frozen Reach policy / WORLD WRIST targets / independent 0-1 fingers / no wrist fixture",
              font=font, fill="white")
    for index, side in enumerate(("left", "right")):
        error = metrics.get("wrist_errors", {}).get(side, {})
        draw.text((14, 60 + 25 * index), f"{side.upper()} WRIST: "
                  f"{_metric(error.get('position_m'), '.1f', 1000.)} mm / "
                  f"{_metric(error.get('orientation_deg'), '.1f')} deg / "
                  f"{_metric(error.get('linear_speed_mps'), '.3f')} m/s", font=font, fill=(160, 210, 240))
    drift = metrics.get("foot_drift_m")
    if isinstance(drift, dict):
        drift_label = " / ".join(f"{side[0].upper()} {_metric(drift.get(side), '.1f', 1000.)}" for side in ("left", "right"))
    else:
        drift_label = _metric(drift, ".1f", 1000.)
    draw.text((650, 60), f"Base tilt {_metric(metrics.get('base_tilt_deg'), '.1f')} deg | feet {drift_label} mm",
              font=font, fill="white")
    draw.text((650, 85), f"Pickup verified {_boolean(metrics.get('grasp_verified'))} | "
              f"slip {_metric(metrics.get('grasp_slip_m'), '.1f', 1000.)} mm", font=font, fill="white")
    draw.text((14, 111), ("Amber table / cyan can = line placeholders, NOT contact evidence" if mode == "air" else
                        f"Object clearance {_metric(metrics.get('clearance_m'), '+.1f', 1000.)} mm | "
                        f"table support {_metric(metrics.get('table_vertical_force_N'), '.2f')} N | "
                        f"opposing contact {_boolean(metrics.get('opposed_contact'))}"), font=font, fill="white")
    draw.text((14, 139), "EXACT SIDE / COMPLETE FREE-BASE ROBOT", font=font, fill=(190, 200, 210))
    draw.text((654, 139), "FIXED OBLIQUE / LEFT-HAND TASK REGION", font=font, fill=(190, 200, 210))
    draw.text((14, 729), "Body policy 50 Hz | independent fingers 100 Hz | physics / torque 1 kHz | actual state recorded",
              font=font, fill="white")
    draw.text((14, 753), "Short thick RGB = ACTUAL wrists; thin green / blue = desired L / R wrists; all axes are render-only",
              font=font, fill=(190, 200, 210))
    status, color = _status(exp, mode, terminal_freeze=terminal_freeze)
    draw.text((14, 777), status[:140], font=font, fill=color)
    return np.asarray(canvas)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("air", "contact"), default="air")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"FULL-BODY CAN GRASP / {args.mode.upper()}: capture completion != acceptance", flush=True)
    print(f"Output: {out}", flush=True)
    exp = renderer = writer = None
    runtime_error = runtime_traceback = None
    cleanup_errors, frame_times, screenshots, camera_configuration = [], [], [], []
    frame_count = freeze_frame_count = 0
    next_video_time_s, last_phase, final_frame = 0., None, None
    reached_terminal_state = False
    report = {"phase": "INITIALIZATION", "success": None}
    mujoco_version = config_digest = None
    config = None
    try:
        config_bytes = args.config.read_bytes()
        config_digest = hashlib.sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes)
        if not isinstance(config, dict):
            raise ValueError("--config must contain a JSON object")
        import mujoco
        from common.r2v2_top_grasp_fullbody import TopGraspFullbodyExperiment

        mujoco_version = mujoco.__version__
        exp = TopGraspFullbodyExperiment(config=config, mode=args.mode, keep_trace=True)
        next_video_time_s = float(exp.data.time)
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont

            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out / VIDEO_NAME), fps=FPS, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-threads", "2"])
            options = mujoco.MjvOption()
            options.geomgroup[3] = 0  # Collision proxy meshes; physics remains untouched.
            if args.mode == "air":
                options.geomgroup[4] = 0  # Virtual props are replaced by render-only outlines.
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
                frame = _render_frame(exp, renderer, cameras, options, font, args.mode)
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
                    final_frame = _render_frame(exp, renderer, cameras, options, font, args.mode, terminal_freeze=True)
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
        # Do not call exp.fail(): recording errors must not mutate experiment
        # state or masquerade as a physical controller failure.
        if renderer is not None and exp is not None:
            try:
                final_frame = _render_frame(exp, renderer, cameras, options, font, args.mode, terminal_freeze=True)
                imageio.imwrite(out / "final.png", final_frame)
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
                report = {"phase": exp.phase, "failure": exp.failure, "success": None}
        report.update({
            "renderer_scope": ("AIR: NO GRASP EVIDENCE" if args.mode == "air" else
                               "CONTACT: FREE ROBOT AND OBJECT"),
            "renderer_runtime_completed": bool(reached_terminal_state and runtime_error is None and not cleanup_errors),
            "renderer_exit_code_semantics": "0 = normal terminal state and artifacts, NOT acceptance or grasp success",
            "requested_mode": args.mode, "requested_config": config,
            "config_json_path": str(args.config.resolve()), "requested_config_file_sha256": config_digest,
            "trace_semantics": "Experiment raw samples, preserved without resampling; no video-frame interpolation",
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
        _write_json(out / "report.json", report)
    print(json.dumps(_json_safe({key: report.get(key) for key in (
        "renderer_runtime_completed", "phase", "failure", "success", "air_passed",
        "grasp_verified", "release_commanded", "duration_s", "runtime_error", "cleanup_errors",
    )}), indent=2))
    return 0 if report["renderer_runtime_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
