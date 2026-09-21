"""Build, validate and render two tables, free cargo cans and an optional shelf."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_views(model, data, cameras, output, layout, has_shelf=False):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # Hide robot collision duplicates, retain physics.
    options.sitegroup[:] = 0
    qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), data.time
    results = {}
    for name, spec in cameras.items():
        start = time.perf_counter()
        camera = mujoco.MjvCamera()
        camera.lookat[:] = spec["lookat"]
        for key in ("distance", "azimuth", "elevation"):
            setattr(camera, key, spec[key])
        with mujoco.Renderer(model, width=spec["width"], height=spec["height"]) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=options)
            picture = Image.fromarray(renderer.render())
        # Labels identify table roles without adding any collision geometry.
        draw = ImageDraw.Draw(picture)
        scale = spec["width"] / 1920
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        font = ImageFont.truetype(str(font_path), round(22 * scale)) if font_path.exists() else ImageFont.load_default()
        if name == "overview":
            title = "R2V2 | PICKUP / DROPOFF" + (" / SHELF" if has_shelf else "")
            detail = (f"PICKUP: {layout['rows']} x {layout['columns']} = {layout['count']} upright cans"
                      "   |   DROPOFF: empty   |   Robot: static reference")
            if has_shelf:
                detail += "   |   SHELF: empty"
        else:
            title = "PICKUP TABLE | INSIDE THE OPEN CRATE"
            radius, half_height = model.geom(layout["cans"][0]["geom"]).size[:2]
            mass = model.body(layout["cans"][0]["body"]).mass[0]
            detail = (f"{layout['rows']} x {layout['columns']} free cans | gap {layout['surface_gap_m']*1000:g} mm | each: "
                      f"{radius*2000:g} mm diameter, {half_height*2000:g} mm height, {mass*1000:g} g")
        draw.rectangle((0, 0, spec["width"], round(85 * scale)), fill=(24, 32, 43))
        draw.text((round(24*scale), round(12*scale)), title, font=font, fill="white")
        draw.text((round(24*scale), round(47*scale)), detail, font=font, fill=(188, 213, 230))
        path = output / (name + ".png")
        picture.save(path)
        results[name] = {"file": path.name, "width": spec["width"], "height": spec["height"],
                         "render_and_save_s": time.perf_counter() - start,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if not np.array_equal(qpos, data.qpos) or not np.array_equal(qvel, data.qvel) or timestamp != data.time:
        raise RuntimeError("Rendering modified the simulation state")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "deploy_mujoco/config/r2v2_static_task_grid.yaml")
    parser.add_argument("--output", type=Path, help="New/empty output directory; default is sibling tmp/r2v2_static_task/<timestamp>")
    parser.add_argument("--no-render", action="store_true", help="Run physics/reload checks and export without OpenGL")
    parser.add_argument("--view", action="store_true", help="Open an interactive viewer after validation")
    args = parser.parse_args()
    if args.view:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    elif not args.no_render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    import yaml
    from common.r2v2_static_task_grid_scene import load_scene_config, build_scene_xml, settle_and_validate, grid_layout
    from common.r2v2_static_task_scene import export_scene

    cfg = load_scene_config(args.config)
    layout = grid_layout(cfg)
    output = (args.output or PROJECT_ROOT.parent / "tmp/r2v2_static_task" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    xml = build_scene_xml(cfg)
    scene_path = export_scene(xml, output)
    sources = ("common/r2v2_static_task_scene.py", "common/r2v2_static_task_grid_scene.py",
               "deploy_mujoco/r2v2_static_task_grid.py",
               "common/r2v2_crate.py", "common/r2v2_can_visual.py", "r2v2_description/model.py")
    if "shelf" in cfg:
        sources += (cfg["shelf"]["asset"],)
    report = {"scope": "static robot reference and passive object contact; no robot motion/FSM",
              "mujoco_version": mujoco.__version__, "python": sys.version,
              "configuration": cfg, "layout": layout, "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() for p in sources}}
    model = mujoco.MjModel.from_xml_string(xml)
    start = time.perf_counter()
    data, report["original"] = settle_and_validate(model, cfg)
    report["original"]["wall_seconds"] = time.perf_counter() - start
    # A fresh model loaded only from the portable export must reproduce the rollout.
    reloaded = mujoco.MjModel.from_xml_path(str(scene_path))
    fresh, report["reloaded"] = settle_and_validate(reloaded, cfg)
    atol = cfg["validation"]["reload_state_atol"]
    report["reload_matches"] = bool(np.allclose(data.qpos, fresh.qpos, rtol=0, atol=atol)
                                     and np.allclose(data.qvel, fresh.qvel, rtol=0, atol=atol))
    report["reload_max_qpos_difference"] = float(np.max(np.abs(data.qpos - fresh.qpos)))
    report["passed"] = report["original"]["passed"] and report["reloaded"]["passed"] and report["reload_matches"]
    report_path = output / "report.json"
    report_path.write_text(json.dumps(_safe(report), indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        print(f"FAIL: inspect {report_path}", flush=True)
        return 1
    if not args.no_render:
        report["images"] = render_views(reloaded, fresh, cfg["cameras"], output, layout, "shelf" in cfg)
    report_path.write_text(json.dumps(_safe(report), indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# R2V2 static task scene\n\n"
        "overview.png: full scene; crate_closeup.png: open crate and a regular layer of upright cans.\n"
        "Both images show the same contact-settled state. See report.json for quantitative checks.\n\n"
        "scene.xml contains the initial state; its assets/ paths are relative. Copy this whole folder.\n"
        "The complete robot is a static neutral-pose reference. The crate and every can have independent free joints. See layout.json for names and world positions.\n\n"
        "If configured, the fixed shelf is embedded in scene.xml; no external shelf XML is needed.\n"
        "This package validates the static scene and passive contact only, not robot task execution.\n\n"
        "Open on a machine with a display and MuJoCo 3.3.7:\n"
        "```sh\npython -m mujoco.viewer --mjcf scene.xml\n```\n"
        "In the generic viewer hide geom group 3 to hide duplicate robot collision visuals.\n"
        "The project entry deploy_mujoco/r2v2_static_task_grid.py --view does this automatically.\n",
        encoding="utf-8")
    print(f"PASS: physical validation and independent XML reload\nOutput: {output}", flush=True)
    if args.view:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(reloaded, fresh) as viewer:
            viewer.opt.geomgroup[3] = 0
            viewer.opt.sitegroup[:] = 0
            spec = cfg["cameras"]["overview"]
            viewer.cam.lookat[:] = spec["lookat"]
            for key in ("distance", "azimuth", "elevation"):
                setattr(viewer.cam, key, spec[key])
            while viewer.is_running():
                tick = time.perf_counter()
                mujoco.mj_step(reloaded, fresh)
                viewer.sync()
                time.sleep(max(0, reloaded.opt.timestep - (time.perf_counter() - tick)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
