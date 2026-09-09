"""Native MuJoCo crate/hand geometry previews, not a grasp or lifting demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.r2v2_crate import add_crate, load_crate_config
from common.r2v2_crate_hand_preview import build_hand_preview, sweep_hand_insertion


def _font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def render(model, data, title, detail, *, azimuth=135, elevation=-30,
           distance=.70, lookat=(0, 0, .055), width=960, height=720):
    """Only the camera/HUD changes; no simulation state is advanced here."""
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = azimuth, elevation
    camera.distance, camera.lookat[:] = distance, lookat
    camera.orthographic = True
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # Robot collision duplicates only; crate uses 0.
    options.sitegroup[:] = 0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        result = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, width, 75), fill=(20, 28, 36))
    draw.text((18, 10), title, font=_font(24), fill="white")
    draw.text((18, 43), detail, font=_font(16), fill=(163, 208, 225))
    return result


def sheet(images, columns=2):
    width, height = images[0].size
    result = Image.new("RGB", (columns * width,
                              ((len(images)+columns-1)//columns) * height), (20, 28, 36))
    for index, picture in enumerate(images):
        result.paste(picture, (index % columns * width, index // columns * height))
    return result


def standalone(params):
    root = ET.fromstring('''<mujoco model="R2V2_cargo_crate_geometry_preview">
      <compiler autolimits="true"/>
      <option timestep=".001" integrator="implicitfast" iterations="100" tolerance="1e-10">
        <flag multiccd="enable"/>
      </option>
      <visual><global offwidth="1280" offheight="960"/>
        <headlight ambient=".45 .45 .45" diffuse=".65 .65 .65" specular=".2 .2 .2"/>
        <rgba haze=".85 .89 .92 1"/>
      </visual>
      <asset><texture name="sky" type="skybox" builtin="gradient"
        rgb1=".82 .87 .92" rgb2=".96 .97 .98" width="128" height="768"/></asset>
      <worldbody>
        <light pos="0 -.5 1" dir="0 .3 -1" diffuse=".6 .6 .6"/>
        <geom name="preview_floor" type="plane" size="1 1 .02"
          rgba=".78 .81 .84 1" friction="1 .005 .0001"/>
      </worldbody></mujoco>''')
    add_crate(root, params, position=(0, 0, .001))
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if np.any(data.warning.number) or not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("Free crate settling produced a numerical warning")
    report = {
        "scope": "unactuated empty crate settling on a plane, not hand lifting",
        "time_s": data.time, "position_m": data.body("cargo_crate").xpos.copy(),
        "quaternion_wxyz": data.body("cargo_crate").xquat.copy(),
        "velocity": data.qvel.copy(), "contact_count": data.ncon,
        "mass_kg": model.body("cargo_crate").mass[0],
        "principal_inertia_kg_m2": model.body("cargo_crate").inertia.copy(),
        "warnings": data.warning.number.copy(),
    }
    return model, data, report, xml


def tabletop(params, parity_report, settle_z, crate_xy=(.585, .16)):
    # Reuse the existing policy's verified standing warmup only. The new
    # scene is a static layout preview: no picking, placing or lifting runs.
    from common.r2v2_reach_sim import require_parity
    from common.r2v2_tabletop_demo import (
        TabletopDemoExperiment, copy_robot_initial_state, load_demo_config,
    )
    from common.r2v2_tabletop_scene import build_tabletop_xml

    cfg = load_demo_config()
    require_parity(parity_report, cfg["reach"])
    exp = TabletopDemoExperiment(cfg)
    xml, _ = build_tabletop_xml(exp.cfg, exp.scene_cfg)
    root = ET.fromstring(xml)
    if len(crate_xy) != 2 or not np.all(np.isfinite(crate_xy)):
        raise ValueError("crate_xy must contain two finite world coordinates")
    position = [*crate_xy, exp.table_height + settle_z]
    center = np.asarray(exp.scene_cfg["table_center_xyz"])
    half = np.asarray(exp.scene_cfg["table_half_size"])
    crate_half = np.array([params.depth, params.width])/2
    if np.any(np.abs(np.array(position[:2])-center[:2]) + crate_half > half[:2]):
        raise ValueError("Crate footprint exceeds the existing tabletop; choose a smaller crate")
    add_crate(root, params, position=position)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    copy_robot_initial_state(exp.model, exp.data, model, data)
    # The crate uses its natural isolated rest height for episode setup, not
    # a per-frame pose reset or an extra support constraint.
    contacts = []
    for c in data.contact:
        names = [model.geom(g).name for g in c.geom]
        if any(n.startswith("crate_") for n in names):
            contacts.append({"geoms": names, "distance_m": float(c.dist)})
    record = {
        "scope": "static layout after existing standing warmup; no crate manipulation",
        "table_height_m": exp.table_height,
        "crate_bottom_world_m": position,
        "table_center_xyz": center, "table_half_size": half,
        "crate_contacts": contacts,
        "robot_mass_unchanged": bool(np.array_equal(
            model.body_mass[:exp.model.nbody], exp.model.body_mass)),
        "warmup_duration_s": exp.warmup_duration_s,
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
    }
    return model, data, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parity-report", type=Path,
                        help="Also render the existing standing robot/table layout")
    parser.add_argument("--crate-xy", type=float, nargs=2, default=(.585, .16),
                        metavar=("WORLD_X", "WORLD_Y"), help="Crate bottom-center XY for layout preview")
    args = parser.parse_args()
    params = load_crate_config(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_crate" / stamp).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": "crate asset and static hand geometry; NOT grasp/lift acceptance",
              "parameters": asdict(params), "mujoco_version": mujoco.__version__,
              "source_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                                for name in ("common/r2v2_crate.py", "common/r2v2_crate_hand_preview.py",
                                             "tools/render_r2v2_crate_preview.py")}}
    model, data, report["free_settling"], xml = standalone(params)
    (out / "crate.xml").write_text(xml + "\n")
    dims = (f"{params.width*1000:.0f} W x {params.depth*1000:.0f} D x "
            f"{params.height*1000:.0f} H mm | OPEN TOP | free rigid body")
    views = []
    for name, az, el in (("overview", 140, -32), ("top", 90, -90),
                         ("side", -90, 0), ("front", 0, 0)):
        detail = dims
        if name == "side":
            detail = (f"THROUGH HOLE {params.handle_opening_width*1000:.0f} W x "
                      f"{params.handle_opening_height*1000:.0f} H mm | upper beam "
                      f"{params.handle_beam_height*1000:.0f} H x "
                      f"{params.handle_beam_thickness*1000:.0f} T mm")
        picture = render(model, data, "CARGO CRATE / " + name.upper(), detail,
                         azimuth=az, elevation=el, distance=.60)
        picture.save(out / f"crate_{name}.png")
        views.append(picture)
    sheet(views).save(out / "crate_views.png")
    report["hand_previews"] = {}
    hand_pictures = []
    for side in ("left", "right"):
        for curl in (.03, .30, .50):
            key = f"{side}_{round(np.rad2deg(curl)):02d}deg"
            hand_model, hand_data, record = build_hand_preview(params, side=side, curl_rad=curl)
            report["hand_previews"][key] = record
            gap = record["minimum_hand_crate_distance_m"]
            gap_text = "N/A" if gap is None else f"{gap*1000:+.2f} mm"
            text = (f"STATIC FK ONLY | curl {np.rad2deg(curl):.1f} deg | "
                    f"minimum hand/crate gap {gap_text}")
            picture = render(hand_model, hand_data, side.upper()+" HAND / "
                             + ("OPEN INSERTION" if curl == .03 else "CURL CANDIDATE"), text,
                             azimuth=-55 if side == "left" else 55, elevation=-30,
                             distance=.68, lookat=(0, .10 if side == "left" else -.10, .065))
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 678, 960, 720), fill=(20, 28, 36))
            if record["hand_crate_contacts"] or record["hand_ground_contacts"]:
                status = "BLOCKED POSE / collision overlap; NOT a successful grasp"
                color = (255, 150, 120)
            else:
                status = "No hand/box or ground collision detected; static pose only"
                color = (170, 225, 185)
            draw.text((16, 689), status, font=_font(18), fill=color)
            picture.save(out / f"hand_{key}.png")
            hand_pictures.append(picture)
    sheet(hand_pictures, columns=3).save(out / "hand_comparison.png")
    report["insertion_sweeps"] = {
        side: sweep_hand_insertion(params, side=side) for side in ("left", "right")
    }
    if args.parity_report:
        model, data, report["tabletop_layout"] = tabletop(
            params, args.parity_report, float(report["free_settling"]["position_m"][2]), args.crate_xy)
        picture = render(model, data, "R2V2 / CRATE ON EXISTING TABLE",
                         "STATIC LAYOUT | previous standing pose | bottle pick/place not implemented",
                         azimuth=125, elevation=-15, distance=3.25, lookat=(.18, .05, .93),
                         width=1280, height=900)
        picture.save(out / "tabletop_layout.png")
    (out / "report.json").write_text(json.dumps(_safe(report), indent=2, allow_nan=False)+"\n")
    print(f"Saved native MuJoCo geometry previews and report: {out}", flush=True)
    print("Static hand poses are not controller rollouts or grasp-success evidence.", flush=True)


if __name__ == "__main__":
    main()
