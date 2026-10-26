"""Build and render a native, reusable vertical-grip crate review candidate."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from common.r2v2_vertical_grip_crate import (
    add_vertical_grip_crate, load_vertical_grip_crate_config,
)
from tools.render_r2v2_crate_preview import render, sheet


def standalone(params):
    root = ET.fromstring('''<mujoco model="r2v2_vertical_grip_crate">
      <compiler autolimits="true"/>
      <option timestep=".001" integrator="implicitfast" iterations="100" tolerance="1e-10"/>
      <visual><global offwidth="1280" offheight="960"/>
        <headlight ambient=".45 .45 .45" diffuse=".65 .65 .65" specular=".2 .2 .2"/>
      </visual>
      <asset><texture name="sky" type="skybox" builtin="gradient"
        rgb1=".82 .87 .92" rgb2=".96 .97 .98" width="128" height="768"/></asset>
      <worldbody><light pos=".4 -.5 1" dir="-.2 .3 -1" diffuse=".6 .6 .6"/>
        <geom name="preview_floor" type="plane" size="1 1 .02" rgba=".78 .81 .84 1"/>
      </worldbody></mujoco>''')
    add_vertical_grip_crate(root, params, position=(0, 0, .001))
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if data.warning.number.any() or not np.isfinite(data.qpos).all():
        raise RuntimeError("Crate settling produced invalid physics")
    return model, data, xml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    params = load_vertical_grip_crate_config(args.config) if args.config else load_vertical_grip_crate_config()
    args.output.mkdir(parents=True, exist_ok=False)
    model, data, xml = standalone(params)
    (args.output/"crate_vertical_grips.xml").write_text(xml+"\n")
    overview = render(model, data, "VERTICAL SIDE GRIPS / OPEN-TOP CRATE",
        "Shell 360 W x 240 D x 160 H mm | 2 vertical grips + 4 connecting rods",
        azimuth=25, elevation=-28, distance=.90, lookat=(0, 0, .08), width=1200, height=900)
    overview.save(args.output/"overview.png")
    views = []
    for name, title, detail, az, el, distance, lookat in (
        ("front", "FRONT / BOTH VERTICAL GRIPS", "Shell width 360 mm | overall width including grips 520 mm", 0, 0, .85, (0, 0, .08)),
        ("top", "TOP / OPEN CAVITY", "Solid side panels; no old through-hole handles", 0, -90, .85, (0, 0, .08)),
        ("close", "SIDE HANDLE / PROFILE", "Grip diameter 40 mm, length 120 mm | wall clearance 40 mm", 0, 0, .36, (0, .21, .08)),
        ("side", "SIDE HANDLE / OBLIQUE", "Two 12 mm rods | 96 mm unobstructed central grasp span", -58, -20, .43, (0, .20, .08)),
    ):
        picture = render(model, data, title, detail, azimuth=az, elevation=el,
            distance=distance, lookat=lookat, width=960, height=720)
        picture.save(args.output/f"{name}.png")
        views.append(picture)
    sheet(views).save(args.output/"detail_views.png")
    report = dict(scope="native MuJoCo geometry preview; NOT IK, hand grasp or loaded walking validation",
        parameters=asdict(params), overall_width_m=params.overall_width,
        unobstructed_grip_span_m=params.unobstructed_grip_span,
        mass_kg=float(model.body("cargo_crate").mass[0]),
        inertia_kg_m2=model.body("cargo_crate").inertia.tolist(),
        settled_position_m=data.body("cargo_crate").xpos.tolist(),
        settled_velocity_norm=float(np.linalg.norm(data.qvel)),
        warnings=data.warning.number.tolist(), model_external_dependencies=False,
        old_crate_unchanged=True, existing_fsm_unchanged=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output.resolve()), **report), indent=2))


if __name__ == "__main__":
    main()
