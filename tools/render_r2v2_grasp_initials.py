"""Static MuJoCo comparisons for user selection; never changes default poses."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.path_config import PROJECT_ROOT

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.r2v2_cylinder_test import CylinderExperiment, CylinderParameters
from r2v2_description.model import CONFIG, load_config

WIDTH, HEIGHT = 640, 520
POSITIONS = {
    "1": (0.015, 0.035, "original cylinder position"),
    "2": (0.025, 0.035, "+10 mm toward fingertips"),
    "3": (0.015, 0.045, "+10 mm away from palm"),
}


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def candidate(degrees, position_code):
    cfg = load_config()
    original = copy.deepcopy(cfg)
    x, offset, description = POSITIONS[position_code]
    # Read the bound from the compiled model, not from a new assumed axis.
    baseline = CylinderExperiment()
    limit = float(baseline.model.joint("left_thumb_metacarpal_joint").range[1])
    applied = min(math.radians(degrees), limit)
    cfg["hands"]["left"]["open"][0] = applied
    exp = CylinderExperiment(CylinderParameters(x=x, palm_offset=offset), cfg)
    assert exp.data.time == 0.0  # Never settle/interpolate away the specified pose.
    assert cfg["hands"]["left"]["open"][1:] == original["hands"]["left"]["open"][1:]
    distances, contacts = [], []
    for g in range(exp.model.ngeom):
        name = exp.model.geom(g).name
        if name.startswith("left_") and "_col_" in name:
            distance = mujoco.mj_geomDistance(exp.model, exp.data, exp.cylinder_geom, g, 1.0, None)
            distances.append((float(distance), name))
    for c in exp.data.contact:
        names = [exp.model.geom(i).name for i in c.geom]
        if any(n.startswith("left_") for n in names) and c.dist <= 0:
            contacts.append({"geoms": names, "distance_m": float(c.dist)})
    wrist = exp.data.body("left_hand_roll_link")
    relative = wrist.xmat.reshape(3, 3).T @ (exp.params.center - wrist.xpos)
    summary = {
        "requested_thumb_rotation_deg": degrees,
        "applied_thumb_rotation_rad": applied,
        "applied_thumb_rotation_deg": math.degrees(applied),
        "joint": "left_thumb_metacarpal_joint",
        "joint_limit_rad": [0.0, limit],
        "open_joint_angles_rad": cfg["hands"]["left"]["open"],
        "position_code": position_code, "position_description": description,
        "cylinder_center_world_m": exp.params.center.tolist(),
        "cylinder_center_wrist_m": relative.tolist(),
        "cylinder_diameter_m": 2 * exp.params.radius, "cylinder_height_m": exp.params.height,
        "minimum_hand_cylinder_gap_m": min(distances)[0],
        "minimum_thumb_cylinder_gap_m": min(d for d, n in distances if "_thumb_" in n),
        "nearest_hand_geom": min(distances)[1],
        "hand_contacts_at_initial_pose": contacts,
        "initial_self_penetration_m": max([-c["distance_m"] for c in contacts
                                           if "cylinder_geom" not in c["geoms"]] or [0.0]),
        "scope": "static t=0 geometry only; no dynamics or grasp success test",
    }
    return exp, summary


def render(exp, summary, identifier, view="oblique"):
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 1
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    # Identical cameras across candidates: apparent size/spacing is comparable.
    camera.lookat[:] = [0.0, 0.115, 0.46]
    camera.distance = 0.40
    if view == "oblique":
        camera.azimuth, camera.elevation = 50, -25
    elif view == "top":
        camera.azimuth, camera.elevation = -90, -88
    else:
        raise ValueError(view)
    with mujoco.Renderer(exp.model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(exp.data, camera=camera, scene_option=opt)
        panel = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, WIDTH, 82), fill=(20, 28, 36))
    angle = summary["applied_thumb_rotation_deg"]
    angle_text = f"{angle:.1f} deg" if angle < 89 else "90 deg (~89.95 model limit)"
    draw.text((14, 10), f"{identifier} | thumb rotation: {angle_text}", font=font(21), fill="white")
    draw.text((14, 45), f"{view.upper()} | {summary['position_description']}", font=font(18), fill=(155, 210, 235))
    draw.rectangle((0, HEIGHT - 86, WIDTH, HEIGHT), fill=(20, 28, 36))
    rel = np.array(summary["cylinder_center_wrist_m"]) * 1000
    draw.text((14, HEIGHT - 81), f"Cylinder/wrist XYZ = ({rel[0]:.0f}, {rel[1]:.0f}, {rel[2]:.0f}) mm", font=font(19), fill="white")
    gap = 1000 * summary["minimum_hand_cylinder_gap_m"]
    thumb = 1000 * summary["minimum_thumb_cylinder_gap_m"]
    color = (130, 230, 160) if gap >= 0 else (255, 110, 100)
    draw.text((14, HEIGHT - 54), f"Object gap {gap:.1f} mm | thumb gap {thumb:.1f} mm", font=font(18), fill=color)
    depth = 1000 * summary["initial_self_penetration_m"]
    draw.text((14, HEIGHT - 27), f"Palm/thumb collision overlap: {depth:.2f} mm | STATIC",
              font=font(17), fill=(255, 160, 100) if depth > 0 else (130, 230, 160))
    return panel


def contact_sheet(panels, columns, heading):
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (WIDTH * columns, HEIGHT * rows + 70), (12, 18, 25))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), heading, font=font(25), fill="white")
    draw.text((18, 43), "Left hand | cylinder diameter 40 mm, height 120 mm | all other joints unchanged", font=font(18), fill=(170, 185, 200))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (i % columns * WIDTH, 70 + i // columns * HEIGHT))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = (args.output or PROJECT_ROOT / "artifacts/r2v2_grasp_initials" / stamp).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config_before = CONFIG.read_bytes()
    summaries, panels = {}, []
    current_angle = math.degrees(load_config()["hands"]["left"]["open"][0])
    for label, degrees in (("CURRENT", current_angle), ("60", 60), ("75", 75), ("90", 90)):
        exp, record = candidate(degrees, "1")
        record["id"] = label
        panel = render(exp, record, label)
        panel.save(out / f"angle_{label}.png")
        panels.append(panel)
        summaries[label] = record
    contact_sheet(panels, 2, "1. Thumb initial angle (same cylinder position)").save(out / "angle_comparison.png")
    overview = {}
    for family, degrees in (("A", 75), ("B", 90)):
        panels = []
        for code in POSITIONS:
            identifier = family + code
            exp, record = candidate(degrees, code)
            record["id"] = identifier
            summaries[identifier] = record
            views = [render(exp, record, identifier, view) for view in ("oblique", "top")]
            overview[identifier] = views[0]
            panels.extend(views)
            contact_sheet(views, 2, f"Candidate {identifier}: initial hand / cylinder placement").save(out / f"{identifier}.png")
        contact_sheet(panels, 2, f"{family}. Thumb {degrees} deg: three cylinder positions, two views each").save(out / f"positions_{degrees}deg.png")
    contact_sheet([overview[family + code] for code in POSITIONS for family in ("A", "B")],
                  2, "2. Choose initial placement: A = 75 deg, B = 90 deg").save(out / "placement_overview.png")
    assert CONFIG.read_bytes() == config_before
    (out / "candidates.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"output": str(out), "candidates": {
        k: {"angle_deg": r["applied_thumb_rotation_deg"],
            "relative_xyz_mm": (np.array(r["cylinder_center_wrist_m"]) * 1000).tolist(),
            "thumb_gap_mm": 1000*r["minimum_thumb_cylinder_gap_m"],
            "hand_gap_mm": 1000*r["minimum_hand_cylinder_gap_m"],
            "initial_contacts": len(r["hand_contacts_at_initial_pose"])} for k, r in summaries.items()
    }, "default_config_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
