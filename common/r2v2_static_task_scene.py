"""Independent two-table scene: static R2V2, free crate and free can.

No controller, policy or FSM imports. The complete robot's source neutral
geometry is compiled as a static hierarchy, without welds or runtime resets.
"""

import copy
from dataclasses import fields
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from common.path_config import PROJECT_ROOT
from common.r2v2_can_visual import add_can_visual
from common.r2v2_crate import CrateParameters, add_crate
from r2v2_description.model import build_model_xml, load_config as load_hand_config


DEFAULT_CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_static_task.yaml"
TABLE_NAMES = ("pickup_table", "dropoff_table")
OBJECTS = (("cargo_crate", "crate_free"), ("test_cylinder", "cylinder_free"))


def _text(values):
    return " ".join(format(float(v), ".17g") for v in values)


def _vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return a


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _keys(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"{label} must contain exactly: {names}")


def load_scene_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def validate_config(config):
    cfg = copy.deepcopy(config)
    _keys(cfg, "schema_version robot table tables crate can simulation validation cameras", "config")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    _keys(cfg["robot"], "mode position_xyz", "robot")
    if cfg["robot"]["mode"] != "static_neutral":
        raise ValueError("This entry supports only a static_neutral robot")
    _vector(cfg["robot"]["position_xyz"], 3, "robot.position_xyz")
    t = cfg["table"]
    _keys(t, "board_size_xyz leg_width friction board_rgba leg_rgba", "table")
    size = _vector(t["board_size_xyz"], 3, "table.board_size_xyz")
    _positive(t["leg_width"], "table.leg_width")
    if np.min(size) <= 0 or t["leg_width"] >= min(size[:2]) / 2:
        raise ValueError("Table dimensions must be positive and leave room between legs")
    for key in ("board_rgba", "leg_rgba"):
        rgba = _vector(t[key], 4, key)
        if np.any((rgba < 0) | (rgba > 1)):
            raise ValueError(f"{key} channels must be in [0, 1]")
    _keys(cfg["tables"], "pickup_table dropoff_table", "tables")
    for name, pos in cfg["tables"].items():
        if _vector(pos, 3, name)[2] <= size[2] / 2:
            raise ValueError("Table underside must be above ground")
    if cfg["tables"]["pickup_table"][2] != cfg["tables"]["dropoff_table"][2]:
        raise ValueError("Identical tables must have identical leg heights")
    _keys(cfg["crate"], "parameters position_xyz quaternion_wxyz", "crate")
    if set(cfg["crate"]["parameters"]) != {f.name for f in fields(CrateParameters)}:
        raise ValueError("All CrateParameters must be explicit in the scene config")
    CrateParameters(**cfg["crate"]["parameters"])
    _keys(cfg["can"], "position_xyz quaternion_wxyz radius height mass friction solref solimp label", "can")
    for name in ("crate", "can"):
        _vector(cfg[name]["position_xyz"], 3, name + ".position_xyz")
        q = _vector(cfg[name]["quaternion_wxyz"], 4, name + ".quaternion_wxyz")
        # This initial layout contract is axis-aligned and upright.
        if not np.allclose(q, [1, 0, 0, 0], atol=1e-12, rtol=0):
            raise ValueError(f"{name} must start upright with quaternion [1,0,0,0]")
    for key in ("radius", "height", "mass"):
        _positive(cfg["can"][key], "can." + key)
    for label, friction in (("table", t["friction"]), ("can", cfg["can"]["friction"])):
        v = _vector(friction, 3, label + ".friction")
        if v[0] <= 0 or min(v) < 0:
            raise ValueError("Friction must be nonnegative with positive sliding friction")
    ref = _vector(cfg["can"]["solref"], 2, "can.solref")
    imp = _vector(cfg["can"]["solimp"], 3, "can.solimp")
    if min(ref) <= 0 or not 0 < imp[0] <= imp[1] < 1 or imp[2] <= 0:
        raise ValueError("Invalid can contact solver parameters")
    if not isinstance(cfg["can"]["label"], str) or not cfg["can"]["label"]:
        raise ValueError("can.label must be a nonempty path")
    _keys(cfg["simulation"], "timestep duration_s stable_window_s", "simulation")
    for key, value in cfg["simulation"].items():
        _positive(value, "simulation." + key)
    sim = cfg["simulation"]
    if sim["timestep"] != 0.001 or not sim["timestep"] <= sim["stable_window_s"] < sim["duration_s"]:
        raise ValueError("Require 1 ms physics and a stable window shorter than the rollout")
    if not np.isclose(sim["duration_s"] / sim["timestep"], round(sim["duration_s"] / sim["timestep"])):
        raise ValueError("duration_s must be a whole number of physics steps")
    _keys(cfg["validation"], "initial_penetration_tolerance_m max_contact_penetration_m max_linear_speed_mps max_angular_speed_radps max_position_drift_m max_tilt_deg support_force_relative_tolerance reload_state_atol", "validation")
    for key, value in cfg["validation"].items():
        _positive(value, "validation." + key)
    _keys(cfg["cameras"], "overview crate_closeup", "cameras")
    for name, camera in cfg["cameras"].items():
        _keys(camera, "width height lookat distance azimuth elevation", name)
        _vector(camera["lookat"], 3, name + ".lookat")
        _vector([camera["azimuth"], camera["elevation"]], 2, name + ".angles")
        _positive(camera["distance"], name + ".distance")
        for key in ("width", "height"):
            if type(camera[key]) is not int or not 1 <= camera[key] <= 8192:
                raise ValueError("Image dimensions must be integers in [1,8192]")
    return cfg


def add_table(world, name, center, parameters):
    """Board + four legs, following r2v2_tabletop_scene's geometry convention."""
    half = np.asarray(parameters["board_size_xyz"]) / 2
    center = np.asarray(center)
    table = ET.SubElement(world, "body", name=name, pos=_text(center))
    common = dict(type="box", contype="1", conaffinity="1", friction=_text(parameters["friction"]))
    ET.SubElement(table, "geom", name=name + "_top", size=_text(half),
                  rgba=_text(parameters["board_rgba"]), **common)
    leg_half_width = parameters["leg_width"] / 2
    leg_half_height = (center[2] - half[2]) / 2
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        position = [sx * (half[0] - leg_half_width), sy * (half[1] - leg_half_width),
                    leg_half_height - center[2]]
        ET.SubElement(table, "geom", name=f"{name}_leg_{index}", pos=_text(position),
                      size=_text([leg_half_width, leg_half_width, leg_half_height]),
                      rgba=_text(parameters["leg_rgba"]), **common)


def build_scene_xml(config):
    cfg = validate_config(config)
    hands = load_hand_config()
    hands["simulation_dt"] = cfg["simulation"]["timestep"]
    root = ET.fromstring(build_model_xml(hands, fixture=False))
    root.set("model", "R2V2_static_two_table_task")
    # Source joint q=0 is the neutral pose; retain every link and mesh.
    # Removing joints at compile time fixes the hierarchy without constraints.
    for body in root.findall(".//worldbody//body"):
        for joint in list(body.findall("joint")) + list(body.findall("freejoint")):
            body.remove(joint)
    for tag in ("actuator", "sensor", "equality", "keyframe"):
        for element in list(root.findall(tag)):
            root.remove(element)
    root.find('.//body[@name="base_link"]').set("pos", _text(cfg["robot"]["position_xyz"]))
    ET.SubElement(root.find("option"), "flag", multiccd="enable")
    # Neutral floor and reduced reflections make support geometry legible.
    ground = root.find('./asset/material[@name="groundplane"]')
    ground.set("reflectance", "0")
    texture = root.find('./asset/texture[@name="groundplane"]')
    if texture is not None:
        texture.set("rgb1", ".72 .76 .80")
        texture.set("rgb2", ".65 .69 .73")
    world = root.find("worldbody")
    for name in TABLE_NAMES:
        add_table(world, name, cfg["tables"][name], cfg["table"])
    crate = cfg["crate"]
    add_crate(root, CrateParameters(**crate["parameters"]), position=crate["position_xyz"],
              quaternion=crate["quaternion_wxyz"], free=True)
    c = cfg["can"]
    can = ET.SubElement(world, "body", name="test_cylinder", pos=_text(c["position_xyz"]),
                        quat=_text(c["quaternion_wxyz"]))
    ET.SubElement(can, "freejoint", name="cylinder_free")
    ET.SubElement(can, "geom", name="cylinder_geom", type="cylinder",
                  size=_text([c["radius"], c["height"] / 2]), mass=str(c["mass"]),
                  rgba="1 0 0 0", contype="1", conaffinity="1", condim="3", priority="1",
                  friction=_text(c["friction"]), solref=_text(c["solref"]), solimp=_text(c["solimp"]))
    add_can_visual(root, can, c["label"])
    visual = root.find("visual/global")
    visual.set("offwidth", str(max(c["width"] for c in cfg["cameras"].values())))
    visual.set("offheight", str(max(c["height"] for c in cfg["cameras"].values())))
    return ET.tostring(root, encoding="unicode")


def export_scene(xml, output):
    """Export initial MJCF with relative asset paths; no repository dependency."""
    output = Path(output)
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "."))
    for kind in ("mesh", "texture"):
        for item in root.findall(f"asset/{kind}[@file]"):
            source = Path(item.get("file"))
            if not source.is_absolute():
                source = (meshdir if kind == "mesh" else PROJECT_ROOT) / source
            relative = Path("assets") / kind / (item.get("name") + source.suffix)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            item.set("file", relative.as_posix())
    for key in ("meshdir", "texturedir", "assetdir"):
        compiler.attrib.pop(key, None)
    path = output / "scene.xml"
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    return path


def _geoms(model, prefix):
    return [g for g in range(model.ngeom) if model.geom(g).name.startswith(prefix)
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def initial_checks(model, data, cfg):
    """Explicit distances also cover static bodies filtered from data.contact."""
    mujoco.mj_forward(model, data)
    tol = cfg["validation"]["initial_penetration_tolerance_m"]
    names = [model.geom(i).name for i in range(model.ngeom)]
    groups = {name: _geoms(model, name) for name in TABLE_NAMES}
    groups["crate"] = _geoms(model, "crate_")
    groups["can"] = [model.geom("cylinder_geom").id]
    task = {g for ids in groups.values() for g in ids}
    floor = model.geom("floor").id
    robot = [g for g in range(model.ngeom) if g not in task and g != floor
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    pairs = {(a, b) for i, ids in enumerate(groups.values())
             for others in list(groups.values())[i + 1:] for a in ids for b in others}
    pairs.update((a, b) for a in robot for b in task)
    pairs.update((g, floor) for g in task | set(robot))
    penetrations = []
    minimum = float("inf")
    for a, b in sorted(pairs):
        distance = float(mujoco.mj_geomDistance(model, data, a, b, 10, None))
        # Coincident boxes can return zero from the convex distance query.
        # Our table boxes are axis-aligned: exact AABB overlap covers this
        # degenerate case, including static-static pairs filtered by MuJoCo.
        if (model.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX
                and model.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX
                and np.allclose(data.geom_xmat[a].reshape(3, 3), np.eye(3), atol=1e-12)
                and np.allclose(data.geom_xmat[b].reshape(3, 3), np.eye(3), atol=1e-12)):
            overlap = model.geom_size[a] + model.geom_size[b] - np.abs(data.geom_xpos[a] - data.geom_xpos[b])
            if min(overlap) > 0:
                distance = min(distance, -float(min(overlap)))
        minimum = min(minimum, distance)
        if distance < -tol:
            penetrations.append({"geoms": [names[a], names[b]], "distance_m": distance})
    p = cfg["crate"]["parameters"]
    center = np.asarray(cfg["tables"]["pickup_table"])
    half = np.asarray(cfg["table"]["board_size_xyz"]) / 2
    pos = np.asarray(cfg["crate"]["position_xyz"])
    can = np.asarray(cfg["can"]["position_xyz"])
    crate_margin = half[:2] - np.abs(pos[:2] - center[:2]) - np.array([p["depth"], p["width"]]) / 2
    can_margin = (np.array([p["depth"], p["width"]]) / 2 - p["wall_thickness"]
                  - np.abs(can[:2] - pos[:2]) - cfg["can"]["radius"])
    crate_gap = pos[2] - center[2] - half[2]
    can_gap = can[2] - cfg["can"]["height"] / 2 - pos[2] - p["bottom_thickness"]
    can_top_margin = pos[2] + p["height"] - can[2] - cfg["can"]["height"] / 2
    free = (model.nq, model.nv, model.njnt, model.nu, model.neq, model.nmocap) == (14, 12, 2, 0, 0, 0)
    free &= all(model.joint(j).type == mujoco.mjtJoint.mjJNT_FREE for _, j in OBJECTS)
    checks = {
        "two_free_objects_no_actuators_constraints_or_mocap": bool(free),
        "unique_geom_names": len(names) == len(set(names)),
        "no_initial_scene_penetration": not penetrations,
        "crate_fully_over_pickup_table": bool(np.min(crate_margin) >= -tol and crate_gap >= -tol),
        "can_fully_inside_crate": bool(np.min(can_margin) >= -tol and can_gap >= -tol and can_top_margin >= -tol),
        "two_identical_collision_tables": all(len(groups[n]) == 5 for n in TABLE_NAMES)
            and bool(np.array_equal(model.geom_size[groups[TABLE_NAMES[0]]], model.geom_size[groups[TABLE_NAMES[1]]])),
    }
    return {"passed": all(checks.values()), "checks": checks, "checked_geom_pairs": len(pairs),
            "scope": "task objects, robot/task and all ground pairs; source robot internal geometry unchanged",
            "minimum_distance_m": minimum, "penetrations": penetrations,
            "crate_table_clearance_m": float(crate_gap), "can_floor_clearance_m": float(can_gap),
            "crate_table_xy_margin_m": crate_margin.tolist(), "can_wall_xy_margin_m": can_margin.tolist()}


def _contact_state(model, data):
    support = {"crate_on_table_N": 0.0, "can_on_crate_N": 0.0}
    unwanted = []
    max_depth = 0.0
    for i, contact in enumerate(data.contact):
        a, b = (model.geom(int(g)).name for g in contact.geom)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        max_depth = max(max_depth, -float(contact.dist))
        pair = {a, b}
        if pair == {"pickup_table_top", "crate_bottom"}:
            vertical = (contact.frame.reshape(3, 3).T @ force[:3])[2]
            support["crate_on_table_N"] += float(vertical if b == "crate_bottom" else -vertical)
        elif pair == {"cylinder_geom", "crate_bottom"}:
            vertical = (contact.frame.reshape(3, 3).T @ force[:3])[2]
            support["can_on_crate_N"] += float(vertical if b == "cylinder_geom" else -vertical)
        elif force[0] > 1e-6:
            unwanted.append([a, b])
    return support, max_depth, unwanted


def _object_geometry(model, data, cfg):
    crate = data.body("cargo_crate")
    can = data.body("test_cylinder")
    rotation = crate.xmat.reshape(3, 3)
    local = rotation.T @ (can.xpos - crate.xpos)
    axis = rotation.T @ can.xmat.reshape(3, 3)[:, 2]
    # Exact projected half extents of the cylinder, including any tilt.
    extent = cfg["can"]["height"] / 2 * np.abs(axis) + cfg["can"]["radius"] * np.sqrt(np.maximum(0, 1 - axis**2))
    p = cfg["crate"]["parameters"]
    # Inward handle beams occupy the top 20 mm; use a conservative Y envelope.
    ywall = p["handle_beam_thickness"] if local[2] + extent[2] >= p["height"] - p["handle_beam_height"] else p["wall_thickness"]
    bounds = np.array([p["depth"] / 2 - p["wall_thickness"], p["width"] / 2 - ywall])
    margins = np.r_[bounds - np.abs(local[:2]) - extent[:2],
                    local[2] - extent[2] - p["bottom_thickness"], p["height"] - local[2] - extent[2]]
    # Outside-bottom corners must stay over the source board after settling.
    corners = np.array([[sx*p["depth"]/2, sy*p["width"]/2, 0]
                        for sx in (-1, 1) for sy in (-1, 1)]) @ rotation.T + crate.xpos
    table = np.asarray(cfg["tables"]["pickup_table"])
    half = np.asarray(cfg["table"]["board_size_xyz"]) / 2
    support_margin = np.min(half[:2] - np.abs(corners[:, :2] - table[:2]))
    return margins, float(support_margin)


def settle_and_validate(model, cfg):
    """Advance only mj_step; never reset qpos/qvel or apply forces in rollout."""
    data = mujoco.MjData(model)
    initial = initial_checks(model, data, cfg)
    report = {"initial": initial, "passed": False}
    if not initial["passed"]:
        return data, report
    sim, limits = cfg["simulation"], cfg["validation"]
    tail_positions, tail_linear, tail_angular, tail_tilt, tail_support = [], [], [], [], []
    finite = clean_forces = contained = supported_footprint = no_unwanted = True
    max_depth = 0.0
    min_containment = np.full(4, np.inf)
    for step in range(round(sim["duration_s"] / model.opt.timestep)):
        mujoco.mj_step(model, data)
        finite &= all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc))
        clean_forces &= not (data.qfrc_applied.any() or data.xfrc_applied.any())
        if not finite or data.warning.number.any():
            break
        # Refresh kinematics and solved forces at the current integration state.
        mujoco.mj_forward(model, data)
        margins, support_margin = _object_geometry(model, data, cfg)
        min_containment = np.minimum(min_containment, margins)
        contained &= bool(min(margins) >= -limits["max_contact_penetration_m"])
        supported_footprint &= support_margin >= 0
        support, depth, unwanted = _contact_state(model, data)
        max_depth = max(max_depth, depth)
        no_unwanted &= not unwanted
        if (step + 1) * model.opt.timestep >= sim["duration_s"] - sim["stable_window_s"]:
            positions, linear, angular, tilts = [], [], [], []
            for body, joint in OBJECTS:
                address = model.joint(joint).dofadr[0]
                positions.append(data.body(body).xpos.copy())
                linear.append(np.linalg.norm(data.qvel[address:address + 3]))
                angular.append(np.linalg.norm(data.qvel[address + 3:address + 6]))
                tilts.append(np.rad2deg(np.arccos(np.clip(data.body(body).xmat[8], -1, 1))))
            tail_positions.append(positions)
            tail_linear.append(linear)
            tail_angular.append(angular)
            tail_tilt.append(tilts)
            tail_support.append([support["crate_on_table_N"], support["can_on_crate_N"]])
    checks = {"finite_state": bool(finite), "zero_numerical_warnings": not data.warning.number.any(),
              "no_external_forces": bool(clean_forces), "can_remains_inside": bool(contained),
              "crate_remains_over_table": bool(supported_footprint), "only_expected_contacts": bool(no_unwanted),
              "bounded_contact_penetration": max_depth <= limits["max_contact_penetration_m"],
              "full_duration": bool(np.isclose(data.time, sim["duration_s"], rtol=0, atol=1e-8))}
    tail = {}
    if tail_positions:
        weights = np.array([cfg["crate"]["parameters"]["mass"] + cfg["can"]["mass"], cfg["can"]["mass"]]) * abs(model.opt.gravity[2])
        force_error = np.max(np.abs(np.asarray(tail_support) / weights - 1), axis=0)
        linear_max, angular_max = np.max(tail_linear, axis=0), np.max(tail_angular, axis=0)
        tilt_max = np.max(tail_tilt, axis=0)
        drift = np.max(np.ptp(tail_positions, axis=0), axis=1)
        checks.update(stable_linear_speed=bool(max(linear_max) <= limits["max_linear_speed_mps"]),
                      stable_angular_speed=bool(max(angular_max) <= limits["max_angular_speed_radps"]),
                      upright=bool(max(tilt_max) <= limits["max_tilt_deg"]),
                      stable_position=bool(max(drift) <= limits["max_position_drift_m"]),
                      supported_by_contact=bool(max(force_error) <= limits["support_force_relative_tolerance"]))
        tail = dict(object_order=[o[0] for o in OBJECTS], samples=len(tail_positions),
                    max_linear_speed_mps=linear_max.tolist(), max_angular_speed_radps=angular_max.tolist(),
                    max_tilt_deg=tilt_max.tolist(), position_drift_m=drift.tolist(),
                    support_force_relative_error=force_error.tolist(),
                    mean_support_force_N=np.mean(tail_support, axis=0).tolist())
    else:
        checks["stable_window_observed"] = False
    report.update(passed=bool(all(checks.values())), checks=checks, time_s=float(data.time),
                  warnings=data.warning.number.tolist(), max_contact_penetration_m=max_depth,
                  min_can_containment_m=min_containment.tolist(), stable_window=tail,
                  final_positions={b: data.body(b).xpos.tolist() for b, _ in OBJECTS},
                  final_qpos=data.qpos.tolist(), final_qvel=data.qvel.tolist())
    return data, report
