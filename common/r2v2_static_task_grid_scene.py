"""Two-table scene with independently free cans and an optional fixed shelf.

The single-can implementation stays intact. Meshes/materials are shared by
cloned visual geoms, while each can owns a free joint and one physical cylinder.
"""

import copy
import itertools
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from common.path_config import PROJECT_ROOT
from common import r2v2_static_task_scene as single


DEFAULT_CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_static_task_grid.yaml"


def load_scene_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def validate_config(config):
    base = copy.deepcopy(config)
    if not isinstance(base, dict) or "cargo_grid" not in base:
        raise ValueError("Grid configuration requires cargo_grid")
    grid = base.pop("cargo_grid")
    shelf = base.pop("shelf", None)
    cfg = single.validate_config(base)
    if shelf is not None:
        single._keys(shelf, "asset position_xyz yaw_rad", "shelf")
        if shelf["asset"] != "assets/shelf/shelf.xml":
            raise ValueError("shelf.asset must reference the shared assets/shelf/shelf.xml")
        single._vector(shelf["position_xyz"], 3, "shelf.position_xyz")
        single._vector([shelf["yaw_rad"]], 1, "shelf.yaw_rad")
        if shelf["position_xyz"][2] != 0:
            raise ValueError("The fixed shelf must stand on the ground at z=0")
        cfg["shelf"] = shelf
    single._keys(grid, "rows columns gap_m minimum_wall_clearance_m", "cargo_grid")
    for key in ("rows", "columns"):
        if type(grid[key]) is not int or not 1 <= grid[key] <= 100:
            raise ValueError(f"cargo_grid.{key} must be an integer in [1,100]")
    for key in ("gap_m", "minimum_wall_clearance_m"):
        single._positive(grid[key], "cargo_grid." + key)
    if grid["rows"] * grid["columns"] > 100:
        raise ValueError("This serial validation entry is limited to 100 cans")
    cfg["cargo_grid"] = grid
    layout = grid_layout(cfg)
    if min(layout["safe_envelope_clearance_xy_m"]) < grid["minimum_wall_clearance_m"] - 1e-12:
        raise ValueError("Grid does not fit with the configured wall clearance")
    p = cfg["crate"]["parameters"]
    relative = np.asarray(cfg["can"]["position_xyz"]) - cfg["crate"]["position_xyz"]
    bottom = relative[2] - cfg["can"]["height"] / 2
    if bottom < p["bottom_thickness"] - 1e-12 or relative[2] + cfg["can"]["height"] / 2 > p["height"]:
        raise ValueError("Grid must start fully inside the crate above its bottom")
    return cfg


def grid_layout(cfg):
    """Compute row-major world poses and capacity under explicit gap constraints."""
    g, c, p = cfg["cargo_grid"], cfg["can"], cfg["crate"]["parameters"]
    count = np.array([g["rows"], g["columns"]])
    diameter = 2 * c["radius"]
    pitch = diameter + g["gap_m"]
    footprint = count * diameter + (count - 1) * g["gap_m"]
    inner = np.array([p["depth"], p["width"]]) - 2 * p["wall_thickness"]
    # Reserve clearance to the inward handle beams even below beam height.
    envelope = np.array([inner[0], p["width"] - 2 * max(p["wall_thickness"], p["handle_beam_thickness"])])
    offset = np.asarray(c["position_xyz"][:2]) - cfg["crate"]["position_xyz"][:2]
    clearances = (envelope - footprint) / 2 - np.abs(offset)
    capacity = np.maximum(0, np.floor((envelope - 2 * (g["minimum_wall_clearance_m"] + np.abs(offset))
                                      + g["gap_m"] + 1e-12) / pitch).astype(int))
    cans = []
    for row in range(g["rows"]):
        for col in range(g["columns"]):
            name = f"cargo_can_r{row + 1:02d}_c{col + 1:02d}"
            position = np.asarray(c["position_xyz"], dtype=float).copy()
            position[:2] += [((row - (g["rows"] - 1) / 2) * pitch),
                             ((col - (g["columns"] - 1) / 2) * pitch)]
            cans.append({"body": name, "joint": name + "_free", "geom": name + "_geom",
                         "row": row + 1, "column": col + 1, "position_xyz": position.tolist()})
    return {"rows": g["rows"], "columns": g["columns"], "count": len(cans), "layers": 1,
            "pitch_m": pitch, "surface_gap_m": g["gap_m"], "footprint_xy_m": footprint.tolist(),
            "inner_floor_xy_m": inner.tolist(), "safe_envelope_xy_m": envelope.tolist(),
            "floor_wall_clearance_xy_m": ((inner - footprint) / 2 - np.abs(offset)).tolist(),
            "safe_envelope_clearance_xy_m": clearances.tolist(),
            "max_rows_columns_with_current_gaps": capacity.tolist(),
            "total_cargo_mass_kg": len(cans) * c["mass"],
            "loaded_crate_mass_kg": p["mass"] + len(cans) * c["mass"], "cans": cans}


def build_scene_xml(config):
    cfg = validate_config(config)
    base = copy.deepcopy(cfg)
    base.pop("cargo_grid")
    shelf = base.pop("shelf", None)
    root = ET.fromstring(single.build_scene_xml(base))
    root.set("model", "R2V2_static_two_table_grid_task")
    world = root.find("worldbody")
    template = world.find('./body[@name="test_cylinder"]')
    world.remove(template)
    for item in grid_layout(cfg)["cans"]:
        can = copy.deepcopy(template)
        can.set("name", item["body"])
        can.set("pos", single._text(item["position_xyz"]))
        can.find("freejoint").set("name", item["joint"])
        for geom in can.findall("geom"):
            if geom.get("name") == "cylinder_geom":
                geom.set("name", item["geom"])
            else:
                geom.set("name", item["body"] + "_visual_" + geom.get("name").removeprefix("r2v2_cola_can_"))
        world.append(can)
    if shelf is not None:
        source = ET.parse(PROJECT_ROOT / shelf["asset"]).getroot()
        root.find("asset").extend(copy.deepcopy(list(source.find("asset"))))
        body = copy.deepcopy(source.find('./worldbody/body[@name="shelf"]'))
        body.set("pos", single._text(shelf["position_xyz"]))
        body.set("euler", single._text([0, 0, shelf["yaw_rad"]]))
        for geom in body.iter("geom"):
            geom.set("name", "shelf_" + geom.get("name"))
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
        world.append(body)
    return ET.tostring(root, encoding="unicode")


def _indexes(model, layout):
    body_ids = np.array([model.body(c["body"]).id for c in layout["cans"]], dtype=int)
    geom_ids = np.array([model.geom(c["geom"]).id for c in layout["cans"]], dtype=int)
    bodies = np.r_[model.body("cargo_crate").id, body_ids]
    joints = ["crate_free"] + [c["joint"] for c in layout["cans"]]
    dofs = np.array([model.joint(j).dofadr[0] for j in joints])
    return body_ids, geom_ids, bodies, dofs


def _containment(model, data, cfg, can_ids):
    crate = data.body("cargo_crate")
    rotation = crate.xmat.reshape(3, 3)
    local = (data.xpos[can_ids] - crate.xpos) @ rotation
    axes = data.xmat[can_ids].reshape(-1, 3, 3)[:, :, 2] @ rotation
    c, p = cfg["can"], cfg["crate"]["parameters"]
    extent = c["height"] / 2 * np.abs(axes) + c["radius"] * np.sqrt(np.maximum(0, 1 - axes**2))
    ywall = np.where(local[:, 2] + extent[:, 2] >= p["height"] - p["handle_beam_height"],
                     max(p["handle_beam_thickness"], p["wall_thickness"]), p["wall_thickness"])
    margins = np.column_stack((p["depth"] / 2 - p["wall_thickness"] - np.abs(local[:, 0]) - extent[:, 0],
                               p["width"] / 2 - ywall - np.abs(local[:, 1]) - extent[:, 1],
                               local[:, 2] - extent[:, 2] - p["bottom_thickness"],
                               p["height"] - local[:, 2] - extent[:, 2]))
    corners = np.array([[sx*p["depth"]/2, sy*p["width"]/2, 0]
                        for sx in (-1, 1) for sy in (-1, 1)]) @ rotation.T + crate.xpos
    table = np.asarray(cfg["tables"]["pickup_table"])
    half = np.asarray(cfg["table"]["board_size_xyz"]) / 2
    footprint_margin = np.min(half[:2] - np.abs(corners[:, :2] - table[:2]))
    return margins, float(footprint_margin), local


def initial_checks(model, data, cfg, layout):
    mujoco.mj_forward(model, data)
    can_ids, geom_ids, bodies, _ = _indexes(model, layout)
    table_groups = [single._geoms(model, n) for n in single.TABLE_NAMES]
    groups = table_groups + [single._geoms(model, "crate_")] + [[int(g)] for g in geom_ids]
    if "shelf" in cfg:
        groups.append(single._geoms(model, "shelf_"))
    task = {g for group in groups for g in group}
    floor = model.geom("floor").id
    robot = [g for g in range(model.ngeom) if g not in task and g != floor
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    pairs = {(a, b) for left, right in itertools.combinations(groups, 2) for a in left for b in right}
    pairs.update((a, b) for a in robot for b in task)
    pairs.update((g, floor) for g in task | set(robot))
    penetrations = []
    tol = cfg["validation"]["initial_penetration_tolerance_m"]
    for a, b in sorted(pairs):
        distance = float(mujoco.mj_geomDistance(model, data, a, b, 10, None))
        if (model.geom_type[a] == mujoco.mjtGeom.mjGEOM_BOX
                and model.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX
                and np.allclose(data.geom_xmat[a].reshape(3, 3), np.eye(3), atol=1e-12)
                and np.allclose(data.geom_xmat[b].reshape(3, 3), np.eye(3), atol=1e-12)):
            overlap = model.geom_size[a] + model.geom_size[b] - np.abs(data.geom_xpos[a] - data.geom_xpos[b])
            if min(overlap) > 0:
                distance = min(distance, -float(min(overlap)))
        if distance < -tol:
            penetrations.append({"geoms": [model.geom(a).name, model.geom(b).name], "distance_m": distance})
    margins, footprint, _ = _containment(model, data, cfg, can_ids)
    # Analytic upright cylinder gaps also catch fully coincident equal cylinders.
    centers = data.xpos[can_ids]
    center_pairs = list(itertools.combinations(range(len(can_ids)), 2))
    min_gap = min((np.linalg.norm(centers[i, :2] - centers[j, :2]) - 2 * cfg["can"]["radius"]
                   for i, j in center_pairs), default=None)
    n = len(bodies)
    free = (model.nq, model.nv, model.njnt, model.nu, model.neq, model.nmocap) == (7*n, 6*n, n, 0, 0, 0)
    free &= bool(np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    names = [model.geom(i).name for i in range(model.ngeom)]
    checks = {"independent_free_crate_and_cans": bool(free),
              "no_actuators_constraints_or_mocap": (model.nu, model.neq, model.nmocap) == (0, 0, 0),
              "unique_geom_names": len(names) == len(set(names)),
              "no_initial_scene_penetration": not penetrations,
              "all_cans_inside": bool(np.min(margins) >= -tol),
              "configured_can_gaps": min_gap is None or min_gap >= cfg["cargo_grid"]["gap_m"] - tol,
              "crate_fully_over_table": footprint >= -tol,
              "two_identical_collision_tables": all(len(g) == 5 for g in table_groups)
                  and bool(np.array_equal(model.geom_size[table_groups[0]], model.geom_size[table_groups[1]]))}
    if "shelf" in cfg:
        shelf_geoms = single._geoms(model, "shelf_")
        checks["fixed_collision_shelf"] = (len(shelf_geoms) == 29
            and model.body("shelf").jntnum[0] == 0
            and bool(np.all(model.geom_contype[shelf_geoms] == 1))
            and bool(np.all(model.geom_conaffinity[shelf_geoms] == 1)))
    return {"passed": bool(all(checks.values())), "checks": checks,
            "checked_geom_pairs": len(pairs), "can_pair_count": len(center_pairs),
            "minimum_can_surface_gap_m": min_gap, "penetrations": penetrations,
            "can_containment_m": margins.tolist(),
            "scope": "all task objects, robot/task and ground; source robot internal geometry unchanged"}


def settle_and_validate(model, cfg):
    layout = grid_layout(cfg)
    data = mujoco.MjData(model)
    initial = initial_checks(model, data, cfg, layout)
    report = {"initial": initial, "passed": False}
    if not initial["passed"]:
        return data, report
    can_ids, geom_ids, bodies, dofs = _indexes(model, layout)
    geom_to_slot = {int(g): i + 1 for i, g in enumerate(geom_ids)}
    bottom, tabletop = model.geom("crate_bottom").id, model.geom("pickup_table_top").id
    order = ["cargo_crate"] + [c["body"] for c in layout["cans"]]
    sim, limits = cfg["simulation"], cfg["validation"]
    expected_weights = np.r_[layout["loaded_crate_mass_kg"], np.full(layout["count"], cfg["can"]["mass"])] * abs(model.opt.gravity[2])
    initial_local = (np.array([c["position_xyz"] for c in layout["cans"]]) - cfg["crate"]["position_xyz"])[:, :2]
    min_margins = np.full((layout["count"], 4), np.inf)
    max_linear, max_angular, max_tilt = np.zeros(len(bodies)), np.zeros(len(bodies)), np.zeros(len(bodies))
    position_min, position_max = np.full((len(bodies), 3), np.inf), np.full((len(bodies), 3), -np.inf)
    force_error, force_sum, min_support = np.zeros(len(bodies)), np.zeros(len(bodies)), np.full(len(bodies), np.inf)
    max_depth = max_grid_deviation = 0.0
    unexpected = set()
    finite = force_free = contained = footprint_ok = True
    samples = 0
    linear_indices = dofs[:, None] + np.arange(3)
    angular_indices = dofs[:, None] + np.arange(3, 6)
    for step in range(round(sim["duration_s"] / model.opt.timestep)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        finite &= all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.qacc))
        force_free &= not (data.qfrc_applied.any() or data.xfrc_applied.any())
        if not finite or data.warning.number.any():
            break
        margins, footprint, local = _containment(model, data, cfg, can_ids)
        min_margins = np.minimum(min_margins, margins)
        contained &= bool(np.min(margins) >= -limits["max_contact_penetration_m"])
        footprint_ok &= footprint >= 0
        max_grid_deviation = max(max_grid_deviation, float(np.max(np.linalg.norm(local[:, :2] - initial_local, axis=1))))
        support = np.zeros(len(bodies))
        for i, contact in enumerate(data.contact):
            a, b = map(int, contact.geom)
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            max_depth = max(max_depth, -float(contact.dist))
            if (a == tabletop and b == bottom) or (a == bottom and b == tabletop):
                support[0] += float((contact.frame.reshape(3, 3).T @ force[:3])[2]) * (1 if b == bottom else -1)
            elif a == bottom and b in geom_to_slot:
                support[geom_to_slot[b]] += float((contact.frame.reshape(3, 3).T @ force[:3])[2])
            elif b == bottom and a in geom_to_slot:
                support[geom_to_slot[a]] -= float((contact.frame.reshape(3, 3).T @ force[:3])[2])
            elif force[0] > 1e-6:
                unexpected.add(tuple(sorted((model.geom(a).name, model.geom(b).name))))
        if (step + 1) * model.opt.timestep >= sim["duration_s"] - sim["stable_window_s"]:
            samples += 1
            max_linear = np.maximum(max_linear, np.linalg.norm(data.qvel[linear_indices], axis=1))
            max_angular = np.maximum(max_angular, np.linalg.norm(data.qvel[angular_indices], axis=1))
            max_tilt = np.maximum(max_tilt, np.rad2deg(np.arccos(np.clip(data.xmat[bodies, 8], -1, 1))))
            position_min = np.minimum(position_min, data.xpos[bodies])
            position_max = np.maximum(position_max, data.xpos[bodies])
            force_error = np.maximum(force_error, np.abs(support / expected_weights - 1))
            force_sum += support
            min_support = np.minimum(min_support, support)
    drift = np.max(position_max - position_min, axis=1)
    checks = {"finite_state": bool(finite), "zero_numerical_warnings": not data.warning.number.any(),
              "no_external_forces": bool(force_free), "all_cans_remain_inside": bool(contained),
              "crate_remains_over_table": bool(footprint_ok), "only_expected_support_contacts": not unexpected,
              "bounded_contact_penetration": max_depth <= limits["max_contact_penetration_m"],
              "full_duration": bool(np.isclose(data.time, sim["duration_s"], rtol=0, atol=1e-8)),
              "stable_window_observed": samples > 0,
              "all_linear_speeds_stable": bool(max(max_linear) <= limits["max_linear_speed_mps"]),
              "all_angular_speeds_stable": bool(max(max_angular) <= limits["max_angular_speed_radps"]),
              "all_objects_upright": bool(max(max_tilt) <= limits["max_tilt_deg"]),
              "all_positions_stable": bool(max(drift) <= limits["max_position_drift_m"]),
              "every_can_and_crate_supported": bool(max(force_error) <= limits["support_force_relative_tolerance"] and min(min_support) > 0),
              "grid_alignment_preserved": max_grid_deviation <= limits["max_position_drift_m"]}
    report.update(passed=bool(all(checks.values())), checks=checks, time_s=float(data.time),
                  warnings=data.warning.number.tolist(), max_contact_penetration_m=max_depth,
                  unexpected_contacts=sorted(unexpected), max_grid_xy_deviation_m=max_grid_deviation,
                  minimum_containment_per_can_m=min_margins.tolist(),
                  stable_window={"samples": samples, "object_order": order,
                                 "max_linear_speed_mps": max_linear.tolist(), "max_angular_speed_radps": max_angular.tolist(),
                                 "max_tilt_deg": max_tilt.tolist(), "position_drift_m": drift.tolist(),
                                 "support_force_relative_error": force_error.tolist(),
                                 "mean_support_force_N": (force_sum / max(1, samples)).tolist(),
                                 "min_support_force_N": min_support.tolist()},
                  final_positions={name: data.xpos[bid].tolist() for name, bid in zip(order, bodies)},
                  final_qpos=data.qpos.tolist(), final_qvel=data.qvel.tolist())
    return data, report
