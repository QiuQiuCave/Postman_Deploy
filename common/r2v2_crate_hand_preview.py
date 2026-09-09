"""Fixed-wrist crate-hole geometry previews; never a lifting/grasping trial.

The chosen wrist is fixed, the crate keeps a free joint so MuJoCo does not
discard static/static contact pairs, and only mj_forward is called. No
controller, policy, force-based success detector, or time integration runs.
"""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict
from functools import lru_cache
import itertools
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters, add_crate
from common.r2v2_grasp_recording import body_transform, transform_pose
from r2v2_description.model import (
    SIDES, build_model_xml, hand_names, load_config, urdf_hand_joints,
)


FINGERS = ("index", "middle", "ring", "pinky")
SCOPE = "static geometry only"
NOMINAL_OPEN_CURL_RAD = 0.03


def _text(values):
    return " ".join(format(float(value), ".15g") for value in values)


def _one_hand_root(side):
    if side not in SIDES:
        raise ValueError(f"Unknown hand side: {side}")
    root = ET.fromstring(build_model_xml(load_config(), fixture=True))
    root.set("model", f"R2V2_{side}_crate_static_hand_preview")
    world = root.find("worldbody")
    for body in list(world.findall("body")):
        if body.get("name") != f"{side}_hand_roll_link":
            world.remove(body)
    wrist = world.find(f'body[@name="{side}_hand_roll_link"]')
    wrist.set("pos", "0 0 0")
    wrist.set("quat", "1 0 0 0")
    for parent_name, attribute in (("actuator", "joint"), ("equality", "joint1"), ("contact", "body1")):
        parent = root.find(parent_name)
        for child in list(parent):
            if not child.get(attribute, "").startswith(side + "_"):
                parent.remove(child)
    used = {geom.get("mesh") for geom in world.iter("geom") if geom.get("mesh")}
    for mesh in list(root.find("asset").findall("mesh")):
        if mesh.get("name") not in used:
            root.find("asset").remove(mesh)
    return root, wrist


def _set_hand_pose(model, data, side, curl_rad, thumb_deg):
    cfg = load_config()
    values = np.array(cfg["hands"][side]["open"], dtype=float)
    values[0], values[2:] = np.deg2rad(thumb_deg), curl_rad
    if not np.all(np.isfinite(values)):
        raise ValueError("Hand preview angles must be finite")
    joints = urdf_hand_joints()
    for name, value in zip(hand_names(side), values):
        joint = model.joint(name)
        if not joint.range[0] <= value <= joint.range[1]:
            raise ValueError(f"{name} preview angle is outside its source limits")
        data.qpos[joint.qposadr[0]] = value
    for name, joint in joints.items():
        mimic = joint.find("mimic")
        if not name.startswith(side + "_") or mimic is None:
            continue
        value = (data.qpos[model.joint(mimic.get("joint")).qposadr[0]]
                 * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0")))
        destination = model.joint(name)
        if not destination.range[0] <= value <= destination.range[1]:
            raise ValueError(f"{name} coupled preview angle is outside source limits")
        data.qpos[destination.qposadr[0]] = value
    mujoco.mj_forward(model, data)


def _hand_geoms(model, side):
    return [geom for geom in range(model.ngeom)
            if model.body(model.geom_bodyid[geom]).name.startswith(side + "_")
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])]


def _world_vertices(model, data, geom):
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
    elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.array(list(itertools.product((-1, 1), repeat=3))) * model.geom_size[geom]
    else:
        raise ValueError(f"Unsupported hand collision primitive: {model.geom(geom).name}")
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def _local_part_clouds(model, data, side):
    wrist = body_transform(data, model.body(f"{side}_hand_roll_link").id)
    parts = {part: [] for part in ("palm", "thumb", *FINGERS)}
    for geom in _hand_geoms(model, side):
        body_name = model.body(model.geom_bodyid[geom]).name
        part = next((part for part in ("thumb", *FINGERS) if f"_{part}_" in body_name), "palm")
        vertices = (_world_vertices(model, data, geom)-wrist[:3, 3]) @ wrist[:3, :3]
        parts[part].append(vertices)
    return {part: np.concatenate(clouds) for part, clouds in parts.items()}


@lru_cache(maxsize=8)
def _nominal_open_geometry(side, thumb_deg):
    root, _ = _one_hand_root(side)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    _set_hand_pose(model, data, side, NOMINAL_OPEN_CURL_RAD, thumb_deg)
    parts = _local_part_clouds(model, data, side)
    four = np.concatenate([parts[finger] for finger in FINGERS])
    return {
        "shortest_tip_x": min(float(parts[finger][:, 0].max()) for finger in FINGERS),
        "palm_front_x": float(parts["palm"][:, 0].max()),
        "four_finger_y_center": float((four[:, 1].min()+four[:, 1].max())/2),
    }


def _wrist_rotation(side):
    # Palm faces +world Z. The mirrored right fingers flex toward local +Y,
    # while the left fingers flex toward local -Y. Both local +X point inward.
    if side == "left":
        return np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    return np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def _minimum_geometry_distance(model, data, hand_geoms, crate_geoms):
    """Bounded, witness-checked distance queries, with conservative failure.

    MuJoCo 3.3.7 can return zero and inconsistent witness points for some
    distant mesh pairs. AABB lower bounds safely eliminate irrelevant pairs;
    an invalid potentially closer result makes the minimum unknown, not zero.
    The collision model and its solver options are never changed.
    """
    radius, tolerance = 0.020, 5e-6
    bounds = {}
    for geom in (*hand_geoms, *crate_geoms):
        points = _world_vertices(model, data, geom)
        bounds[geom] = (points.min(0), points.max(0))
    pairs = []
    for hand in hand_geoms:
        for crate in crate_geoms:
            hmin, hmax = bounds[hand]
            cmin, cmax = bounds[crate]
            separation = np.maximum(0., np.maximum(hmin-cmax, cmin-hmax))
            pairs.append((float(np.linalg.norm(separation)), hand, crate))
    closest, best, invalid = None, radius, []
    for lower_bound, hand, crate in sorted(pairs):
        if lower_bound > max(0., best)+tolerance:
            continue
        points = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, hand, crate, radius, points))
        if distance == radius:
            continue  # A censored query is a lower bound, never an exact gap.
        witness_distance = float(np.linalg.norm(points[3:]-points[:3]))
        valid = (np.isfinite(distance) and np.all(np.isfinite(points))
                 and abs(abs(distance)-witness_distance) <= tolerance
                 and (lower_bound <= tolerance or distance >= lower_bound-tolerance))
        pair = [model.geom(hand).name, model.geom(crate).name]
        if not valid:
            invalid.append({"geoms": pair, "returned_distance_m": distance,
                            "witness_distance_m": witness_distance,
                            "aabb_distance_lower_bound_m": lower_bound})
            continue
        if distance < best:
            best = distance
            closest = {"distance_m": distance, "geoms": pair,
                       "points_world_m": points.reshape(2, 3).tolist()}
    unresolved = [item for item in invalid
                  if item["aabb_distance_lower_bound_m"] <= max(0., best)+tolerance]
    reliable = closest is not None and not unresolved
    return {"minimum_hand_crate_distance_m": best if reliable else None,
            "minimum_hand_crate_distance_reliable": reliable,
            "closest_hand_crate_pair": closest,
            "distance_query_radius_m": radius,
            "minimum_hand_crate_distance_lower_bound_m": radius if closest is None and not unresolved else None,
            "unresolved_distance_pairs": unresolved,
            "distance_method": "Collision mesh/box AABB lower-bound pruning, 20 mm MuJoCo mj_geomDistance "
                               "queries, witness-length consistency checked within 5 micrometres; "
                               "unknown or beyond-radius minimum is null, never a false zero."}


def _geometry_record(model, data, params, side, wall_x, insertion_m, curl_rad, thumb_deg):
    wrist = body_transform(data, model.body(f"{side}_hand_roll_link").id)
    hand_geoms = _hand_geoms(model, side)
    hand_set = set(hand_geoms)
    crate_body = model.body("cargo_crate").id
    crate_geoms = [geom for geom in range(model.ngeom)
                   if model.geom_bodyid[geom] == crate_body
                   and (model.geom_contype[geom] or model.geom_conaffinity[geom])]
    crate_set = set(crate_geoms)
    collisions, self_contacts, ground_contacts = [], [], []
    for contact in data.contact:
        if contact.dist > 0:
            continue
        pair = set(map(int, contact.geom))
        item = {"geoms": [model.geom(int(geom)).name for geom in contact.geom],
                "distance_m": float(contact.dist), "penetration_m": max(0., -float(contact.dist)),
                "position_world_m": contact.pos.tolist()}
        if pair & hand_set and pair & crate_set:
            collisions.append(item)
        elif pair <= hand_set:
            self_contacts.append(item)
        elif pair & hand_set:
            ground_contacts.append(item)
    distance_record = _minimum_geometry_distance(model, data, hand_geoms, crate_geoms)
    parts = _local_part_clouds(model, data, side)
    four = np.concatenate([parts[finger] for finger in FINGERS])
    per_finger = {}
    for finger in FINGERS:
        tip = float(parts[finger][:, 0].max())
        per_finger[finger] = {
            "tip_x_wrist_m": tip,
            "past_outer_wall_m": tip-wall_x,
            "past_wall_inner_face_m": tip-wall_x-params.wall_thickness,
            "past_beam_inner_face_m": tip-wall_x-params.handle_beam_thickness,
        }
    return {
        "scope": SCOPE, "side": side, "time_s": float(data.time), "simulation_steps": 0,
        "grasp_success_evaluated": False, "crate_is_free_but_not_integrated": True,
        "parameters": asdict(params), "nominal_insertion_m": float(insertion_m),
        "nominal_open_curl_rad": NOMINAL_OPEN_CURL_RAD,
        "curl_rad": float(curl_rad), "thumb_deg": float(thumb_deg),
        "wrist_transform_world": wrist.tolist(), "wrist_pose_world": transform_pose(wrist),
        "wrist_local_axes": {"x": "finger extension / into box", "y": "palm normal axis (mirrored)",
                             "z": "across the four fingers"},
        "outer_wall_x_wrist_m": float(wall_x),
        "hand_crate_contacts": collisions, "hand_self_contacts": self_contacts,
        "hand_ground_contacts": ground_contacts,
        "collision_free": not collisions and not ground_contacts,
        "maximum_hand_crate_penetration_m": max((item["penetration_m"] for item in collisions), default=0.),
        **distance_record,
        "four_finger_collision_aabb_wrist_m": {"min": four.min(0).tolist(), "max": four.max(0).tolist()},
        "four_finger_width_m": float(np.ptp(four[:, 2])),
        "four_finger_full_swept_thickness_m": float(np.ptp(four[:, 1])),
        "per_finger": per_finger,
        "palm_to_outer_wall_m": float(wall_x-parts["palm"][:, 0].max()),
        "thumb_to_outer_wall_m": float(wall_x-parts["thumb"][:, 0].max()),
        "note": "Curled candidates retain the identical wrist pose: finger retraction is measured, not compensated. "
                "Nonzero box penetration is an infeasible pose/contact candidate, not a successful hook.",
    }


def build_hand_preview(params=None, side="left", insertion_m=0.030, curl_rad=0.03, thumb_deg=75):
    """Return ``(model, data, record)`` for one palm-up hand and the real crate.

    ``insertion_m`` is nominal penetration of the shortest *open* fingertip
    beyond the ordinary sidewall's inner plane. The reinforced beam extends
    farther inward, so its effective insertion depth is reported separately.
    Different curl candidates share this open-pose-based wrist transform.
    The crate's local/world origin is the centre of its bottom surface.
    """
    params = params or CrateParameters()
    if side not in SIDES:
        raise ValueError(f"Unknown hand side: {side}")
    if not np.isfinite(insertion_m) or not -0.25 <= insertion_m <= 0.10:
        raise ValueError("Preview insertion must be finite and between -0.25 and 0.10 m")
    nominal = _nominal_open_geometry(side, float(thumb_deg))
    wall_x = nominal["shortest_tip_x"]-params.wall_thickness-insertion_m
    rotation = _wrist_rotation(side)
    sign = 1 if side == "left" else -1
    hole_center_z = (params.handle_opening_bottom+params.handle_opening_top)/2
    position = np.array([0., sign*(params.width/2+wall_x),
                         hole_center_z-rotation[2, 1]*nominal["four_finger_y_center"]])
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    root, wrist = _one_hand_root(side)
    wrist.set("pos", _text(position))
    wrist.set("quat", _text(quaternion))
    add_crate(root, params, position=(0, 0, 0), quaternion=(1, 0, 0, 0), free=True)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    _set_hand_pose(model, data, side, float(curl_rad), float(thumb_deg))
    record = _geometry_record(model, data, params, side, wall_x, insertion_m, curl_rad, thumb_deg)
    return model, data, record


def sweep_hand_insertion(params=None, side="left", insertion_m=0.030, curl_rad=0.03,
                         thumb_deg=75, start_palm_clearance_m=0.060, samples=61):
    """Sample the straight insertion path with no dynamics or collision edits.

    Begin with the palm's front edge 60 mm outside the sidewall; finish at
    the requested open-pose-based insertion. Contacts include any initial
    fingertip/edge overlap instead of silently skipping it.
    """
    if samples < 2 or int(samples) != samples:
        raise ValueError("Insertion sweep requires at least two integer samples")
    if not np.isfinite(start_palm_clearance_m) or start_palm_clearance_m <= 0:
        raise ValueError("Initial palm clearance must be finite and positive")
    params = params or CrateParameters()
    model, final_data, final_record = build_hand_preview(params, side, insertion_m, curl_rad, thumb_deg)
    nominal = _nominal_open_geometry(side, float(thumb_deg))
    final_wall = final_record["outer_wall_x_wrist_m"]
    start_wall = nominal["palm_front_x"]+start_palm_clearance_m
    if start_wall < final_wall:
        raise ValueError("Sweep start must be outside the final wrist position")
    # Transform the static wrist on a private model copy for each FK sample.
    # Neither the returned preview model nor the free crate pose is changed.
    model = copy.copy(model)
    wrist_id = model.body(f"{side}_hand_roll_link").id
    final_wrist_position = model.body_pos[wrist_id].copy()
    direction = np.array(final_record["wrist_transform_world"])[:3, 0]
    scratch = mujoco.MjData(model)
    rows = []
    for index, wall in enumerate(np.linspace(start_wall, final_wall, int(samples))):
        scratch.qpos[:] = final_data.qpos
        model.body_pos[wrist_id] = final_wrist_position-direction*(wall-final_wall)
        mujoco.mj_forward(model, scratch)
        row = _geometry_record(model, scratch, params, side, float(wall),
                               nominal["shortest_tip_x"]-params.wall_thickness-wall,
                               curl_rad, thumb_deg)
        rows.append({"index": index, "outer_wall_x_wrist_m": float(wall),
                     "wrist_transform_world": row["wrist_transform_world"],
                     "palm_to_outer_wall_m": row["palm_to_outer_wall_m"],
                     "collision_free": row["collision_free"],
                     "minimum_hand_crate_distance_m": row["minimum_hand_crate_distance_m"],
                     "minimum_hand_crate_distance_reliable": row["minimum_hand_crate_distance_reliable"],
                     "minimum_hand_crate_distance_lower_bound_m": row["minimum_hand_crate_distance_lower_bound_m"],
                     "unresolved_distance_pairs": row["unresolved_distance_pairs"],
                     "maximum_hand_crate_penetration_m": row["maximum_hand_crate_penetration_m"],
                     "hand_crate_contacts": row["hand_crate_contacts"]})
    return {"scope": SCOPE, "side": side, "simulation_steps": 0,
            "curl_rad": float(curl_rad), "thumb_deg": float(thumb_deg),
            "start_palm_clearance_m": float(start_palm_clearance_m),
            "final_nominal_insertion_m": float(insertion_m),
            "collision_free": all(row["collision_free"] for row in rows),
            "colliding_samples": sum(not row["collision_free"] for row in rows), "samples": rows,
            "distance_method": final_record["distance_method"],
            "note": "Wrist-frame FK sweep on a private model only; the free crate pose never changes."}


__all__ = ["build_hand_preview", "sweep_hand_insertion"]
