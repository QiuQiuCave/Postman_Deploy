"""Passive, instantaneous measurements for the two-hand crate fixture.

The caller must supply already-forwarded MuJoCo data so poses and solved
contact forces share a timestamp. Nothing is forwarded, integrated, reset,
or written here. An upward finger contact is not a successful-grasp latch:
the experiment must additionally verify sustained clearance and low slip.
"""

from common.path_config import PROJECT_ROOT

import mujoco
import numpy as np

from common.r2v2_grasp_recording import body_transform, relative_transform


SIDES = ("left", "right")
FINGERS = ("index", "middle", "ring", "pinky")
PARTS = (*FINGERS, "thumb", "palm")
FORCE_EPS_N = 1e-8


def _descendant(model, body, ancestor):
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _geom_minimum_z(model, data, geom):
    """Exact lowest collision point for the crate's boxes/convex mesh geoms."""
    rotation_z = data.geom_xmat[geom].reshape(3, 3)[2]
    center_z = float(data.geom_xpos[geom, 2])
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        # Equivalent to testing all eight oriented corners, without an AABB
        # approximation in body coordinates or a fixed upright height.
        return center_z-float(np.abs(rotation_z) @ model.geom_size[geom])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+count]
        return center_z+float(np.min(vertices @ rotation_z))
    raise ValueError(f"Unsupported crate collision shape: {model.geom(geom).name}")


def _contact_force_on_geom(model, data, index, geom):
    """Return raw local wrench and world force on a specified contact geom.

    Positive MuJoCo contact force acts on geom2. Contact axes are the rows
    of contact.frame, so conversion to world coordinates uses its transpose.
    Tangential forces are included: normal force alone is not bearing load.
    """
    contact = data.contact[index]
    if geom not in (int(contact.geom1), int(contact.geom2)):
        raise ValueError("Requested force target is not part of this contact")
    local = np.zeros(6)
    mujoco.mj_contactForce(model, data, index, local)
    world = contact.frame.reshape(3, 3).T @ local[:3]
    if geom == int(contact.geom1):
        world *= -1
    return local, world


def _new_part():
    return {"normal_force_N": 0.0, "force_on_crate_world_N": np.zeros(3),
            "vertical_force_N": 0.0, "contact_count": 0,
            "loaded_contact_count": 0, "contact_indices": []}


def measure_crate_lift(model, data, table_top_m, *, virtual_props=False):
    """Return JSON-ready measurements, without claiming lift success.

    ``hands[side].finger_vertical_force_N`` is the net world +Z force on
    the crate from its four non-thumb fingers. Positive normal force can be
    purely horizontal or even downward; it is never used as a substitute.
    ``bearing_finger_contacts`` further identifies upward-loaded contacts
    specifically against a handle beam. All thresholds here only distinguish
    numerical zero (1e-8 N), not stable support or grasp success.
    In explicit virtual-prop mode, noncolliding crate geometry is used for
    pose/outline measurements only; zero contact values are not grasp evidence.
    """
    if not np.isfinite(table_top_m):
        raise ValueError("table_top_m must be finite")
    crate = model.body("cargo_crate").id
    table = model.geom("lift_table_geom").id
    floor = model.geom("floor").id
    wrist_ids = {side: model.body(f"{side}_hand_roll_link").id for side in SIDES}
    crate_geoms, hand_geoms = set(), {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if virtual_props and _descendant(model, body, crate):
            if model.geom_contype[geom] or model.geom_conaffinity[geom]:
                raise ValueError("Virtual crate geometry must have both contact masks disabled")
            crate_geoms.add(geom)
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if _descendant(model, body, crate):
            crate_geoms.add(geom)
        for side, wrist in wrist_ids.items():
            if _descendant(model, body, wrist):
                name = model.body(body).name
                part = next((part for part in PARTS[:-1] if f"_{part}_" in name), "palm")
                hand_geoms[geom] = (side, part)
                break
    if not crate_geoms:
        raise ValueError("cargo_crate has no collision geometry to measure")

    crate_transform = body_transform(data, crate)
    quaternion = data.xquat[crate].copy()
    if quaternion[0] < 0:
        quaternion *= -1
    jacobian_pos, jacobian_rot = np.empty((3, model.nv)), np.empty((3, model.nv))
    # Body-frame origin velocity, not COM-based cvel with an omitted offset.
    mujoco.mj_jacBody(model, data, jacobian_pos, jacobian_rot, crate)
    linear, angular = jacobian_pos @ data.qvel, jacobian_rot @ data.qvel
    bottom = min(_geom_minimum_z(model, data, geom) for geom in crate_geoms)
    # atan2 avoids arccos' precision loss near perfectly upright/inverted.
    tilt = float(np.arctan2(np.linalg.norm(crate_transform[:2, 2]), crate_transform[2, 2]))
    hands = {}
    for side in SIDES:
        wrist_transform = body_transform(data, wrist_ids[side])
        hands[side] = {"parts": {part: _new_part() for part in PARTS},
                       "T_world_wrist": wrist_transform.tolist(),
                       "T_wrist_crate": relative_transform(wrist_transform, crate_transform).tolist(),
                       "contacts": [], "bearing_finger_contacts": [], "upward_finger_contacts": []}

    crate_contacts, table_contacts, floor_contacts = [], [], []
    hand_table_contacts, hand_floor_contacts, hand_self_contacts = [], [], []
    crate_force, table_force, floor_force = np.zeros(3), np.zeros(3), np.zeros(3)
    for index, contact in enumerate(data.contact):
        g1, g2 = int(contact.geom1), int(contact.geom2)
        target = g1 if g1 in crate_geoms else g2 if g2 in crate_geoms else None
        hand_table = (g1 in hand_geoms and g2 == table) or (g2 in hand_geoms and g1 == table)
        hand_floor = (g1 in hand_geoms and g2 == floor) or (g2 in hand_geoms and g1 == floor)
        hand_self = g1 in hand_geoms and g2 in hand_geoms
        if target is None and not (hand_table or hand_floor or hand_self):
            continue
        local, on_geom2 = _contact_force_on_geom(model, data, index, g2)
        normal = max(0., float(local[0]))
        active = bool(normal > FORCE_EPS_N or contact.dist <= 0)
        item = {"index": index, "geoms": [model.geom(g1).name, model.geom(g2).name],
                "distance_m": float(contact.dist), "penetration_m": max(0., -float(contact.dist)),
                "position_world_m": contact.pos.tolist(), "normal_force_N": normal,
                "force_on_geom2_world_N": on_geom2.tolist(), "active": active,
                "loaded": bool(normal > FORCE_EPS_N)}
        if hand_table and active:
            hand_table_contacts.append(item.copy())
        if hand_floor and active:
            hand_floor_contacts.append(item.copy())
        if hand_self and active:
            item["sides"] = [hand_geoms[g1][0], hand_geoms[g2][0]]
            hand_self_contacts.append(item.copy())
        if target is None:
            continue
        other = g2 if target == g1 else g1
        force = -on_geom2 if target == g1 else on_geom2
        crate_force += force
        item.update({"crate_geom": model.geom(target).name, "other_geom": model.geom(other).name,
                     "force_on_crate_world_N": force.tolist(), "vertical_force_N": float(force[2]),
                     "side": None, "part": None})
        if other == table:
            table_contacts.append(item)
            table_force += force
        elif other == floor:
            floor_contacts.append(item)
            floor_force += force
        elif other in hand_geoms:
            side, part = hand_geoms[other]
            item.update({"side": side, "part": part,
                         "handle_beam_contact": model.geom(target).name.endswith("_handle_beam")})
            summary = hands[side]["parts"][part]
            summary["normal_force_N"] += normal
            summary["force_on_crate_world_N"] += force
            summary["vertical_force_N"] += float(force[2])
            summary["contact_count"] += 1
            summary["loaded_contact_count"] += int(normal > FORCE_EPS_N)
            summary["contact_indices"].append(index)
            hands[side]["contacts"].append(item)
            if part in FINGERS and normal > FORCE_EPS_N and force[2] > FORCE_EPS_N:
                hands[side]["upward_finger_contacts"].append(item)
                if item["handle_beam_contact"]:
                    hands[side]["bearing_finger_contacts"].append(item)
        crate_contacts.append(item)

    for side in SIDES:
        hand = hands[side]
        hand["normal_force_N"] = sum(part["normal_force_N"] for part in hand["parts"].values())
        hand["finger_normal_force_N"] = sum(hand["parts"][part]["normal_force_N"] for part in FINGERS)
        total = sum((part["force_on_crate_world_N"] for part in hand["parts"].values()), np.zeros(3))
        finger = sum((hand["parts"][part]["force_on_crate_world_N"] for part in FINGERS), np.zeros(3))
        hand.update({"force_on_crate_world_N": total.tolist(), "vertical_force_N": float(total[2]),
                     "finger_force_on_crate_world_N": finger.tolist(), "finger_vertical_force_N": float(finger[2]),
                     "finger_handle_vertical_force_N": sum(contact["vertical_force_N"] for contact in hand["contacts"]
                         if contact["part"] in FINGERS and contact["handle_beam_contact"]),
                     "has_upward_finger_contact": bool(hand["upward_finger_contacts"]),
                     "has_bearing_finger_contact": bool(hand["bearing_finger_contacts"])})
        for part in hand["parts"].values():
            part["force_on_crate_world_N"] = part["force_on_crate_world_N"].tolist()

    maximum_depth = lambda contacts: max((contact["penetration_m"] for contact in contacts), default=0.)
    return {"scope": ("virtual static crate pose; contacts/load/lift NOT evaluated" if virtual_props else
                       "instantaneous two-hand crate fixture measurements; not a grasp-success verdict"),
            "prop_contact_physics_evaluated": not virtual_props,
            "grasp_success_evaluated": False, "time_s": float(data.time), "table_top_m": float(table_top_m),
            "T_world_crate": crate_transform.tolist(), "crate_position_m": crate_transform[:3, 3].tolist(),
            "crate_quaternion_wxyz": quaternion.tolist(), "crate_tilt_rad": tilt,
            "crate_tilt_deg": float(np.rad2deg(tilt)), "bottom_height_m": bottom,
            "clearance_m": bottom-float(table_top_m),
            "bottom_height_method": "minimum over every collision box's rotated corners and every convex-mesh vertex",
            "crate_linear_velocity_world_m_s": linear.tolist(), "crate_angular_velocity_world_rad_s": angular.tolist(),
            "crate_linear_speed_m_s": float(np.linalg.norm(linear)), "crate_angular_speed_rad_s": float(np.linalg.norm(angular)),
            "crate_mass_kg": float(model.body_mass[crate]),
            "crate_weight_N": float(-model.body_mass[crate]*model.opt.gravity[2]),
            "hands": hands, "crate_contacts": crate_contacts, "table_contacts": table_contacts,
            "floor_contacts": floor_contacts, "hand_table_contacts": hand_table_contacts,
            "hand_floor_contacts": hand_floor_contacts, "hand_self_contacts": hand_self_contacts,
            "table_contact": any(contact["active"] for contact in table_contacts),
            "floor_contact": any(contact["active"] for contact in floor_contacts),
            "table_bearing_contact": any(contact["vertical_force_N"] > FORCE_EPS_N for contact in table_contacts),
            "crate_contact_force_world_N": crate_force.tolist(),
            "table_force_on_crate_world_N": table_force.tolist(), "table_vertical_force_N": float(table_force[2]),
            "floor_force_on_crate_world_N": floor_force.tolist(), "floor_vertical_force_N": float(floor_force[2]),
            "max_hand_crate_penetration_m": maximum_depth([c for c in crate_contacts if c["side"] is not None]),
            "max_hand_table_penetration_m": maximum_depth(hand_table_contacts),
            "max_hand_floor_penetration_m": maximum_depth(hand_floor_contacts),
            "max_hand_self_penetration_m": maximum_depth(hand_self_contacts),
            "force_convention": "force_on_crate = +/- contact.frame.T @ local_force[:3]; plus for crate geom2; "
                                "all tangential forces included; positive vertical is world +Z",
            "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
                                 and np.all(np.isfinite(data.xpos)) and np.all(np.isfinite(data.xmat))
                                 and np.all(np.isfinite(crate_force)))}


__all__ = ["measure_crate_lift"]
