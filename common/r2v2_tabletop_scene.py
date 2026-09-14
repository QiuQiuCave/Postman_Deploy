"""Unmodified new R2V2 asset, a fixed table, and a genuinely free cylinder.

This module constructs and measures the scene only. It contains no object
pose resets, mocap movement, external forces, welds, or grasp-success latch.
Metrics expect MuJoCo kinematics/contact forces to have been refreshed by the
caller; they never forward or step the caller's live simulation data.
"""

from common.path_config import PROJECT_ROOT

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_cylinder_test import CylinderParameters, load_cylinder_profile
from common.r2v2_grasp_recording import CONTACT_PARTS, body_transform
from common.r2v2_reach_sim import TCP_OFFSETS
from r2v2_description.model import SIDES, build_model_xml, load_config


DEFAULT_TABLE_CENTER = (0.48, 0.12, 1.084)
DEFAULT_TABLE_HALF_SIZE = (0.16, 0.22, 0.03)
DEFAULT_CYLINDER_POSITION = (0.40, 0.20, 1.175)


def _vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result.copy()


def _text(value):
    return " ".join(map(str, value))


def build_tabletop_xml(reach_cfg, demo_cfg):
    """Return ``(MJCF text, hand_cfg)`` without mutating either input config.

    ``demo_cfg`` accepts world-coordinate ``table_center_xyz``, positive box
    ``table_half_size`` (including half thickness), ``cylinder_position_xyz``,
    and optional ``cylinder_quaternion_wxyz``. ``cylinder_profile`` accepts a
    profile name, YAML path or resolved profile mapping; omission retains the
    40 mm / 120 mm / 100 g baseline. Without ``cylinder_position_xyz``, an
    upright object is placed 1 mm above the configured tabletop (or the
    specified nonnegative ``cylinder_initial_clearance_m``).
    ``object_appearance`` defaults to ``orange_cylinder``; ``cola_can`` adds a
    purely visual can skin scaled from the actual collision dimensions.
    The table remains axis-aligned
    and fixed, with four collision-enabled legs. The selected profile owns
    dimensions/mass; friction and contact solver values retain the baseline.
    """
    appearance = demo_cfg.get("object_appearance", "orange_cylinder")
    if appearance not in ("orange_cylinder", "cola_can"):
        raise ValueError(f"Unknown object_appearance: {appearance}")
    center = _vector(demo_cfg.get("table_center_xyz", DEFAULT_TABLE_CENTER), 3, "table_center_xyz")
    half_size = _vector(demo_cfg.get("table_half_size", DEFAULT_TABLE_HALF_SIZE), 3, "table_half_size")
    profile = load_cylinder_profile(demo_cfg.get("cylinder_profile"))
    params = CylinderParameters.from_profile(profile)
    clearance = demo_cfg.get("cylinder_initial_clearance_m", 0.001)
    if (isinstance(clearance, bool) or not isinstance(clearance, (int, float))
            or not np.isfinite(clearance) or clearance < 0):
        raise ValueError("cylinder_initial_clearance_m must be finite and nonnegative")
    default_position = [*DEFAULT_CYLINDER_POSITION[:2],
                        params.upright_center_height(center[2] + half_size[2], clearance)]
    cylinder_position = _vector(demo_cfg.get("cylinder_position_xyz", default_position), 3,
                                "cylinder_position_xyz")
    quaternion = _vector(demo_cfg.get("cylinder_quaternion_wxyz", [1, 0, 0, 0]), 4,
                         "cylinder_quaternion_wxyz")
    if np.any(half_size <= 0) or center[2]-half_size[2] <= 0:
        raise ValueError("Table half sizes must be positive and its underside above the floor")
    if np.linalg.norm(quaternion) < 1e-12:
        raise ValueError("Cylinder quaternion must have nonzero norm")
    quaternion /= np.linalg.norm(quaternion)
    hand_cfg = copy.deepcopy(load_config())
    hand_cfg["simulation_dt"] = float(reach_cfg["simulation_dt"])
    hand_cfg["control_dt"] = float(reach_cfg["hand_dt"])
    if (not np.isclose(hand_cfg["simulation_dt"], 0.001, rtol=0, atol=1e-12)
            or not np.isclose(hand_cfg["control_dt"], 0.01, rtol=0, atol=1e-12)):
        raise ValueError("Tabletop contact baseline requires 1 kHz physics and 100 Hz hands")
    root = ET.fromstring(build_model_xml(hand_cfg, fixture=False))
    root.set("model", "R2V2_new_asset_fixed_table_free_cylinder")
    # Cylinder/box surface support needs a contact manifold, not a single
    # convex CCD point. This keeps the original friction/solref/geometry; it
    # changes contact detection only in this standalone tabletop scene.
    ET.SubElement(root.find("option"), "flag", multiccd="enable")
    for side, position in TCP_OFFSETS.items():
        wrist = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        ET.SubElement(wrist, "site", name=f"{side}_tcp", pos=_text(position),
                      quat="1 0 0 0", size="0.008", rgba="0.1 0.8 0.3 1")
    world = root.find("worldbody")
    table = ET.SubElement(world, "body", name="tabletop", pos=_text(center))
    ET.SubElement(table, "geom", name="tabletop_geom", type="box", size=_text(half_size),
                  rgba="0.48 0.34 0.22 1", friction="1 0.005 0.0001")
    # Legs end at z=0 and at the underside of the board. These are static
    # collision geoms, not a moving support that can carry the object away.
    leg_half_width = min(0.012, half_size[0]/4, half_size[1]/4)
    leg_half_height = (center[2]-half_size[2])/2
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        leg_position = [sx*(half_size[0]-leg_half_width), sy*(half_size[1]-leg_half_width),
                        leg_half_height-center[2]]
        ET.SubElement(table, "geom", name=f"tabletop_leg_{index}", type="box",
                      pos=_text(leg_position), size=_text([leg_half_width, leg_half_width, leg_half_height]),
                      rgba="0.25 0.28 0.30 1", friction="1 0.005 0.0001")
    cylinder = ET.SubElement(world, "body", name="test_cylinder", pos=_text(cylinder_position),
                             quat=_text(quaternion))
    ET.SubElement(cylinder, "freejoint", name="cylinder_free")
    collider = ET.SubElement(cylinder, "geom", name="cylinder_geom", type="cylinder",
                  size=_text([params.radius, params.half_height]), mass=str(params.mass),
                  rgba="0.95 0.45 0.08 1", friction="1 0.005 0.0001", condim="3",
                  priority="1", solref="0.008 1", solimp="0.95 0.99 0.001")
    if appearance == "cola_can":
        from common.r2v2_can_visual import add_can_visual

        # Hide only the orange rendering; the original collider still owns
        # the complete mass, inertia and contact response of the object.
        collider.set("rgba", "0.95 0.45 0.08 0")
        add_can_visual(root, cylinder,
                       PROJECT_ROOT / "r2v2_description/visuals/cola_can/label.png")
    return ET.tostring(root, encoding="unicode"), hand_cfg


def build_tabletop_model(reach_cfg, demo_cfg):
    """Compile the unchanged tabletop scene; XML is reusable for asset previews."""
    xml, hand_cfg = build_tabletop_xml(reach_cfg, demo_cfg)
    return mujoco.MjModel.from_xml_string(xml), hand_cfg


def object_metrics(model, data):
    """Cylinder pose/twist in world coordinates, with analytic bottom height.

    Array fields: T_world_cylinder (4x4), position_m (3), quaternion_wxyz (4),
    linear_velocity_mps (3), angular_velocity_radps (3). Scalar fields:
    tilt_deg (0 upright, 180 inverted), bottom_height_m, vertical_extent_m.
    ``vertical_extent_m`` is the *half* extent, not total bounding-box height.
    For axial unit vector a: extent = half_height*abs(a_z)
    + radius*sqrt(1-a_z**2); using center_z-half_height is wrong when tilted.
    """
    body_id = model.body("test_cylinder").id
    geom_id = model.geom("cylinder_geom").id
    transform = body_transform(data, body_id)
    quaternion = data.xquat[body_id].copy()
    if quaternion[0] < 0:
        quaternion *= -1
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    axis_z = float(np.clip(data.geom_xmat[geom_id].reshape(3, 3)[2, 2], -1, 1))
    radius, half_height = model.geom_size[geom_id, :2]
    extent = float(half_height*abs(axis_z) + radius*np.sqrt(max(0.0, 1-axis_z**2)))
    return {
        "T_world_cylinder": transform,
        "position_m": transform[:3, 3].copy(),
        "quaternion_wxyz": quaternion,
        "linear_velocity_mps": jacp @ data.qvel,
        "angular_velocity_radps": jacr @ data.qvel,
        "tilt_deg": float(np.rad2deg(np.arccos(axis_z))),
        "bottom_height_m": float(data.geom_xpos[geom_id, 2]-extent),
        "vertical_extent_m": extent,
    }


def _descendant(model, body, ancestor):
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def contact_metrics(model, data, side="left"):
    """Classify solved contacts; opposite finger contact is not grasp success.

    Table/floor flags refer only to cylinder contacts. Robot-table contact is
    separately listed for safety. A palm contact or an arm collision cannot
    count as an opposite finger. Returned net contact force acts on the object
    in world coordinates; maxima/sums of normal forces are not object weight.
    """
    if side not in SIDES:
        raise ValueError(f"Unknown hand side: {side}")
    cylinder_geom = model.geom("cylinder_geom").id
    floor_geom = model.geom("floor").id
    table_body = model.body("tabletop").id
    robot_body = model.body("base_link").id
    wrist_body = model.body(f"{side}_hand_roll_link").id
    fingers = dict.fromkeys(CONTACT_PARTS, 0.0)
    table_contact = floor_contact = False
    robot_table_contacts, contacts = [], []
    hand_depth = max_normal = 0.0
    total_world_force = np.zeros(3)
    for index, contact in enumerate(data.contact):
        geom1, geom2 = map(int, contact.geom)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        table1, table2 = _descendant(model, body1, table_body), _descendant(model, body2, table_body)
        robot_table = ((table1 and _descendant(model, body2, robot_body))
                       or (table2 and _descendant(model, body1, robot_body)))
        is_object = cylinder_geom in (geom1, geom2)
        if not (is_object or robot_table):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        normal = max(0.0, float(force[0]))
        active = normal > 1e-8 or contact.dist <= 0.0
        depth = max(0.0, -float(contact.dist))
        geoms = [model.geom(geom1).name, model.geom(geom2).name]
        if robot_table and active:
            robot_table_contacts.append({"geoms": geoms, "normal_force_N": normal,
                                         "penetration_m": depth})
        if not is_object:
            continue
        other = geom2 if geom1 == cylinder_geom else geom1
        other_body = int(model.geom_bodyid[other])
        hand = _descendant(model, other_body, wrist_body)
        part = None
        if hand:
            body_name = model.body(other_body).name
            part = next((finger for finger in CONTACT_PARTS[1:] if f"_{finger}_" in body_name), "palm")
            fingers[part] += normal
            hand_depth = max(hand_depth, depth)
        # MuJoCo's positive contact force acts on geom2. Rows of frame are
        # contact axes expressed in world coordinates, hence frame.T below.
        object_force = contact.frame.reshape(3, 3).T @ force[:3]
        if geom1 == cylinder_geom:
            object_force *= -1
        total_world_force += object_force
        max_normal = max(max_normal, normal)
        table_contact |= bool(active and _descendant(model, other_body, table_body))
        floor_contact |= bool(active and other == floor_geom)
        contacts.append({"geoms": geoms, "other_geom": model.geom(other).name, "part": part,
                         "normal_force_N": normal, "force_world_N": object_force.tolist(),
                         "penetration_m": depth, "active": bool(active)})
    opposed = fingers["thumb"] > 0.05 and any(fingers[name] > 0.05 for name in CONTACT_PARTS[2:])
    return {
        "fingers_normal_force_N": fingers,
        "opposed": bool(opposed),
        "table_contact": bool(table_contact),
        "floor_contact": bool(floor_contact),
        "robot_table_contacts": robot_table_contacts,
        "max_hand_object_penetration_m": hand_depth,
        "max_object_contact_normal_force_N": max_normal,
        "object_contact_force_world_N": total_world_force,
        "contacts": contacts,
    }
