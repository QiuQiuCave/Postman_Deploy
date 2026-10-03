"""Exact archived front-Reach scene with either real or visual-only props.

The full source robot remains dynamic and unmodified. Contact mode adds two
free rigid objects, not grasp welds, prescribed wrists, or auxiliary forces.
Air mode retains the same robot but hides/disables and fixes the props so a
caller can draw an explicitly labelled wireframe Reach-only preview.
"""

import copy
from collections.abc import Mapping
from itertools import product
from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters, add_crate
from common.r2v2_cylinder_test import load_cylinder_profile
from common.r2v2_tabletop_scene import build_tabletop_xml
from r2v2_description.model import SIDES


def _scalar(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _vector(value, size, label, *, positive=False):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {size} finite numbers")
    result = np.asarray(value, dtype=object)
    if result.shape != (size,):
        raise ValueError(f"{label} must contain {size} finite numbers")
    return np.array([_scalar(item, label, positive=positive) for item in result])


def _quaternion(value, label):
    quat = _vector(value, 4, label)
    scale = np.max(np.abs(quat))
    if scale == 0:
        raise ValueError(f"{label} must have nonzero norm")
    quat /= scale
    return quat / np.linalg.norm(quat)


def _world_bounds(bounds, position, quaternion):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    corners = np.array(list(product(*zip(bounds["min"], bounds["max"]))))
    world = corners @ rotation.reshape(3, 3).T + position
    return {"min": world.min(axis=0), "max": world.max(axis=0)}


def _build_front_drop_model(reach_cfg, scene, mode="contact", *, diagnostic_prop_contacts=False):
    """Return ``(MjModel, hand_cfg, layout)`` without mutating caller data.

    ``scene`` is the archived path manifest's ``provenance.scene`` mapping;
    every placement/dimension comes from it, with no implicit repositioning.
    ``mode='contact'`` creates a free can and crate above a fixed table.
    ``mode='air'`` creates hidden, contact-disabled static props. In both
    modes the body's 28 and hands' 12 torque channels remain independent,
    source hand mimics remain the only equalities, and the true wrist sites
    ``left_wrist`` / ``right_wrist`` replace the obsolete offset TCP sites.

    Bounds named ``*_world_m`` are INITIAL axis-aligned bounding boxes, not
    live measurements of a moving crate. For landing tests transform the
    can into the current crate frame and use ``*_local_m``. Ordinary cavity
    bounds and the narrower top opening (handle beams) are both reported.
    """
    if mode not in ("air", "contact"):
        raise ValueError("mode must be 'air' or 'contact'")
    if not isinstance(scene, Mapping):
        raise ValueError("scene must be the archived scene mapping")
    raw = copy.deepcopy(dict(scene))
    table_center = _vector(raw["table_center_world_m"], 3, "table_center_world_m")
    table_half = _vector(raw["table_half_size_m"], 3, "table_half_size_m", positive=True)
    table_top = _scalar(raw["tabletop_height_m"], "tabletop_height_m", positive=True)
    if not np.isclose(table_center[2] + table_half[2], table_top, rtol=0, atol=1e-9):
        raise ValueError("tabletop_height_m disagrees with table geometry")
    can_position = _vector(raw["can_initial_world_m"], 3, "can_initial_world_m")
    can_quat = _quaternion(raw["can_initial_quaternion_wxyz"], "can_initial_quaternion_wxyz")
    diameter, height, mass = _vector(raw["can_dimensions_diameter_height_mass"], 3,
                                    "can_dimensions_diameter_height_mass", positive=True)
    crate_position = _vector(raw["crate_base_world_m"], 3, "crate_base_world_m")
    crate_quat = _quaternion(raw["crate_quaternion_wxyz"], "crate_quaternion_wxyz")
    dimensions = _vector(raw["crate_dimensions_depth_width_height_m"], 3,
                         "crate_dimensions_depth_width_height_m", positive=True)
    opening = _vector(raw["crate_handle_opening_width_height_m"], 2,
                      "crate_handle_opening_width_height_m", positive=True)
    crate = CrateParameters(
        depth=dimensions[0], width=dimensions[1], height=dimensions[2],
        wall_thickness=raw["crate_wall_thickness_m"],
        bottom_thickness=raw["crate_bottom_thickness_m"],
        handle_opening_width=opening[0], handle_opening_height=opening[1],
        mass=raw["crate_mass_kg"],
    )
    profile = load_cylinder_profile()
    if (diameter / 2, height, mass) != (profile["radius_m"], profile["height_m"], profile["mass_kg"]):
        profile.update(profile_id="archived_front_drop_scene", radius_m=diameter / 2,
                       height_m=height, mass_kg=mass, grasp_calibration_status="unvalidated",
                       geometry_source="Archived front-manipulation path provenance.scene",
                       mass_provenance="Archived simulation scene specification")
    xml, hand_cfg = build_tabletop_xml(reach_cfg, {
        "table_center_xyz": table_center, "table_half_size": table_half,
        "cylinder_position_xyz": can_position, "cylinder_quaternion_wxyz": can_quat,
        "cylinder_profile": profile, "object_appearance": "cola_can",
    })
    root = ET.fromstring(xml)
    root.set("model", f"R2V2_front_drop_{mode}_full_body_true_wrist")
    for side in SIDES:
        wrist = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        tcp = wrist.find(f'site[@name="{side}_tcp"]')
        if tcp is None:
            raise RuntimeError("Tabletop builder's expected legacy TCP site is missing")
        wrist.remove(tcp)
        ET.SubElement(wrist, "site", name=f"{side}_wrist", pos="0 0 0", quat="1 0 0 0",
                      size="0.005", rgba="0.1 0.8 0.3 0")
    add_crate(root, crate, position=crate_position, quaternion=crate_quat, free=mode == "contact")
    if mode == "air":
        world = root.find("worldbody")
        for name in ("tabletop", "test_cylinder", "cargo_crate"):
            body = world.find(f'body[@name="{name}"]')
            for joint in body.findall("freejoint"):
                body.remove(joint)
            for geom in body.iter("geom"):
                if not diagnostic_prop_contacts:
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
                # Materials can supply their own opaque RGBA. Remove only
                # prop material references so alpha zero is unambiguous.
                geom.attrib.pop("material", None)
                geom.set("rgba", "0 0 0 0")
            for site in body.iter("site"):
                site.set("rgba", "0 0 0 0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    expected = (71, 68, 40, 10, 0) if mode == "contact" else (57, 56, 40, 10, 0)
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != expected:
        raise RuntimeError("Unexpected full-robot/prop state dimensions")
    if np.any(model.eq_type != mujoco.mjtEq.mjEQ_JOINT):
        raise RuntimeError("Only the original finger mimic equalities are permitted")
    interior = {
        "min": np.array([-crate.inner_depth / 2, -crate.inner_width / 2, crate.bottom_thickness]),
        "max": np.array([crate.inner_depth / 2, crate.inner_width / 2, crate.height]),
    }
    top_opening = {"min": np.array([-crate.inner_depth / 2,
                                    -crate.width / 2 + crate.handle_beam_thickness, crate.height]),
                   "max": np.array([crate.inner_depth / 2,
                                    crate.width / 2 - crate.handle_beam_thickness, crate.height])}
    layout = {
        "scene": raw, "mode": mode, "driver": "reach_policy_and_independent_binary_hands",
        "object_is_free": mode == "contact", "object_welds": False,
        "props_contact_enabled": mode == "contact" or diagnostic_prop_contacts,
        "diagnostic_only": bool(diagnostic_prop_contacts),
        "wrist_sites": {s: f"{s}_wrist" for s in SIDES},
        "table_top_m": table_top, "table_center_xyz": table_center.copy(),
        "table_half_size": table_half.copy(), "cylinder_initial_position": can_position.copy(),
        "cylinder_initial_quaternion_wxyz": can_quat.copy(), "cylinder_profile": copy.deepcopy(profile),
        "crate_initial_position": crate_position.copy(), "crate_quaternion_wxyz": crate_quat.copy(),
        "crate_dimensions_depth_width_height_m": dimensions.copy(),
        "crate_handle_opening_width_height_m": opening.copy(),
        "crate_interior_bounds_local_m": interior,
        "crate_interior_bounds_world_m": _world_bounds(interior, crate_position, crate_quat),
        "crate_opening_bounds_local_m": top_opening,
        "crate_opening_bounds_world_m": _world_bounds(top_opening, crate_position, crate_quat),
    }
    return model, hand_cfg, layout


def build_front_drop_model(reach_cfg, scene, mode="contact"):
    """Build ``(model, hand_cfg, layout)`` for real contact or wire-only AIR.

    Source robot physics is unchanged. Contact props have genuine free
    joints; AIR props are hidden, static and non-colliding. Positions and
    physical dimensions are taken exactly from the archived scene mapping.
    """
    return _build_front_drop_model(reach_cfg, scene, mode)


def build_front_drop_collision_diagnostic(reach_cfg, scene):
    """Compile a separate static-prop geometry model matching AIR state layout.

    Use only a private ``MjData`` with ``mj_forward``; never step it or use
    its contact response to control the live AIR experiment. It has AIR's
    57-qpos / 56-velocity layout and identical robot and geom names, but
    prop collision flags are enabled *before compilation*. Toggling only
    ``geom_contype`` afterwards is insufficient: MuJoCo also compiles body
    aggregate contact masks and collision BVHs, which AIR omits entirely.
    Original visual-only can skin geoms remain non-colliding.
    """
    return _build_front_drop_model(reach_cfg, scene, "air", diagnostic_prop_contacts=True)[0]


__all__ = ["build_front_drop_model", "build_front_drop_collision_diagnostic"]
