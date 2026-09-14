"""Unmodified full R2V2 robot, a fixed table, and a freely moving cargo crate.

The wrist transforms below are Reach *commands*, never prescribed motion of
the robot or object. This scene has no mocap targets, free wrists, grasp welds,
or auxiliary object forces. Its only equalities are the original finger
mimic constraints.
"""

from common.path_config import PROJECT_ROOT

from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters, add_crate
from common.r2v2_crate_hand_preview import (
    NOMINAL_OPEN_CURL_RAD, _nominal_open_geometry, _wrist_rotation,
)
from common.r2v2_tabletop_scene import build_tabletop_xml
from r2v2_description.model import SIDES, load_config


CRATE_INITIAL_TABLE_GAP_M = 0.001
MINIMUM_TABLE_FRONT_X_M = 0.26


def _scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _vector(value, size, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result.copy()


def crate_wrist_targets(crate_params, *, table_top_m, crate_center_xy,
                        insertion_m=0.060, start_palm_clearance_m=0.080):
    """Return initial/inserted 4×4 world wrist transforms for both hands.

    The shortest nominal-open fingertip extends ``insertion_m`` beyond the
    ordinary sidewall's inner face. Initial placement measures palm-to-wall
    clearance; it is not an instruction to spawn a robot wrist there. Both
    palms face upward and local +X points inward through its handle opening.
    Targets assume the crate has settled upright onto ``table_top_m``.

    Calibration uses the same source one-hand collision geometry and 75°
    thumb/open-finger pose as the independent lifting experiment, without
    constructing that two-wrist fixture or its welds. The returned arrays
    are independent of the cached calibration and of caller-owned inputs.
    This geometric helper cannot certify full-body reachability or dynamics.
    """
    if not isinstance(crate_params, CrateParameters):
        raise TypeError("crate_params must be CrateParameters")
    p = crate_params
    table_top = _scalar(table_top_m, "table_top_m")
    crate_xy = _vector(crate_center_xy, 2, "crate_center_xy")
    insertion = _scalar(insertion_m, "insertion_m")
    clearance = _scalar(start_palm_clearance_m, "start_palm_clearance_m")
    if table_top <= 0:
        raise ValueError("table_top_m must be above the floor")
    if not 0 <= insertion <= 0.10:
        raise ValueError("insertion_m must be between 0 and 0.10 m")
    if clearance <= 0:
        raise ValueError("start_palm_clearance_m must be positive")

    hand_cfg = load_config()
    initial, inserted = {}, {}
    hole_center_z = (p.handle_opening_bottom + p.handle_opening_top) / 2
    for side in SIDES:
        open_values = np.asarray(hand_cfg["hands"][side]["open"], dtype=float)
        if not np.allclose(open_values[2:], NOMINAL_OPEN_CURL_RAD, rtol=0, atol=1e-12):
            raise ValueError("Default open fingers differ from the insertion calibration")
        nominal = _nominal_open_geometry(side, float(np.rad2deg(open_values[0])))
        initial_wall = nominal["palm_front_x"] + clearance
        inserted_wall = nominal["shortest_tip_x"] - p.wall_thickness - insertion
        if initial_wall <= inserted_wall:
            raise ValueError(f"{side} initial wrist must be outside its inserted target")
        if inserted_wall <= 0:
            raise ValueError("Insertion would move the wrist origin inside the sidewall")
        rotation = _wrist_rotation(side)
        sign = 1 if side == "left" else -1
        wrist_z = table_top + hole_center_z - rotation[2, 1] * nominal["four_finger_y_center"]
        for transforms, wall in ((initial, initial_wall), (inserted, inserted_wall)):
            target = np.eye(4)
            target[:3, :3] = rotation
            target[:3, 3] = [crate_xy[0], crate_xy[1] + sign * (p.width / 2 + wall), wrist_z]
            transforms[side] = target
    return {"initial_wrist_transforms": initial, "inserted_wrist_transforms": inserted}


def build_crate_reach_model(reach_cfg, crate_params, *, table_top_m,
                            crate_center_xy=(0.43, 0), table_half_size=(0.22, 0.35, 0.02),
                            table_center_xy=(0.48, 0)):
    """Compile ``(model, hand_cfg, layout)`` with the original full robot.

    The fixed tabletop must contain the complete axis-aligned crate footprint
    and its near X edge must stay at or beyond 0.26 m. These placement checks
    do not assert that a dynamic robot cannot hit the table: the experiment
    must still monitor all real robot/table contacts. Hand controller config
    is the private copy supplied by ``build_tabletop_xml``; no grasp targets,
    actuator limits, inertias, collision geoms, or body joint limits change.
    """
    if not isinstance(crate_params, CrateParameters):
        raise TypeError("crate_params must be CrateParameters")
    table_top = _scalar(table_top_m, "table_top_m")
    crate_xy = _vector(crate_center_xy, 2, "crate_center_xy")
    table_xy = _vector(table_center_xy, 2, "table_center_xy")
    half_size = _vector(table_half_size, 3, "table_half_size")
    if np.any(half_size <= 0) or table_top <= 2 * half_size[2]:
        raise ValueError("Table sizes must be positive and its underside above the floor")
    if table_xy[0] - half_size[0] < MINIMUM_TABLE_FRONT_X_M - 1e-12:
        raise ValueError("Table front must be at least 0.26 m ahead of the robot origin")
    crate_half_size = np.array([crate_params.depth / 2, crate_params.width / 2])
    if np.any(np.abs(crate_xy - table_xy) + crate_half_size > half_size[:2] + 1e-12):
        raise ValueError("The full crate footprint must be supported by the fixed table")

    table_center = np.array([*table_xy, table_top - half_size[2]])
    initial_position = np.array([*crate_xy, table_top + CRATE_INITIAL_TABLE_GAP_M])
    xml, hand_cfg = build_tabletop_xml(reach_cfg, {
        "object_appearance": "orange_cylinder",
        "table_center_xyz": table_center.tolist(),
        "table_half_size": half_size.tolist(),
        # Only required by the reused builder; this cylinder is removed below.
        "cylinder_position_xyz": initial_position.tolist(),
    })
    root = ET.fromstring(xml)
    root.set("model", "R2V2_full_body_reach_fixed_table_free_crate")
    world = root.find("worldbody")
    cylinder = world.find('body[@name="test_cylinder"]')
    if cylinder is None:
        raise RuntimeError("Tabletop template did not contain the expected removable cylinder")
    world.remove(cylinder)
    table_geom = world.find('body[@name="tabletop"]/geom[@name="tabletop_geom"]')
    if table_geom is None:
        raise RuntimeError("Tabletop template did not contain the expected tabletop geometry")
    table_geom.set("name", "lift_table_geom")
    add_crate(root, crate_params, position=initial_position, free=True)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (64, 62, 40, 10, 0):
        raise RuntimeError("Expected unchanged full-body R2V2, 10 hand mimics and one free crate")
    if np.any(model.eq_type != mujoco.mjtEq.mjEQ_JOINT):
        raise RuntimeError("Only the original finger mimic joint equalities are allowed")
    targets = crate_wrist_targets(crate_params, table_top_m=table_top, crate_center_xy=crate_xy)
    layout = {
        **targets,
        "table_top_m": table_top,
        "table_center_xyz": table_center.copy(),
        "table_center_xy": table_xy.copy(),
        "table_half_size": half_size.copy(),
        "crate_initial_position": initial_position.copy(),
        "crate_center_xy": crate_xy.copy(),
        "driver": "dual_arm_reach_policy_no_wrist_fixtures",
    }
    return model, hand_cfg, layout


__all__ = ["crate_wrist_targets", "build_crate_reach_model"]
