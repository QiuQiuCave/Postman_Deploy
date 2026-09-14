"""True-wrist full-body scene for replaying recorded crate-relative motion.

The robot and crate remain dynamic; only the table is fixed. No fixture,
mocap, support force or object/robot weld is introduced. The sole scene
change from the established full-body crate baseline is the versioned wrist
observation sites. Motion targets must come from the recorded trajectory,
not from the old parallel-insertion geometric calibration.
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters, add_crate
from common.r2v2_crate_reach_scene import (
    CRATE_INITIAL_TABLE_GAP_M, MINIMUM_TABLE_FRONT_X_M, _scalar, _vector,
)
from common.r2v2_reach_policy import WRIST_ENDPOINT_CONTRACT
from common.r2v2_tabletop_scene import build_tabletop_xml
from r2v2_description.model import SIDES


def build_motion_replay_model(
    reach_cfg, crate_params, *, table_top_m=1.0109189696536514,
    crate_center_xy=(.38, 0), table_half_size=(.22, .35, .02),
    table_center_xy=(.48, 0), virtual_props=False, allow_near_table=False,
):
    """Return ``(model, private_hand_cfg, layout)`` for wrist-world v2 only.

    All body inertias, joint limits, contact geometry and actuator channels
    are inherited unchanged from the new complete R2V2 model. The free crate
    starts 1 mm above the tabletop, as in the existing crate experiment.
    Layout checks certify table support only, not arm reachability or grasp.
    Explicit ``virtual_props`` creates static, invisible, noncolliding layout
    placeholders. Their outlines belong to the renderer, not to physics. The
    articulated robot, its floor contacts and self-collisions remain unchanged.
    """
    if reach_cfg.get("endpoint_contract") != WRIST_ENDPOINT_CONTRACT:
        raise ValueError("Recorded wrist replay requires explicit wrist_world_v2")
    if not isinstance(crate_params, CrateParameters):
        raise TypeError("crate_params must be CrateParameters")
    if not isinstance(virtual_props, bool):
        raise TypeError("virtual_props must be an explicit boolean")
    if not isinstance(allow_near_table, bool):
        raise TypeError("allow_near_table must be an explicit boolean")
    table_top = _scalar(table_top_m, "table_top_m")
    crate_xy = _vector(crate_center_xy, 2, "crate_center_xy")
    table_xy = _vector(table_center_xy, 2, "table_center_xy")
    half_size = _vector(table_half_size, 3, "table_half_size")
    if np.any(half_size <= 0) or table_top <= 2 * half_size[2]:
        raise ValueError("Table sizes must be positive and its underside above the floor")
    if not virtual_props and not allow_near_table and table_xy[0] - half_size[0] < MINIMUM_TABLE_FRONT_X_M - 1e-12:
        raise ValueError("Table front must be at least 0.26 m ahead of the robot origin")
    crate_half_size = np.array([crate_params.depth / 2, crate_params.width / 2])
    if np.any(np.abs(crate_xy - table_xy) + crate_half_size > half_size[:2] + 1e-12):
        raise ValueError("The full crate footprint must be supported by the fixed table")

    table_center = np.array([*table_xy, table_top - half_size[2]])
    initial_position = np.array([*crate_xy, table_top + (0. if virtual_props else CRATE_INITIAL_TABLE_GAP_M)])
    xml, hand_cfg = build_tabletop_xml(reach_cfg, {
        "object_appearance": "orange_cylinder",
        "table_center_xyz": table_center.tolist(),
        "table_half_size": half_size.tolist(),
        "cylinder_position_xyz": initial_position.tolist(),
    })
    root = ET.fromstring(xml)
    root.set("model", "R2V2_wrist_world_v2_recorded_crate_motion_replay")
    for side in SIDES:
        wrist = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        site = None if wrist is None else wrist.find(f'site[@name="{side}_tcp"]')
        if site is None or root.find(f'.//site[@name="{side}_wrist"]') is not None:
            raise RuntimeError("Tabletop wrist-site template changed; revalidate endpoint placement")
        site.set("name", f"{side}_wrist")
        site.set("pos", "0 0 0")
        site.set("quat", "1 0 0 0")
    world = root.find("worldbody")
    cylinder = world.find('body[@name="test_cylinder"]')
    table_geom = world.find('body[@name="tabletop"]/geom[@name="tabletop_geom"]')
    if cylinder is None or table_geom is None:
        raise RuntimeError("Tabletop template lacks its expected removable cylinder or table")
    world.remove(cylinder)
    table_geom.set("name", "lift_table_geom")
    add_crate(root, crate_params, position=initial_position, free=not virtual_props)
    if virtual_props:
        # No dynamic crate to fall through a noncolliding table. Static bodies
        # have no connections to the robot and cannot support or move it.
        for body_name in ("tabletop", "cargo_crate"):
            for geom in world.find(f'body[@name="{body_name}"]').iter("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("rgba", "0 0 0 0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    expected_dimensions = (57, 56, 40, 10, 0) if virtual_props else (64, 62, 40, 10, 0)
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != expected_dimensions:
        raise RuntimeError("Expected unchanged full-body R2V2 and 10 hand mimics; crate free only in physical mode")
    if np.any(model.eq_type != mujoco.mjtEq.mjEQ_JOINT):
        raise RuntimeError("Only original finger mimic joint equalities are allowed")
    for side in SIDES:
        site = model.site(f"{side}_wrist")
        if (int(site.bodyid[0]) != model.body(f"{side}_hand_roll_link").id
                or not np.array_equal(site.pos, np.zeros(3))
                or not np.array_equal(site.quat, np.array([1., 0., 0., 0.]))):
            raise RuntimeError("Wrist endpoint must exactly match its link origin and axes")
    layout = {
        "table_top_m": table_top,
        "table_center_xyz": table_center.copy(),
        "table_center_xy": table_xy.copy(),
        "table_half_size": half_size.copy(),
        "crate_initial_position": initial_position.copy(),
        "crate_center_xy": crate_xy.copy(),
        "driver": "dual_arm_reach_policy_no_wrist_fixtures",
        "endpoint_contract": WRIST_ENDPOINT_CONTRACT,
        "target_source": "recorded_crate_relative_wrist_motion",
        "virtual_props": virtual_props,
        "allow_near_table": allow_near_table,
        "near_table_note": "Explicit exploratory layout waiver only; all contact physics remain active" if allow_near_table else None,
        "prop_physics": "static_noncolliding_render_only_outlines" if virtual_props else "physical_table_and_free_crate",
    }
    return model, hand_cfg, layout


__all__ = ["build_motion_replay_model"]
