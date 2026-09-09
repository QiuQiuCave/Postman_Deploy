"""Two moving wrist fixtures, real dynamic fingers, and one free cargo crate.

This is a prescribed *wrist-fixture* experiment, not whole-body Reach control.
Two massless mocap targets drive independent dynamic free-joint wrists through
wrist-only weld constraints. Real wrist velocities therefore enter finger
contact Jacobians, unlike directly translating a mocap parent of the hand.
The original hand subtrees keep their masses, inertias, collision geometry,
actuators and mimic joints; the crate receives no weld or auxiliary force.

Placement deliberately reuses the static preview's private geometry helpers
``_nominal_open_geometry``, ``_wrist_rotation`` and ``_local_part_clouds`` so
the insertion definition and palm-up frame cannot drift from that preview.
"""

from common.path_config import PROJECT_ROOT

import copy
from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters, add_crate, load_crate_config
from common.r2v2_crate_hand_preview import (
    NOMINAL_OPEN_CURL_RAD, _local_part_clouds, _nominal_open_geometry, _wrist_rotation,
)
from r2v2_description.model import SIDES, build_model_xml, initialize_hands, load_config


MINIMUM_INITIAL_TIP_CLEARANCE_M = 0.015
CRATE_INITIAL_TABLE_GAP_M = 0.001
TABLE_HALF_SIZE = (0.25, 0.45, 0.02)
WRIST_WELD_SOLREF = (0.005, 1.0)
WRIST_WELD_SOLIMP = (0.95, 0.99, 0.001)
WRIST_WELD_TORQUESCALE_M = 1.0


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _text(values):
    return " ".join(format(float(value), ".17g") for value in values)


def build_crate_lift_model(crate_params=None, insertion_m=0.06,
                           start_palm_clearance_m=0.08, table_height=0.4):
    """Return ``(model, hand_cfg, layout)``; do not run a grasp or simulation.

    ``insertion_m`` is the shortest nominal-open fingertip's depth beyond
    the ordinary sidewall's inner face, exactly as in ``build_hand_preview``.
    The initial palm clearance is exact, not an automatically enlarged value;
    inputs leaving any leading hand geometry less than 15 mm outside the
    crate are rejected. Wrist poses use the *resting* crate bottom at the
    table top. The free crate alone starts 1 mm higher to settle naturally.

    ``layout`` contains left/right dictionaries ``initial_wrist_positions``,
    ``inserted_wrist_positions``, ``wrist_quaternions`` (wxyz), ``mocap_ids``,
    ``initial_leading_finger_clearance_m`` and ``initial_palm_clearance_m``;
    ``table_top_m`` and ``crate_initial_position`` are world coordinates.
    The caller must initialize hand qpos using ``initialize_hands`` and then
    drive the returned mocap target IDs. Actual wrists are independent free
    bodies; their qpos initializes from XML to the same world pose as their
    targets. ``wrist_free_joints`` and ``wrist_welds`` map sides to names, and
    ``driver`` / ``wrist_weld_parameters`` record this fixture architecture.
    The caller must measure actual wrists, not mocap targets. All returned
    arrays/configs are private.
    No default hand pose/profile is changed; only a copied simulation_dt is
    set to 1 ms and copied control_dt to 10 ms.
    """
    p = load_crate_config() if crate_params is None else crate_params
    if not isinstance(p, CrateParameters):
        raise TypeError("crate_params must be CrateParameters or None")
    insertion_m = _scalar(insertion_m, "insertion_m")
    clearance = _scalar(start_palm_clearance_m, "start_palm_clearance_m")
    table_height = _scalar(table_height, "table_height")
    if not 0 <= insertion_m <= 0.10:
        raise ValueError("insertion_m must be between 0 and 0.10 m")
    if clearance <= 0:
        raise ValueError("start_palm_clearance_m must be positive")
    if table_height <= 2 * TABLE_HALF_SIZE[2]:
        raise ValueError("table_height must leave the 4 cm table slab above the floor")
    if p.depth >= 2 * TABLE_HALF_SIZE[0] or p.width >= 2 * TABLE_HALF_SIZE[1]:
        raise ValueError("Crate footprint must fit on the fixed table slab")

    hand_cfg = copy.deepcopy(load_config())
    hand_cfg["simulation_dt"] = 0.001
    hand_cfg["control_dt"] = 0.01
    root = ET.fromstring(build_model_xml(hand_cfg, fixture=True))
    root.set("model", "R2V2_dynamic_weld_tracked_wrists_hands_free_crate")
    world = root.find("worldbody")
    # Measure actual source collision vertices at the unchanged default open
    # pose, including the thumb and the longest of the four fingertips.
    baseline = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    baseline_data = mujoco.MjData(baseline)
    initialize_hands(baseline, baseline_data, hand_cfg)
    layout = {
        "initial_wrist_positions": {}, "inserted_wrist_positions": {},
        "wrist_quaternions": {}, "mocap_ids": {},
        "driver": "mocap_targets_with_dynamic_free_wrist_welds",
        "wrist_free_joints": {}, "wrist_welds": {},
        "wrist_weld_parameters": {"solref": list(WRIST_WELD_SOLREF),
                                  "solimp": list(WRIST_WELD_SOLIMP),
                                  "torquescale_m": WRIST_WELD_TORQUESCALE_M},
        "initial_leading_finger_clearance_m": {}, "initial_palm_clearance_m": {},
        "table_top_m": table_height,
        "crate_initial_position": np.array([0., 0., table_height + CRATE_INITIAL_TABLE_GAP_M]),
    }
    hole_center_z = (p.handle_opening_bottom + p.handle_opening_top) / 2
    for side in SIDES:
        open_values = np.asarray(hand_cfg["hands"][side]["open"], dtype=float)
        if not np.allclose(open_values[2:], NOMINAL_OPEN_CURL_RAD, rtol=0, atol=1e-12):
            raise ValueError("Default open finger pose differs from the static insertion calibration")
        nominal = _nominal_open_geometry(side, float(np.rad2deg(open_values[0])))
        parts = _local_part_clouds(baseline, baseline_data, side)
        front = max(float(cloud[:, 0].max()) for cloud in parts.values())
        initial_wall = nominal["palm_front_x"] + clearance
        inserted_wall = nominal["shortest_tip_x"] - p.wall_thickness - insertion_m
        tip_clearance = initial_wall - front
        if tip_clearance < MINIMUM_INITIAL_TIP_CLEARANCE_M - 1e-12:
            minimum = front - nominal["palm_front_x"] + MINIMUM_INITIAL_TIP_CLEARANCE_M
            raise ValueError(f"{side} initial fingertip clearance is below 15 mm; "
                             f"start_palm_clearance_m must be at least {minimum:.6f}")
        if initial_wall <= inserted_wall:
            raise ValueError(f"{side} initial wrist must be outside the inserted position")
        if front - inserted_wall >= p.width / 2:
            raise ValueError("Insertion would bring the opposing hand geometries across the crate midplane")
        rotation = _wrist_rotation(side)
        sign = 1 if side == "left" else -1
        wrist_z = table_height + hole_center_z - rotation[2, 1] * nominal["four_finger_y_center"]
        initial = np.array([0., sign * (p.width/2 + initial_wall), wrist_z])
        inserted = np.array([0., sign * (p.width/2 + inserted_wall), wrist_z])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        layout["initial_wrist_positions"][side] = initial
        layout["inserted_wrist_positions"][side] = inserted
        layout["wrist_quaternions"][side] = quaternion
        layout["initial_leading_finger_clearance_m"][side] = tip_clearance
        layout["initial_palm_clearance_m"][side] = clearance

        old_support = world.find(f'body[@name="{side}_fixture"]')
        if old_support is None:
            raise ValueError(f"Source fixture is missing {side}_fixture")
        world.remove(old_support)
        wrist = world.find(f'body[@name="{side}_hand_roll_link"]')
        if wrist is None or wrist.findall("joint") or wrist.findall("freejoint"):
            raise ValueError(f"Expected a jointless {side} wrist fixture root")
        world.remove(wrist)
        # A directly mocap-parented wrist has no simulated root velocity.
        # Instead, preserve its actual inertia and make the whole real hand
        # an independent free body. Only its target is kinematic/massless.
        wrist.set("pos", _text(initial))
        wrist.set("quat", _text(quaternion))
        free_joint = f"{side}_wrist_free"
        weld_name = f"{side}_wrist_tracking_weld"
        target_name = f"{side}_wrist_fixture"
        ET.SubElement(wrist, "freejoint", name=free_joint)
        ET.SubElement(world, "body", name=target_name, mocap="true",
                      pos=_text(initial), quat=_text(quaternion))
        world.append(wrist)
        # Identical initial poses and explicit identity relpose eliminate any
        # inherited constraint offset. Five dt of time constant is resolved
        # at 1 kHz. MuJoCo's standard 1 m torque scale keeps orientation
        # clamped; a 0.1 m scale was too compliant under hand gravity load.
        ET.SubElement(root.find("equality"), "weld", name=weld_name,
                      body1=f"{side}_hand_roll_link", body2=target_name,
                      relpose="0 0 0 1 0 0 0", solref=_text(WRIST_WELD_SOLREF),
                      solimp=_text(WRIST_WELD_SOLIMP), torquescale=str(WRIST_WELD_TORQUESCALE_M))
        layout["wrist_free_joints"][side] = free_joint
        layout["wrist_welds"][side] = weld_name

    option = root.find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "enable")
    table = ET.SubElement(world, "body", name="lift_table", pos=_text((0, 0, table_height-TABLE_HALF_SIZE[2])))
    ET.SubElement(table, "geom", name="lift_table_geom", type="box", size=_text(TABLE_HALF_SIZE),
                  rgba="0.48 0.34 0.22 1", group="0", contype="1", conaffinity="1",
                  friction="1 0.005 0.0001")
    add_crate(root, p, position=layout["crate_initial_position"], free=True)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for side in SIDES:
        mocap_id = int(model.body(f"{side}_wrist_fixture").mocapid[0])
        if mocap_id < 0:
            raise RuntimeError(f"{side} wrist fixture was not compiled as a mocap body")
        layout["mocap_ids"][side] = mocap_id
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (43, 40, 12, 12, 2):
        raise RuntimeError("Lift fixture requires two dynamic free wrists, 12 hand motors, "
                           "10 mimic joints and exactly two wrist-only tracking welds")
    return model, hand_cfg, layout


__all__ = ["build_crate_lift_model"]
