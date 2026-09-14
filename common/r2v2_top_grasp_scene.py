"""Real-hand, free-cylinder scenes for top-entry grasp calibration.

Only the wrists are tracked by kinematic targets. Each wrist itself remains
a dynamic free body with the source inertia; the cylinder is never attached
to a target, welded, or externally forced. Geometry records are static FK
diagnostics, not grasp-success claims.
"""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict, dataclass
from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_can_visual import add_can_visual
from common.r2v2_crate_hand_preview import _hand_geoms, _local_part_clouds
from common.r2v2_crate_lift_scene import (
    WRIST_WELD_SOLIMP, WRIST_WELD_SOLREF, WRIST_WELD_TORQUESCALE_M,
)
from common.r2v2_cylinder_test import load_cylinder_profile
from r2v2_description.model import SIDES, build_model_xml, initialize_hands, load_config


TABLE_HALF_SIZE = (0.35, 0.30, 0.02)
CYLINDER_INITIAL_CLEARANCE_M = 0.001
INITIAL_WRIST_RAISE_M = 0.10


def _scalar(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class TopGraspCandidate:
    """A top-entry pose; angles are degrees and all offsets are metres.

    ``depth_m`` is the depth below the cylinder top of the furthest source
    thumb collision vertex along local +X at the nominal open posture. It
    is a *nominal* geometric reference, not a contact location. Tilt rotates
    the complete hand about that reference point; the object stays upright.
    ``lateral_m`` increases the palm-to-cylinder gap along local mirrored Y.
    ``across_m`` locates the cylinder axis along wrist local Z (finger spread).
    """

    side: str = "left"
    depth_m: float = 0.0
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    spread_tilt_deg: float = 0.0
    lateral_m: float = 0.0
    across_m: float = 0.012
    open_curl_rad: float = 0.03

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError("side must be left or right")
        for key in ("depth_m", "yaw_deg", "tilt_deg", "spread_tilt_deg", "lateral_m", "across_m", "open_curl_rad"):
            _scalar(getattr(self, key), key)
        if not -0.08 <= self.depth_m <= 0.04:
            raise ValueError("depth_m must be between -0.08 and 0.04 m")
        if not 0.03 <= self.open_curl_rad <= 0.60:
            raise ValueError("open_curl_rad must be between 0.03 and 0.60 rad")
        if max(abs(self.tilt_deg), abs(self.spread_tilt_deg)) > 60 or abs(self.yaw_deg) > 180:
            raise ValueError("Top-entry tilt components must be <= 60 degrees; yaw <= 180 degrees")
        if np.cos(np.deg2rad(self.tilt_deg))*np.cos(np.deg2rad(self.spread_tilt_deg)) < .5-1e-12:
            raise ValueError("Combined inclination from downward must be <= 60 degrees")
        if abs(self.lateral_m) > 0.04 or abs(self.across_m) > 0.05:
            raise ValueError("Candidate alignment offsets exceed the calibration window")


def _text(values):
    return " ".join(format(float(value), ".17g") for value in values)


def top_wrist_rotation(candidate):
    """Right-handed wrist rotation with +X fingers down at zero tilt.

    At zero angles, left palm closure is toward world -Y and right toward
    +Y. Positive tilt inclines extension away from that palm-closure direction
    (toward +world Y for the left hand, -world Y for the right before yaw).
    Spread tilt rotates about world Y before yaw, inclining along the finger
    spread / thumb side instead of the flexion plane. These two inclinations
    are physically distinct; yaw alone is redundant for an isolated cylinder.
    """
    side = 1.0 if candidate.side == "left" else -1.0
    base = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    yaw, tilt = np.deg2rad([candidate.yaw_deg, candidate.tilt_deg * side])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                   [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(tilt), -np.sin(tilt)],
                   [0., np.sin(tilt), np.cos(tilt)]])
    spread = np.deg2rad(candidate.spread_tilt_deg)
    ry = np.array([[np.cos(spread), 0., np.sin(spread)], [0., 1., 0.],
                   [-np.sin(spread), 0., np.cos(spread)]])
    return rz @ ry @ rx @ base


def _contacts(model, data, hand, cylinder, table):
    groups = {"hand_cylinder": [], "hand_table": [], "hand_self": []}
    for contact in data.contact:
        pair = set(map(int, contact.geom))
        group = ("hand_cylinder" if pair & hand and cylinder in pair else
                 "hand_table" if pair & hand and table in pair else
                 "hand_self" if pair <= hand else None)
        if group is not None and contact.dist <= 0:
            groups[group].append({
                "geoms": [model.geom(int(g)).name for g in contact.geom],
                "distance_m": float(contact.dist),
                "penetration_m": max(0., -float(contact.dist)),
            })
    return groups


def build_top_grasp_model(profile=None, candidate=None, table_height=0.4):
    """Return ``(model, hand_cfg, layout)`` without performing any simulation.

    The active hand starts exactly 10 cm above its candidate target. A second
    hand is retained with identical controls and parked more than 1 m away.
    Inputs leaving the initial hand intersecting table/object are rejected;
    nominal target collisions are recorded unchanged for candidate rejection.
    Only a private ``MjData`` is positioned for this static diagnostic. The
    caller's experiment must never reset the free object's runtime state.
    """
    profile = load_cylinder_profile(profile)
    candidate = TopGraspCandidate() if candidate is None else candidate
    if not isinstance(candidate, TopGraspCandidate):
        raise TypeError("candidate must be TopGraspCandidate or None")
    table_height = _scalar(table_height, "table_height")
    if table_height <= 2 * TABLE_HALF_SIZE[2]:
        raise ValueError("Table slab must be above the floor")
    radius, height = profile["radius_m"], profile["height_m"]
    cfg = copy.deepcopy(load_config())
    cfg["simulation_dt"], cfg["control_dt"] = .001, .01
    # Candidate-only four-finger pre-curl; retain the 75-degree thumb opposition
    # and its opening angle, closing targets, motor limits and trajectory.
    cfg["hands"][candidate.side]["open"][2:] = [candidate.open_curl_rad] * 4
    root = ET.fromstring(build_model_xml(cfg, fixture=True))
    root.set("model", "R2V2_top_grasp_dynamic_wrist_free_cylinder")
    world, equality = root.find("worldbody"), root.find("equality")
    baseline = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    baseline_data = mujoco.MjData(baseline)
    initialize_hands(baseline, baseline_data, cfg)
    parts = _local_part_clouds(baseline, baseline_data, candidate.side)
    thumb_forward = float(parts["thumb"][:, 0].max())
    sign = 1. if candidate.side == "left" else -1.
    reference_local = np.array([thumb_forward,
                                -sign * (radius + .030 + candidate.lateral_m),
                                candidate.across_m])
    rotation = top_wrist_rotation(candidate)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    reference_world = np.array([0., 0., table_height + height - candidate.depth_m])
    grasp_position = reference_world - rotation @ reference_local
    cylinder_initial = np.array([0., 0., table_height + height / 2 + CYLINDER_INITIAL_CLEARANCE_M])
    layout = {
        "candidate": asdict(candidate), "cylinder_profile": copy.deepcopy(profile),
        "table_top_m": table_height, "table_half_size_m": list(TABLE_HALF_SIZE),
        "cylinder_initial_position": cylinder_initial,
        "grasp_wrist_position": grasp_position,
        "wrist_quaternion": quaternion,
        "wrist_quaternions": {}, "initial_wrist_positions": {}, "mocap_ids": {},
        "wrist_free_joints": {}, "wrist_welds": {},
        "driver": "mocap_targets_with_dynamic_free_wrist_welds",
        "wrist_weld_parameters": {"solref": list(WRIST_WELD_SOLREF),
                                  "solimp": list(WRIST_WELD_SOLIMP),
                                  "torquescale_m": WRIST_WELD_TORQUESCALE_M},
    }
    for side in SIDES:
        world.remove(world.find(f'body[@name="{side}_fixture"]'))
        wrist = world.find(f'body[@name="{side}_hand_roll_link"]')
        if wrist is None or wrist.findall("joint") or wrist.findall("freejoint"):
            raise ValueError("Expected untouched jointless fixture wrist")
        initial = (grasp_position + [0, 0, INITIAL_WRIST_RAISE_M]
                   if side == candidate.side else np.array([1.2, 1.2 if side == "left" else -1.2, table_height+.45]))
        quat = quaternion.copy() if side == candidate.side else np.array([1., 0., 0., 0.])
        wrist.set("pos", _text(initial))
        wrist.set("quat", _text(quat))
        free_name, target_name = f"{side}_wrist_free", f"{side}_wrist_fixture"
        weld_name = f"{side}_wrist_tracking_weld"
        ET.SubElement(wrist, "freejoint", name=free_name)
        ET.SubElement(world, "body", name=target_name, mocap="true", pos=_text(initial), quat=_text(quat))
        ET.SubElement(equality, "weld", name=weld_name,
                      body1=f"{side}_hand_roll_link", body2=target_name,
                      relpose="0 0 0 1 0 0 0", solref=_text(WRIST_WELD_SOLREF),
                      solimp=_text(WRIST_WELD_SOLIMP), torquescale=str(WRIST_WELD_TORQUESCALE_M))
        layout["initial_wrist_positions"][side] = initial.copy()
        layout["wrist_quaternions"][side] = quat.copy()
        layout["wrist_free_joints"][side] = free_name
        layout["wrist_welds"][side] = weld_name
    ET.SubElement(root.find("option"), "flag", multiccd="enable")
    table = ET.SubElement(world, "body", name="tabletop", pos=_text([0, 0, table_height-TABLE_HALF_SIZE[2]]))
    ET.SubElement(table, "geom", name="table_geom", type="box", size=_text(TABLE_HALF_SIZE),
                  rgba="0.48 0.34 0.22 1", group="0", friction="1 0.005 0.0001")
    cylinder = ET.SubElement(world, "body", name="test_cylinder", pos=_text(cylinder_initial))
    ET.SubElement(cylinder, "freejoint", name="cylinder_free")
    ET.SubElement(cylinder, "geom", name="cylinder_geom", type="cylinder",
                  size=_text([radius, height/2]), mass=str(profile["mass_kg"]),
                  rgba="0.95 0.45 0.08 0", friction="1 0.005 0.0001", condim="3",
                  priority="1", solref="0.008 1", solimp="0.95 0.99 0.001")
    add_can_visual(root, cylinder, PROJECT_ROOT / "r2v2_description/visuals/cola_can/label.png")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (43, 40, 12, 12, 2):
        raise RuntimeError("Expected two dynamic free wrists, one free cylinder, and original 12 motors / 10 mimics")
    for side in SIDES:
        layout["mocap_ids"][side] = int(model.body(f"{side}_wrist_fixture").mocapid[0])
    data = mujoco.MjData(model)
    initialize_hands(model, data, cfg)
    hand = set(_hand_geoms(model, candidate.side))
    cylinder_geom, table_geom = model.geom("cylinder_geom").id, model.geom("table_geom").id
    initial_contacts = _contacts(model, data, hand, cylinder_geom, table_geom)
    if initial_contacts["hand_cylinder"] or initial_contacts["hand_table"]:
        raise ValueError("Candidate start 10 cm above target intersects table or cylinder")
    # Private static FK preview only, before any dynamics. Deliberately retain
    # invalid nominal candidates in diagnostics instead of correcting them.
    wrist_free = model.joint(layout["wrist_free_joints"][candidate.side])
    address = int(wrist_free.qposadr[0])
    data.qpos[address:address+7] = np.r_[grasp_position, quaternion]
    data.mocap_pos[layout["mocap_ids"][candidate.side]] = grasp_position
    # Candidate calibration references a settled upright object, not its
    # initial 1 mm release gap; this is a separate, never-integrated MjData.
    cylinder_address = int(model.joint("cylinder_free").qposadr[0])
    data.qpos[cylinder_address+2] = table_height + height/2
    mujoco.mj_forward(model, data)
    contacts = _contacts(model, data, hand, cylinder_geom, table_geom)
    clouds_world = {part: cloud @ rotation.T + grasp_position for part, cloud in parts.items()}
    layout["geometry"] = {
        "scope": "Static open-hand FK only; no grasp-success inference",
        "initial_contacts": initial_contacts, "nominal_target_contacts": contacts,
        "nominal_open_target_collision_free": not (contacts["hand_cylinder"] or contacts["hand_table"]),
        "valid_pregrasp": not (contacts["hand_cylinder"] or contacts["hand_table"]),
        "nominal_maximum_hand_object_penetration_m": max((c["penetration_m"] for c in contacts["hand_cylinder"]), default=0.),
        "nominal_maximum_hand_table_penetration_m": max((c["penetration_m"] for c in contacts["hand_table"]), default=0.),
        "source_open_collision_aabbs_wrist_m": {
            part: {"min": cloud.min(0).tolist(), "max": cloud.max(0).tolist()}
            for part, cloud in parts.items()},
        "nominal_part_minimum_table_clearance_m": {
            part: float(cloud[:, 2].min()-table_height) for part, cloud in clouds_world.items()},
        "nominal_part_deepest_below_object_top_m": {
            part: float(table_height+height-cloud[:, 2].min()) for part, cloud in clouds_world.items()},
        "nominal_thumb_reference_local_m": reference_local.tolist(),
        "nominal_thumb_reference_world_m": reference_world.tolist(),
        "open_thumb_angle_deg": float(np.rad2deg(cfg["hands"][candidate.side]["open"][0])),
        "depth_definition": "furthest nominal-open thumb collision vertex along local +X minus top; tilt is about its aligned reference",
        "wrist_local_finger_extension_axis_world": rotation[:, 0].tolist(),
        "object_is_free": True, "object_welds": False,
    }
    return model, cfg, layout


__all__ = ["TopGraspCandidate", "build_top_grasp_model", "top_wrist_rotation"]
