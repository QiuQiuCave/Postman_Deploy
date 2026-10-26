"""Bilateral bottle-style grip of a free crate; driven wrists, no body policy.

Only dynamic wrists are welded to prescribed fixture targets. No constraint,
applied force, pose reset, or collision suppression assists the crate. The
timeline deliberately finishes failed attempts so contact failures remain
visible; success is judged separately from measured lift/contact/slip.
"""
from dataclasses import asdict, dataclass
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_crate_lift import _blend
from common.r2v2_crate_lift_metrics import measure_crate_lift
from common.r2v2_crate_lift_scene import (
    WRIST_WELD_SOLIMP, WRIST_WELD_SOLREF, WRIST_WELD_TORQUESCALE_M,
)
from common.r2v2_hand_control import DualHandControl
from common.r2v2_vertical_grip_crate import load_vertical_grip_crate_config, add_vertical_grip_crate
from r2v2_description.model import SIDES, build_model_xml, initialize_hands, load_config, urdf_hand_joints


def text(values):
    return " ".join(str(float(v)) for v in values)


@dataclass(frozen=True)
class GripTestParameters:
    grasp: bool = True
    table_height: float = .4
    approach_m: float = .08
    wrist_z_offset_m: float = 0.
    wrist_outward_offset_m: float = 0.
    yaw_deg: float = 0.
    pitch_deg: float = 0.
    lift_m: float = .10

    def __post_init__(self):
        if not isinstance(self.grasp, bool):
            raise ValueError("grasp must be boolean")
        if not all(np.isfinite(v) for k, v in asdict(self).items() if k != "grasp"):
            raise ValueError("Parameters must be finite")
        if self.table_height <= .04 or not .01 <= self.approach_m <= .2 or not .02 <= self.lift_m <= .2:
            raise ValueError("Invalid table height or travel")
        if abs(self.wrist_z_offset_m) > .04 or abs(self.wrist_outward_offset_m) > .04 or max(abs(self.yaw_deg), abs(self.pitch_deg)) > 60:
            raise ValueError("Candidate outside the local alignment search")


def build_scene(p, crate=None):
    crate = load_vertical_grip_crate_config() if crate is None else crate
    cfg = copy.deepcopy(load_config())
    cfg["simulation_dt"], cfg["control_dt"] = .001, .01
    root = ET.fromstring(build_model_xml(cfg, fixture=True))
    root.set("model", "R2V2_vertical_handle_grasp_fixture")
    world, equality = root.find("worldbody"), root.find("equality")
    layout = {"target": {}, "initial": {}, "quaternion": {}, "calibration": {}}
    for side, sign in (("left", 1), ("right", -1)):
        record = PROJECT_ROOT / f"reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/{side}/grasp_record.json"
        relation = json.loads(record.read_text())["keyframes"]["initial"]["cylinder_in_wrist"]
        local = np.asarray(relation["position_m"])
        angle = np.deg2rad(sign*p.yaw_deg)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        pitch = np.deg2rad(p.pitch_deg)
        rotation = rotation @ np.array([[np.cos(pitch), 0., np.sin(pitch)],
                                       [0., 1., 0.], [-np.sin(pitch), 0., np.cos(pitch)]])
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())
        handle = np.array([0., sign*crate.grip_center_abs_y, p.table_height+crate.height/2])
        target = handle-rotation@local + [0., sign*p.wrist_outward_offset_m, p.wrist_z_offset_m]
        # Approach from outside the side wall, through the open palm's normal.
        # Translating along finger extension would sweep fingertips through
        # the shaft before its calibrated position between fingers and thumb.
        initial = target + rotation@np.array([0., sign*p.approach_m, 0.])
        layout["calibration"][side] = {"record": str(record), "handle_in_wrist_initial": relation}
        layout["target"][side], layout["initial"][side], layout["quaternion"][side] = target, initial, quaternion
        world.remove(world.find(f'body[@name="{side}_fixture"]'))
        wrist = world.find(f'body[@name="{side}_hand_roll_link"]')
        wrist.set("pos", text(initial))
        wrist.set("quat", text(quaternion))
        ET.SubElement(wrist, "freejoint", name=f"{side}_wrist_free")
        ET.SubElement(world, "body", name=f"{side}_wrist_fixture", mocap="true", pos=text(initial), quat=text(quaternion))
        ET.SubElement(equality, "weld", name=f"{side}_wrist_tracking_weld",
                      body1=f"{side}_hand_roll_link", body2=f"{side}_wrist_fixture",
                      relpose="0 0 0 1 0 0 0", solref=text(WRIST_WELD_SOLREF),
                      solimp=text(WRIST_WELD_SOLIMP), torquescale=str(WRIST_WELD_TORQUESCALE_M))
    option = root.find("option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "enable")
    table = ET.SubElement(world, "body", name="lift_table", pos=text((0, 0, p.table_height-.02)))
    ET.SubElement(table, "geom", name="lift_table_geom", type="box", size=".35 .45 .02",
                  rgba=".48 .34 .22 1", friction="1 .005 .0001")
    add_vertical_grip_crate(root, crate, position=(0, 0, p.table_height+.001))
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (43, 40, 12, 12, 2):
        raise RuntimeError("Unexpected fixture topology")
    return model, cfg, layout, xml


class VerticalGripExperiment:
    # Absolute interval ends; all ramps use smooth quintic interpolation.
    timeline = ((1., "READY"), (3., "APPROACH"), (3.5, "SETTLE"),
                (6.5, "CLOSE"), (8.5, "TRIAL_LIFT"), (9.5, "TRIAL_HOLD"),
                (12.5, "LIFT"), (15.5, "HOLD"), (18.5, "LOWER"),
                (19.5, "TABLE_SETTLE"), (22.5, "OPEN"), (24.5, "RETRACT"))

    def __init__(self, params=None):
        self.params = params or GripTestParameters()
        self.crate_params = load_vertical_grip_crate_config()
        self.model, self.cfg, self.layout, self.xml = build_scene(self.params, self.crate_params)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_hands(self.model, self.data, self.cfg)
        self.hands = DualHandControl(self.model, self.data, self.cfg)
        self.mocap = {s: self.model.body(f"{s}_wrist_fixture").mocapid[0] for s in SIDES}
        self.joints = [self.model.joint(n).id for n in urdf_hand_joints()]
        self.mimics = [(self.model.joint(n).qposadr[0], self.model.joint(j.find("mimic").get("joint")).qposadr[0],
                        float(j.find("mimic").get("multiplier", "1")), float(j.find("mimic").get("offset", "0")))
                       for n, j in urdf_hand_joints().items() if j.find("mimic") is not None]
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.spec))
        self.peak_joint_violation_rad = self.peak_mimic_error_rad = 0.
        self.steps, self.baseline, self.failure = 0, None, None
        self.samples, self.states, self.transitions = [], [], []
        self.measure()
        self.initial_metrics = copy.deepcopy(self.metrics)
        if max(self.metrics["hand_crate_penetration_m"], self.metrics["hand_table_penetration_m"]) > 1e-6:
            raise ValueError("Initial open hands intersect crate or table")
        self.last_phase = None

    @property
    def phase(self):
        return next((name for end, name in self.timeline if self.data.time < end-1e-8), "COMPLETE")

    @property
    def done(self):
        return self.phase == "COMPLETE" or self.failure is not None

    def motion(self):
        t, p = self.data.time, self.params
        approach = _blend((t-1.)/2.)*(1-_blend((t-22.5)/2.))
        lift = .02*_blend((t-6.5)/2.) + (p.lift_m-.02)*_blend((t-9.5)/3.) - p.lift_m*_blend((t-15.5)/3.)
        for side in SIDES:
            initial, target = self.layout["initial"][side], self.layout["target"][side]
            self.data.mocap_pos[self.mocap[side]] = initial+approach*(target-initial)+[0, 0, lift]

    def measure(self):
        m, d = self.model, self.scratch
        mujoco.mj_getState(m, self.data, self.state, self.spec)
        mujoco.mj_setState(m, d, self.state, self.spec)
        mujoco.mj_forward(m, d)
        raw = measure_crate_lift(m, d, self.params.table_height)
        if self.baseline is None and d.time >= 6.5-1e-8:
            self.baseline = {s: np.asarray(raw["hands"][s]["T_wrist_crate"]) for s in SIDES}
        sides = {}
        for side in SIDES:
            hand = raw["hands"][side]
            shaft = [c for c in hand["contacts"] if c["crate_geom"] == f"crate_{side}_vertical_grip"]
            thumb = sum(c["normal_force_N"] for c in shaft if c["part"] == "thumb")
            fingers = sum(c["normal_force_N"] for c in shaft if c["part"] in ("index", "middle", "ring", "pinky"))
            slip = 0. if self.baseline is None else float(np.linalg.norm(
                np.asarray(hand["T_wrist_crate"])[:3, 3]-self.baseline[side][:3, 3]))
            bid = m.body(f"{side}_hand_roll_link").id
            sides[side] = dict(thumb_shaft_normal_N=thumb, fingers_shaft_normal_N=fingers,
                opposed=bool(thumb>.05 and fingers>.05), vertical_N=hand["vertical_force_N"],
                shaft_vertical_N=sum(c["vertical_force_N"] for c in shaft),
                connector_vertical_N=sum(c["vertical_force_N"] for c in hand["contacts"] if c["crate_geom"].endswith("_connector")),
                slip_m=slip, wrist_position_m=d.xpos[bid].tolist(), wrist_quaternion_wxyz=d.xquat[bid].tolist(),
                wrist_tracking_m=float(np.linalg.norm(d.xpos[bid]-d.mocap_pos[self.mocap[side]])))
        q = d.qpos[m.jnt_qposadr[self.joints]]
        limits = m.jnt_range[self.joints]
        self.metrics = dict(time_s=float(d.time), phase=self.phase, clearance_m=raw["clearance_m"],
            crate_position_m=raw["crate_position_m"], crate_quaternion_wxyz=raw["crate_quaternion_wxyz"],
            tilt_deg=raw["crate_tilt_deg"], table_contact=raw["table_contact"], table_vertical_N=raw["table_vertical_force_N"],
            hands=sides, hand_crate_penetration_m=raw["max_hand_crate_penetration_m"],
            hand_table_penetration_m=raw["max_hand_table_penetration_m"], hand_self_penetration_m=raw["max_hand_self_penetration_m"],
            joint_violation_rad=float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1]))),
            mimic_error_rad=float(max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a,b,ratio,offset in self.mimics)),
            max_motor_torque_Nm=float(np.max(np.abs(self.data.actuator_force))),
            finite=raw["finite_state"])
        return self.metrics

    def record(self):
        self.measure()
        if self.last_phase != self.phase:
            self.transitions.append({"time_s": float(self.data.time), "phase": self.phase})
            self.last_phase = self.phase
        self.samples.append(copy.deepcopy(self.metrics))
        self.states.append((float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy(),
                            self.data.mocap_pos.copy(), self.data.mocap_quat.copy()))

    def step(self):
        if self.done:
            return
        self.motion()
        if self.steps % 10 == 0:
            for side in SIDES:
                self.hands.command(side, int(self.params.grasp and 3.5-1e-8 <= self.data.time < 19.5-1e-8))
            self.hands.update()
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        q = self.data.qpos[self.model.jnt_qposadr[self.joints]]
        limits = self.model.jnt_range[self.joints]
        self.peak_joint_violation_rad = float(max(self.peak_joint_violation_rad,
                                                 np.max(limits[:, 0]-q), np.max(q-limits[:, 1])))
        self.peak_mimic_error_rad = max(self.peak_mimic_error_rad, max(
            abs(self.data.qpos[a]-ratio*self.data.qpos[b]-offset) for a,b,ratio,offset in self.mimics))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or self.data.warning.number.any():
            self.failure = "Nonfinite state or MuJoCo warning"
        if self.steps % 10 == 0 or self.done:
            self.record()

    def report(self):
        hold = [r for r in self.samples if 12.5-1e-8 <= r["time_s"] < 15.5-1e-8]
        good = lambda r: (r["clearance_m"] >= .08 and not r["table_contact"] and r["tilt_deg"] < 8
            and all(h["opposed"] and h["vertical_N"] > .2 and h["slip_m"] < .015 for h in r["hands"].values()))
        safe = all(r["hand_crate_penetration_m"] < .003 and r["hand_table_penetration_m"] < .003
                   and r["joint_violation_rad"] < .01 and r["mimic_error_rad"] < .015
                   and all(h["wrist_tracking_m"] < .003 for h in r["hands"].values()) for r in self.samples)
        safe = safe and self.peak_joint_violation_rad < .01 and self.peak_mimic_error_rad < .015
        passed = bool(self.phase == "COMPLETE" and len(hold) >= 299 and all(good(r) for r in hold)
                      and safe and self.failure is None)
        peak = lambda key: max((r[key] for r in self.samples), default=0.)
        return dict(scope="DYNAMIC WRIST FIXTURE ONLY / no whole-body policy or arm IK", passed=passed,
            failure=self.failure, completed=self.phase == "COMPLETE", parameters=asdict(self.params),
            crate_parameters=asdict(self.crate_params), hand_configuration=self.cfg, layout=self.layout,
            initial_metrics=self.initial_metrics, duration_s=float(self.data.time),
            hold_samples=len(hold), hold_valid_fraction=float(np.mean([good(r) for r in hold])) if hold else 0.,
            hold_min_clearance_m=min((r["clearance_m"] for r in hold), default=None),
            hold_max_tilt_deg=max((r["tilt_deg"] for r in hold), default=None),
            hold_max_slip_m=max((h["slip_m"] for r in hold for h in r["hands"].values()), default=None),
            hold_mean_load_N={s: float(np.mean([r["hands"][s]["vertical_N"] for r in hold])) if hold else None for s in SIDES},
            hold_mean_shaft_load_N={s: float(np.mean([r["hands"][s]["shaft_vertical_N"] for r in hold])) if hold else None for s in SIDES},
            hold_mean_connector_load_N={s: float(np.mean([r["hands"][s]["connector_vertical_N"] for r in hold])) if hold else None for s in SIDES},
            peak_clearance_m=peak("clearance_m"), peak_hand_crate_penetration_m=peak("hand_crate_penetration_m"),
            peak_hand_table_penetration_m=peak("hand_table_penetration_m"), peak_joint_violation_rad=self.peak_joint_violation_rad,
            peak_mimic_error_rad=self.peak_mimic_error_rad, safety_checks_passed=safe,
            joint_measurement_hz=1000, contact_pose_measurement_hz=100,
            grasp_acceptance={"hold_window_s": [12.5, 15.5], "minimum_clearance_m": .08,
                "maximum_tilt_deg": 8., "maximum_relative_translation_m": .015,
                "minimum_each_hand_load_N": .2, "thumb_and_fingers_on_each_shaft_required": True,
                "table_contact_allowed": False},
            no_crate_weld=True, no_crate_external_force=not bool(self.data.xfrc_applied.any() or self.data.qfrc_applied.any()),
            warnings=self.data.warning.number.tolist(), mujoco_version=mujoco.__version__)
