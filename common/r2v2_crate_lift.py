"""Dynamic bilateral hand fixture test. No full-body policy or crate assistance."""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_lift_metrics import measure_crate_lift
from common.r2v2_crate_lift_scene import build_crate_lift_model
from common.r2v2_hand_control import DualHandControl
from r2v2_description.model import SIDES, initialize_hands, urdf_hand_joints

CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_crate_lift.yaml"


@dataclass(frozen=True)
class CrateLiftParameters:
    insertion_m: float = .060
    start_palm_clearance_m: float = .080
    table_height: float = .4
    closed_curl_rad: float = .80
    closure_seating_m: float = .0175
    grasp_enabled: bool = True
    ready_s: float = 1.0
    insert_s: float = 2.5
    insert_settle_s: float = .5
    close_minimum_s: float = 2.5
    close_timeout_s: float = 6.0
    contact_stable_s: float = .3
    trial_height_m: float = .02
    trial_lift_s: float = 2.0
    trial_timeout_s: float = 2.0
    trial_clearance_m: float = .008
    lift_height_m: float = .10
    lift_s: float = 4.0
    hold_s: float = 3.0
    hold_clearance_m: float = .08
    hold_tilt_deg: float = 8.0
    min_side_vertical_force_N: float = .2
    max_slip_m: float = .015
    violation_s: float = .2
    max_penetration_m: float = .003
    max_tilt_deg: float = 15.0
    max_joint_violation_rad: float = .01
    max_mimic_error_rad: float = .015
    max_wrist_tracking_m: float = .003
    max_wrist_tracking_deg: float = 3.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "grasp_enabled":
                if not isinstance(value, bool):
                    raise ValueError("grasp_enabled must be boolean")
            elif (isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value)
                  or (value <= 0 and field.name != "closure_seating_m")):
                raise ValueError(f"{field.name} must be finite and positive")
        if not 0 <= self.closure_seating_m <= .03:
            raise ValueError("closure_seating_m must be between 0 and 0.03 m")
        if not 0 < self.trial_clearance_m < self.trial_height_m < self.lift_height_m:
            raise ValueError("Require trial_clearance < trial_height < lift_height")
        if not self.trial_height_m < self.hold_clearance_m < self.lift_height_m:
            raise ValueError("Invalid hold_clearance")
        if self.close_minimum_s >= self.close_timeout_s or self.hold_tilt_deg >= self.max_tilt_deg:
            raise ValueError("Invalid closure timeout or tilt thresholds")
        if self.closed_curl_rad > 1.4:
            raise ValueError("closed_curl_rad exceeds source joint range")


def load_lift_config(path=None):
    path = Path(path or CONFIG)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Lift config must be a mapping")
    return CrateLiftParameters(**values)


def _blend(value):
    t = float(np.clip(value, 0., 1.))
    return t*t*t*(10+t*(-15+6*t))


class CrateLiftExperiment:
    """Support-driven dynamic wrists, torque-limited fingers and a free crate."""

    def __init__(self, params=None):
        self.params = params or load_lift_config()
        self.crate_params = load_crate_config()
        p = self.params
        self.model, cfg, self.layout = build_crate_lift_model(
            self.crate_params, p.insertion_m, p.start_palm_clearance_m, p.table_height)
        self.hand_cfg = copy.deepcopy(cfg)
        # Independent hook profile: do not overwrite the existing bottle pose.
        for side in SIDES:
            self.hand_cfg["hands"][side]["closed"] = list(self.hand_cfg["hands"][side]["open"])
            self.hand_cfg["hands"][side]["closed"][2:] = [p.closed_curl_rad]*4
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_hands(self.model, self.data, self.hand_cfg)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.table_height = float(self.layout["table_top_m"])
        self.initial_wrists = {s: np.asarray(self.layout["initial_wrist_positions"][s], float) for s in SIDES}
        self.inserted_wrists = {s: np.asarray(self.layout["inserted_wrist_positions"][s], float) for s in SIDES}
        self.phase, self.phase_start, self.failure = "READY", 0., None
        self.steps = 0
        self.transitions = [{"time_s": 0., "state": self.phase}]
        self.samples, self.hold_samples = [], []
        self.baseline_relations = None
        self.trial_confirmed = False
        self.stable_since = self.bad_since = None
        self.current_metrics = {}
        self.peaks = dict.fromkeys(("hand_crate_penetration_m", "hand_self_penetration_m",
                                   "joint_violation_rad", "mimic_error_rad", "slip_m",
                                   "crate_tilt_deg", "clearance_m", "actuator_torque_Nm",
                                   "wrist_tracking_error_m", "wrist_tracking_error_deg"), 0.)
        self.joints = [self.model.joint(n).id for n in urdf_hand_joints()]
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0],
                                    self.model.joint(mimic.get("joint")).qposadr[0],
                                    float(mimic.get("multiplier", "1")),
                                    float(mimic.get("offset", "0"))))
        self.sync()
        self.initial_crate_position = np.asarray(self.current_metrics["crate_position_m"]).copy()
        self.record()

    @property
    def done(self):
        return self.phase in ("COMPLETE", "FAILED")

    def sync(self):
        # MuJoCo 3.3.7 Python bindings have no mj_copyData. Copy the complete
        # integration state, including controls, mocap and warm-start state.
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        mujoco.mj_setState(self.model, self.scratch, state, spec)
        mujoco.mj_forward(self.model, self.scratch)
        metrics = measure_crate_lift(self.model, self.scratch, self.table_height)
        slips = {}
        if self.baseline_relations is not None:
            for side in SIDES:
                current = np.asarray(metrics["hands"][side]["T_wrist_crate"])
                slips[side] = float(np.linalg.norm(current[:3, 3]-self.baseline_relations[side][:3, 3]))
        metrics["grasp_slip_m"] = max(slips.values()) if slips else None
        metrics["side_slip_m"] = slips
        tracking = {}
        for side in SIDES:
            wrist = self.model.body(f"{side}_hand_roll_link").id
            target = self.layout["mocap_ids"][side]
            dot = abs(float(np.dot(self.scratch.xquat[wrist], self.data.mocap_quat[target])))
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.scratch, mujoco.mjtObj.mjOBJ_XBODY,
                                    wrist, velocity, 0)
            tracking[side] = {
                "position_m": float(np.linalg.norm(self.scratch.xpos[wrist]-self.data.mocap_pos[target])),
                "orientation_deg": float(np.rad2deg(2*np.arccos(np.clip(dot, 0., 1.)))),
                "linear_velocity_world_m_s": velocity[3:].copy(),
                "angular_velocity_world_rad_s": velocity[:3].copy(),
            }
        metrics["wrist_tracking"] = tracking
        metrics["wrist_tracking_error_m"] = max(v["position_m"] for v in tracking.values())
        metrics["wrist_tracking_error_deg"] = max(v["orientation_deg"] for v in tracking.values())
        self.current_metrics = metrics
        return metrics

    def enter(self, phase):
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since = self.bad_since = None
        self.transitions.append({"time_s": self.phase_start, "state": phase})
        if phase == "CLOSE":
            for side in SIDES:
                self.hands.command(side, int(self.params.grasp_enabled))
        if phase == "TRIAL_LIFT":
            self.baseline_relations = {
                s: np.array(self.current_metrics["hands"][s]["T_wrist_crate"], copy=True) for s in SIDES}

    def fail(self, reason):
        if not self.done:
            self.failure = str(reason)
            self.enter("FAILED")

    def _stable(self, condition, duration):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = float(self.data.time)
        return self.data.time-self.stable_since >= duration-1e-8

    def _bilateral_contact(self):
        return all(self.current_metrics["hands"][s]["finger_normal_force_N"] > .05 for s in SIDES)

    def _bilateral_load(self):
        # Internal finger/palm clamp forces can be large and opposite. Both
        # fingers must engage the handle AND each whole hand must carry load.
        return all(min(self.current_metrics["hands"][s]["finger_handle_vertical_force_N"],
                       self.current_metrics["hands"][s]["vertical_force_N"])
                   >= self.params.min_side_vertical_force_N for s in SIDES)

    def _table_supported(self):
        return self.current_metrics["table_vertical_force_N"] > .05

    def _gate(self):
        p, m = self.params, self.current_metrics
        elapsed = float(self.data.time-self.phase_start)
        if self.phase == "READY" and elapsed >= p.ready_s:
            self.enter("INSERT")
        elif self.phase == "INSERT" and elapsed >= p.insert_s:
            self.enter("INSERT_SETTLE")
        elif self.phase == "INSERT_SETTLE" and elapsed >= p.insert_settle_s:
            self.enter("CLOSE")
        elif self.phase == "CLOSE":
            if self._stable(elapsed >= p.close_minimum_s and self._bilateral_contact(), p.contact_stable_s):
                self.enter("TRIAL_LIFT")
            elif elapsed >= p.close_timeout_s:
                self.fail("Closure timeout: both hands did not establish finger/handle contact")
        elif self.phase == "TRIAL_LIFT" and elapsed >= p.trial_lift_s:
            self.enter("TRIAL_HOLD")
        elif self.phase == "TRIAL_HOLD":
            good = (m["clearance_m"] >= p.trial_clearance_m and not self._table_supported()
                    and self._bilateral_load() and m["grasp_slip_m"] < p.max_slip_m
                    and m["crate_tilt_deg"] <= p.hold_tilt_deg)
            if self._stable(good, p.contact_stable_s):
                self.trial_confirmed = True
                self.enter("LIFT")
            elif elapsed >= p.trial_timeout_s:
                self.fail("Trial lift failed: clearance, bilateral load, tilt or slip")
        elif self.phase == "LIFT" and elapsed >= p.lift_s:
            self.enter("HOLD")
        elif self.phase == "HOLD":
            good = (m["clearance_m"] >= p.hold_clearance_m and not self._table_supported()
                    and self._bilateral_load() and m["grasp_slip_m"] < p.max_slip_m
                    and m["crate_tilt_deg"] <= p.hold_tilt_deg)
            self.hold_samples.append({"time_s": float(self.data.time), "valid": bool(good),
                                     "clearance_m": m["clearance_m"], "tilt_deg": m["crate_tilt_deg"],
                                     "position_m": m["crate_position_m"], "slip_m": m["grasp_slip_m"],
                                     "side_vertical_force_N": {
                                         s: m["hands"][s]["finger_handle_vertical_force_N"] for s in SIDES}})
            if not good:
                self.fail("Hold failed: crate is not stably supported by both hands")
            elif elapsed >= p.hold_s:
                self.enter("COMPLETE")
        if self.phase in ("LIFT", "HOLD") and not self.done:
            bad = not self._bilateral_load() or self._table_supported() or m["grasp_slip_m"] >= p.max_slip_m
            if bad:
                self.bad_since = self.data.time if self.bad_since is None else self.bad_since
                if self.data.time-self.bad_since >= p.violation_s:
                    self.fail("Lost bilateral grasp or excessive slip during lift")
            else:
                self.bad_since = None

    def _motion(self):
        elapsed, p = self.data.time-self.phase_start, self.params
        alpha, lift = 0., 0.
        if self.phase == "INSERT":
            alpha = _blend(elapsed/p.insert_s)
        elif self.phase != "READY":
            alpha = 1.
        if self.phase == "TRIAL_LIFT":
            lift = p.trial_height_m*_blend(elapsed/p.trial_lift_s)
        elif self.phase == "TRIAL_HOLD":
            lift = p.trial_height_m
        elif self.phase == "LIFT":
            lift = p.trial_height_m+(p.lift_height_m-p.trial_height_m)*_blend(elapsed/p.lift_s)
        elif self.phase in ("HOLD", "COMPLETE"):
            lift = p.lift_height_m
        # Insert at the original collision-free height, then seat underneath
        # the handle while closing. Retain this offset for the lift; never
        # push the hand horizontally into the upper beam during insertion.
        if self.phase == "CLOSE" and p.grasp_enabled:
            lift += p.closure_seating_m*_blend(elapsed/p.close_minimum_s)
        elif p.grasp_enabled and self.phase in ("TRIAL_LIFT", "TRIAL_HOLD", "LIFT", "HOLD", "COMPLETE"):
            lift += p.closure_seating_m
        for side in SIDES:
            mocap = self.layout["mocap_ids"][side]
            self.data.mocap_pos[mocap] = ((1-alpha)*self.initial_wrists[side]
                                         + alpha*self.inserted_wrists[side] + [0., 0., lift])
            self.data.mocap_quat[mocap] = self.layout["wrist_quaternions"][side]

    def _safety(self):
        p, m, d = self.params, self.current_metrics, self.data
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or np.any(d.warning.number):
            self.fail("Nonfinite state or MuJoCo warning")
            return
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            self.fail("Unexpected applied force")
            return
        q = d.qpos[self.model.jnt_qposadr[self.joints]]
        bounds = self.model.jnt_range[self.joints]
        violation = max(0., float(np.max(bounds[:, 0]-q)), float(np.max(q-bounds[:, 1])))
        mimic = max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a, b, ratio, offset in self.mimics)
        penetration = m["max_hand_crate_penetration_m"]
        self_penetration = m["max_hand_self_penetration_m"]
        for key, value in (("joint_violation_rad", violation), ("mimic_error_rad", mimic),
                           ("hand_crate_penetration_m", penetration), ("hand_self_penetration_m", self_penetration),
                           ("crate_tilt_deg", m["crate_tilt_deg"]), ("clearance_m", m["clearance_m"]),
                           ("slip_m", m["grasp_slip_m"] or 0.),
                           ("wrist_tracking_error_m", m["wrist_tracking_error_m"]),
                           ("wrist_tracking_error_deg", m["wrist_tracking_error_deg"]),
                           ("actuator_torque_Nm", np.max(np.abs(d.actuator_force)))):
            self.peaks[key] = max(self.peaks[key], float(value))
        reasons = []
        if violation > p.max_joint_violation_rad: reasons.append("finger joint limit")
        if mimic > p.max_mimic_error_rad: reasons.append("finger mimic error")
        if penetration > p.max_penetration_m: reasons.append("hand/crate penetration")
        if self_penetration > p.max_penetration_m: reasons.append("hand self penetration")
        if m["crate_tilt_deg"] > p.max_tilt_deg: reasons.append("crate tilt")
        if m["hand_table_contacts"]: reasons.append("hand/table collision")
        if m["hand_floor_contacts"]: reasons.append("hand/floor collision")
        if m["floor_contacts"]: reasons.append("crate hit floor")
        if m["wrist_tracking_error_m"] > p.max_wrist_tracking_m: reasons.append("wrist position tracking")
        if m["wrist_tracking_error_deg"] > p.max_wrist_tracking_deg: reasons.append("wrist orientation tracking")
        if reasons:
            self.fail("Safety stop: " + ", ".join(reasons))

    def step(self):
        if self.done:
            return
        if self.steps % 10 == 0:
            self.sync()
            self._safety()
            if not self.done:
                self._gate()
            if self.done:
                self.record()
                return
            self.hands.update()
        self._motion()
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps % 10 == 0:
            self.sync()
            self.record()

    def record(self):
        states = {}
        for side, mapping in self.hands.maps.items():
            ref = self.hands.controllers[side].reference
            states[side] = {"q_rad": self.data.qpos[mapping.qpos].copy(),
                            "q_ref_rad": ref.position.copy(),
                            "torque_Nm": self.data.ctrl[mapping.actuators].copy(),
                            "command": self.hands.controllers[side].command,
                            "wrist_position_m": np.asarray(self.current_metrics["hands"][side]["T_world_wrist"])[:3, 3].copy(),
                            "wrist_target_position_m": self.data.mocap_pos[self.layout["mocap_ids"][side]].copy()}
        self.samples.append({"time_s": float(self.data.time), "phase": self.phase,
                             "metrics": copy.deepcopy(self.current_metrics), "hands": states})

    def report(self):
        held = self.hold_samples
        complete = self.phase == "COMPLETE" and self.failure is None
        checks = {"experiment_completed": complete, "trial_lift_confirmed": self.trial_confirmed,
                  "full_hold_valid": bool(held) and all(row["valid"] for row in held),
                  "hold_duration": bool(held) and held[-1]["time_s"]-held[0]["time_s"] >= self.params.hold_s-.011,
                  "no_applied_force_fields": not np.any(self.data.qfrc_applied) and not np.any(self.data.xfrc_applied),
                  "no_warnings": not np.any(self.data.warning.number)}
        checks = {k: bool(v) for k, v in checks.items()}
        paths = ("common/r2v2_crate_lift.py", "common/r2v2_crate_lift_scene.py",
                 "common/r2v2_crate_lift_metrics.py", "common/r2v2_crate.py")
        return {"lift_passed": all(checks.values()), "experiment_completed": complete,
                "scope": "support-driven dynamic wrists, torque-limited fingers, free crate; NOT full-body policy",
                "checks": checks, "phase": self.phase, "failure": self.failure,
                "duration_s": float(self.data.time), "parameters": asdict(self.params),
                "crate_parameters": asdict(self.crate_params), "hand_configuration": self.hand_cfg,
                "layout": self.layout, "peaks": self.peaks, "final_metrics": self.current_metrics,
                "hold_samples": held, "transitions": self.transitions,
                "crate_start_position_m": self.initial_crate_position,
                "crate_runtime_resets": 0, "crate_welds": 0, "mocap_wrist_count": 2,
                "wrist_support_welds": 2,
                "wrist_drives": "mocap targets drive dynamic freejoint wrists through wrist-only support welds; robot arms are not simulated",
                "mujoco_version": mujoco.__version__,
                "source_sha256": {name: hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() for name in paths}}
