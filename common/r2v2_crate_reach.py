"""Attempt the committed crate grasp with the real free-base Reach robot.

The frozen policy is the sole body controller. No IK, mocap wrists, body
welds, support forces, runtime pose resets, or automatic retraining.
"""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import yaml

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_lift import CrateLiftExperiment, load_lift_config
from common.r2v2_crate_lift_metrics import _descendant, measure_crate_lift
from common.r2v2_crate_reach_scene import build_crate_reach_model, crate_wrist_targets
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import (
    ReachCompatibilityExperiment, TCP_OFFSETS, angle_error, foot_collision_ids,
    load_reach_config, quaternion_from_matrix, require_parity, resolve_asset, sha256,
)
from common.r2v2_tabletop_demo import copy_robot_initial_state
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, urdf_hand_joints


CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_crate_reach.yaml"
PARITY = PROJECT_ROOT / "artifacts/r2v2_reach/parity_12000/report.json"


def load_crate_reach_config(path=None):
    path = Path(path or CONFIG)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Crate Reach configuration must be a mapping")
    vectors = {"crate_center_xy": 2, "table_center_xy": 2, "table_half_size": 3}
    keys = {"warmup_standing_s", "table_above_waist_m", "phase_timeout_s", "prealign_position_m",
            "prealign_orientation_deg", "insertion_position_m", "insertion_orientation_deg",
            "motion_speed_mps", "motion_stable_s", *vectors}
    optional = {"crate_width_m"}
    if keys-set(cfg) or set(cfg)-keys-optional:
        raise ValueError(f"Incorrect Crate Reach keys: {(keys-set(cfg)) | (set(cfg)-keys-optional)}")
    for key, value in cfg.items():
        arr = np.asarray(value)
        if arr.dtype.kind not in "fiu" or not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} must be finite numeric values")
        if key in vectors:
            if arr.shape != (vectors[key],):
                raise ValueError(f"Invalid {key} shape")
        elif key == "table_above_waist_m":
            if arr.shape != () or float(arr) < 0:
                raise ValueError("table_above_waist_m must be nonnegative")
        elif arr.shape != () or float(arr) <= 0:
            raise ValueError(f"{key} must be positive")
    if np.any(np.asarray(cfg["table_half_size"]) <= 0):
        raise ValueError("table_half_size must be positive")
    return cfg


def wrist_goal_to_tcp(side, wrist):
    wrist = np.asarray(wrist, dtype=float)
    if side not in SIDES or wrist.shape != (4, 4) or not np.all(np.isfinite(wrist)):
        raise ValueError("Invalid world wrist target")
    tool = np.eye(4)
    tool[:3, 3] = TCP_OFFSETS[side]
    return wrist @ tool


class CrateReachExperiment(CrateLiftExperiment):
    """Reuse physical grasp gates, replacing fixture motion with policy goals.

Targets are set once on phase entry; the policy's trained smooth reference
continues without resetting its histories or reference state. Consequently
the exact wrist trajectories differ from the hand-fixture quintic paths.
"""

    def __init__(self, config=None, parity_report=None):
        self.fullbody_cfg = copy.deepcopy(config if config is not None else load_crate_reach_config())
        self.cfg = load_reach_config()
        self.parity_path = Path(parity_report or PARITY).resolve()
        self.parity_evidence = require_parity(self.parity_path, self.cfg)
        self.params, self.crate_params = load_lift_config(), load_crate_config()
        if "crate_width_m" in self.fullbody_cfg:
            # Per-experiment geometry override; preserve the committed
            # 36 cm fixture and every other crate/contact/grasp parameter.
            self.crate_params = replace(self.crate_params, width=self.fullbody_cfg["crate_width_m"])
        warm = ReachCompatibilityExperiment(self.cfg)
        while not (warm.phase == "STANDING_CHECK" and
                   warm.data.time-warm.phase_start >= self.fullbody_cfg["warmup_standing_s"]-1e-8):
            warm.step()
            if warm.done:
                raise RuntimeError(f"Free-standing preparation failed: {warm.failure}")
        warm.sync()
        self.warmup_duration_s = float(warm.data.time)
        self.warmup_peaks = copy.deepcopy(warm.peaks)
        self.table_height = float(warm.scratch.xpos[warm.model.body("waist_pitch_link").id, 2]
                                  + self.fullbody_cfg["table_above_waist_m"])
        self.model, cfg, self.layout = build_crate_reach_model(
            self.cfg, self.crate_params, table_top_m=self.table_height,
            crate_center_xy=self.fullbody_cfg["crate_center_xy"],
            table_center_xy=self.fullbody_cfg["table_center_xy"],
            table_half_size=self.fullbody_cfg["table_half_size"])
        targets = crate_wrist_targets(self.crate_params, table_top_m=self.table_height,
            crate_center_xy=self.fullbody_cfg["crate_center_xy"], insertion_m=self.params.insertion_m,
            start_palm_clearance_m=self.params.start_palm_clearance_m)
        self.layout.update(targets)
        self.hand_cfg = copy.deepcopy(cfg)
        for side in SIDES:
            self.hand_cfg["hands"][side]["closed"] = self.hand_cfg["hands"][side]["open"][:2]+[self.params.closed_curl_rad]*4
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        copy_robot_initial_state(warm.model, warm.data, self.model, self.data)
        self.policy = ReachPolicy(self.model, self.data, resolve_asset(self.cfg["policy_path"]))
        for key in ("last_action", "q_des", "last_torque", "references", "histories", "last_observation"):
            setattr(self.policy, key, copy.deepcopy(getattr(warm.policy, key)))
        # Fresh private hand controllers start at the already-open measured
        # pose, with the crate profile, not the old bottle closing target.
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        self.base = self.model.body("base_link").id
        self.floor = self.model.geom("floor").id
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.robot_geoms = {g for g in range(self.model.ngeom)
                            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.hand_geoms = {g for g in self.robot_geoms if any(
            _descendant(self.model, int(self.model.geom_bodyid[g]), self.model.body(f"{s}_hand_roll_link").id)
            for s in SIDES)}
        self.table_geoms = {g for g in range(self.model.ngeom)
                           if _descendant(self.model, int(self.model.geom_bodyid[g]), self.model.body("tabletop").id)}
        self.crate_geoms = {g for g in range(self.model.ngeom)
                           if int(self.model.geom_bodyid[g]) == self.model.body("cargo_crate").id}
        self.joints = [self.model.joint(n).id for n in urdf_hand_joints()]
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0],
                    self.model.joint(mimic.get("joint")).qposadr[0], float(mimic.get("multiplier", "1")),
                    float(mimic.get("offset", "0"))))
        self.phase, self.phase_start, self.failure = "READY", 0., None
        self.failure_phase = None
        self.steps = 0
        self.transitions = [{"time_s": 0., "state": "READY"}]
        self.samples, self.hold_samples, self.targets = [], [], []
        self.baseline_relations = None
        self.trial_confirmed = self.prealign_passed = False
        self.stable_since = self.bad_since = None
        self.current_metrics = {}
        self.goal_wrist_transforms = {}
        for side in SIDES:
            ref = self.policy.references[side]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, ref.goal_quaternion)
            goal = np.eye(4)
            goal[:3, :3] = rotation.reshape(3, 3)
            goal[:3, 3] = ref.goal_position-goal[:3, :3]@TCP_OFFSETS[side]
            self.goal_wrist_transforms[side] = goal
        self.peaks = dict.fromkeys(("hand_crate_penetration_m", "hand_self_penetration_m", "joint_violation_rad",
            "mimic_error_rad", "slip_m", "crate_tilt_deg", "clearance_m", "actuator_torque_Nm",
            "wrist_tracking_error_m", "wrist_tracking_error_deg", "base_tilt_deg",
            "body_joint_violation_rad", "robot_self_penetration_m"), 0.)
        self.sync()
        self.initial_crate_position = np.asarray(self.current_metrics["crate_position_m"]).copy()
        self.record()
        self._safety()
        if self.done:
            self.record()

    def sync(self):
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        mujoco.mj_setState(self.model, self.scratch, state, spec)
        mujoco.mj_forward(self.model, self.scratch)
        m = measure_crate_lift(self.model, self.scratch, self.table_height,
                               virtual_props=self.layout.get("virtual_props", False))
        errors, slips = {}, {}
        for side in SIDES:
            wrist = self.model.body(f"{side}_hand_roll_link").id
            actual = body_transform(self.scratch, wrist)
            target = self.goal_wrist_transforms[side]
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.scratch, mujoco.mjtObj.mjOBJ_XBODY, wrist, velocity, 0)
            errors[side] = {"position_m": float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
                "orientation_deg": float(np.rad2deg(angle_error(quaternion_from_matrix(actual[:3, :3]),
                                                                quaternion_from_matrix(target[:3, :3])))),
                "linear_speed_mps": float(np.linalg.norm(velocity[3:]))}
            if self.baseline_relations is not None:
                relation = np.asarray(m["hands"][side]["T_wrist_crate"])
                slips[side] = float(np.linalg.norm(relation[:3, 3]-self.baseline_relations[side][:3, 3]))
        m["wrist_errors"] = errors
        m["wrist_tracking_error_m"] = max(v["position_m"] for v in errors.values())
        m["wrist_tracking_error_deg"] = max(v["orientation_deg"] for v in errors.values())
        m["side_slip_m"], m["grasp_slip_m"] = slips, max(slips.values()) if slips else None
        m["base_tilt_deg"] = float(np.rad2deg(np.arccos(np.clip(self.scratch.xmat[self.base].reshape(3, 3)[2, 2], -1, 1))))
        self.current_metrics = m
        return m

    def enter(self, phase):
        super().enter(phase)
        if phase not in ("PREALIGN", "INSERT", "CLOSE", "TRIAL_LIFT", "LIFT"):
            return
        source = "initial_wrist_transforms" if phase == "PREALIGN" else "inserted_wrist_transforms"
        lift = 0.
        if phase in ("CLOSE", "TRIAL_LIFT", "LIFT") and self.params.grasp_enabled:
            lift += self.params.closure_seating_m
        if phase == "TRIAL_LIFT": lift += self.params.trial_height_m
        if phase == "LIFT": lift += self.params.lift_height_m
        for side in SIDES:
            wrist = np.array(self.layout[source][side], copy=True)
            wrist[2, 3] += lift
            self.goal_wrist_transforms[side] = wrist
            tcp = wrist_goal_to_tcp(side, wrist)
            self.policy.set_target_world(side, tcp[:3, 3], quaternion_from_matrix(tcp[:3, :3]))
        self.targets.append({"time_s": float(self.data.time), "phase": phase,
                             "wrist_targets_world": copy.deepcopy(self.goal_wrist_transforms)})

    def fail(self, reason):
        if not self.done:
            self.failure_phase = self.phase
        super().fail(reason)

    def _at_wrist_targets(self, prefix):
        c = self.fullbody_cfg
        reference_done = all(np.linalg.norm(r.goal_position-r.position) < 1e-5 and
                             angle_error(r.goal_quaternion, r.quaternion) < 1e-4
                             for r in self.policy.references.values())
        return reference_done and all(e["position_m"] < c[f"{prefix}_position_m"] and
            e["orientation_deg"] < c[f"{prefix}_orientation_deg"] and e["linear_speed_mps"] < c["motion_speed_mps"]
            for e in self.current_metrics["wrist_errors"].values())

    def _gate(self):
        elapsed = self.data.time-self.phase_start
        if self.phase == "READY":
            if elapsed >= self.params.ready_s: self.enter("PREALIGN")
        elif self.phase == "PREALIGN":
            if self._stable(self._at_wrist_targets("prealign"), self.fullbody_cfg["motion_stable_s"]):
                self.prealign_passed = True
                self.enter("INSERT")
            elif elapsed >= self.fullbody_cfg["phase_timeout_s"]:
                self.fail("Dual-arm Reach could not attain the outside-hole palm-up wrist poses; insertion not attempted")
        elif self.phase == "INSERT_SETTLE":
            if self._stable(elapsed >= self.params.insert_settle_s and self._at_wrist_targets("insertion"),
                            self.fullbody_cfg["motion_stable_s"]):
                self.enter("CLOSE")
            elif elapsed >= self.fullbody_cfg["phase_timeout_s"]:
                self.fail("Actual wrists did not reach the insertion targets; grasp command withheld")
        else:
            super()._gate()

    def _safety(self):
        p, m, d = self.params, self.current_metrics, self.data
        if not m["finite_state"] or np.any(d.warning.number):
            self.fail("Nonfinite state or MuJoCo warning"); return
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or self.model.nmocap or np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD):
            self.fail("Unexpected assistance or wrist fixture"); return
        q = d.qpos[self.model.jnt_qposadr[self.joints]]
        limits = self.model.jnt_range[self.joints]
        violation = max(0., float(np.max(limits[:, 0]-q)), float(np.max(q-limits[:, 1])))
        mimic = max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a, b, ratio, offset in self.mimics)
        body_q, body_limits = d.qpos[self.body_map.qpos], self.model.jnt_range[self.body_map.joints]
        body_violation = max(0., float(np.max(body_limits[:, 0]-body_q)), float(np.max(body_q-body_limits[:, 1])))
        reasons, self_depth = [], 0.
        for contact in self.scratch.contact:
            if contact.dist > 0: continue
            pair = set(map(int, contact.geom))
            if self.floor in pair and pair & self.robot_geoms and not (pair & self.foot_geoms):
                reasons.append("non-foot ground contact")
            if pair <= self.robot_geoms: self_depth = max(self_depth, -float(contact.dist))
            if pair & self.robot_geoms and pair & self.table_geoms: reasons.append("robot/table collision")
            if pair & self.crate_geoms and pair & (self.robot_geoms-self.hand_geoms): reasons.append("non-hand robot/crate collision")
        values = {"joint_violation_rad": violation, "body_joint_violation_rad": body_violation,
                  "mimic_error_rad": mimic, "robot_self_penetration_m": self_depth,
                  "hand_crate_penetration_m": m["max_hand_crate_penetration_m"],
                  "hand_self_penetration_m": m["max_hand_self_penetration_m"], "base_tilt_deg": m["base_tilt_deg"],
                  "crate_tilt_deg": m["crate_tilt_deg"], "clearance_m": m["clearance_m"],
                  "slip_m": m["grasp_slip_m"] or 0., "wrist_tracking_error_m": m["wrist_tracking_error_m"],
                  "wrist_tracking_error_deg": m["wrist_tracking_error_deg"],
                  "actuator_torque_Nm": max(float(np.max(np.abs(d.actuator_force[h.actuators]))) for h in self.hands.maps.values())}
        for k, v in values.items(): self.peaks[k] = max(self.peaks[k], float(v))
        g = self.cfg["gates"]
        if violation > p.max_joint_violation_rad: reasons.append("hand joint limit")
        if mimic > p.max_mimic_error_rad: reasons.append("hand mimic error")
        if body_violation > g["max_joint_violation_rad"]: reasons.append("body joint limit")
        if self_depth > g["max_self_penetration_m"]: reasons.append("robot self penetration")
        if m["max_hand_self_penetration_m"] > p.max_penetration_m: reasons.append("hand self penetration")
        if m["max_hand_crate_penetration_m"] > p.max_penetration_m: reasons.append("hand/crate penetration")
        if m["base_tilt_deg"] > g["max_base_tilt_deg"] or self.scratch.xpos[self.base, 2] < g["min_base_height_m"]:
            reasons.append("base tilt/height")
        if m["crate_tilt_deg"] > p.max_tilt_deg: reasons.append("crate tilt")
        if m["floor_contacts"]: reasons.append("crate hit floor")
        if reasons: self.fail("Safety stop: " + ", ".join(sorted(set(reasons))))

    def step(self):
        if self.done: return
        if self.steps % 10 == 0:
            self.sync(); self._safety()
            if not self.done: self._gate()
            if self.done:
                self.sync(); self.record(); return
            self.hands.update()
        if self.steps % 20 == 0:
            self.sync(); self.policy.act(self.scratch)
        self.policy.apply(self.data)
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps % 10 == 0:
            self.sync(); self.record()

    def record(self):
        hands = {}
        for side, mapping in self.hands.maps.items():
            ref = self.hands.controllers[side].reference
            hands[side] = {"q_rad": self.data.qpos[mapping.qpos].copy(), "q_ref_rad": ref.position.copy(),
                "torque_Nm": self.data.ctrl[mapping.actuators].copy(), "command": self.hands.controllers[side].command,
                "wrist_target_world": self.goal_wrist_transforms[side].copy(),
                "tcp_reference_position_m": self.policy.references[side].position.copy()}
        self.samples.append({"time_s": float(self.data.time), "phase": self.phase,
            "metrics": copy.deepcopy(self.current_metrics), "hands": hands,
            "base_position_world_m": self.scratch.xpos[self.base].copy(),
            "body_q_rad": self.data.qpos[self.body_map.qpos].copy(),
            "body_tau_Nm": self.data.ctrl[self.body_map.actuators].copy(), "body_action": self.policy.last_action.copy()})

    def report(self):
        r = super().report()
        grasp_commanded = any(any(h["command"] == 1 for h in row["hands"].values())
                              for row in self.samples)
        r["checks"].update({"prealign_passed": bool(self.prealign_passed),
                            "grasp_commanded": grasp_commanded,
                            "no_wrist_fixtures": self.model.nmocap == 0 and
                                not np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD)})
        r["lift_passed"] = all(r["checks"].values())
        r.update({"scope": "real new free-base robot, frozen dual-arm Reach, independent torque-limited hands, free crate",
            "failure_phase": self.failure_phase, "prealign_passed": self.prealign_passed,
            "insertion_attempted": any(t["state"] == "INSERT" for t in self.transitions),
            "grasp_commanded": grasp_commanded,
            "mocap_wrist_count": 0, "wrist_support_welds": 0,
            "wrist_drives": "28-DOF ONNX whole-body torque policy; no wrist fixtures or IK",
            "model_dimensions": {"nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu,
                                 "neq": self.model.neq, "nmocap": self.model.nmocap},
            "fullbody_configuration": self.fullbody_cfg, "reach_configuration": self.cfg,
            "warmup_duration_s": self.warmup_duration_s, "warmup_peaks": self.warmup_peaks,
            "final_wrist_errors": self.current_metrics["wrist_errors"], "targets": self.targets,
            "baseline_commit": "bfc4888", "parity_report": str(self.parity_path),
            "parity_report_sha256": sha256(self.parity_path), "onnx_sha256": self.parity_evidence["onnx_sha256"],
            "checkpoint_sha256": self.parity_evidence["checkpoint_sha256"],
            "target_schedule": "once per phase; continuous trained Reach reference, not fixture quintic timing"})
        for file in ("common/r2v2_crate_reach.py", "common/r2v2_crate_reach_scene.py", "common/r2v2_reach_policy.py"):
            r["source_sha256"][file] = sha256(PROJECT_ROOT/file)
        return r
