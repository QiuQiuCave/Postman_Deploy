"""New-asset free-standing/empty-hand gate. Does not run a grasp task."""

from common.path_config import PROJECT_ROOT

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from common.r2v2_hand_control import DualHandControl
from common.r2v2_grasp_recording import body_transform
from r2v2_description.model import (
    BODY_JOINTS, JointMap, SIDES, build_model_xml, initialize_hands, load_config,
)


# The training tool frame is intentionally not the new hand's palm center.
TCP_OFFSETS = {
    "left": np.array([0.1735, -0.0000004, -0.0317504]),
    "right": np.array([0.1735, 0.0000004, -0.0317504]),
}


def load_reach_config(path=None):
    path = Path(path or PROJECT_ROOT / "deploy_mujoco/config/r2v2_reach.yaml")
    cfg = yaml.safe_load(path.read_text())
    if (cfg["simulation_dt"], cfg["policy_dt"], cfg["hand_dt"]) != (0.001, 0.02, 0.01):
        raise ValueError("Validated schedule requires 1 kHz physics, 50 Hz policy, 100 Hz hands")
    return cfg


def resolve_asset(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_parity(path, cfg):
    report = json.loads(Path(path).read_text())
    if not report.get("passed"):
        raise ValueError("Training/deployment numerical parity has not passed")
    for key, file in (("onnx_sha256", resolve_asset(cfg["policy_path"])),
                      ("checkpoint_sha256", resolve_asset(cfg["checkpoint_path"])),
                      ("adapter_sha256", PROJECT_ROOT / "common/r2v2_reach_policy.py")):
        if report.get(key) != sha256(file):
            raise ValueError(f"Stale or mismatched parity evidence: {key}")
    return report


def build_reach_model(cfg):
    hands = copy.deepcopy(load_config())
    hands["simulation_dt"] = cfg["simulation_dt"]
    hands["control_dt"] = cfg["hand_dt"]
    root = ET.fromstring(build_model_xml(hands, fixture=False))
    root.set("model", "R2V2_new_asset_reach_compatibility")
    for side, position in TCP_OFFSETS.items():
        body = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        ET.SubElement(body, "site", name=f"{side}_tcp", pos=" ".join(map(str, position)),
                      quat="1 0 0 0", size="0.008", rgba="0.1 0.8 0.3 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, hands


def foot_collision_ids(model):
    ids = [g for g in range(model.ngeom)
           if model.body(model.geom_bodyid[g]).name in
           ("left_ankle_roll_link", "right_ankle_roll_link")
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if len(ids) != 28 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in ids):
        raise ValueError("New-asset foot geometry changed; revalidate ground initialization")
    return ids


def initialize_robot(model, data, hands, clearance=0.001):
    mapping = JointMap.create(model, BODY_JOINTS)
    q = np.zeros(28)
    for side in SIDES:
        for suffix, angle in (("hip_pitch", -0.1), ("knee", 0.3), ("ankle_pitch", -0.2)):
            q[BODY_JOINTS.index(f"{side}_{suffix}_joint")] = angle
    data.qpos[mapping.qpos] = q
    initialize_hands(model, data, hands)
    ids = foot_collision_ids(model)
    bottom = min(data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in ids)
    base_qpos = model.joint("floating_base_joint").qposadr[0]
    data.qpos[base_qpos + 2] += clearance - bottom
    mujoco.mj_forward(model, data)
    return float(data.qpos[base_qpos + 2])


def quaternion_from_matrix(matrix):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(matrix).ravel())
    return q if q[0] >= 0 else -q


def rotation_from_rpy_deg(rpy):
    """Extrinsic XYZ / roll-pitch-yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rpy = np.asarray(rpy, dtype=float)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("RPY must contain three finite degree values")
    r, p, y = np.deg2rad(rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def angle_error(a, b):
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)), 0, 1)))


class ReachCompatibilityExperiment:
    """Hard-gated sequence; failure stops instead of advancing to grasping."""

    def __init__(self, cfg, policy_path=None):
        from common.r2v2_reach_policy import ReachPolicy
        self.cfg = copy.deepcopy(cfg)
        self.model, self.hand_cfg = build_reach_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.initial_base_height = initialize_robot(
            self.model, self.data, self.hand_cfg, cfg["foot_clearance_m"])
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.policy = ReachPolicy(self.model, self.data, policy_path or resolve_asset(cfg["policy_path"]))
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.floor = self.model.geom("floor").id
        self.base = self.model.body("base_link").id
        self.scratch = mujoco.MjData(self.model)
        self.phase, self.phase_start = "RESET_SETTLE", 0.0
        self.steps = 0
        self.stable_time = self.inactive_bad_time = 0.0
        self.stable_since = self.inactive_bad_since = None
        self.standing_passed = False
        self.failure = None
        self.failure_phase = None
        self.transitions = [{"time_s": 0.0, "state": self.phase}]
        self.samples = []
        self.peaks = {"self_penetration_m": 0.0, "joint_violation_rad": 0.0,
                      "base_tilt_deg": 0.0, "inactive_wrist_position_error_m": 0.0,
                      "inactive_orientation_error_deg": 0.0}
        self.saturation_count = self.torque_count = 0
        self.contact_pairs = {}
        self.initial_contacts = self._contacts()
        self.air_targets = []
        self.air_index = -1
        self.right_lock = None
        self.initial_base_xy = self.data.xpos[self.base, :2].copy()
        self.sync()
        self.record()

    @property
    def done(self):
        return self.phase in ("FAILED", "PASSED")

    def sync(self):
        """Current-state observations on scratch, no forwarding of live physics."""
        d, s = self.data, self.scratch
        s.time = d.time
        s.qpos[:] = d.qpos
        s.qvel[:] = d.qvel
        s.ctrl[:] = d.ctrl
        mujoco.mj_forward(self.model, s)

    def transition(self, phase, exit_errors=None):
        previous = self.phase
        self.phase, self.phase_start = phase, self.data.time
        self.stable_time = 0.0
        self.stable_since = None
        self.transitions.append({"time_s": self.data.time, "state": phase, "from_state": previous,
                                 "exit_errors": exit_errors})
        self.record()  # Replace same-time old-state row, keep transition exact.

    def fail(self, reason):
        self.failure = reason
        self.failure_phase = self.phase
        self.transition("FAILED", {side: self.errors(side) for side in SIDES})
        # Caller stops physics immediately. Do not release hands or fall back
        # to low-gain PASSIVE while carrying an object in a future task.

    def tcp(self, side):
        site = self.model.site(f"{side}_tcp").id
        s = self.scratch
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, s, jp, jr, site)
        return (s.site_xpos[site].copy(), quaternion_from_matrix(s.site_xmat[site].reshape(3, 3)),
                jp @ s.qvel, jr @ s.qvel)

    def errors(self, side):
        p, q, v, w = self.tcp(side)
        ref = self.policy.references[side]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, ref.goal_quaternion)
        wrist_goal = ref.goal_position - rotation.reshape(3, 3) @ TCP_OFFSETS[side]
        wrist = self.model.body(f"{side}_hand_roll_link").id
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.scratch, jp, jr, wrist)
        return {"position_m": float(np.linalg.norm(p - ref.goal_position)),
                "orientation_deg": float(np.rad2deg(angle_error(q, ref.goal_quaternion))),
                "linear_speed_mps": float(np.linalg.norm(v)), "angular_speed_radps": float(np.linalg.norm(w)),
                "wrist_position_m": float(np.linalg.norm(self.scratch.xpos[wrist] - wrist_goal)),
                "wrist_linear_speed_mps": float(np.linalg.norm(jp @ self.scratch.qvel))}

    def _contacts(self):
        contacts = []
        for c in self.data.contact:
            if c.dist > 0:
                continue
            names = [self.model.geom(g).name for g in c.geom]
            depth = max(0.0, -float(c.dist))
            contacts.append({"geoms": names, "penetration_m": depth})
            key = " | ".join(names)
            self.contact_pairs[key] = max(depth, self.contact_pairs.get(key, 0.0))
        return contacts

    def safety(self):
        m, d = self.model, self.data
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or np.any(d.warning.number):
            return "Nonfinite state or MuJoCo numerical warning"
        q = d.qpos[self.body_map.qpos]
        limits = m.jnt_range[self.body_map.joints]
        violation = float(max(0.0, np.max(limits[:, 0] - q), np.max(q - limits[:, 1])))
        self.peaks["joint_violation_rad"] = max(self.peaks["joint_violation_rad"], violation)
        if violation > self.cfg["gates"]["max_joint_violation_rad"]:
            return "Measured body joint exceeded physical limit tolerance"
        for c in d.contact:
            if c.dist > 0:
                continue
            if self.floor in c.geom:
                other = c.geom2 if c.geom1 == self.floor else c.geom1
                if other not in self.foot_geoms:
                    return f"Non-foot ground contact: {m.geom(other).name}"
            else:
                self.peaks["self_penetration_m"] = max(self.peaks["self_penetration_m"], -float(c.dist))
        if self.peaks["self_penetration_m"] > self.cfg["gates"]["max_self_penetration_m"]:
            return "Deep robot self-contact"
        return None

    def set_standby(self):
        exit_errors = {side: self.errors(side) for side in SIDES}
        base = self.scratch.xpos[self.base]
        rotation = self.scratch.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.initial_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for side in SIDES:
            pose = self.cfg["standby_xyz_rpy_deg"][side]
            self.policy.set_target_world(side, base + self.initial_yaw @ pose[:3],
                                         quaternion_from_matrix(self.initial_yaw @ rotation_from_rpy_deg(pose[3:])))
        self.transition("STANDBY_REACH", exit_errors)

    def prepare_air_targets(self):
        cfg, s = self.cfg["placement"], self.scratch
        waist = self.model.body("waist_pitch_link").id
        cylinder = np.eye(4)
        cylinder[:3, :3] = self.initial_yaw
        xy = self.initial_base_xy + (self.initial_yaw @ [*cfg["cylinder_xy"], 0])[:2]
        self.table_height = float(s.xpos[waist, 2] + cfg["table_above_waist_m"])
        cylinder[:3, 3] = [*xy, self.table_height + cfg["cylinder_height_m"] / 2]
        relation = np.eye(4)
        relation[:3, 3] = [0.145, -0.035, 0]
        wrist = cylinder @ np.linalg.inv(relation)
        tool = np.eye(4)
        tool[:3, 3] = TCP_OFFSETS["left"]
        grasp = wrist @ tool
        pre = grasp.copy()
        pre[:3, 3] -= self.initial_yaw[:, 0] * cfg["pregrasp_retreat_m"]
        lift = grasp.copy()
        lift[2, 3] += cfg["lift_m"]
        transfer = lift.copy()
        transfer[:3, 3] -= self.initial_yaw[:, 1] * cfg["inward_translation_m"]
        place = transfer.copy()
        place[2, 3] -= cfg["lift_m"]
        self.air_targets = [("AIR_PREGRASP", pre), ("AIR_GRASP", grasp),
                            ("AIR_LIFT", lift), ("AIR_TRANSFER", transfer), ("AIR_PLACE", place)]

    def next_air_target(self):
        exit_errors = {side: self.errors(side) for side in SIDES}
        self.air_index += 1
        if self.air_index == len(self.air_targets):
            self.transition("PASSED", exit_errors)
            return
        phase, target = self.air_targets[self.air_index]
        self.policy.set_target_world("left", target[:3, 3], quaternion_from_matrix(target[:3, :3]))
        self.transition(phase, exit_errors)

    def stable_for_required_time(self, condition):
        if not condition:
            self.stable_since, self.stable_time = None, 0.0
            return False
        if self.stable_since is None:
            self.stable_since = self.data.time
        self.stable_time = self.data.time - self.stable_since
        return self.stable_time >= self.cfg["gates"]["stable_s"] - 1e-8

    def update_gate(self):
        cfg, s = self.cfg, self.scratch
        g = cfg["gates"]
        up = s.xmat[self.base].reshape(3, 3)[2, 2]
        tilt = float(np.rad2deg(np.arccos(np.clip(up, -1, 1))))
        self.peaks["base_tilt_deg"] = max(self.peaks["base_tilt_deg"], tilt)
        if tilt > g["max_base_tilt_deg"] or s.xpos[self.base, 2] < g["min_base_height_m"]:
            self.fail("Base tilt/height safety limit")
            return
        if self.phase == "RESET_SETTLE":
            if s.time < cfg["reset_settle_s"] - 1e-8:
                self.policy.follow_current(s)
            else:
                self.set_standby()
            return
        errors = {side: self.errors(side) for side in SIDES}
        if self.right_lock is not None:
            right = errors["right"]
            for metric, key in (("wrist_position_m", "inactive_wrist_position_error_m"),
                                ("orientation_deg", "inactive_orientation_error_deg")):
                self.peaks[key] = max(self.peaks[key], right[metric])
            bad = (right["wrist_position_m"] > g["inactive_position_m"]
                   or right["orientation_deg"] > g["inactive_orientation_deg"])
            if not bad:
                self.inactive_bad_since = None
            elif self.inactive_bad_since is None:
                self.inactive_bad_since = s.time
            self.inactive_bad_time = s.time - self.inactive_bad_since if bad else 0.0
            if self.inactive_bad_time >= g["inactive_violation_s"] - 1e-8:
                self.fail("Inactive hand did not maintain its locked world pose")
                return
        if self.phase == "STANDBY_REACH":
            at_goal = all(e["position_m"] < g["standby_position_m"]
                          and e["orientation_deg"] < g["standby_orientation_deg"]
                          and e["linear_speed_mps"] < 0.05 for e in errors.values())
            if self.stable_for_required_time(at_goal):
                p, q, _, _ = self.tcp("right")
                self.right_lock = (p.copy(), q.copy())
                self.policy.set_target_world("right", p, q)
                self.transition("STANDING_CHECK", errors)
        elif self.phase == "STANDING_CHECK":
            if s.time - self.phase_start >= cfg["standing_duration_s"] - 1e-8:
                self.standing_passed = True
                self.prepare_air_targets()
                self.next_air_target()
        else:
            e = errors["left"]
            at_goal = (e["wrist_position_m"] < g["position_m"] and e["orientation_deg"] < g["orientation_deg"]
                       and e["wrist_linear_speed_mps"] < g["linear_speed_mps"])
            if self.stable_for_required_time(at_goal):
                self.next_air_target()
        if self.phase != "STANDING_CHECK" and s.time - self.phase_start >= cfg["phase_timeout_s"] - 1e-8:
            self.fail(f"Target did not settle within {cfg['phase_timeout_s']:.1f}s")

    def step(self):
        if self.done:
            raise RuntimeError("Compatibility episode has already stopped")
        if self.steps % 20 == 0:
            self.sync()
            self.update_gate()
            if self.done:
                self.record()
                return
            self.policy.act(self.scratch)
        if self.steps % 10 == 0:
            self.hands.update()
        self.policy.apply(self.data)
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        caps = self.model.actuator_ctrlrange[self.body_map.actuators, 1]
        self.saturation_count += int(np.count_nonzero(np.abs(self.data.ctrl[self.body_map.actuators]) >= caps * 0.99))
        self.torque_count += len(caps)
        reason = self.safety()
        if reason:
            self.sync()  # Failure may occur between the 100 Hz record ticks.
            self.fail(reason)
        if self.steps % 10 == 0 or self.done:
            self.sync()
            self.record()

    def record(self):
        s = self.scratch
        if self.samples and abs(self.samples[-1]["time_s"] - s.time) < 1e-10:
            self.samples.pop()
        arms = {}
        for side in SIDES:
            p, q, v, w = self.tcp(side)
            ref = self.policy.references[side]
            wrist = body_transform(s, self.model.body(f"{side}_hand_roll_link").id)
            arms[side] = {**self.errors(side), "tcp_position_world_m": p.tolist(),
                          "tcp_quaternion_world_wxyz": q.tolist(), "tcp_velocity_world_mps": v.tolist(),
                          "goal_position_world_m": ref.goal_position.tolist(),
                          "goal_quaternion_world_wxyz": ref.goal_quaternion.tolist(),
                          "reference_position_world_m": ref.position.tolist(),
                          "wrist_transform_world": wrist.tolist()}
        self.samples.append({"time_s": float(s.time), "phase": self.phase,
                             "base_position_world_m": s.xpos[self.base].tolist(),
                             "body_q_rad": self.data.qpos[self.body_map.qpos].tolist(),
                             "body_tau_Nm": self.data.ctrl[self.body_map.actuators].tolist(),
                             "body_action": self.policy.last_action.tolist(), "arms": arms,
                             "contacts": self._contacts()})

    def report(self):
        return {"passed": self.phase == "PASSED", "standing_passed": self.standing_passed,
                "phase": self.phase, "failure_phase": self.failure_phase,
                "failure": self.failure, "duration_s": float(self.data.time),
                "scope": "new complete asset, free base, empty hands; no table, object or grasp task executed",
                "configuration": self.cfg, "hand_configuration": self.hand_cfg,
                "initial_base_height_m": self.initial_base_height,
                "initial_contacts": self.initial_contacts, "peaks": self.peaks,
                "body_torque_saturation_fraction": self.saturation_count / max(1, self.torque_count),
                "final_errors": {side: self.errors(side) for side in SIDES},
                "transitions": self.transitions, "contact_pairs_max_depth_m": self.contact_pairs,
                "warnings": self.data.warning.number.tolist(), "mujoco_version": mujoco.__version__,
                "onnx_sha256": sha256(resolve_asset(self.cfg["policy_path"])),
                "table_height_for_future_task_m": getattr(self, "table_height", None),
                "air_targets_world_tcp": {name: pose.tolist() for name, pose in self.air_targets},
                "body_joint_names": list(BODY_JOINTS),
                "runtime_sha256": sha256(Path(__file__)),
                "model_dimensions": {"nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu},
                "body_mass_kg": float(np.sum(self.model.body_mass))}
