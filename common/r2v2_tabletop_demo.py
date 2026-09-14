"""Simulation-only physical pick/place demo; Reach accuracy is diagnostic.

The strict compatibility experiment remains unchanged. This demo is explicitly
authorized to proceed despite its millimetre/degree end-point precision failure.
Objects are initialized once after free-standing warmup, never pinned or reset
while grasping. Contact/lift/slip checks, not the hand's CLOSED flag, prove grip.
"""

from common.path_config import PROJECT_ROOT

import copy
from pathlib import Path
import warnings

import mujoco
import numpy as np
import yaml

from common.r2v2_cylinder_test import CylinderParameters, load_cylinder_profile
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import (
    ReachCompatibilityExperiment, TCP_OFFSETS, angle_error, foot_collision_ids,
    load_reach_config, quaternion_from_matrix, resolve_asset, sha256,
)
from common.r2v2_tabletop_scene import build_tabletop_model, contact_metrics, object_metrics
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES


def load_demo_config(path=None):
    path = Path(path or PROJECT_ROOT / "deploy_mujoco/config/r2v2_tabletop_demo.yaml")
    cfg = yaml.safe_load(path.read_text())
    cfg["reach"] = load_reach_config()
    cfg["cylinder_profile"] = load_cylinder_profile(cfg.get("cylinder_profile"))
    positive = ("warmup_standing_s", "phase_timeout_s", "motion_settle_s", "motion_speed_mps",
                "approach_step_m", "trial_lift_m", "lift_m", "lower_step_m", "touch_step_m",
                "close_minimum_s", "contact_stable_s", "max_slip_m", "max_slip_deg")
    if any(not np.isfinite(cfg[k]) or cfg[k] <= 0 for k in positive):
        raise ValueError("Demo durations, step sizes and physical tolerances must be positive")
    if not 0 < cfg["trial_lift_m"] < cfg["lift_m"]:
        raise ValueError("Trial lift must be smaller than the total lift")
    return cfg


def build_demo_scene_config(cfg, table_height):
    """Pure reset-time placement; no policy, physics stepping or runtime resets."""
    profile = load_cylinder_profile(cfg.get("cylinder_profile"))
    params = CylinderParameters.from_profile(profile)
    return {
        "object_appearance": cfg.get("object_appearance", "orange_cylinder"),
        "cylinder_profile": profile,
        "table_center_xyz": [*cfg["table_center_xy"], table_height - cfg["table_half_size"][2]],
        "table_half_size": copy.deepcopy(cfg["table_half_size"]),
        "cylinder_position_xyz": [*cfg["cylinder_xy"], params.upright_center_height(
            table_height, cfg["cylinder_initial_clearance_m"])],
        "cylinder_initial_clearance_m": cfg["cylinder_initial_clearance_m"],
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def copy_robot_initial_state(source_model, source, model, data):
    """One-time scene initialization by joint/actuator names, never in rollout."""
    for j in range(source_model.njnt):
        target = model.joint(source_model.joint(j).name).id
        assert source_model.jnt_type[j] == model.jnt_type[target]
        free = source_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        nq, nv = (7, 6) if free else (1, 1)
        oldq, newq = source_model.jnt_qposadr[j], model.jnt_qposadr[target]
        oldv, newv = source_model.jnt_dofadr[j], model.jnt_dofadr[target]
        data.qpos[newq:newq+nq] = source.qpos[oldq:oldq+nq]
        data.qvel[newv:newv+nv] = source.qvel[oldv:oldv+nv]
        data.qacc_warmstart[newv:newv+nv] = source.qacc_warmstart[oldv:oldv+nv]
    for a in range(source_model.nu):
        data.ctrl[model.actuator(source_model.actuator(a).name).id] = source.ctrl[a]
    mujoco.mj_forward(model, data)


class TabletopDemoExperiment(ReachCompatibilityExperiment):
    def __init__(self, cfg):
        self.demo_cfg = copy.deepcopy(cfg)
        self.cfg = copy.deepcopy(cfg["reach"])
        self.demo_cfg["cylinder_profile"] = load_cylinder_profile(cfg.get("cylinder_profile"))
        self.cylinder_params = CylinderParameters.from_profile(self.demo_cfg["cylinder_profile"])
        if (self.demo_cfg["cylinder_profile"]["grasp_calibration_status"] == "unvalidated"
                and cfg["grasp_enabled"]):
            warnings.warn("Cylinder profile is unvalidated: the baseline 75-degree hand trajectory "
                          "and wrist-cylinder relation require new grasp calibration. "
                          "This run is an unvalidated simulation experiment, not a verified grasp.",
                          RuntimeWarning, stacklevel=2)
        # The source neutral arms would intersect a waist-high table. Warm up
        # the free robot first, then create the fixed scene at episode reset.
        warm = ReachCompatibilityExperiment(self.cfg)
        while not (warm.phase == "STANDING_CHECK"
                   and warm.data.time - warm.phase_start >= cfg["warmup_standing_s"] - 1e-8):
            warm.step()
            if warm.done:
                raise RuntimeError(f"Free-standing scene preparation failed: {warm.failure}")
        warm.sync()
        self.warmup_duration_s = float(warm.data.time)
        self.table_height = float(warm.scratch.xpos[warm.model.body("waist_pitch_link").id, 2]
                                  + cfg["table_above_waist_m"])
        self.scene_cfg = build_demo_scene_config(self.demo_cfg, self.table_height)
        self.model, self.hand_cfg = build_tabletop_model(self.cfg, self.scene_cfg)
        self.data = mujoco.MjData(self.model)
        copy_robot_initial_state(warm.model, warm.data, self.model, self.data)
        self.policy = ReachPolicy(self.model, self.data, resolve_asset(self.cfg["policy_path"]))
        for key in ("last_action", "q_des", "last_torque", "references", "histories", "last_observation"):
            setattr(self.policy, key, copy.deepcopy(getattr(warm.policy, key)))
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.hands.controllers = warm.hands.controllers  # Model-independent Ruckig references.
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.floor = self.model.geom("floor").id
        self.table_geom = self.model.geom("tabletop_geom").id
        self.cylinder_geom = self.model.geom("cylinder_geom").id
        self.base = self.model.body("base_link").id
        self.scratch = mujoco.MjData(self.model)
        self.phase, self.phase_start = "READY", 0.0
        self.steps = 0
        self.stable_time = self.inactive_bad_time = 0.0
        self.stable_since = self.inactive_bad_since = None
        self.standing_passed = True
        self.failure = self.failure_phase = None
        self.transitions = [{"time_s": 0.0, "state": self.phase}]
        self.samples = []
        self.peaks = {"self_penetration_m": 0.0, "joint_violation_rad": 0.0,
                      "base_tilt_deg": 0.0, "inactive_wrist_position_error_m": 0.0,
                      "inactive_orientation_error_deg": 0.0, "slip_m": 0.0, "slip_deg": 0.0,
                      "object_clearance_m": 0.0, "hand_object_penetration_m": 0.0}
        self.saturation_count = self.torque_count = 0
        self.contact_pairs = {}
        self.initial_contacts = self._contacts()
        self.initial_base_height = warm.initial_base_height
        self.air_targets = []
        self.right_lock = copy.deepcopy(warm.right_lock)
        self.initial_base_xy = warm.initial_base_xy.copy()
        self.object_state = self.contact_state = {}
        self.condition_since = {}
        self.precision_events = []
        self.grasp_relation = self.closing_relation = None
        self.grasp_confirmed = False
        self.released = False
        self.slip_bad_since = None
        self.current_slip_m = self.current_slip_deg = 0.0
        self.opposed_hold_samples = self.hold_samples = 0
        self.approach_index = self.lower_index = self.touch_index = 0
        self.targets = []
        self.sync()
        self.initial_object_transform = self.object_state["T_world_cylinder"].copy()
        self.destination_transform = self.initial_object_transform.copy()
        self.destination_transform[1, 3] -= cfg["inward_translation_m"]
        self.destination_transform[2, 3] = self.cylinder_params.upright_center_height(self.table_height)
        self.record()
        if self.contact_state["robot_table_contacts"]:
            self.fail("Robot/table overlap at prepared scene initialization")

    @property
    def done(self):
        return self.phase in ("FAILED", "COMPLETE")

    def sync(self):
        super().sync()
        self.object_state = object_metrics(self.model, self.scratch)
        # Forces belong to the live just-completed physics step, not an extra solve.
        self.contact_state = contact_metrics(self.model, self.data, side="left")

    def wrist(self):
        return body_transform(self.scratch, self.model.body("left_hand_roll_link").id)

    def relative_object(self):
        return np.linalg.inv(self.wrist()) @ self.object_state["T_world_cylinder"]

    def condition_held(self, key, condition, duration):
        if not condition:
            self.condition_since.pop(key, None)
            return False
        since = self.condition_since.setdefault(key, self.data.time)
        return self.data.time - since >= duration - 1e-8

    def transition(self, phase, exit_errors=None):
        # A motion-stage boundary must not forgive an ongoing grip loss.
        self.condition_since = {k: v for k, v in self.condition_since.items() if k in ("slip", "right_hold")}
        super().transition(phase, exit_errors)

    def set_wrist_target(self, phase, target):
        exit_errors = {side: self.errors(side) for side in SIDES}
        tool = np.eye(4)
        tool[:3, 3] = TCP_OFFSETS["left"]
        tcp = target @ tool
        self.policy.set_target_world("left", tcp[:3, 3], quaternion_from_matrix(tcp[:3, :3]))
        self.targets.append({"time_s": float(self.data.time), "phase": phase,
                             "wrist_transform_world": target.tolist()})
        self.transition(phase, exit_errors)

    def set_object_target(self, phase, target):
        if self.grasp_relation is None:
            raise RuntimeError("Cannot transport before contact-and-lift grasp confirmation")
        self.set_wrist_target(phase, target @ np.linalg.inv(self.grasp_relation))

    def motion_settled(self):
        ref, e = self.policy.references["left"], self.errors("left")
        reference_done = (np.linalg.norm(ref.goal_position-ref.position) < 1e-5
                          and angle_error(ref.goal_quaternion, ref.quaternion) < 1e-4)
        settled = self.condition_held("motion", reference_done
                                     and e["wrist_linear_speed_mps"] < self.demo_cfg["motion_speed_mps"],
                                     self.demo_cfg["motion_settle_s"])
        if settled:
            g = self.cfg["gates"]
            self.precision_events.append({"time_s": float(self.data.time), "phase": self.phase,
                "errors": e, "strict_reach_passed": bool(e["wrist_position_m"] < g["position_m"]
                                                          and e["orientation_deg"] < g["orientation_deg"]),
                "mode": "diagnostic_only"})
        return settled

    def prepare_approach(self):
        relation = np.eye(4)
        relation[:3, 3] = [0.145, -0.035, 0]
        self.grasp_wrist_target = self.object_state["T_world_cylinder"] @ np.linalg.inv(relation)
        self.pregrasp_wrist_target = self.grasp_wrist_target.copy()
        # The left palm faces local -Y, not the fingers' longitudinal +X.
        # First clear the cylinder laterally before descending/advancing.
        self.pregrasp_wrist_target[:3, 3] += self.grasp_wrist_target[:3, 1] * self.demo_cfg["pregrasp_retreat_m"]
        align = self.wrist()
        align[:3, :3] = self.grasp_wrist_target[:3, :3]
        axis = self.grasp_wrist_target[:3, 1]
        align[:3, 3] += axis * np.dot(self.pregrasp_wrist_target[:3, 3] - align[:3, 3], axis)
        self.set_wrist_target("OUTWARD_ALIGN", align)

    def next_approach(self):
        self.approach_index += 1
        remaining = max(0.0, self.demo_cfg["pregrasp_retreat_m"]
                        - self.approach_index * self.demo_cfg["approach_step_m"])
        target = self.grasp_wrist_target.copy()
        target[:3, 3] += target[:3, 1] * remaining
        self.set_wrist_target(f"APPROACH_{self.approach_index}", target)

    def update_slip(self):
        if not self.grasp_confirmed or self.released:
            return
        relation = self.relative_object()
        self.current_slip_m = float(np.linalg.norm(relation[:3, 3] - self.grasp_relation[:3, 3]))
        self.current_slip_deg = float(np.rad2deg(angle_error(quaternion_from_matrix(relation[:3, :3]),
                                               quaternion_from_matrix(self.grasp_relation[:3, :3]))))
        self.peaks["slip_m"] = max(self.peaks["slip_m"], self.current_slip_m)
        self.peaks["slip_deg"] = max(self.peaks["slip_deg"], self.current_slip_deg)
        supported = self.contact_state["table_contact"]
        placing = self.phase.startswith(("LOWER_", "TOUCH_"))
        if not supported:
            self.hold_samples += 1
            self.opposed_hold_samples += int(self.contact_state["opposed"])
        if not supported or not placing:
            bad = (self.current_slip_m > self.demo_cfg["max_slip_m"]
                   or self.current_slip_deg > self.demo_cfg["max_slip_deg"]
                   or not self.contact_state["opposed"] or (supported and not placing))
            if self.condition_held("slip", bad, self.demo_cfg["slip_violation_s"]):
                self.fail("Grasp lost: sustained relative slip or opposing contact loss")
        else:
            self.condition_since.pop("slip", None)

    def update_gate(self):
        c, s = self.demo_cfg, self.scratch
        tilt = float(np.rad2deg(np.arccos(np.clip(s.xmat[self.base].reshape(3, 3)[2, 2], -1, 1))))
        self.peaks["base_tilt_deg"] = max(self.peaks["base_tilt_deg"], tilt)
        if tilt > self.cfg["gates"]["max_base_tilt_deg"] or s.xpos[self.base, 2] < self.cfg["gates"]["min_base_height_m"]:
            self.fail("Base tilt/height safety limit")
            return
        er = self.errors("right")
        self.peaks["inactive_wrist_position_error_m"] = max(self.peaks["inactive_wrist_position_error_m"], er["wrist_position_m"])
        self.peaks["inactive_orientation_error_deg"] = max(self.peaks["inactive_orientation_error_deg"], er["orientation_deg"])
        g = self.cfg["gates"]
        if self.condition_held("right_hold", er["wrist_position_m"] > g["inactive_position_m"]
                               or er["orientation_deg"] > g["inactive_orientation_deg"], g["inactive_violation_s"]):
            self.fail("Inactive right wrist did not maintain its locked world pose")
            return
        clearance = self.object_state["bottom_height_m"] - self.table_height
        self.peaks["object_clearance_m"] = max(self.peaks["object_clearance_m"], clearance)
        self.peaks["hand_object_penetration_m"] = max(self.peaks["hand_object_penetration_m"],
                                                      self.contact_state["max_hand_object_penetration_m"])
        self.update_slip()
        if self.done:
            return
        elapsed = self.data.time - self.phase_start
        phase = self.phase
        if phase == "READY":
            if elapsed >= c["ready_s"]:
                self.prepare_approach()
        elif phase == "OUTWARD_ALIGN":
            if self.motion_settled():
                self.set_wrist_target("SIDE_PREGRASP", self.pregrasp_wrist_target)
        elif phase == "SIDE_PREGRASP":
            if self.motion_settled():
                self.next_approach()
        elif phase.startswith("APPROACH_"):
            if self.motion_settled():
                if self.approach_index * c["approach_step_m"] < c["pregrasp_retreat_m"] - 1e-8:
                    self.next_approach()
                else:
                    self.hands.command("left", int(c["grasp_enabled"]))
                    self.transition("CLOSE")
        elif phase == "CLOSE":
            opposed = self.condition_held("closure_contact", self.contact_state["opposed"], c["contact_stable_s"])
            if elapsed >= c["close_minimum_s"] and opposed:
                self.closing_relation = self.relative_object()
                trial = self.wrist()
                trial[2, 3] += c["trial_lift_m"]
                self.set_wrist_target("TRIAL_LIFT", trial)
        elif phase == "TRIAL_LIFT":
            relative = self.relative_object()
            slip = float(np.linalg.norm(relative[:3, 3] - self.closing_relation[:3, 3]))
            slip_angle = float(np.rad2deg(angle_error(quaternion_from_matrix(relative[:3, :3]),
                                                      quaternion_from_matrix(self.closing_relation[:3, :3]))))
            real_grip = (clearance > c["trial_clearance_m"] and not self.contact_state["table_contact"]
                         and self.contact_state["opposed"] and slip < c["max_slip_m"]
                         and slip_angle < c["max_slip_deg"])
            if self.condition_held("real_grip", real_grip, c["contact_stable_s"]):
                self.grasp_confirmed = True
                self.grasp_relation = self.relative_object()
                self.carry_transform = self.object_state["T_world_cylinder"].copy()
                self.carry_transform[:3, 3] = self.initial_object_transform[:3, 3]
                self.carry_transform[2, 3] = self.cylinder_params.upright_center_height(self.table_height, c["lift_m"])
                self.set_object_target("LIFT", self.carry_transform)
        elif phase == "LIFT":
            if self.motion_settled():
                self.carry_transform[:2, 3] = self.destination_transform[:2, 3]
                self.set_object_target("TRANSFER", self.carry_transform)
        elif phase == "TRANSFER":
            if self.motion_settled():
                self.carry_transform[:3, :3] = self.destination_transform[:3, :3]
                self.set_object_target("ORIENT_UPRIGHT", self.carry_transform)
        elif phase == "ORIENT_UPRIGHT" or phase.startswith("LOWER_"):
            if self.motion_settled():
                self.lower_index += 1
                remaining = max(0.0, c["lift_m"] - self.lower_index * c["lower_step_m"])
                target = self.destination_transform.copy()
                target[2, 3] += remaining
                self.set_object_target(f"LOWER_{self.lower_index}" if remaining > 0 else "TOUCH_0", target)
        elif phase.startswith("TOUCH_"):
            supported = (self.contact_state["table_contact"]
                         and np.linalg.norm(self.object_state["linear_velocity_mps"]) < 0.02
                         and self.object_state["tilt_deg"] < c["final_tilt_deg"])
            if self.condition_held("supported", supported, c["contact_stable_s"]):
                # Stop issuing descent increments and retain the final goal.
                # Re-targeting the biased actual wrist would compound its
                # orientation error, rather than hold the previous reference.
                self.hands.command("left", 0)
                self.released = True
                self.transition("RELEASE")
            elif self.motion_settled() and not self.contact_state["table_contact"]:
                self.touch_index += 1
                descent = self.touch_index * c["touch_step_m"]
                if descent > c["max_touch_descent_m"]:
                    self.fail("No table support within bounded contact-seeking descent")
                else:
                    target = self.destination_transform.copy()
                    target[2, 3] -= descent
                    self.set_object_target(f"TOUCH_{self.touch_index}", target)
        elif phase == "RELEASE":
            detached = (self.hands.controllers["left"].finished
                        and max(self.contact_state["fingers_normal_force_N"].values(), default=0) < 0.05)
            if self.condition_held("detached", detached, c["contact_stable_s"]):
                target = self.wrist()
                target[:3, 3] += target[:3, 1] * c["retreat_distance_m"]
                target[:3, 3] -= target[:3, 0] * c["retreat_backoff_m"]
                self.set_wrist_target("RETREAT", target)
        elif phase == "RETREAT":
            if self.motion_settled():
                self.transition("FINAL_HOLD")
        elif phase == "FINAL_HOLD":
            good = (np.linalg.norm(self.object_state["position_m"] - self.destination_transform[:3, 3]) < c["final_position_m"]
                    and self.object_state["tilt_deg"] < c["final_tilt_deg"]
                    and np.linalg.norm(self.object_state["linear_velocity_mps"]) < 0.01
                    and self.contact_state["table_contact"]
                    and max(self.contact_state["fingers_normal_force_N"].values(), default=0) < 0.05)
            if self.condition_held("final", good, c["final_stable_s"]):
                self.transition("COMPLETE")
        if not self.done and self.data.time - self.phase_start >= c["phase_timeout_s"] - 1e-8:
            self.fail(f"{self.phase} did not complete within {c['phase_timeout_s']:.1f}s")

    def safety(self):
        m, d = self.model, self.data
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or np.any(d.warning.number):
            return "Nonfinite state or MuJoCo numerical warning"
        q, limits = d.qpos[self.body_map.qpos], m.jnt_range[self.body_map.joints]
        violation = float(max(0.0, np.max(limits[:, 0] - q), np.max(q - limits[:, 1])))
        self.peaks["joint_violation_rad"] = max(self.peaks["joint_violation_rad"], violation)
        if violation > self.cfg["gates"]["max_joint_violation_rad"]:
            return "Measured body joint exceeded physical limit tolerance"
        for c in d.contact:
            if c.dist > 0:
                continue
            names = [m.geom(g).name for g in c.geom]
            if self.cylinder_geom in c.geom:
                if self.floor in c.geom:
                    return "Cylinder fell to the floor"
                continue
            if any(name.startswith("tabletop") for name in names):
                other = c.geom2 if names[0].startswith("tabletop") else c.geom1
                if other != self.floor and not m.geom(other).name.startswith("tabletop"):
                    return f"Robot/table contact: {m.geom(other).name}"
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

    def record(self):
        super().record()
        hands = {}
        for side in SIDES:
            mapping = self.hands.maps[side]
            ref = self.hands.controllers[side].reference
            hands[side] = {"joint_names": [self.model.joint(j).name for j in mapping.joints],
                           "position_rad": self.data.qpos[mapping.qpos].tolist(),
                           "velocity_radps": self.data.qvel[mapping.dofs].tolist(),
                           "reference_position_rad": ref.position.tolist(),
                           "reference_velocity_radps": ref.velocity.tolist(),
                           "torque_Nm": self.data.ctrl[mapping.actuators].tolist()}
        self.samples[-1].update({"object": serializable(self.object_state),
                                 "object_contacts": serializable(self.contact_state),
                                 "hands": hands,
                                 "left_hand_command": int(self.hands.controllers["left"].command),
                                 "grasp_confirmed": self.grasp_confirmed, "release_commanded": self.released,
                                 "slip_m": self.current_slip_m, "slip_deg": self.current_slip_deg})

    def report(self):
        result = super().report()
        result.update({"passed": self.phase == "COMPLETE", "demo_completed": self.phase == "COMPLETE",
                       "strict_task_acceptance": False, "reach_precision_gate_bypassed": True,
                       "scope": "Single physical simulation demo; not strict precision or three-trial acceptance",
                       "demo_configuration": self.demo_cfg, "scene_configuration": self.scene_cfg,
                       "cylinder_profile": load_cylinder_profile(self.demo_cfg.get("cylinder_profile")),
                       "warmup_duration_s": self.warmup_duration_s, "table_height_m": self.table_height,
                       "grasp_confirmed": self.grasp_confirmed, "release_commanded": self.released,
                       "grasp_relation": serializable(self.grasp_relation), "object_final": serializable(self.object_state),
                       "object_contacts_final": serializable(self.contact_state),
                       "object_start_position_m": self.initial_object_transform[:3, 3].tolist(),
                       "object_target_position_m": self.destination_transform[:3, 3].tolist(),
                       "object_final_position_error_m": float(np.linalg.norm(self.object_state["position_m"]
                                                                              - self.destination_transform[:3, 3])),
                       "opposed_contact_fraction_while_unsupported": self.opposed_hold_samples / max(1, self.hold_samples),
                       "precision_events": self.precision_events, "wrist_targets": self.targets,
                       "demo_code_sha256": sha256(Path(__file__)),
                       "scene_code_sha256": sha256(PROJECT_ROOT / "common/r2v2_tabletop_scene.py"),
                       "multiccd_enabled": bool(self.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD),
                       "object_runtime_resets": 0, "object_welds": 0, "external_support_forces": False})
        return result
