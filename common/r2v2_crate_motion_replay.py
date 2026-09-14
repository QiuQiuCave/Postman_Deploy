"""Replay a measured hand/crate motion with an unassisted free-base policy.

Only world wrist goals and binary finger commands are driven. Recorded poses
never become simulator state, IK solutions, mocap drives or external forces.
The source clock pauses at physical gates; spatial targets remain unchanged.
"""
import copy
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_crate import CrateParameters
from common.r2v2_crate_lift import load_lift_config
from common.r2v2_crate_lift_metrics import _descendant
from common.r2v2_crate_motion_recording import load_crate_motion, pose_from_transform
from common.r2v2_crate_motion_replay_scene import build_motion_replay_model
from common.r2v2_crate_reach import CrateReachExperiment
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import (
    angle_error, foot_collision_ids, initialize_robot, load_reach_config,
    quaternion_from_matrix, require_parity, resolve_asset, sha256,
)
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, urdf_hand_joints


SOURCE_PHASES = ("READY", "INSERT", "INSERT_SETTLE", "CLOSE", "PROBE_LIFT", "HOLD")


def relocated_targets(sample, world_anchor):
    """Retain object motion: live-crate-relative targets alone erase the lift."""
    desired_crate = world_anchor @ sample["T_anchor_crate"]
    return desired_crate, desired_crate @ sample["T_crate_wrist"]


class CrateMotionReplayExperiment(CrateReachExperiment):
    def __init__(self, motion_path, reach_config, parity_report, *, table_top_m=None, prepared_state=None,
                 crate_center_xy=None, table_center_xy=None, virtual_props=False, allow_near_table=False):
        self.motion_path = Path(motion_path).resolve()
        if self.motion_path.is_dir():
            self.motion_path /= "manifest.json"
        self.motion = load_crate_motion(self.motion_path)
        self.cfg = load_reach_config(reach_config)
        if self.cfg.get("endpoint_contract") != "wrist_world_v2":
            raise ValueError("Recorded link-origin motion requires wrist_world_v2")
        self.parity_path = Path(parity_report).resolve()
        self.parity_evidence = require_parity(self.parity_path, self.cfg)
        manifest = self.motion.manifest
        if manifest["sides"] != list(SIDES) or manifest["candidate"]["active_sides"] != list(SIDES):
            raise ValueError("This experiment requires the successful bilateral recording")
        if [e["state"] for e in manifest["transitions"]] != [*SOURCE_PHASES, "COMPLETE"]:
            raise ValueError("Unexpected source phase sequence")
        self.source_boundaries = {e["state"]: e["time_s"] for e in manifest["transitions"]}
        candidate = manifest["candidate"]
        # Preserve production safety (15deg), not the fixture's exploratory
        # 45deg stop. The actual success gate remains an 8deg level lift.
        self.params = replace(load_lift_config(), insertion_m=candidate["insertion_m"],
            closed_curl_rad=candidate["closed_curl_rad"], closure_seating_m=candidate["seating_m"], hold_s=2.)
        self.crate_params = CrateParameters(**manifest["crate_parameters"])
        self.fullbody_cfg = dict(phase_timeout_s=10., prealign_position_m=.02,
            prealign_orientation_deg=10., insertion_position_m=.005,
            insertion_orientation_deg=3., motion_speed_mps=.02, motion_stable_s=.3,
            reset_settle_s=.4, standing_minimum_s=2.)
        scene_options = {} if table_top_m is None else {"table_top_m": table_top_m}
        if crate_center_xy is not None:
            scene_options["crate_center_xy"] = crate_center_xy
        if table_center_xy is not None:
            scene_options["table_center_xy"] = table_center_xy
        if virtual_props:
            scene_options["virtual_props"] = True
        if allow_near_table:
            scene_options["allow_near_table"] = True
        self.model, _, self.layout = build_motion_replay_model(self.cfg, self.crate_params, **scene_options)
        self.table_height = self.layout["table_top_m"]
        self.hand_cfg = copy.deepcopy(manifest["hand_configuration"])
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, self.hand_cfg)
        # Native mjlab reset_joints_by_offset clamps reset values into the
        # 90% soft range. Do this once at initialization, never change limits.
        initial_map = JointMap.create(self.model, BODY_JOINTS)
        limits = self.model.jnt_range[initial_map.joints]
        center, half = limits.mean(axis=1), .45*(limits[:, 1]-limits[:, 0])
        self.data.qpos[initial_map.qpos] = np.clip(self.data.qpos[initial_map.qpos], center-half, center+half)
        mujoco.mj_forward(self.model, self.data)
        self.policy = ReachPolicy(self.model, self.data, resolve_asset(self.cfg["policy_path"]),
                                  expected_endpoint_contract="wrist_world_v2")
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        self.base = self.model.body("base_link").id
        self.floor = self.model.geom("floor").id
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.robot_geoms = {g for g in range(self.model.ngeom)
            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.hand_geoms = {g for g in self.robot_geoms if any(_descendant(self.model,
            int(self.model.geom_bodyid[g]), self.model.body(f"{s}_hand_roll_link").id) for s in SIDES)}
        self.table_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model,
            int(self.model.geom_bodyid[g]), self.model.body("tabletop").id)}
        self.crate_geoms = {g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == self.model.body("cargo_crate").id}
        self.joints = [self.model.joint(n).id for n in urdf_hand_joints()]
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0],
                    self.model.joint(mimic.get("joint")).qposadr[0],
                    float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
        self.phase, self.phase_start, self.failure = "RESET_SETTLE", 0., None
        self.failure_phase = None
        self.steps = 0
        self.source_time_s = 0.
        self.world_anchor = None
        self.desired_crate_world = None
        self.goal_wrist_transforms = {s: body_transform(self.data,
            self.model.body(f"{s}_hand_roll_link").id) for s in SIDES}
        self.transitions = [{"time_s": 0., "state": self.phase, "source_time_s": 0.}]
        self.samples, self.hold_samples, self.targets = [], [], []
        self.baseline_relations = None
        self.trial_confirmed = self.prealign_passed = False
        self.stable_since = self.bad_since = None
        self.current_metrics = {}
        self.peaks = dict.fromkeys(("hand_crate_penetration_m", "hand_self_penetration_m",
            "joint_violation_rad", "mimic_error_rad", "slip_m", "crate_tilt_deg", "clearance_m",
            "actuator_torque_Nm", "wrist_tracking_error_m", "wrist_tracking_error_deg", "base_tilt_deg",
            "body_joint_violation_rad", "robot_self_penetration_m"), 0.)
        if prepared_state is not None:
            from common.r2v2_crate_height_state import load_common_start, initialize_from_common
            from common.r2v2_crate_motion_recording import transform_from_pose
            self.prepared_state_info = initialize_from_common(self,
                load_common_start(prepared_state, self.cfg, self.parity_path))
            self.goal_wrist_transforms = {s: transform_from_pose(r.goal_position, r.goal_quaternion)
                                          for s, r in self.policy.references.items()}
        self.sync()
        self.initial_crate_position = np.asarray(self.current_metrics["crate_position_m"]).copy()
        self._safety()
        self.record()

    def enter(self, phase):
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since = self.bad_since = None
        if phase in SOURCE_PHASES:
            self.source_time_s = self.source_boundaries[phase]
        self.transitions.append({"time_s": self.phase_start, "state": phase,
                                 "source_time_s": self.source_time_s})
        if phase == "STAND":
            # Settling follows the measured wrist as in training. Afterwards
            # lock world goals; never keep following a falling or drifting hand.
            for side in SIDES:
                self._set_goal(side, body_transform(self.scratch,
                    self.model.body(f"{side}_hand_roll_link").id))
        elif phase == "PREALIGN":
            settled = self.motion.sample(self.source_boundaries["INSERT"])
            actual_crate = body_transform(self.scratch, self.model.body("cargo_crate").id)
            self.world_anchor = actual_crate @ np.linalg.inv(settled["T_anchor_crate"])
            self.source_time_s = 0.
            self._stream_targets()
        elif phase == "CLOSE":
            for side in SIDES:
                self.hands.command(side, 1)
        elif phase == "PROBE_LIFT":
            self.baseline_relations = {s: np.array(self.current_metrics["hands"][s]["T_wrist_crate"],
                                                  copy=True) for s in SIDES}

    def _set_goal(self, side, transform):
        self.goal_wrist_transforms[side] = transform.copy()
        position, quaternion = pose_from_transform(transform)
        self.policy.set_target_world(side, position, quaternion)

    def _stream_targets(self):
        if self.world_anchor is None:
            return
        sample = self.motion.sample(self.source_time_s)
        self.desired_crate_world, targets = relocated_targets(sample, self.world_anchor)
        for side, target in zip(SIDES, targets):
            self._set_goal(side, target)
        self.targets.append({"time_s": float(self.data.time), "source_time_s": self.source_time_s,
            "phase": self.phase, "T_world_crate_desired": self.desired_crate_world.copy(),
            "T_world_wrist_goal": targets.copy(), "T_crate_wrist_source": sample["T_crate_wrist"].copy()})

    def _update_source_clock(self):
        if self.phase in SOURCE_PHASES:
            index = SOURCE_PHASES.index(self.phase)
            end = self.source_boundaries[((*SOURCE_PHASES, "COMPLETE"))[index+1]]
            self.source_time_s = min(end, self.source_boundaries[self.phase]+float(self.data.time-self.phase_start))

    def _source_segment_finished(self):
        index = SOURCE_PHASES.index(self.phase)
        end = self.source_boundaries[((*SOURCE_PHASES, "COMPLETE"))[index+1]]
        return self.source_time_s >= end-1e-8

    def _lift_good(self):
        m, p = self.current_metrics, self.params
        slip, rotation_slip = m.get("grasp_slip_m"), m.get("grasp_rotation_slip_deg")
        if slip is None or rotation_slip is None or not np.all(np.isfinite([slip, rotation_slip])):
            return False
        net_hand_load = sum(m["hands"][s]["vertical_force_N"] for s in SIDES)
        return bool(m["clearance_m"] >= p.hold_clearance_m and not self._table_supported()
            and self._bilateral_load() and net_hand_load >= .8*self.crate_params.mass*9.81
            and slip < p.max_slip_m and rotation_slip < 5. and m["crate_tilt_deg"] <= p.hold_tilt_deg
            and m["crate_linear_speed_m_s"] < .02 and m["crate_angular_speed_rad_s"] < .1
            and self._at_wrist_targets("insertion"))

    def _gate(self):
        elapsed = float(self.data.time-self.phase_start)
        if self.phase == "RESET_SETTLE":
            if elapsed >= self.fullbody_cfg["reset_settle_s"]-1e-8:
                self.enter("STAND")
        elif self.phase == "STAND":
            good = self.current_metrics["base_tilt_deg"] < 8. and np.linalg.norm(self.scratch.qvel[:6]) < .2
            if self._stable(good, self.fullbody_cfg["standing_minimum_s"]):
                self.enter("PREALIGN")
        elif self.phase == "PREALIGN":
            if self._stable(self._at_wrist_targets("prealign"), self.fullbody_cfg["motion_stable_s"]):
                self.prealign_passed = True
                self.enter("READY")
        elif self.phase in ("READY", "INSERT"):
            if self._source_segment_finished():
                self.enter(SOURCE_PHASES[SOURCE_PHASES.index(self.phase)+1])
        elif self.phase == "INSERT_SETTLE":
            if self._stable(self._source_segment_finished() and self._at_wrist_targets("insertion"), .3):
                self.enter("CLOSE")
        elif self.phase == "CLOSE":
            if self._stable(self._source_segment_finished() and self._bilateral_contact(), .3):
                self.enter("PROBE_LIFT")
        elif self.phase == "PROBE_LIFT":
            if self._source_segment_finished():
                self.enter("HOLD")
        elif self.phase == "HOLD":
            good = self._lift_good()
            self.hold_samples.append({"time_s": float(self.data.time), "valid": good,
                "clearance_m": self.current_metrics["clearance_m"],
                "slip_m": self.current_metrics["grasp_slip_m"]})
            if self._stable(good, self.params.hold_s):
                self.trial_confirmed = True
                self.enter("COMPLETE")
        if self.phase in ("PROBE_LIFT", "HOLD") and not self.done:
            m = self.current_metrics
            bad = (m["grasp_slip_m"] or 0.) >= self.params.max_slip_m or m.get("grasp_rotation_slip_deg", 0.) >= 5.
            if elapsed > .5:
                bad |= not self._bilateral_contact()
            if bad:
                self.bad_since = float(self.data.time) if self.bad_since is None else self.bad_since
                if self.data.time-self.bad_since >= self.params.violation_s:
                    self.fail("Recorded lift lost bilateral contact or exceeded slip limits; hands remain closed")
            else:
                self.bad_since = None
        # Recompute after transitions, not against the previous phase's elapsed.
        if not self.done and self.data.time-self.phase_start >= self.fullbody_cfg["phase_timeout_s"]:
            self.fail(f"{self.phase} timeout: physical pose/contact/stability gate not met; no gate relaxation")

    def sync(self):
        m = super().sync()
        if self.baseline_relations is not None:
            angles = {}
            for side in SIDES:
                current = np.asarray(m["hands"][side]["T_wrist_crate"])
                angles[side] = float(np.rad2deg(angle_error(quaternion_from_matrix(current[:3, :3]),
                    quaternion_from_matrix(self.baseline_relations[side][:3, :3]))))
            m["grasp_rotation_slip_deg"] = max(angles.values())
            m["side_rotation_slip_deg"] = angles
        if self.desired_crate_world is not None:
            m["crate_path_error_m"] = float(np.linalg.norm(np.asarray(m["crate_position_m"])-self.desired_crate_world[:3, 3]))
        return m

    def step(self):
        if self.done:
            return
        if self.steps % 10 == 0:
            self.sync()
            self._safety()
            if not self.done:
                self._update_source_clock()
                self._gate()
            if self.done:
                self.record()
                return
            self.hands.update()
        if self.steps % 20 == 0:
            self.sync()
            if self.phase == "RESET_SETTLE":
                self.policy.follow_current(self.scratch)
                self.goal_wrist_transforms = {s: body_transform(self.scratch,
                    self.model.body(f"{s}_hand_roll_link").id) for s in SIDES}
            else:
                self._stream_targets()
            self.policy.act(self.scratch)
        self.policy.apply(self.data)
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        # Non-finite physics terminates immediately, detailed contact gates at
        # the measurement rate (100Hz). No post-failure integration is allowed.
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or np.any(self.data.warning.number):
            self.fail("Nonfinite state or MuJoCo warning")
        if self.steps % 10 == 0 or self.done:
            self.sync()
            self._safety()
            self.record()

    def record(self):
        super().record()
        self.samples[-1].update(source_time_s=self.source_time_s,
            desired_crate_world=None if self.desired_crate_world is None else self.desired_crate_world.copy())

    def report(self):
        commanded = any(any(h["command"] == 1 for h in row["hands"].values()) for row in self.samples)
        return dict(scope="measured fixture path retargeted to real free-base wrist-v2 policy; no assistance",
            phase=self.phase, failure=self.failure, failure_phase=self.failure_phase,
            experiment_completed=self.phase == "COMPLETE",
            duration_s=float(self.data.time), source_time_s=self.source_time_s,
            source_duration_s=self.motion.duration_s, prealign_passed=self.prealign_passed,
            insertion_attempted=any(e["state"] == "INSERT" for e in self.transitions),
            grasp_commanded=commanded, lift_passed=self.phase == "COMPLETE" and self.trial_confirmed,
            transitions=self.transitions, final_metrics=self.current_metrics,
            final_wrist_errors=self.current_metrics["wrist_errors"], peaks=self.peaks,
            source_motion_manifest=str(self.motion_path), source_motion_sha256=sha256(self.motion_path),
            trajectory_sha256=self.motion.manifest["trajectory_sha256"],
            candidate=self.motion.manifest["candidate"],
            source_signal="measured wrist-link poses, not commanded fixture mocap targets",
            reconstruction="T_W_C_desired(t)=T_W_anchor*T_anchor_C_source(t); T_W_wrist=T_W_C_desired*T_C_wrist_source",
            T_world_anchor=self.world_anchor,
            target_schedule="100Hz recorded path sampled at 50Hz, shortest SLERP; source clock pauses at physical gates",
            hand_schedule="recorded binary close event gated by actual insertion, same 100Hz Ruckig profile; not measured-q teleport",
            hand_configuration=self.hand_cfg, parameters=asdict(self.params), crate_parameters=asdict(self.crate_params),
            layout=self.layout, fullbody_configuration=self.fullbody_cfg, reach_configuration=self.cfg,
            parity_report=str(self.parity_path), parity_report_sha256=sha256(self.parity_path),
            onnx_sha256=self.parity_evidence["onnx_sha256"], checkpoint_sha256=self.parity_evidence["checkpoint_sha256"],
            model_dimensions=dict(nq=self.model.nq, nv=self.model.nv, nu=self.model.nu,
                                  neq=self.model.neq, nmocap=self.model.nmocap),
            no_external_forces=not np.any(self.data.xfrc_applied) and not np.any(self.data.qfrc_applied),
            no_wrist_fixtures=self.model.nmocap == 0 and not np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD),
            contact_safety_check_hz=100, physics_torque_hz=1000, hand_hz=100, policy_hz=50,
            reset_body_q="native default, clamped to 90% soft range at time zero only; real model limits unchanged",
            source_sha256={f: sha256(PROJECT_ROOT/f) for f in (
                "common/r2v2_crate_motion_replay.py", "common/r2v2_crate_motion_recording.py",
                "common/r2v2_crate_motion_replay_scene.py", "common/r2v2_reach_policy.py")})
