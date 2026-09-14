"""Exploratory full-schedule crate contact test driven solely by the Reach actor.

Prop contact, poor tracking, failed closure, crate tilt/drop, slip and transient
constraint deviations are measurements, NOT schedule stops. Falling/non-foot
ground contact and numerical failure still stop this simulation-only test.
There are no fixtures, state resets, IK controls or hidden object motion.
"""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from common.r2v2_crate_height_path import DEFAULT_MOTION_PATH
from common.r2v2_crate_motion_recording import (
    _validate_transform, interpolate_transform, transform_from_pose,
)
from common.r2v2_crate_motion_replay import CrateMotionReplayExperiment
from common.r2v2_grasp_recording import body_transform
from common.r2v2_reach_sim import sha256
from r2v2_description.model import SIDES

HOME = np.array([[.246, .19698, 1.087902013081883], [.246, -.19643, 1.087902013081883]])
SAFE = np.array([[.18, .27, 1.13], [.18, -.27, 1.13]])
PREPARATION_S = 15.4
PATH_TASK = "R2V2-Reach-CrateWristPath-v1-28DoF"
PAYLOAD_TASK = "R2V2-Reach-CrateWristPayload-v2-28DoF"
PAYLOAD_CONTRACT = "wrist_payload_path_v2"
HOLDING_PHASES = ("PROBE_LIFT", "HOLD", "HOLD_PROBE", "LIFT_HIGHER", "HOLD_HIGHER")


def validate_contact_policy_path(path, parity, metadata):
    """A new path must retain its real task identity and exported archive hash."""
    if parity.get("task") != path.task_id:
        raise ValueError("Parity evidence is not for this path task")
    if metadata.get("path_trajectory_sha256") != path.sha256:
        raise ValueError("Path archive differs from the checkpoint's exported training path")
    if path.task_id == PAYLOAD_TASK:
        if metadata.get("task_id") != PAYLOAD_TASK or metadata.get("path_contract") != PAYLOAD_CONTRACT:
            raise ValueError("Payload ONNX task/path contract mismatch")


class ArchivedContactPath:
    """NumPy mirror of native WristPathData, targets only, not robot poses."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        archive = self.manifest_path.parent/self.manifest["trajectory_file"]
        self.sha256 = sha256(archive)
        if self.sha256 != self.manifest["trajectory_sha256"]:
            raise ValueError("Desired path archive hash mismatch")
        if self.manifest["delta_x_m"] != -.05 or self.manifest["delta_z_m"] != -.05:
            raise ValueError("This physical test is for the reviewed closer5/down5 scene")
        with np.load(archive, allow_pickle=False) as data:
            self.time = data["time_s"].copy()
            self.wrists = _validate_transform(data["T_world_wrist_goal"]).copy()
            self.crate = _validate_transform(data["T_world_crate_desired"]).copy()
            phases = data["phase"].copy()
        if not np.all(np.isfinite(self.time)) or not np.all(np.diff(self.time) > 0):
            raise ValueError("Nonfinite or nonmonotonic path clock")
        phases[phases == "COMPLETE"] = "HOLD"
        source_phases = tuple(dict.fromkeys(phases.tolist()))
        payload_contract = self.manifest.get("payload_training", {}).get("contract")
        if payload_contract is not None:
            if payload_contract != PAYLOAD_CONTRACT or source_phases[-5:] != (
                    "CLOSE_SEAT", "PROBE_LIFT", "HOLD_PROBE", "LIFT_HIGHER", "HOLD_HIGHER"):
                raise ValueError("Unsupported payload contact path contract or phase sequence")
            self.task_id, self.close_phase = PAYLOAD_TASK, "CLOSE_SEAT"
        else:
            if "CLOSE" not in source_phases or "CLOSE_SEAT" in source_phases:
                raise ValueError("Unsupported contact path phase sequence")
            self.task_id, self.close_phase = PATH_TASK, "CLOSE"
        self.phase_names = ("STAND", "HOME", "SAFE_WAIT", "START_HOLD", *source_phases)
        self.phase_start = np.r_[0., .4, 5.4, 13.4,
            [self.time[np.flatnonzero(phases == p)[0]]+PREPARATION_S for p in source_phases]]
        self.duration_s = PREPARATION_S+float(self.time[-1])
        self.close_time = self.phase_start[self.phase_names.index(self.close_phase)]

    def sample(self, time_s):
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Expected finite nonnegative simulation time")
        phase = self.phase_names[min(len(self.phase_names)-1, np.searchsorted(self.phase_start, time_s+1e-9, side="right")-1)]
        if time_s < PREPARATION_S:
            t = np.clip((time_s-5.4)/8., 0., 1.)
            p = HOME+(SAFE-HOME)*(t*t*(3.-2.*t))
            wrists = np.array([transform_from_pose(x, [1., 0., 0., 0.]) for x in p])
            return phase, wrists, self.crate[0].copy()
        t = np.clip(time_s-PREPARATION_S, 0., self.time[-1])
        upper = int(np.clip(np.searchsorted(self.time, t, side="right"), 1, len(self.time)-1))
        lower = upper-1
        w = float(np.clip((t-self.time[lower])/(self.time[upper]-self.time[lower]), 0., 1.))
        return (phase, interpolate_transform(self.wrists[lower], self.wrists[upper], w),
                interpolate_transform(self.crate[lower], self.crate[upper], w))


def contact_lift_conditions(metrics, *, baseline_contact_verified, weight_N):
    """Physical pickup and accurate Reach are deliberately separate verdicts."""
    m = metrics
    slip, rotation = m.get("grasp_slip_m"), m.get("grasp_rotation_slip_deg")
    valid_slip = (slip is not None and rotation is not None and np.isfinite(slip)
                  and np.isfinite(rotation) and slip < .015 and rotation < 5.)
    bilateral = all(min(m["hands"][s]["finger_handle_vertical_force_N"],
                        m["hands"][s]["vertical_force_N"]) >= .2 for s in SIDES)
    hand_load = sum(m["hands"][s]["vertical_force_N"] for s in SIDES)
    physical = bool(baseline_contact_verified and valid_slip and bilateral
        and hand_load >= .8*weight_N and m["clearance_m"] >= .008
        and not m["table_contact"] and not m["floor_contact"]
        and m.get("nonhand_crate_contact_count", 0) == 0
        and m.get("robot_table_contact_count", 0) == 0
        and m["crate_tilt_deg"] <= 8. and m["crate_linear_speed_m_s"] < .02
        and m["crate_angular_speed_rad_s"] < .1)
    accurate = all(m["wrist_errors"][s]["position_m"] < .005
        and m["wrist_errors"][s]["orientation_deg"] < 3.
        and m["wrist_errors"][s]["linear_speed_mps"] < .02 for s in SIDES)
    return physical, bool(physical and accurate)


class WristPathContactExperiment(CrateMotionReplayExperiment):
    def __init__(self, path_manifest, reach_config, parity_report):
        self.path = ArchivedContactPath(path_manifest)
        self.physical_hold_s = self.strict_hold_s = self.contact_hold_s = 0.
        self.max_physical_hold_s = self.max_strict_hold_s = 0.
        self._hold_since = dict(contact=None, physical=None, strict=None)
        self.baseline_contact_verified = False
        self.constraint_events, self.constraint_active, self.constraint_violations_seen = [], set(), set()
        self.foot_anchor = None
        meta = self.path.manifest["path_metadata"]
        super().__init__(DEFAULT_MOTION_PATH, reach_config, parity_report,
            table_top_m=meta["table_top_m"], crate_center_xy=meta["crate_center_xy_m"],
            table_center_xy=(.48+meta["delta_x_m"], 0.), virtual_props=False, allow_near_table=True)
        validate_contact_policy_path(self.path, self.parity_evidence, self.policy.metadata)
        self.initial_metrics = copy.deepcopy(self.current_metrics)
        # Never clear an initial robot-safety failure. Prop contact cannot
        # produce one in this explicitly authorized exploratory subclass.
        if not self.done:
            self.phase = "STAND"
            self.phase_start = 0.
            self.transitions = [dict(time_s=0., state="STAND", source_time_s=0.)]
            self.samples = []
            self.foot_anchor = np.array([self.scratch.xpos[self.model.body(f"{s}_ankle_roll_link").id] for s in SIDES])
            self.record()

    def enter(self, phase):
        if phase == "FAILED":
            self.failure_phase = self.phase
        self.phase, self.phase_start = phase, float(self.data.time)
        self.transitions.append(dict(time_s=self.phase_start, state=phase, source_time_s=self.source_time_s))
        if phase == self.path.close_phase:
            for side in SIDES:
                self.hands.command(side, 1)
        if phase == "PROBE_LIFT":
            self.baseline_contact_verified = self.contact_hold_s >= .3-1e-8
            self.baseline_relations = {s: np.asarray(self.current_metrics["hands"][s]["T_wrist_crate"]).copy() for s in SIDES}

    def _gate(self):
        """Clock only. No grasp/pose/prop collision stops or goal chasing."""
        t = float(self.data.time)
        phase, _, _ = self.path.sample(t)
        if phase != self.phase:
            self.enter(phase)
        if self.phase == self.path.close_phase:
            self.contact_hold_s = self._continuous_hold("contact", self._bilateral_contact(), t)
        if self.phase in HOLDING_PHASES:
            physical, strict = contact_lift_conditions(self.current_metrics,
                baseline_contact_verified=self.baseline_contact_verified,
                weight_N=self.crate_params.mass*9.81)
            self.physical_hold_s = self._continuous_hold("physical", physical, t)
            self.strict_hold_s = self._continuous_hold("strict", strict, t)
            self.max_physical_hold_s = max(self.max_physical_hold_s, self.physical_hold_s)
            self.max_strict_hold_s = max(self.max_strict_hold_s, self.strict_hold_s)
            self.hold_samples.append(dict(time_s=t, physical=physical, strict=strict))
        # Evaluate the terminal state's real contacts before declaring the
        # clock complete; never reuse the previous tick's successful hold.
        if t >= self.path.duration_s+2.-1e-8:
            self.enter("COMPLETE")

    def _continuous_hold(self, name, valid, time_s):
        if not valid:
            self._hold_since[name] = None
            return 0.
        if self._hold_since[name] is None:
            self._hold_since[name] = time_s
        return time_s-self._hold_since[name]

    def _update_source_clock(self):
        self.source_time_s = max(0., float(self.data.time)-PREPARATION_S)

    def _stream_targets(self):
        if self.done:
            return
        _, wrists, crate = self.path.sample(float(self.data.time))
        self.desired_crate_world = crate
        for side, target in zip(SIDES, wrists):
            self._set_goal(side, target)
        self.targets.append(dict(time_s=float(self.data.time), phase=self.phase,
            T_world_wrist_goal=wrists, T_world_crate_desired=crate,
            executed_hand_command=[self.hands.controllers[s].command for s in SIDES]))

    def _safety(self):
        """Measure contact/constraint deviations; stop falls and invalid physics.

        These are software acceptance thresholds, not MuJoCo constraints. All
        real joint limits, mimic equations, collision and effort limits remain
        unchanged even when their instantaneous residual exceeds a threshold.
        """
        m, d, g = self.current_metrics, self.data, self.cfg["gates"]
        if not m["finite_state"] or np.any(d.warning.number):
            self.fail("Nonfinite state or MuJoCo warning"); return
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or self.model.nmocap or np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD):
            self.fail("Unexpected assistance or wrist fixture"); return
        q, limits = d.qpos[self.body_map.qpos], self.model.jnt_range[self.body_map.joints]
        body_violation = max(0., float(np.max(limits[:, 0]-q)), float(np.max(q-limits[:, 1])))
        hq, hl = d.qpos[self.model.jnt_qposadr[self.joints]], self.model.jnt_range[self.joints]
        hand_violation = max(0., float(np.max(hl[:, 0]-hq)), float(np.max(hq-hl[:, 1])))
        mimic = max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a, b, ratio, offset in self.mimics)
        reasons, self_depth, table_contacts, nonhand_crate = [], 0., 0, 0
        for contact in self.scratch.contact:
            if contact.dist > 0:
                continue
            pair = set(map(int, contact.geom))
            if self.floor in pair and pair & self.robot_geoms and not pair & self.foot_geoms:
                reasons.append("non-foot ground contact")
            if pair <= self.robot_geoms:
                self_depth = max(self_depth, -float(contact.dist))
            table_contacts += int(bool(pair & self.robot_geoms and pair & self.table_geoms))
            nonhand_crate += int(bool(pair & self.crate_geoms and pair & (self.robot_geoms-self.hand_geoms)))
        m.update(robot_table_contact_count=table_contacts, nonhand_crate_contact_count=nonhand_crate,
                 body_joint_violation_rad=body_violation, robot_self_penetration_m=self_depth)
        values = dict(joint_violation_rad=hand_violation, body_joint_violation_rad=body_violation,
            mimic_error_rad=mimic, robot_self_penetration_m=self_depth,
            hand_crate_penetration_m=m["max_hand_crate_penetration_m"],
            hand_self_penetration_m=m["max_hand_self_penetration_m"], base_tilt_deg=m["base_tilt_deg"],
            crate_tilt_deg=m["crate_tilt_deg"], clearance_m=m["clearance_m"], slip_m=m["grasp_slip_m"] or 0.,
            wrist_tracking_error_m=m["wrist_tracking_error_m"], wrist_tracking_error_deg=m["wrist_tracking_error_deg"],
            actuator_torque_Nm=max(float(np.max(np.abs(d.actuator_force[h.actuators]))) for h in self.hands.maps.values()))
        for key, value in values.items():
            self.peaks[key] = max(self.peaks[key], value)
        deviations = set()
        if body_violation > min(.01, g["max_joint_violation_rad"]): deviations.add("body joint limit")
        if hand_violation > self.params.max_joint_violation_rad: deviations.add("hand joint limit")
        if mimic > self.params.max_mimic_error_rad: deviations.add("hand mimic error")
        if self_depth > min(.002, g["max_self_penetration_m"]): deviations.add("robot self penetration")
        if m["max_hand_crate_penetration_m"] > self.params.max_penetration_m:
            deviations.add("hand-crate penetration")
        m["constraint_deviations"] = sorted(deviations)
        if deviations != self.constraint_active:
            self.constraint_events.append(dict(time_s=float(d.time), phase=self.phase,
                active=sorted(deviations), values=values.copy()))
            self.constraint_active = deviations
        self.constraint_violations_seen.update(deviations)
        if m["base_tilt_deg"] > g["max_base_tilt_deg"] or self.scratch.xpos[self.base, 2] < g["min_base_height_m"]:
            reasons.append("base tilt/height")
        if reasons:
            self.fail("Robot safety stop: "+", ".join(sorted(set(reasons))))

    def step(self):
        if self.done:
            return
        if self.steps % 10 == 0:
            self.sync(); self._safety()
            if not self.done:
                self._update_source_clock(); self._gate()
            if self.done:
                self.record(); return
            self.hands.update()
        if self.steps % 20 == 0:
            self.sync()
            if self.phase == "STAND":
                self.policy.follow_current(self.scratch)
                self.goal_wrist_transforms = {s: body_transform(self.scratch, self.model.body(f"{s}_hand_roll_link").id) for s in SIDES}
            else:
                self._stream_targets()
            self.policy.act(self.scratch)
        self.policy.apply(self.data)
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or np.any(self.data.warning.number):
            self.fail("Nonfinite state or MuJoCo warning")
        if self.steps % 10 == 0 or self.done:
            self.sync(); self._safety(); self.record()

    def report(self):
        completed = self.phase == "COMPLETE" and not self.failure
        return dict(scope="timed exploratory REAL-CONTACT full-body trial, not pose-gated grasp permission",
            phase=self.phase, failure=self.failure, failure_phase=self.failure_phase,
            experiment_completed=completed, schedule_completed=completed,
            duration_s=float(self.data.time), planned_duration_s=self.path.duration_s+2.,
            grasp_commanded=any(any(h["command"] for h in row["hands"].values()) for row in self.samples),
            lift_passed=completed and self.physical_hold_s >= 2.-1e-8,
            physical_pickup_verified=completed and self.physical_hold_s >= 2.-1e-8,
            pickup_observed=self.max_physical_hold_s >= 2.-1e-8,
            strict_success=completed and self.strict_hold_s >= 2.-1e-8 and not self.constraint_violations_seen,
            final_physical_hold_s=self.physical_hold_s, final_strict_hold_s=self.strict_hold_s,
            maximum_physical_hold_s=self.max_physical_hold_s, maximum_strict_hold_s=self.max_strict_hold_s,
            baseline_contact_verified=self.baseline_contact_verified,
            physical_success_criteria=dict(clearance_m=.008, hold_s=2., slip_m=.015, rotation_slip_deg=5.,
                tilt_deg=8., minimum_side_handle_load_N=.2, minimum_weight_fraction=.8,
                no_table_or_floor_support=True, no_nonhand_crate_or_robot_table_contact=True),
            strict_success_adds=dict(wrist_position_m=.005, wrist_orientation_deg=3., wrist_speed_mps=.02),
            prop_collision_stops=False, prop_contacts_enabled=True, pose_error_stops=False,
            slip_drop_tilt_stops=False, robot_numerical_safety_stops=True,
            software_constraint_deviation_stops=False, physical_constraints_unchanged=True,
            constraints_passed=not self.constraint_violations_seen,
            constraint_violations_seen=sorted(self.constraint_violations_seen), constraint_events=self.constraint_events,
            no_external_forces=not np.any(self.data.xfrc_applied) and not np.any(self.data.qfrc_applied),
            no_wrist_fixtures=self.model.nmocap == 0 and not np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD),
            current_goal_clock_is_not_live_crate_pose=True, transitions=self.transitions,
            initial_metrics=self.initial_metrics, final_metrics=self.current_metrics, peaks=self.peaks, layout=self.layout,
            hand_configuration=self.hand_cfg, crate_parameters=asdict(self.crate_params),
            reach_configuration=self.cfg, path_manifest=str(self.path.manifest_path), trajectory_sha256=self.path.sha256,
            training_task_id=self.path.task_id, close_command_phase=self.path.close_phase,
            payload_training_external_force_replayed=False,
            load_source="Actual free crate contact only; no extra equivalent payload force",
            checkpoint_sha256=self.parity_evidence["checkpoint_sha256"], onnx_sha256=self.parity_evidence["onnx_sha256"],
            parity_report=str(self.parity_path), model_dimensions=dict(nq=self.model.nq,nv=self.model.nv,nu=self.model.nu,neq=self.model.neq,nmocap=self.model.nmocap),
            timing_hz=dict(physics_torque=1000, hands=100, body_policy=50),
            note="No automatic opening after a failed loaded attempt; hands remain commanded closed after the close phase.")
