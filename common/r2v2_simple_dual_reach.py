"""Fixed-duration simple wrist goals on the real articulated-hand robot.

This is an empty-hand diagnostic, not a grasp gate or collision-avoidance task.
Each world goal is set once. Hard safety stops are retained at 1 kHz; inaccurate
but safe segments are measured as failures, not cut out of the video.
"""
import copy
from pathlib import Path

import mujoco
import numpy as np

from common.r2v2_crate_motion_recording import (
    pose_from_transform, slerp_wxyz, transform_from_pose,
)
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_sim import (
    ReachCompatibilityExperiment, load_reach_config, require_parity,
    rotation_from_rpy_deg, sha256,
)
from r2v2_description.model import SIDES, initialize_hands, urdf_hand_joints


# Calibration from AMO_R2/src/r2v2_loco/tasks/reach/wrist_commands.py.
# World wrist-link origins; NOT 17.35cm-offset legacy TCPs.
HOME = np.array([[.246, .19698, 1.087902013081883], [.246, -.19643, 1.087902013081883]])
PREALIGN = np.array([[.38, .37798823450034097, 1.117407458185944],
                    [.38, -.3779882350100798, 1.1173909153107733]])
INSERT = np.array([[.38, .27624134968839875, 1.117407458185944],
                  [.38, -.2762413493904989, 1.1173909153107733]])
FULL_QUATERNIONS = ((.5, -.5, .5, -.5), (.5, .5, .5, .5))
DEFAULT_CASES = ("HOME_HOLD", "SYMMETRIC_PREALIGN_30", "SYMMETRIC_INSERT_30",
                 "SYMMETRIC_PREALIGN_50", "ASYMMETRIC_LEFT_HIGH", "ASYMMETRIC_RIGHT_HIGH", "RETURN_HOME")


def paired_targets(fraction, inserted=False):
    if not 0. <= fraction <= 1.:
        raise ValueError("Fraction outside [0, 1]")
    positions = HOME + fraction*((INSERT if inserted else PREALIGN)-HOME)
    return {side: transform_from_pose(p, slerp_wxyz([1, 0, 0, 0], q, fraction))
            for side, p, q in zip(SIDES, positions, FULL_QUATERNIONS)}


def make_cases():
    result = []
    def add(name, targets, note, duration=6.):
        result.append(dict(name=name, duration_s=duration, targets=targets, note=note))
    add("HOME_HOLD", paired_targets(0.), "Nominal world HOME, open hands", 5.)
    add("SYMMETRIC_PREALIGN_30", paired_targets(.3), "Trained paired course, 30% prealign")
    add("SYMMETRIC_INSERT_30", paired_targets(.3, True), "Trained paired course, 30% insert")
    add("SYMMETRIC_PREALIGN_50", paired_targets(.5), "Trained paired course, 50% prealign")
    for raised, other in (("left", "right"), ("right", "left")):
        targets = paired_targets(.3, True)
        targets[raised][:3, 3] += [.02, 0., .03]
        targets[other][:3, 3] += [-.01, 0., 0.]
        # Local wrist yaw variation is independent of the paired box command:
        # intentionally a small out-of-training-distribution probe.
        targets[raised][:3, :3] = targets[raised][:3, :3] @ rotation_from_rpy_deg([0., 0., 5. if raised == "left" else -5.])
        add(f"ASYMMETRIC_{raised.upper()}_HIGH", targets,
            f"Generalization probe: {raised} +2cm forward/+3cm up/local yaw 5deg; {other} -1cm forward")
    add("RETURN_HOME", paired_targets(0.), "Return to original world HOME")
    return result


def sustained_duration(times, good):
    longest = 0.
    start = None
    for time_s, valid in zip(times, good):
        if not valid:
            start = None
        else:
            start = float(time_s) if start is None else start
            longest = max(longest, float(time_s)-start)
    return longest


def summarize_case(rows, duration_s, expected_duration_s):
    """Tail metrics exclude motion transients but never replace missed goals."""
    if not rows:
        return dict(completed=False, simple_reach_passed=False, precise_reach_passed=False, samples=0)
    times = np.array([r["time_s"] for r in rows])
    tail_mask = times >= times[-1]-2.-1e-8
    simple, precise = np.ones(len(rows), dtype=bool), np.ones(len(rows), dtype=bool)
    arms = {}
    for side in SIDES:
        values = {key: np.array([r["arms"][side][key] for r in rows])
                  for key in ("position_m", "orientation_deg", "linear_speed_mps")}
        simple &= (values["position_m"] < .02) & (values["orientation_deg"] < 10.) & (values["linear_speed_mps"] < .02)
        precise &= (values["position_m"] < .005) & (values["orientation_deg"] < 3.) & (values["linear_speed_mps"] < .02)
        positions = np.array([r["arms"][side]["tcp_position_world_m"] for r in rows])[tail_mask]
        arms[side] = {f"tail_{stat}_{key}": float(func(array[tail_mask]))
            for key, array in values.items() for stat, func in (("mean", np.mean), ("max", np.max), ("p95", lambda a: np.percentile(a, 95)))}
        arms[side]["tail_position_jitter_rms_m"] = float(np.sqrt(np.mean(np.sum((positions-positions.mean(axis=0))**2, axis=1))))
        arms[side]["final"] = rows[-1]["arms"][side]
    completed = duration_s >= expected_duration_s-1e-6
    simple_hold = sustained_duration(times, simple)
    precise_hold = sustained_duration(times, precise)
    return dict(completed=completed, samples=len(rows), duration_s=duration_s,
        tail_window_s=min(2., float(times[-1]-times[0])), arms=arms,
        simple_reach_passed=bool(completed and simple_hold >= .3-1e-8),
        precise_reach_passed=bool(completed and precise_hold >= .3-1e-8),
        simple_max_hold_s=simple_hold, precise_max_hold_s=precise_hold,
        tail_simple_fraction=float(simple[tail_mask].mean()), tail_precise_fraction=float(precise[tail_mask].mean()),
        max_base_tilt_deg=max(r["base_tilt_deg"] for r in rows),
        max_foot_drift_m=max(r["foot_drift_m"] for r in rows))


class SimpleDualReachExperiment(ReachCompatibilityExperiment):
    def __init__(self, reach_config, parity_report, case_names=None):
        cfg = load_reach_config(reach_config)
        if cfg.get("endpoint_contract") != "wrist_world_v2":
            raise ValueError("Simple wrist test requires wrist_world_v2")
        self.parity_path = Path(parity_report).resolve()
        self.parity = require_parity(self.parity_path, cfg)
        self.cases = make_cases()
        if case_names is not None:
            requested = list(case_names)
            by_name = {c["name"]: c for c in self.cases}
            if not requested or len(set(requested)) != len(requested) or set(requested)-set(by_name):
                raise ValueError(f"Cases must be distinct names from {list(by_name)}")
            self.cases = [by_name[n] for n in requested]
        self.case_names = [c["name"] for c in self.cases]
        self.case_index = -1
        self.case_results = []
        self.goal_wrist_transforms = {}
        self.foot_anchor = None
        self.limit_failure_details = None
        super().__init__(cfg)
        limits = self.model.jnt_range[self.body_map.joints]
        center, half = limits.mean(axis=1), .45*(limits[:, 1]-limits[:, 0])
        self.data.qpos[self.body_map.qpos] = np.clip(self.data.qpos[self.body_map.qpos], center-half, center+half)
        initialize_hands(self.model, self.data, self.hand_cfg)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.policy.reset(self.data)
        self.goal_wrist_transforms = {s: transform_from_pose(*self.policy.tcp_pose(self.data, s)) for s in SIDES}
        self.hand_joints = [self.model.joint(n).id for n in urdf_hand_joints()]
        self.mimics = []
        for name, element in urdf_hand_joints().items():
            mimic = element.find("mimic")
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0], self.model.joint(mimic.get("joint")).qposadr[0],
                    float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
        self.peaks.update(hand_joint_violation_rad=0., hand_mimic_error_rad=0., foot_drift_m=0.)
        self.sync()
        self.initial_contacts = self._contacts()
        self.samples.clear()
        self.record()

    @property
    def done(self):
        return self.phase in ("COMPLETE", "FAILED")

    def _finish_case(self):
        if self.case_index < 0 or len(self.case_results) > self.case_index:
            return
        case = self.cases[self.case_index]
        rows = [r for r in self.samples if r["phase"] == case["name"]]
        summary = summarize_case(rows, float(self.data.time-self.phase_start), case["duration_s"])
        self.case_results.append(dict(name=case["name"], note=case["note"],
            target_transforms_world=case["targets"], **summary))

    def _next_case(self):
        self._finish_case()
        self.case_index += 1
        if self.case_index >= len(self.cases):
            self.transition("COMPLETE")
            return
        case = self.cases[self.case_index]
        for side, target in case["targets"].items():
            self.policy.set_target_world(side, *pose_from_transform(target))
        self.goal_wrist_transforms = copy.deepcopy(case["targets"])
        self.transition(case["name"])

    def update_gate(self):
        if self.phase == "RESET_SETTLE":
            if self.data.time < .4-1e-8:
                self.policy.follow_current(self.scratch)
                self.goal_wrist_transforms = {s: transform_from_pose(*self.policy.tcp_pose(self.scratch, s)) for s in SIDES}
            else:
                self.foot_anchor = self.scratch.geom_xpos[sorted(self.foot_geoms)].copy()
                self._next_case()
            return
        case = self.cases[self.case_index]
        if self.data.time-self.phase_start >= case["duration_s"]-1e-8:
            self._next_case()

    def safety(self):
        reason = super().safety()
        if reason:
            if reason == "Measured body joint exceeded physical limit tolerance":
                limits = self.model.jnt_range[self.body_map.joints]
                q = self.data.qpos[self.body_map.qpos]
                excess = np.maximum(limits[:, 0]-q, q-limits[:, 1])
                index = int(np.argmax(excess))
                self.limit_failure_details = dict(joint=self.model.joint(int(self.body_map.joints[index])).name,
                    measured_rad=float(q[index]), range_rad=limits[index].copy(), excess_rad=float(excess[index]))
            return reason
        d, m, g = self.data, self.model, self.cfg["gates"]
        tilt = float(np.degrees(np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1., 1.))))
        self.peaks["base_tilt_deg"] = max(self.peaks["base_tilt_deg"], tilt)
        if tilt > g["max_base_tilt_deg"] or d.xpos[self.base, 2] < g["min_base_height_m"]:
            return "Base tilt/height safety limit"
        if m.nmocap or np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD) or np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
            return "Unexpected support fixture or auxiliary force"
        limits = m.jnt_range[self.hand_joints]
        q = d.qpos[m.jnt_qposadr[self.hand_joints]]
        excess = float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1])))
        mimic = max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a, b, ratio, offset in self.mimics)
        self.peaks["hand_joint_violation_rad"] = max(self.peaks["hand_joint_violation_rad"], excess)
        self.peaks["hand_mimic_error_rad"] = max(self.peaks["hand_mimic_error_rad"], mimic)
        if excess > .01 or mimic > .015:
            return "Finger limit or mimic constraint safety limit"
        return None

    def fail(self, reason):
        # Include the actual last safety-stop frame in the failed segment's
        # measurements before the inherited transition labels it FAILED.
        self.record()
        self._finish_case()
        super().fail(reason)

    def record(self):
        super().record()
        tilt = float(np.degrees(np.arccos(np.clip(self.scratch.xmat[self.base].reshape(3, 3)[2, 2], -1., 1.))))
        drift = 0. if self.foot_anchor is None else float(np.max(np.linalg.norm(
            self.scratch.geom_xpos[sorted(self.foot_geoms)]-self.foot_anchor, axis=1)))
        self.samples[-1].update(base_tilt_deg=tilt, foot_drift_m=drift,
            hand_commands={s: self.hands.controllers[s].command for s in SIDES})
        if "foot_drift_m" in self.peaks:
            self.peaks["foot_drift_m"] = max(self.peaks["foot_drift_m"], drift)

    def report(self):
        r = super().report()
        r.update(scope="simple empty-hand dual-wrist reach, real new free-base articulated-hand model, no obstacles or grasp",
            sequence_completed=self.phase == "COMPLETE", case_results=self.case_results,
            cases=self.cases, case_names=self.case_names, limit_failure_details=self.limit_failure_details,
            passed=self.phase == "COMPLETE" and all(c["simple_reach_passed"] for c in self.case_results),
            simple_criteria=dict(position_m=.02, orientation_deg=10., speed_mps=.02, continuous_s=.3),
            precise_criteria=dict(position_m=.005, orientation_deg=3., speed_mps=.02, continuous_s=.3),
            segment_schedule="fixed duration, inaccuracies recorded; only hard safety ends episode, no grasp commands",
            motion_reference="original trained 0.25m/s and 0.8rad/s smooth reference, goal updated once per segment",
            standing_passed=False, standing_passed_meaning="No separate 20-second standing certification performed",
            initialization="Native 90% soft-range initial q clamp only; real limits unchanged",
            parity_report=str(self.parity_path), parity_report_sha256=sha256(self.parity_path),
            checkpoint_sha256=self.parity["checkpoint_sha256"], simple_runtime_sha256=sha256(Path(__file__)),
            active_case_unfinished=self.phase != "COMPLETE", safety_hz=1000, policy_hz=50, hands_hz=100)
        return r
