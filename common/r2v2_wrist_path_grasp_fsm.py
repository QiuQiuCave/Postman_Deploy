"""Contact-verified crate pickup followed by an explicitly exploratory high goal.

The trained path remains immutable through insertion. A separately hash-bound
plan supplies absolute world wrist goals for seating, probing and higher lift.
Only policy goals and binary fingers are controlled: no IK, fixtures or object
state writes. Pure state logic is independently testable without simulation.
"""

from dataclasses import asdict, dataclass, fields
import copy
import json
import math
from pathlib import Path

import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_crate_motion_recording import _validate_transform, interpolate_transform
from common.r2v2_reach_sim import sha256
from common.r2v2_wrist_path_contact import (
    PREPARATION_S, WristPathContactExperiment, contact_lift_conditions,
)
from r2v2_description.model import SIDES


@dataclass(frozen=True)
class GraspFSMTiming:
    close_seat_s: float = 4.8
    probe_timeout_s: float = 10.
    bilateral_contact_hold_s: float = .3
    grasp_verified_hold_s: float = .3
    verified_dwell_s: float = .2
    higher_motion_s: float = 4.
    higher_hold_s: float = 2.
    higher_timeout_s: float = 10.
    failed_observe_s: float = 2.
    maximum_sample_gap_s: float = .05

    def __post_init__(self):
        if any(not math.isfinite(float(getattr(self, f.name))) or getattr(self, f.name) <= 0 for f in fields(self)):
            raise ValueError("FSM durations must be finite and positive")
        if any(getattr(self, name) > 10. for name in (
                "close_seat_s", "probe_timeout_s", "higher_motion_s", "higher_timeout_s", "failed_observe_s")):
            raise ValueError("Each exploratory motion/observation stage is bounded by 10 seconds")
        if self.higher_hold_s > self.higher_timeout_s:
            raise ValueError("Higher-lift hold cannot exceed its observation timeout")


@dataclass(frozen=True)
class GraspObservation:
    bilateral_contact: bool = False
    physical_pickup: bool = False
    strict_pickup: bool = False
    grasp_retained: bool = False
    clearance_m: float | None = None


class GraspFSM:
    """Event-only supervisor; it cannot access policy, robot or crate state."""

    def __init__(self, start_time_s, timing=None, minimum_additional_lift_m=.008):
        if not math.isfinite(start_time_s) or start_time_s < 0:
            raise ValueError("Invalid FSM start time")
        if not math.isfinite(minimum_additional_lift_m) or minimum_additional_lift_m <= 0:
            raise ValueError("Additional real lift must be positive")
        self.timing = timing or GraspFSMTiming()
        self.minimum_additional_lift_m = float(minimum_additional_lift_m)
        self.phase = "CLOSE_SEAT"
        self.phase_start_s = self.last_time_s = float(start_time_s)
        self.baseline_captured = False
        self.grasp_verified_ever = self.higher_lift_verified = False
        self.verified_clearance_m = self.verified_time_s = None
        self.failure_reason = None
        self.contact_hold_s = self.physical_hold_s = self.strict_hold_s = self.higher_hold_s = 0.
        self.max_physical_hold_s = self.max_strict_hold_s = 0.
        self._since = dict(contact=None, physical=None, strict=None, higher=None)
        self.events = [dict(kind="enter", time_s=float(start_time_s), state="CLOSE_SEAT")]

    @property
    def done(self):
        return self.phase == "COMPLETE"

    def _hold(self, name, valid, time_s):
        if not valid:
            self._since[name] = None
            return 0.
        if self._since[name] is None:
            self._since[name] = time_s
        return time_s-self._since[name]

    def _enter(self, state, time_s, reason=None):
        self.phase, self.phase_start_s = state, time_s
        event = dict(kind="enter", time_s=time_s, state=state)
        if reason is not None:
            event["reason"] = reason
        self.events.append(event)

    def _capture_baseline(self, time_s):
        if not self.baseline_captured and self.contact_hold_s >= self.timing.bilateral_contact_hold_s-1e-8:
            self.baseline_captured = True
            self.events.append(dict(kind="capture_grasp_baseline", time_s=time_s, state=self.phase))

    def _failed_observe(self, reason, time_s):
        self.failure_reason = reason
        self._enter("OBSERVE_FAILED", time_s, reason)

    def update(self, time_s, observation):
        if not math.isfinite(time_s) or time_s < self.last_time_s-1e-9:
            raise ValueError("FSM requires finite monotonic observation time")
        first_event = len(self.events)
        if self.done:
            return []
        if time_s-self.last_time_s > self.timing.maximum_sample_gap_s+1e-9:
            # Sparse/missing samples cannot establish uninterrupted grasp.
            self._since = dict.fromkeys(self._since, None)
        self.last_time_s = time_s
        elapsed = time_s-self.phase_start_s
        o, t = observation, self.timing
        valid_clearance = o.clearance_m is not None and math.isfinite(o.clearance_m)
        self.contact_hold_s = self._hold("contact", o.bilateral_contact, time_s)
        physical = bool(self.baseline_captured and o.bilateral_contact and o.physical_pickup and valid_clearance)
        self.physical_hold_s = self._hold("physical", physical, time_s)
        self.strict_hold_s = self._hold("strict", physical and o.strict_pickup, time_s)
        self.max_physical_hold_s = max(self.max_physical_hold_s, self.physical_hold_s)
        self.max_strict_hold_s = max(self.max_strict_hold_s, self.strict_hold_s)
        if self.phase == "CLOSE_SEAT":
            if elapsed >= t.close_seat_s-1e-8:
                self._enter("PROBE_LIFT", time_s)
                self._capture_baseline(time_s)
        elif self.phase == "PROBE_LIFT":
            self._capture_baseline(time_s)
            if self.physical_hold_s >= t.grasp_verified_hold_s-1e-8:
                self.grasp_verified_ever = True
                self.verified_clearance_m = float(o.clearance_m)
                self.verified_time_s = time_s
                self._enter("GRASP_VERIFIED", time_s)
            elif elapsed >= t.probe_timeout_s-1e-8:
                self._failed_observe("Probe timeout: real bilateral pickup was not verified", time_s)
        elif self.phase == "GRASP_VERIFIED":
            if not o.grasp_retained:
                self._failed_observe("Verified grasp was lost before higher-lift command", time_s)
            elif elapsed >= t.verified_dwell_s-1e-8:
                if physical:
                    self._since["higher"] = None
                    self._enter("LIFT_HIGHER", time_s)
                else:
                    self._failed_observe("Verified grasp was lost before higher-lift command", time_s)
        elif self.phase == "LIFT_HIGHER":
            # A real loss/slip ends the higher-lift state, but no opening or
            # new wrist goal is commanded: observe the existing goal safely.
            if not o.grasp_retained:
                self._failed_observe("Grasp slipped or lost support during higher lift", time_s)
            elif elapsed >= t.higher_motion_s-1e-8:
                self._since["higher"] = None
                self._enter("HOLD_HIGHER", time_s)
        elif self.phase == "HOLD_HIGHER":
            actually_higher = physical and o.clearance_m >= self.verified_clearance_m+self.minimum_additional_lift_m-1e-8
            self.higher_hold_s = self._hold("higher", actually_higher, time_s)
            if not o.grasp_retained:
                self._failed_observe("Grasp slipped or lost support during higher hold", time_s)
            elif self.higher_hold_s >= t.higher_hold_s-1e-8:
                self.higher_lift_verified = True
                self._enter("COMPLETE", time_s)
            elif elapsed >= t.higher_timeout_s-1e-8:
                self._failed_observe("Higher-lift timeout: additional stable real lift was not verified", time_s)
        elif self.phase == "OBSERVE_FAILED":
            if elapsed >= t.failed_observe_s-1e-8:
                self._enter("COMPLETE", time_s)
        else:
            raise RuntimeError(f"Unknown grasp FSM state: {self.phase}")
        return self.events[first_event:]


def load_grasp_plan(plan_path, path_manifest):
    """Validate explicit target/evidence bindings without changing trained data."""
    plan_path, path_manifest = Path(plan_path).resolve(), Path(path_manifest).resolve()
    plan = json.loads(plan_path.read_text())
    trained = json.loads(path_manifest.read_text())
    if plan.get("schema_version") != 1 or plan.get("goal_frame") != "world":
        raise ValueError("Expected schema_version 1 absolute world grasp targets")
    if plan.get("training_path_trajectory_sha256") != trained["trajectory_sha256"]:
        raise ValueError("Grasp plan does not match the checkpoint's training path")
    bindings = plan["bindings"]
    if bindings.get("path_manifest_sha256") != sha256(path_manifest):
        raise ValueError("Grasp plan path manifest binding mismatch")
    if not bindings.get("model_sources"):
        raise ValueError("Missing model-source evidence bindings")
    for name, expected in bindings["model_sources"].items():
        source = (PROJECT_ROOT/name).resolve()
        if not source.is_relative_to(PROJECT_ROOT.resolve()) or sha256(source) != expected:
            raise ValueError(f"Grasp plan model source binding mismatch: {name}")
    targets = {}
    for stage in ("close_seat", "probe", "higher"):
        targets[stage] = _validate_transform(np.asarray(plan["targets"][stage], dtype=float)).copy()
        if targets[stage].shape != (2, 4, 4):
            raise ValueError(f"Expected bilateral world wrist transforms for {stage}")
    additional = float(plan.get("minimum_additional_lift_m", .008))
    if not np.all(targets["higher"][:, 2, 3] > targets["probe"][:, 2, 3]):
        raise ValueError("Higher wrist targets must be genuinely above both probe targets")
    timing = GraspFSMTiming(**plan.get("timing", {}))
    GraspFSM(0., timing, additional)  # Validate scalar success parameters as well.
    evidence_path = (plan_path.parent/plan["evidence"]["file"]).resolve()
    if sha256(evidence_path) != plan["evidence"]["sha256"]:
        raise ValueError("Grasp plan geometry evidence hash mismatch")
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("candidate_id") != plan.get("candidate_id"):
        raise ValueError("Geometry evidence belongs to a different candidate")
    if evidence.get("training_path_trajectory_sha256") != plan["training_path_trajectory_sha256"]:
        raise ValueError("Geometry evidence belongs to a different trained path")
    for stage, target in targets.items():
        if not np.allclose(np.asarray(evidence["targets"][stage]), target, atol=1e-10, rtol=0.):
            raise ValueError(f"Geometry evidence target mismatch: {stage}")
    if evidence.get("source_bindings", evidence.get("bindings")) != bindings:
        raise ValueError("Geometry evidence/model bindings do not match plan")
    validate_geometry_samples(plan, evidence, targets)
    return dict(path=str(plan_path), sha256=sha256(plan_path), document=plan, targets=targets,
                evidence_path=str(evidence_path), evidence=evidence, timing=timing,
                minimum_additional_lift_m=additional)


def validate_geometry_samples(plan, evidence, targets):
    scope = "sampled_static_kinematics_not_dynamic_contact_success"
    if (plan.get("nominal_sampled_geometry_passed") is not True
            or plan.get("geometry_scope") != scope or evidence.get("scope") != scope):
        raise ValueError("Candidate lacks the required passing sampled-static geometry scope")
    configurations = evidence.get("configurations", [])
    if len(configurations) != 2 or {c.get("feet") for c in configurations} != {"nominal", "prepared"}:
        raise ValueError("Geometry evidence must cover nominal and prepared feet configurations")
    for configuration in configurations:
        points, summary = configuration.get("points", []), configuration.get("summary", {})
        if (not points or summary.get("sample_count") != len(points) or summary.get("passed") != len(points)
                or any(p.get("strict_static_candidate") is not True for p in points)):
            raise ValueError("A missing/failed geometry sample cannot certify a grasp plan")
        if {p.get("stage") for p in points} != set(targets):
            raise ValueError("Geometry samples do not cover all three grasp stages")
        previous = _validate_transform(np.asarray(evidence["T_world_wrist_insert"]))
        for stage in ("close_seat", "probe", "higher"):
            samples = [p for p in points if p["stage"] == stage]
            fractions = np.asarray([p["fraction"] for p in samples], dtype=float)
            if (len(samples) < 11 or not np.all(np.isfinite(fractions))
                    or abs(fractions[0]) > 1e-9 or abs(fractions[-1]-1.) > 1e-9
                    or np.any(np.diff(fractions) <= 0.) or np.max(np.diff(fractions)) > .1+1e-9):
                raise ValueError(f"Geometry stage {stage} requires full start/interior/end sampling")
            for sample, fraction in zip(samples, fractions):
                pose = _validate_transform(np.asarray(sample["desired_wrist_world"]))
                expected = interpolate_transform(previous, targets[stage], float(fraction))
                if pose.shape != (2, 4, 4) or not np.allclose(pose, expected, atol=1e-8, rtol=0.):
                    raise ValueError(f"Geometry sample does not lie on the bound {stage} target segment")
            previous = targets[stage]


def bilateral_handle_contact(metrics):
    """Finger/handle-beam bearing, not arbitrary palm/box contact or CLOSED."""
    return all(bool(metrics["hands"][s].get("has_bearing_finger_contact"))
               and metrics["hands"][s].get("finger_handle_vertical_force_N", 0.) > .05 for s in SIDES)


def retained_grasp_conditions(metrics, *, baseline_contact_verified, weight_N):
    """Moving-load retention check: no requirement to stand still during lift."""
    m = metrics
    slip, rotation = m.get("grasp_slip_m"), m.get("grasp_rotation_slip_deg")
    valid_slip = slip is not None and rotation is not None and np.isfinite(slip) and np.isfinite(rotation) and slip < .015 and rotation < 5.
    bilateral = all(min(m["hands"][s]["finger_handle_vertical_force_N"],
                        m["hands"][s]["vertical_force_N"]) >= .2 for s in SIDES)
    return bool(baseline_contact_verified and valid_slip and bilateral
        and sum(m["hands"][s]["vertical_force_N"] for s in SIDES) >= .8*weight_N
        and m["clearance_m"] >= .008 and not m["table_contact"] and not m["floor_contact"]
        and m.get("nonhand_crate_contact_count", 0) == 0 and m.get("robot_table_contact_count", 0) == 0
        and m["crate_tilt_deg"] <= 8.)


class WristPathGraspFSMExperiment(WristPathContactExperiment):
    """Real policy/contact experiment with event-set exploratory higher goals."""

    def __init__(self, path_manifest, reach_config, parity_report, grasp_plan):
        self.grasp_plan = load_grasp_plan(grasp_plan, path_manifest)
        self.grasp_fsm = None
        self.fsm_goal_events = []
        self.fsm_observations = []
        super().__init__(path_manifest, reach_config, parity_report)
        if self.path.sha256 != self.grasp_plan["document"]["training_path_trajectory_sha256"]:
            raise ValueError("Actual archived training path disagrees with exploratory plan")

    def enter(self, phase):
        # Parent PROBE_LIFT handler would silently reset the slip baseline.
        # Baselines here are set solely by a sustained-contact FSM event.
        if phase == "FAILED":
            self.failure_phase = self.phase
        self.phase, self.phase_start = phase, float(self.data.time)
        self.transitions.append(dict(time_s=self.phase_start, state=phase,
                                     source_time_s=self.source_time_s))

    def _apply_fsm_event(self, event):
        if event["kind"] == "capture_grasp_baseline":
            self.baseline_contact_verified = True
            self.baseline_relations = {s: np.asarray(self.current_metrics["hands"][s]["T_wrist_crate"]).copy()
                                       for s in SIDES}
            return
        if event["kind"] != "enter":
            raise ValueError(f"Unexpected FSM event: {event}")
        self.enter(event["state"])
        if self.phase == "CLOSE_SEAT":
            for side in SIDES:
                self.hands.command(side, 1)
        stage = dict(CLOSE_SEAT="close_seat", PROBE_LIFT="probe", LIFT_HIGHER="higher").get(self.phase)
        if stage is not None:
            goals = self.grasp_plan["targets"][stage]
            for side, transform in zip(SIDES, goals):
                self._set_goal(side, transform)
            self.fsm_goal_events.append(dict(time_s=float(self.data.time), phase=self.phase,
                goal_stage=stage, T_world_wrist_goal=goals.copy(), set_once_on_phase_entry=True,
                source="explicit hash-bound exploratory plan, NOT original training path"))

    def _gate(self):
        t = float(self.data.time)
        if self.grasp_fsm is None:
            if t < self.path.close_time-1e-8:
                phase, _, _ = self.path.sample(t)
                if self.phase != phase:
                    self.enter(phase)
                return
            self.grasp_fsm = GraspFSM(t, self.grasp_plan["timing"], self.grasp_plan["minimum_additional_lift_m"])
            self._apply_fsm_event(self.grasp_fsm.events[0])
        physical, strict = contact_lift_conditions(self.current_metrics,
            baseline_contact_verified=self.baseline_contact_verified, weight_N=self.crate_params.mass*9.81)
        observation = GraspObservation(bilateral_contact=bilateral_handle_contact(self.current_metrics),
            physical_pickup=physical, strict_pickup=strict, clearance_m=self.current_metrics["clearance_m"],
            grasp_retained=retained_grasp_conditions(self.current_metrics,
                baseline_contact_verified=self.baseline_contact_verified, weight_N=self.crate_params.mass*9.81))
        for event in self.grasp_fsm.update(t, observation):
            self._apply_fsm_event(event)
        fsm = self.grasp_fsm
        self.contact_hold_s, self.physical_hold_s, self.strict_hold_s = fsm.contact_hold_s, fsm.physical_hold_s, fsm.strict_hold_s
        self.max_physical_hold_s, self.max_strict_hold_s = fsm.max_physical_hold_s, fsm.max_strict_hold_s
        self.fsm_observations.append(dict(time_s=t, phase=self.phase, **asdict(observation),
            contact_hold_s=fsm.contact_hold_s, physical_hold_s=fsm.physical_hold_s,
            higher_hold_s=fsm.higher_hold_s, baseline_contact_verified=self.baseline_contact_verified))

    def _update_source_clock(self):
        # Once the FSM owns targets, freeze the archived source clock at CLOSE.
        self.source_time_s = max(0., min(float(self.data.time), self.path.close_time)-PREPARATION_S)

    def _stream_targets(self):
        if self.grasp_fsm is None:
            return super()._stream_targets()
        # Do not call set_target_world again: ReachPolicy keeps the original
        # trained velocity/acceleration filter evolving toward each final goal.
        if not self.done:
            self.targets.append(dict(time_s=float(self.data.time), phase=self.phase,
                T_world_wrist_goal=np.array([self.goal_wrist_transforms[s] for s in SIDES]),
                T_world_crate_desired=self.desired_crate_world,
                executed_hand_command=[self.hands.controllers[s].command for s in SIDES],
                target_source="explicit exploratory grasp FSM", target_set_this_tick=False))

    def record(self):
        super().record()
        if self.grasp_fsm is not None:
            fsm = self.grasp_fsm
            self.samples[-1]["grasp_fsm"] = dict(phase=fsm.phase,
                grasp_verified_ever=fsm.grasp_verified_ever, higher_lift_verified=fsm.higher_lift_verified,
                failure_reason=fsm.failure_reason, contact_hold_s=fsm.contact_hold_s,
                physical_hold_s=fsm.physical_hold_s, higher_hold_s=fsm.higher_hold_s)

    def report(self):
        result = super().report()
        fsm = self.grasp_fsm
        completed = bool(self.phase == "COMPLETE" and not self.failure and fsm is not None and fsm.done)
        verified = bool(fsm is not None and fsm.grasp_verified_ever)
        current_physical, _ = contact_lift_conditions(self.current_metrics,
            baseline_contact_verified=self.baseline_contact_verified, weight_N=self.crate_params.mass*9.81)
        pickup_now = bool(completed and verified and current_physical
            and fsm.physical_hold_s >= fsm.timing.grasp_verified_hold_s-1e-8)
        higher = bool(pickup_now and fsm.higher_lift_verified
            and fsm.higher_hold_s >= fsm.timing.higher_hold_s-1e-8
            and self.current_metrics["clearance_m"] >= fsm.verified_clearance_m+self.grasp_plan["minimum_additional_lift_m"]-1e-8)
        strict = bool(higher and fsm.strict_hold_s >= fsm.timing.higher_hold_s-1e-8 and not self.constraint_violations_seen)
        result.update(scope="contact-verified GRASP FSM with exploratory absolute higher wrist goals",
            mode="contact_gated_grasp_fsm", schedule_completed=completed, experiment_completed=completed,
            grasp_verified=verified, grasp_verified_ever=verified, grasp_retained_now=pickup_now,
            grasp_verified_field_semantics="historical event, not current/final grasp success",
            higher_lift_commanded=any(e["phase"] == "LIFT_HIGHER" for e in self.fsm_goal_events),
            higher_lift_verified=higher, physical_pickup_verified=pickup_now,
            pickup_observed=verified, lift_passed=higher, strict_success=strict,
            grasp_fsm_failure=None if fsm is None else fsm.failure_reason,
            grasp_verified_time_s=None if fsm is None else fsm.verified_time_s,
            grasp_verified_clearance_m=None if fsm is None else fsm.verified_clearance_m,
            higher_lift_minimum_additional_clearance_m=self.grasp_plan["minimum_additional_lift_m"],
            grasp_verification_hold_s=self.grasp_plan["timing"].grasp_verified_hold_s,
            higher_lift_verification_hold_s=self.grasp_plan["timing"].higher_hold_s,
            final_higher_hold_s=0. if fsm is None else fsm.higher_hold_s,
            grasp_fsm_events=[] if fsm is None else copy.deepcopy(fsm.events),
            fsm_world_goal_events=self.fsm_goal_events, fsm_observations=self.fsm_observations,
            grasp_plan_path=self.grasp_plan["path"], grasp_plan_sha256=self.grasp_plan["sha256"],
            grasp_plan_document=self.grasp_plan["document"], geometry_evidence=self.grasp_plan["evidence"],
            nominal_geometry_does_not_guarantee_actual_policy_tracking=True,
            independent_policy_filter_path_not_certified_by_static_samples=True,
            higher_goal_is_exploratory_not_the_training_path=True,
            world_targets_set_once_per_fsm_motion_stage=True,
            fingers_never_automatically_open_after_close=True,
            planned_duration_s=self.path.close_time+sum(getattr(self.grasp_plan["timing"], name) for name in
                ("close_seat_s", "probe_timeout_s", "verified_dwell_s", "higher_motion_s", "higher_timeout_s", "failed_observe_s")))
        result["physical_success_criteria"]["hold_s"] = self.grasp_plan["timing"].grasp_verified_hold_s
        result["physical_success_criteria"]["higher_lift_adds"] = dict(
            additional_clearance_m=self.grasp_plan["minimum_additional_lift_m"],
            hold_s=self.grasp_plan["timing"].higher_hold_s)
        return result
