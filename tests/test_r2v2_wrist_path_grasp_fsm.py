"""Pure supervisor, goal-channel and evidence tests; no physics is stepped."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from common.r2v2_crate_motion_recording import interpolate_transform
from common.r2v2_wrist_path_grasp_fsm import (
    GraspFSM, GraspFSMTiming, GraspObservation, WristPathGraspFSMExperiment,
    bilateral_handle_contact, retained_grasp_conditions, validate_geometry_samples,
)
from common.r2v2_wrist_path_contact import WristPathContactExperiment


TIMING = GraspFSMTiming(close_seat_s=.4, probe_timeout_s=.8, bilateral_contact_hold_s=.03,
    grasp_verified_hold_s=.03, verified_dwell_s=.02, higher_motion_s=.2,
    higher_hold_s=.05, higher_timeout_s=.5, failed_observe_s=.1)
GOOD = GraspObservation(bilateral_contact=True, physical_pickup=True, strict_pickup=True,
                        grasp_retained=True, clearance_m=.012)


def drive(fsm, until, observation):
    for t in np.arange(round(fsm.last_time_s+.01, 6), until+.000001, .01):
        fsm.update(round(float(t), 6), observation)
    return fsm


def entered(fsm):
    return [event["state"] for event in fsm.events if event["kind"] == "enter"]


def test_closed_or_contact_without_off_table_pickup_never_commands_higher():
    fsm = GraspFSM(0., TIMING)
    # Arbitrary physical=True is insufficient when contact baseline is absent.
    drive(fsm, 1.4, GraspObservation(physical_pickup=True, clearance_m=.012))
    assert fsm.done and not fsm.grasp_verified_ever
    assert entered(fsm) == ["CLOSE_SEAT", "PROBE_LIFT", "OBSERVE_FAILED", "COMPLETE"]
    assert not fsm.baseline_captured
    table_supported = GraspFSM(0., TIMING)
    drive(table_supported, 1.4, GraspObservation(bilateral_contact=True, clearance_m=0.))
    assert table_supported.baseline_captured
    assert not table_supported.grasp_verified_ever


def test_baseline_requires_sustained_beam_contact_and_is_captured_only_once():
    fsm = GraspFSM(0., TIMING)
    drive(fsm, .43, GraspObservation())
    assert fsm.phase == "PROBE_LIFT" and not fsm.baseline_captured
    drive(fsm, .45, GOOD)
    assert not fsm.baseline_captured
    drive(fsm, .6, GOOD)
    captures = [e for e in fsm.events if e["kind"] == "capture_grasp_baseline"]
    assert len(captures) == 1
    assert captures[0]["time_s"] == pytest.approx(.47)
    assert fsm.verified_time_s > captures[0]["time_s"]


def test_verified_pickup_precedes_higher_and_requires_additional_actual_clearance():
    fsm = GraspFSM(0., TIMING)
    drive(fsm, .7, GOOD)
    assert "GRASP_VERIFIED" in entered(fsm) and "LIFT_HIGHER" in entered(fsm)
    assert entered(fsm).index("GRASP_VERIFIED") < entered(fsm).index("LIFT_HIGHER")
    assert not fsm.higher_lift_verified  # Same box height is not higher lift.
    drive(fsm, .8, replace(GOOD, clearance_m=.025))
    assert fsm.done and fsm.higher_lift_verified
    assert fsm.higher_hold_s >= TIMING.higher_hold_s-1e-8


def test_slip_after_verification_prevents_new_high_goal_and_retains_failure():
    fsm = GraspFSM(0., TIMING)
    while fsm.phase != "GRASP_VERIFIED":
        drive(fsm, fsm.last_time_s+.01, GOOD)
    drive(fsm, fsm.last_time_s+.01, replace(GOOD, physical_pickup=False, grasp_retained=False))
    assert fsm.phase == "OBSERVE_FAILED"
    assert "LIFT_HIGHER" not in entered(fsm)
    drive(fsm, fsm.last_time_s+.2, GOOD)
    assert fsm.done and not fsm.higher_lift_verified
    assert fsm.failure_reason is not None


def test_terminal_tick_rechecks_real_grasp_before_claiming_higher_success():
    fsm = GraspFSM(0., TIMING)
    while fsm.phase != "HOLD_HIGHER":
        drive(fsm, fsm.last_time_s+.01, GOOD)
    drive(fsm, fsm.last_time_s+.05, replace(GOOD, clearance_m=.025))
    assert not fsm.done
    fsm.update(round(fsm.last_time_s+.01, 6), replace(GOOD, grasp_retained=False, physical_pickup=False))
    assert fsm.phase == "OBSERVE_FAILED" and not fsm.higher_lift_verified


def test_observation_gaps_cannot_fabricate_continuous_verified_pickup():
    fsm = GraspFSM(0., TIMING)
    fsm.update(0., GOOD)
    fsm.update(.4, GOOD)
    assert not fsm.baseline_captured
    fsm.update(.8, GOOD)
    assert not fsm.grasp_verified_ever
    with pytest.raises(ValueError, match="monotonic"):
        fsm.update(.7, GOOD)


def _metrics():
    return dict(hands={side: dict(has_bearing_finger_contact=True,
        finger_handle_vertical_force_N=2., vertical_force_N=2.) for side in ("left", "right")},
        grasp_slip_m=.001, grasp_rotation_slip_deg=1., clearance_m=.02,
        table_contact=False, floor_contact=False, crate_tilt_deg=2.,
        crate_linear_speed_m_s=.3, crate_angular_speed_rad_s=.4,
        wrist_errors={s: dict(position_m=.001, orientation_deg=1., linear_speed_mps=.01) for s in ("left", "right")})


def test_beam_contacts_and_retention_are_physical_not_finger_commands():
    m = _metrics()
    assert bilateral_handle_contact(m)
    assert retained_grasp_conditions(m, baseline_contact_verified=True, weight_N=3.924)
    m["hands"]["right"]["has_bearing_finger_contact"] = False
    assert not bilateral_handle_contact(m)
    m["grasp_slip_m"] = .016
    assert not retained_grasp_conditions(m, baseline_contact_verified=True, weight_N=3.924)


def test_motion_stage_sends_one_higher_world_goal_without_restarting_per_tick():
    exp = WristPathGraspFSMExperiment.__new__(WristPathGraspFSMExperiment)
    targets = {stage: np.tile(np.eye(4), (2, 1, 1)) for stage in ("close_seat", "probe", "higher")}
    targets["probe"][:, 2, 3] = 1.
    targets["higher"][:, 2, 3] = 1.03
    exp.grasp_plan = {"targets": targets}
    exp.data = SimpleNamespace(time=1., qpos=np.array([.123]), qvel=np.array([.456]))
    exp.phase, exp.source_time_s, exp.transitions = "GRASP_VERIFIED", 0., []
    exp.fsm_goal_events, exp.targets = [], []
    exp.grasp_fsm = object()
    exp.desired_crate_world = np.eye(4)
    exp.goal_wrist_transforms = {side: np.eye(4) for side in ("left", "right")}
    calls = []
    exp.policy = SimpleNamespace(set_target_world=lambda *args: calls.append(args))
    exp.hands = SimpleNamespace(controllers={s: SimpleNamespace(command=1) for s in ("left", "right")})
    exp._apply_fsm_event(dict(kind="enter", state="LIFT_HIGHER", time_s=1.))
    assert len(calls) == 2  # The real inherited _set_goal reaches set_target_world.
    assert all(args[1][2] > 1. for args in calls)
    for _ in range(20):
        exp._stream_targets()
    assert len(calls) == 2
    exp._apply_fsm_event(dict(kind="enter", state="OBSERVE_FAILED", time_s=1.))
    assert len(calls) == 2
    assert all(c.command == 1 for c in exp.hands.controllers.values())
    np.testing.assert_array_equal(exp.data.qpos, [.123])
    np.testing.assert_array_equal(exp.data.qvel, [.456])


def _geometry():
    targets = {stage: np.tile(np.eye(4), (2, 1, 1)) for stage in ("close_seat", "probe", "higher")}
    for stage, height in zip(targets, (.005, .02, .03)):
        targets[stage][:, 2, 3] = height
    insert = np.tile(np.eye(4), (2, 1, 1))
    points, previous = [], insert
    for stage, target in targets.items():
        for fraction in np.linspace(0., 1., 11):
            points.append(dict(stage=stage, fraction=float(fraction), strict_static_candidate=True,
                desired_wrist_world=interpolate_transform(previous, target, fraction)))
        previous = target
    scope = "sampled_static_kinematics_not_dynamic_contact_success"
    plan = dict(nominal_sampled_geometry_passed=True, geometry_scope=scope)
    evidence = dict(scope=scope, T_world_wrist_insert=insert, configurations=[
        dict(feet=feet, points=copy.deepcopy(points), summary=dict(sample_count=33, passed=33))
        for feet in ("nominal", "prepared")])
    return plan, evidence, targets


def test_geometry_requires_true_pass_for_every_sample_not_only_summary():
    plan, evidence, targets = _geometry()
    validate_geometry_samples(plan, evidence, targets)
    evidence["configurations"][0]["points"][4]["strict_static_candidate"] = False
    with pytest.raises(ValueError, match="missing/failed"):
        validate_geometry_samples(plan, evidence, targets)


def test_endpoint_only_or_wrong_intermediate_geometry_is_rejected():
    plan, evidence, targets = _geometry()
    evidence["configurations"][0]["points"] = [p for p in evidence["configurations"][0]["points"] if p["fraction"] in (0., 1.)]
    evidence["configurations"][0]["summary"] = dict(sample_count=6, passed=6)
    with pytest.raises(ValueError, match="interior"):
        validate_geometry_samples(plan, evidence, targets)
    plan, evidence, targets = _geometry()
    evidence["configurations"][0]["points"][5]["desired_wrist_world"][0, 0, 3] += .1
    with pytest.raises(ValueError, match="target segment"):
        validate_geometry_samples(plan, evidence, targets)


def test_report_keeps_historical_pickup_but_never_claims_final_success_after_drop(monkeypatch):
    monkeypatch.setattr(WristPathContactExperiment, "report", lambda self: {"physical_success_criteria": {}})
    exp = WristPathGraspFSMExperiment.__new__(WristPathGraspFSMExperiment)
    exp.grasp_fsm = GraspFSM(0., TIMING)
    exp.grasp_fsm.phase = "COMPLETE"
    exp.grasp_fsm.grasp_verified_ever = True
    exp.grasp_fsm.verified_clearance_m = .012
    # Even stale previous hold counters cannot override actual final contact.
    exp.grasp_fsm.physical_hold_s = exp.grasp_fsm.higher_hold_s = 2.
    exp.grasp_fsm.higher_lift_verified = True
    exp.phase, exp.failure = "COMPLETE", None
    exp.current_metrics = _metrics()
    exp.current_metrics.update(table_contact=True, clearance_m=0., grasp_slip_m=.1)
    exp.baseline_contact_verified = True
    exp.crate_params = SimpleNamespace(mass=.4)
    exp.constraint_violations_seen = set()
    exp.fsm_goal_events, exp.fsm_observations = [], []
    exp.path = SimpleNamespace(close_time=10.)
    exp.grasp_plan = dict(minimum_additional_lift_m=.008, timing=TIMING, path="bound-test-plan",
        sha256="test-sha", document={}, evidence={})
    report = exp.report()
    assert report["pickup_observed"] is True
    assert report["grasp_verified_ever"] is True
    assert report["physical_pickup_verified"] is False
    assert report["higher_lift_verified"] is False
    assert report["lift_passed"] is False
    assert report["strict_success"] is False
