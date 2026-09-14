"""Pure world-target and reporting contracts; no ONNX loading or rollout claims."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from common.r2v2_crate_motion_recording import pose_from_transform
from common.r2v2_reach_sim import rotation_from_rpy_deg
from common.r2v2_simple_dual_reach import (
    DEFAULT_CASES, HOME, INSERT, PREALIGN, SimpleDualReachExperiment,
    make_cases, paired_targets, summarize_case, sustained_duration,
)
from r2v2_description.model import SIDES


def make_rows(times, position=.003, orientation=2., speed=.01):
    return [dict(time_s=float(time), base_tilt_deg=1., foot_drift_m=.001,
        arms={side: dict(position_m=position, orientation_deg=orientation,
            linear_speed_mps=speed, tcp_position_world_m=[.3, sign*.2, 1.1])
              for side, sign in zip(SIDES, (1., -1.))}) for time in times]


@pytest.mark.parametrize("inserted", [False, True])
@pytest.mark.parametrize("fraction", [0., .3, .5, 1.])
def test_paired_targets_match_native_wrist_course_position_and_quaternion(fraction, inserted):
    # Independent native contract constants and closed-form unit-axis rotation.
    home = np.array([[.246, .19698, 1.087902013081883], [.246, -.19643, 1.087902013081883]])
    endpoints = np.array(
        [[.38, .27624134968839875, 1.117407458185944], [.38, -.2762413493904989, 1.1173909153107733]]
        if inserted else
        [[.38, .37798823450034097, 1.117407458185944], [.38, -.3779882350100798, 1.1173909153107733]])
    targets = paired_targets(fraction, inserted)
    np.testing.assert_array_equal(HOME, home)
    np.testing.assert_array_equal(INSERT if inserted else PREALIGN, endpoints)
    for index, side in enumerate(SIDES):
        p, q = pose_from_transform(targets[side])
        expected_p = home[index]+fraction*(endpoints[index]-home[index])
        axis = np.array([-1., 1., -1.] if side == "left" else [1., 1., 1.])/np.sqrt(3.)
        expected_q = np.r_[np.cos(fraction*np.pi/3), axis*np.sin(fraction*np.pi/3)]
        np.testing.assert_allclose(p, expected_p, rtol=0, atol=1e-14)
        assert abs(float(q @ expected_q)) == pytest.approx(1., abs=1e-14)
        np.testing.assert_array_equal(targets[side][3], [0., 0., 0., 1.])
        assert np.linalg.det(targets[side][:3, :3]) == pytest.approx(1., abs=1e-14)


@pytest.mark.parametrize("fraction", [-.01, 1.01, np.nan, np.inf])
def test_paired_targets_reject_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        paired_targets(fraction)


def test_named_cases_include_home_course30_course50_and_return_without_aliasing():
    cases = make_cases()
    assert tuple(case["name"] for case in cases) == DEFAULT_CASES
    assert cases[0]["duration_s"] == 5.
    assert all(case["duration_s"] == 6. for case in cases[1:])
    mapping = {case["name"]: case for case in cases}
    for name, fraction, inserted in (("HOME_HOLD", 0., False),
        ("SYMMETRIC_PREALIGN_30", .3, False), ("SYMMETRIC_INSERT_30", .3, True),
        ("SYMMETRIC_PREALIGN_50", .5, False), ("RETURN_HOME", 0., False)):
        expected = paired_targets(fraction, inserted)
        for side in SIDES:
            np.testing.assert_array_equal(mapping[name]["targets"][side], expected[side])
    cases[0]["targets"]["left"][:] = 0
    np.testing.assert_array_equal(mapping["RETURN_HOME"]["targets"]["left"], paired_targets(0.)["left"])
    np.testing.assert_array_equal(make_cases()[0]["targets"]["left"], paired_targets(0.)["left"])


@pytest.mark.parametrize("raised,other,yaw", [("left", "right", 5.), ("right", "left", -5.)])
def test_asymmetric_probe_changes_only_documented_small_offsets(raised, other, yaw):
    case = next(c for c in make_cases() if c["name"] == f"ASYMMETRIC_{raised.upper()}_HIGH")
    baseline = paired_targets(.3, True)
    raised_pose, other_pose = case["targets"][raised], case["targets"][other]
    np.testing.assert_allclose(raised_pose[:3, 3]-baseline[raised][:3, 3], [.02, 0, .03], atol=1e-14)
    np.testing.assert_allclose(other_pose[:3, 3]-baseline[other][:3, 3], [-.01, 0, 0], atol=1e-14)
    np.testing.assert_allclose(raised_pose[:3, :3], baseline[raised][:3, :3] @ rotation_from_rpy_deg([0, 0, yaw]), atol=1e-14)
    np.testing.assert_array_equal(other_pose[:3, :3], baseline[other][:3, :3])
    assert "Generalization probe" in case["note"]


def test_world_targets_set_once_per_case_not_per_policy_tick():
    exp = SimpleDualReachExperiment.__new__(SimpleDualReachExperiment)
    exp.cases = make_cases()[:2]
    exp.case_index = -1
    exp.phase, exp.phase_start = "RESET_SETTLE", 0.
    exp.data = SimpleNamespace(time=.4)
    exp.scratch = SimpleNamespace(geom_xpos=np.array([[0., .1, 0.], [0., -.1, 0.]]))
    exp.foot_geoms = {0, 1}
    exp.policy = SimpleNamespace(set_target_world=Mock(), follow_current=Mock())
    exp._finish_case = Mock()
    def transition(phase):
        exp.phase, exp.phase_start = phase, exp.data.time
    exp.transition = Mock(side_effect=transition)
    exp.update_gate()
    assert exp.phase == "HOME_HOLD"
    assert exp.policy.set_target_world.call_count == 2
    for time in .4 + np.arange(1, 250)*.02:
        exp.data.time = float(time)
        exp.update_gate()
    assert exp.policy.set_target_world.call_count == 2
    exp.data.time = 5.4
    exp.update_gate()
    assert exp.phase == "SYMMETRIC_PREALIGN_30"
    assert exp.policy.set_target_world.call_count == 4
    for time in 5.4 + np.arange(1, 300)*.02:
        exp.data.time = float(time)
        exp.update_gate()
    assert exp.policy.set_target_world.call_count == 4
    exp.data.time = 11.4
    exp.update_gate()
    assert exp.phase == "COMPLETE"
    assert exp.policy.set_target_world.call_count == 4
    for index, case in enumerate(exp.cases):
        for side_index, side in enumerate(SIDES):
            args = exp.policy.set_target_world.call_args_list[2*index+side_index].args
            assert args[0] == side
            p, q = pose_from_transform(case["targets"][side])
            np.testing.assert_allclose(args[1], p, atol=1e-14)
            assert abs(args[2] @ q) == pytest.approx(1., abs=1e-14)


def test_sustained_duration_resets_after_any_failed_sample():
    times = np.arange(11)*.1
    good = [True, True, True, False, True, True, True, False, True, True, True]
    assert sustained_duration(times, good) == pytest.approx(.2)
    assert sustained_duration([0., .1, .2, .3], [True]*4) == pytest.approx(.3)
    assert sustained_duration([], []) == 0.


def test_completed_case_pass_requires_continuous_point_three_seconds():
    rows = make_rows(np.arange(11)*.1)
    for index in (3, 7):
        rows[index]["arms"]["left"]["position_m"] = .1
    report = summarize_case(rows, 1., 1.)
    assert report["completed"]
    assert report["simple_max_hold_s"] == pytest.approx(.2)
    assert not report["simple_reach_passed"]
    assert not report["precise_reach_passed"]
    good = summarize_case(make_rows([0., .1, .2, .3]), .3, .3)
    assert good["simple_reach_passed"] and good["precise_reach_passed"]
    short = summarize_case(make_rows([0., .1, .2, .29]), .29, .29)
    assert not short["simple_reach_passed"] and not short["precise_reach_passed"]


def test_incomplete_case_cannot_pass_despite_long_good_hold():
    report = summarize_case(make_rows(np.arange(10)*.1), .9, 1.)
    assert report["simple_max_hold_s"] == pytest.approx(.9)
    assert not report["completed"]
    assert not report["simple_reach_passed"] and not report["precise_reach_passed"]
    empty = summarize_case([], 6., 6.)
    assert not empty["completed"] and not empty["simple_reach_passed"] and not empty["precise_reach_passed"]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("key,value", [("position_m", .02), ("orientation_deg", 10.), ("linear_speed_mps", .02)])
def test_both_wrists_must_pass_all_strict_simple_bounds(side, key, value):
    rows = make_rows(np.arange(11)*.1)
    for row in rows:
        row["arms"][side][key] = value
    report = summarize_case(rows, 1., 1.)
    assert report["completed"]
    assert not report["simple_reach_passed"] and not report["precise_reach_passed"]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("key,value", [("position_m", .005), ("orientation_deg", 3.)])
def test_simple_success_is_not_mislabeled_precise_reach(side, key, value):
    rows = make_rows(np.arange(11)*.1)
    for row in rows:
        row["arms"][side][key] = value
    report = summarize_case(rows, 1., 1.)
    assert report["simple_reach_passed"] and not report["precise_reach_passed"]


def test_tail_two_seconds_excludes_early_motion_errors_and_jitter():
    rows = make_rows(np.arange(61)*.1)
    for row in rows:
        if row["time_s"] < 4.-1e-8:
            for side in SIDES:
                row["arms"][side].update(position_m=.4, orientation_deg=80., linear_speed_mps=.8,
                                         tcp_position_world_m=[row["time_s"], 0., .8])
    report = summarize_case(rows, 6., 6.)
    assert report["tail_window_s"] == 2.
    assert report["tail_simple_fraction"] == 1. and report["tail_precise_fraction"] == 1.
    assert report["simple_max_hold_s"] == pytest.approx(2.)
    for side in SIDES:
        arm = report["arms"][side]
        for stat in ("mean", "max", "p95"):
            assert arm[f"tail_{stat}_position_m"] == pytest.approx(.003)
            assert arm[f"tail_{stat}_orientation_deg"] == pytest.approx(2.)
            assert arm[f"tail_{stat}_linear_speed_mps"] == pytest.approx(.01)
        assert arm["tail_position_jitter_rms_m"] < 1e-14
