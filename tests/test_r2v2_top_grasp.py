"""CPU-only grasp gates/FSM regressions; synthetic metrics are not physics evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from common.r2v2_hand_control import BinaryHandController
from common.r2v2_top_grasp import (
    TopGraspExperiment, blend, contact_opposition_angle, interpolate_pose, pickup_conditions,
    placed_conditions, rotation_error_deg, unsupported_load_conditions,
)
from common.r2v2_top_grasp_scene import TopGraspCandidate
from r2v2_description.model import load_config


WEIGHT = .1 * 9.81


def metrics(**updates):
    result = dict(
        finite_state=True, finger_pair_contact=True, directional_opposition_deg=180.,
        opposed_contact=True, clearance_m=.02,
        table_contact=False, floor_contact=False,
        hand_table_contact_count=0, parked_hand_contact_count=0,
        hand_vertical_force_N=WEIGHT, table_vertical_force_N=0.,
        object_tilt_deg=0., object_linear_speed_mps=0.,
        object_angular_speed_radps=0., object_position_m=[.1, 0., .5],
        active_hand_object_contact_count=2, grasp_slip_m=0.,
        grasp_rotation_slip_deg=0., T_wrist_object=np.eye(4).tolist(),
        alignment_slip_m=None, alignment_rotation_slip_deg=None,
        hand_object_penetration_m=0., hand_self_penetration_m=0.,
        hand_joint_violation_rad=0., mimic_error_rad=0.,
    )
    result.update(updates)
    return result


def supported(**updates):
    values = dict(table_contact=True, table_vertical_force_N=WEIGHT,
                  clearance_m=0., active_hand_object_contact_count=0,
                  hand_vertical_force_N=0., finger_pair_contact=False,
                  directional_opposition_deg=None, opposed_contact=False)
    values.update(updates)
    return metrics(**values)


def experiment(phase="READY", initial_metrics=None):
    """Exercise the actual supervisor without constructing or stepping MuJoCo."""
    exp = TopGraspExperiment.__new__(TopGraspExperiment)
    exp.data = SimpleNamespace(
        time=0., warning=SimpleNamespace(number=np.zeros(8, dtype=int)),
        xfrc_applied=np.zeros((2, 6)), qfrc_applied=np.zeros(6),
        qpos=np.arange(7, dtype=float), qvel=np.arange(6, dtype=float),
    )
    exp.side, exp.phase, exp.phase_start = "left", phase, 0.
    exp.failure = exp.failure_phase = None
    exp.stable_since = exp.bad_since = None
    exp.grasp_verified = exp.release_commanded = exp.completed = False
    exp.baseline_relation = None
    exp.alignment_relation = None
    exp.upright_attempted = False
    exp.max_verified_hold_s = 0.
    exp.active_goal = np.eye(4)
    exp.active_goal[:3, 3] = [.2, .1, .5]
    exp.move_start = exp.active_goal.copy()
    exp.move_end = exp.active_goal.copy()
    exp.grasp_pose = exp.active_goal.copy()
    exp.motion_duration = 0.
    exp.weight_N, exp.table_height = WEIGHT, .4
    exp.profile = dict(height_m=.12, radius_m=.02, mass_kg=.1)
    exp.place_xy = np.array([.1, 0.])
    exp.transitions, exp.command_calls, exp.peaks = [], [], {}
    exp.current_metrics = metrics() if initial_metrics is None else initial_metrics
    exp.initial_metrics = exp.current_metrics.copy()
    exp.candidate, exp.layout, exp.hand_cfg = TopGraspCandidate(depth_m=0), {}, {}

    def command(side, value):
        exp.command_calls.append((side, value))

    exp.hands = SimpleNamespace(command=command)

    def sync():
        # Recompute exactly the relative-pose diagnostics used by the gate.
        if exp.baseline_relation is not None:
            relation = np.asarray(exp.current_metrics["T_wrist_object"])
            exp.current_metrics["grasp_slip_m"] = float(np.linalg.norm(
                relation[:3, 3] - exp.baseline_relation[:3, 3]))
            exp.current_metrics["grasp_rotation_slip_deg"] = rotation_error_deg(
                relation[:3, :3], exp.baseline_relation[:3, :3])
        if exp.alignment_relation is not None:
            relation = np.asarray(exp.current_metrics["T_wrist_object"])
            exp.current_metrics["alignment_slip_m"] = float(np.linalg.norm(
                relation[:3, 3] - exp.alignment_relation[:3, 3]))
            exp.current_metrics["alignment_rotation_slip_deg"] = rotation_error_deg(
                relation[:3, :3], exp.alignment_relation[:3, :3])
        return exp.current_metrics

    exp.sync = sync
    return exp


def tick(exp, time, measurement=None):
    exp.data.time = float(time)
    if measurement is not None:
        exp.current_metrics = measurement
    exp.sync()
    exp._safety()
    if not exp.done:
        exp._gate()
    return exp


def test_pickup_requires_real_unsupported_weight_bearing():
    assert pickup_conditions(metrics(), WEIGHT)
    assert not pickup_conditions(supported(opposed_contact=True), WEIGHT)


@pytest.mark.parametrize("thumb,fingers", [
    ([], []), ([], [[1., 0., 0.]]), ([[1., 0., 0.]], []),
    (np.empty((0, 3)), np.empty((0, 3))),
    (np.empty((0, 3)), np.array([[1., 0., 0.]])),
])
def test_no_contact_normals_have_no_directional_opposition(thumb, fingers):
    assert contact_opposition_angle(thumb, fingers) is None


@pytest.mark.parametrize("angle", [0., 63., 90., 119.9, 120., 120.1, 180.])
def test_opposition_angle_distinguishes_same_side_from_opposing_contacts(angle):
    radians = np.deg2rad(angle)
    thumb = [[1., 0., 0.]]
    fingers = [[np.cos(radians), np.sin(radians), 0.]]
    measured = contact_opposition_angle(thumb, fingers)
    assert measured == pytest.approx(angle, abs=1e-10)
    if angle < 120.:
        assert measured < 120., "Finger-pair contact on the same side is not opposition"
    elif angle > 120.:
        assert measured > 120.


def test_opposition_normalizes_each_vector_and_uses_maximum_cross_group_angle():
    thumb = np.array([[3., 0., 0.], [0., .0001, 0.]])
    fingers = np.array([[100., 0., 0.], [-2., 0., 0.], [0., 0., 9.]])
    thumb_before, fingers_before = thumb.copy(), fingers.copy()
    assert contact_opposition_angle(thumb, fingers) == pytest.approx(180.)
    assert contact_opposition_angle(fingers, thumb) == pytest.approx(180.)
    # The diagnostic must not normalize caller-owned contact arrays in-place.
    np.testing.assert_array_equal(thumb, thumb_before)
    np.testing.assert_array_equal(fingers, fingers_before)


def test_directional_opposition_is_invariant_to_common_world_rotation():
    thumb = np.array([[1., .2, .3], [.4, .5, .6]])
    fingers = np.array([[-.1, -.8, .4], [.9, -.3, -.2]])
    angle = .73
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
                         [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    expected = contact_opposition_angle(thumb, fingers)
    measured = contact_opposition_angle(thumb @ rotation.T, fingers @ rotation.T)
    assert measured == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("bad", [
    [[0., 0., 0.]], [[float("nan"), 0., 1.]], [[0., float("inf"), 1.]],
    [[1., 0.]], [[1., 0., 0., 0.]], [1., 0., 0.],
    [[[1., 0., 0.]]], [[1., 0., 0.], [0., 0., 0.]],
])
@pytest.mark.parametrize("bad_side", ["thumb", "finger"])
def test_invalid_contact_normals_fail_closed(bad, bad_side):
    good = [[1., 0., 0.]]
    with pytest.raises(ValueError):
        contact_opposition_angle(bad, good) if bad_side == "thumb" else contact_opposition_angle(good, bad)


def test_forceful_finger_pair_on_same_side_cannot_start_probe_or_capture_baseline():
    measured_angle = contact_opposition_angle([[1., 0., 0.]],
        [[np.cos(np.deg2rad(63.)), np.sin(np.deg2rad(63.)), 0.]])
    m = metrics(finger_pair_contact=True, directional_opposition_deg=measured_angle,
                opposed_contact=False, hand_vertical_force_N=10. * WEIGHT)
    assert not pickup_conditions(m, WEIGHT)
    exp = experiment("CLOSE", m)
    for time in np.arange(0., 5.011, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and exp.failure_phase == "CLOSE"
    assert not exp.grasp_verified and exp.baseline_relation is None
    assert not any(event["state"] == "PROBE_LIFT" for event in exp.transitions)


@pytest.mark.parametrize("change", [
    {"finite_state": False}, {"opposed_contact": False},
    {"clearance_m": .0079}, {"table_contact": True}, {"floor_contact": True},
    {"hand_table_contact_count": 1}, {"parked_hand_contact_count": 1},
    {"hand_vertical_force_N": .79 * WEIGHT}, {"object_tilt_deg": 10.01},
    {"object_linear_speed_mps": .02}, {"object_angular_speed_radps": .15},
])
def test_pickup_rejects_each_required_missing_evidence(change):
    assert not pickup_conditions(metrics(**change), WEIGHT)


@pytest.mark.parametrize("change", [
    {"finite_state": False}, {"table_contact": False},
    {"table_vertical_force_N": .79 * WEIGHT}, {"floor_contact": True},
    {"object_tilt_deg": 10.01}, {"object_linear_speed_mps": .02},
    {"object_angular_speed_radps": .15}, {"object_position_m": [.121, 0., .46]},
])
def test_placed_requires_stable_table_support_and_target_region(change):
    assert placed_conditions(supported(), WEIGHT, np.array([.1, 0.]))
    assert not placed_conditions(supported(**change), WEIGHT, np.array([.1, 0.]))


def test_interpolation_is_bounded_smooth_and_preserves_pose_endpoints():
    start, end = np.eye(4), np.eye(4)
    angle = np.deg2rad(179.)
    end[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                   [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    end[:3, 3] = [.2, -.1, .5]
    np.testing.assert_allclose(interpolate_pose(start, end, -1.), start, atol=1e-15)
    np.testing.assert_allclose(interpolate_pose(start, end, 2.), end, atol=1e-15)
    mid = interpolate_pose(start, end, .5)
    np.testing.assert_allclose(mid[:3, 3], end[:3, 3] / 2.)
    np.testing.assert_allclose(mid[:3, :3].T @ mid[:3, :3], np.eye(3), atol=1e-15)
    assert rotation_error_deg(mid[:3, :3], start[:3, :3]) == pytest.approx(89.5)
    assert blend(.001) < 1e-7 and 1. - blend(.999) < 1e-7


def test_close_command_is_not_grasp_success_and_is_not_reissued_each_tick():
    exp = experiment(initial_metrics=metrics(opposed_contact=False))
    exp.enter("CLOSE")
    for time in np.arange(0., 5.011, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and exp.failure_phase == "CLOSE"
    assert exp.command_calls == [("left", 1)]
    assert not exp.grasp_verified and exp.baseline_relation is None
    assert not exp.release_commanded and not exp.report()["success"]


def test_repeated_binary_close_preserves_inflight_reference_and_generator_state():
    cfg = load_config()
    controller = BinaryHandController(cfg, "left", cfg["hands"]["left"]["open"])
    assert controller.set_command(1)
    for _ in range(20):
        controller.step()
    reference = controller.reference
    position = np.asarray(controller.input.current_position).copy()
    velocity = np.asarray(controller.input.current_velocity).copy()
    acceleration = np.asarray(controller.input.current_acceleration).copy()
    for _ in range(20):
        assert not controller.set_command(1)
    assert controller.command_changes == 1 and controller.reference is reference
    np.testing.assert_array_equal(controller.input.current_position, position)
    np.testing.assert_array_equal(controller.input.current_velocity, velocity)
    np.testing.assert_array_equal(controller.input.current_acceleration, acceleration)


def test_sustained_opposed_contact_only_permits_probe_not_verified_pickup():
    exp = experiment("CLOSE", supported(opposed_contact=True))
    start = exp.active_goal.copy()
    for time in np.arange(0., 2.811, .01):
        tick(exp, time)
    assert exp.phase == "PROBE_LIFT"
    assert exp.motion_duration == 1.5
    np.testing.assert_allclose(exp.move_end[:3, 3] - start[:3, 3], [0., 0., .02])
    assert not exp.grasp_verified and exp.baseline_relation is None
    phase_start, target = exp.phase_start, exp.move_end.copy()
    for time in np.arange(2.82, 3.311, .01):
        tick(exp, time)
    assert exp.phase_start == phase_start
    np.testing.assert_array_equal(exp.move_end, target)


def test_table_supported_probe_cannot_capture_a_false_grasp_baseline():
    exp = experiment("VERIFY", supported(opposed_contact=True))
    for time in np.arange(0., 3.011, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and exp.failure_phase == "VERIFY"
    assert exp.baseline_relation is None and not exp.grasp_verified
    assert not exp.command_calls


def test_baseline_capture_is_provisional_until_full_stable_unsupported_second():
    exp = experiment("VERIFY")
    relation = np.eye(4)
    relation[:3, 3] = [.03, -.02, -.10]
    tick(exp, 0., metrics(T_wrist_object=relation.tolist()))
    np.testing.assert_array_equal(exp.baseline_relation, relation)
    assert not exp.grasp_verified
    tick(exp, .4)
    tick(exp, .5, metrics(opposed_contact=False, T_wrist_object=relation.tolist()))
    assert exp.stable_since is None
    for time in np.arange(.51, 1.511, .01):
        tick(exp, time, metrics(T_wrist_object=relation.tolist()))
    assert exp.phase == "LIFT" and exp.grasp_verified
    assert exp.max_verified_hold_s >= 1. - 1e-8


def test_provisional_baseline_is_not_recaptured_to_hide_slip():
    exp = experiment("VERIFY")
    tick(exp, 0.)
    first = exp.baseline_relation.copy()
    slipped = np.eye(4)
    slipped[0, 3] = .02
    for time in np.arange(.01, 3.011, .01):
        tick(exp, time, metrics(T_wrist_object=slipped.tolist()))
    assert exp.phase == "FAILED" and not exp.grasp_verified
    np.testing.assert_array_equal(exp.baseline_relation, first)
    assert not exp.release_commanded


def test_tentative_tilted_load_is_not_final_verified_pickup():
    assert unsupported_load_conditions(metrics(object_tilt_deg=20.), WEIGHT)
    assert unsupported_load_conditions(metrics(object_tilt_deg=30.), WEIGHT)
    assert not unsupported_load_conditions(metrics(object_tilt_deg=30.01), WEIGHT)
    assert not pickup_conditions(metrics(object_tilt_deg=20.), WEIGHT)


@pytest.mark.parametrize("change", [
    {"finite_state": False}, {"opposed_contact": False}, {"clearance_m": .0079},
    {"table_contact": True}, {"floor_contact": True},
    {"hand_table_contact_count": 1}, {"parked_hand_contact_count": 1},
    {"hand_vertical_force_N": .79*WEIGHT}, {"object_linear_speed_mps": .02},
    {"object_angular_speed_radps": .15},
])
def test_tentative_load_keeps_all_physical_support_conditions(change):
    assert not unsupported_load_conditions(metrics(object_tilt_deg=20., **change), WEIGHT)


def start_upright():
    exp = experiment("VERIFY")
    relation = np.eye(4)
    angle = np.deg2rad(20.)
    relation[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                        [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    relation[:3, 3] = [.03, -.02, -.10]
    exp.current_metrics = metrics(object_tilt_deg=20., T_wrist_object=relation.tolist())
    tick(exp, 1.)
    assert exp.phase == "UPRIGHT"
    return exp


def test_upright_uses_measured_relation_only_to_command_wrist_not_object_state():
    exp = experiment("VERIFY")
    qpos, qvel = exp.data.qpos.copy(), exp.data.qvel.copy()
    relation = np.eye(4)
    relation[:3, 3] = [.035, -.018, -.105]
    m = metrics(object_tilt_deg=20., T_wrist_object=relation.tolist())
    tick(exp, .99, m)
    assert exp.phase == "VERIFY" and not exp.upright_attempted
    tick(exp, 1.)
    assert exp.phase == "UPRIGHT" and exp.upright_attempted
    assert exp.motion_duration == 2.
    np.testing.assert_array_equal(exp.alignment_relation, relation)
    desired_object = exp.move_end @ relation
    np.testing.assert_allclose(desired_object[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(desired_object[:3, 3], m["object_position_m"], atol=1e-15)
    np.testing.assert_array_equal(exp.data.qpos, qpos)
    np.testing.assert_array_equal(exp.data.qvel, qvel)
    assert not exp.grasp_verified and exp.baseline_relation is None
    assert not exp.release_commanded and not exp.report()["success"]
    assert not exp.command_calls


@pytest.mark.parametrize("change", [
    {"opposed_contact": False}, {"table_contact": True}, {"floor_contact": True},
    {"clearance_m": .0079}, {"object_tilt_deg": 30.01},
])
def test_upright_correction_never_continues_after_loss_of_opposition_or_clearance(change):
    exp = start_upright()
    measurement = exp.current_metrics.copy()
    measurement.update(change)
    tick(exp, 1.01, measurement)
    assert exp.phase == "FAILED" and exp.failure_phase == "UPRIGHT"
    assert not exp.grasp_verified and not exp.release_commanded
    assert not exp.command_calls


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_upright_retains_fifteen_mm_five_degree_relative_slip_limits(kind):
    exp = start_upright()
    changed_relation = exp.alignment_relation.copy()
    if kind == "translation":
        changed_relation[0, 3] += .016
    else:
        angle = np.deg2rad(5.1)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        changed_relation[:3, :3] = rotation @ changed_relation[:3, :3]
    measurement = exp.current_metrics.copy()
    measurement["T_wrist_object"] = changed_relation.tolist()
    tick(exp, 1.01, measurement)
    assert exp.phase == "FAILED" and not exp.grasp_verified and not exp.release_commanded


def test_upright_target_not_restarted_and_final_upright_verification_needs_new_full_second():
    exp = start_upright()
    target, phase_start = exp.move_end.copy(), exp.phase_start
    for time in np.arange(1.01, 3., .01):
        tick(exp, time)
    assert exp.phase == "UPRIGHT" and exp.phase_start == phase_start
    np.testing.assert_array_equal(exp.move_end, target)
    assert not exp.grasp_verified and exp.baseline_relation is None
    corrected = exp.current_metrics.copy()
    corrected["object_tilt_deg"] = 0.
    tick(exp, 3., corrected)
    assert exp.phase == "VERIFY" and exp.stable_since is None
    tick(exp, 3.01, corrected)
    assert exp.baseline_relation is not None and not exp.grasp_verified
    for time in np.arange(3.02, 4.01, .01):
        tick(exp, time, corrected)
    assert not exp.grasp_verified
    tick(exp, 4.01, corrected)
    assert exp.phase == "LIFT" and exp.grasp_verified
    assert exp.max_verified_hold_s >= 1. - 1e-8
    assert sum(event["state"] == "UPRIGHT" for event in exp.transitions) == 1


def test_unsuccessful_upright_attempt_cannot_loop_or_relax_final_tilt_gate():
    exp = start_upright()
    for time in np.arange(1.01, 6.011, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and exp.failure_phase == "VERIFY"
    assert sum(event["state"] == "UPRIGHT" for event in exp.transitions) == 1
    assert not exp.grasp_verified and exp.baseline_relation is None
    assert not exp.release_commanded and not exp.command_calls


def test_lower_goal_uses_current_measured_wrist_object_relation_and_upright_object():
    exp = experiment("HOLD_MOVED")
    exp.grasp_verified = True
    held = np.eye(4)
    angle = np.deg2rad(2.)
    held[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                    [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    held[:3, 3] = [.035, -.018, -.105]
    exp.baseline_relation = held.copy()
    exp.baseline_relation[0, 3] -= .003
    for time in np.arange(0., 1.011, .01):
        tick(exp, time, metrics(T_wrist_object=held.tolist()))
    assert exp.phase == "LOWER"
    np.testing.assert_array_equal(exp.placement_relation, held)
    desired_object = exp.move_end @ held
    np.testing.assert_allclose(desired_object[:3, :3], np.eye(3), atol=1e-15)
    np.testing.assert_allclose(desired_object[:3, 3], [.1, 0., .46], atol=1e-15)
    assert exp.command_calls == [] and not exp.release_commanded


def test_lower_timeout_without_table_support_keeps_hand_closed():
    exp = experiment("LOWER")
    exp.grasp_verified = True
    for time in np.arange(0., 6.021, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and exp.failure_phase == "PLACE_SETTLE"
    assert exp.command_calls == [] and not exp.release_commanded


def test_place_support_must_persist_before_single_open_command():
    exp = experiment("PLACE_SETTLE")
    tick(exp, 0., supported())
    tick(exp, .2, supported())
    tick(exp, .21, metrics())
    assert not exp.release_commanded
    for time in np.arange(.22, .531, .01):
        tick(exp, time, supported())
    assert exp.phase == "OPEN" and exp.release_commanded
    assert exp.command_calls == [("left", 0)]
    for time in np.arange(.54, 2., .01):
        tick(exp, time, supported())
    assert exp.command_calls == [("left", 0)]


@pytest.mark.parametrize("phase", ["LIFT", "HOLD_LIFT", "TRANSLATE", "HOLD_MOVED", "LOWER", "PLACE_SETTLE"])
def test_loss_of_verified_unsupported_grasp_fails_without_opening(phase):
    exp = experiment(phase)
    exp.grasp_verified = True
    exp.baseline_relation = np.eye(4)
    exp.current_metrics = metrics(opposed_contact=False, hand_vertical_force_N=0.)
    tick(exp, 0.)
    assert exp.phase == "FAILED", "Dropping during lowering must not become successful placement"
    assert not exp.release_commanded and exp.command_calls == []


def test_measured_table_support_allows_load_transfer_before_opening():
    exp = experiment("LOWER", supported())
    exp.grasp_verified = True
    # After genuine table contact the closed hand may stop bearing weight;
    # this is a support hand-off, not an unsupported drop.
    tick(exp, .5)
    assert exp.phase == "PLACE_SETTLE" and not exp.release_commanded
    for time in np.arange(.51, .821, .01):
        tick(exp, time, supported())
    assert exp.phase == "OPEN" and exp.command_calls == [("left", 0)]


@pytest.mark.parametrize("phase", ["OPEN", "RETREAT", "FINAL_HOLD"])
def test_dropped_or_unsupported_object_cannot_complete_release_sequence(phase):
    exp = experiment(phase)
    exp.grasp_verified = exp.release_commanded = True
    exp.current_metrics = metrics(opposed_contact=False, active_hand_object_contact_count=0,
                                  hand_vertical_force_N=0.)
    for time in np.arange(0., 5.021, .01):
        tick(exp, time)
    assert exp.phase == "FAILED" and not exp.completed
    assert not exp.report()["success"]


def test_final_success_requires_full_stable_second_after_retreat_and_clear_hand():
    exp = experiment("FINAL_HOLD", supported())
    exp.grasp_verified = exp.release_commanded = True
    tick(exp, 0.)
    tick(exp, .8)
    tick(exp, .81, supported(active_hand_object_contact_count=1))
    assert not exp.completed and exp.stable_since is None
    for time in np.arange(.82, 1.831, .01):
        tick(exp, time, supported())
    assert exp.phase == "COMPLETE" and exp.completed and exp.report()["success"]


@pytest.mark.parametrize("change", [
    {"finite_state": False}, {"floor_contact": True},
    {"hand_table_contact_count": 1}, {"parked_hand_contact_count": 1},
    {"hand_object_penetration_m": .0031}, {"hand_self_penetration_m": .0031},
    {"hand_joint_violation_rad": .0101}, {"mimic_error_rad": .0151},
    {"object_tilt_deg": 30.1},
])
def test_numerical_contact_joint_and_drop_safety_fail_closed(change):
    exp = experiment("TRANSLATE", metrics(**change))
    exp.grasp_verified = True
    tick(exp, 0.)
    assert exp.phase == "FAILED" and not exp.release_commanded
    assert not exp.command_calls


@pytest.mark.parametrize("kind", ["warning", "xfrc", "qfrc"])
def test_unexpected_external_assistance_or_solver_warning_is_failure(kind):
    exp = experiment("LIFT")
    if kind == "warning":
        exp.data.warning.number[0] = 1
    elif kind == "xfrc":
        exp.data.xfrc_applied[0, 2] = 1.
    else:
        exp.data.qfrc_applied[0] = 1.
    tick(exp, 0.)
    assert exp.phase == "FAILED" and not exp.release_commanded


def test_failure_is_sticky_and_all_motion_phases_have_finite_deadline():
    exp = experiment("APPROACH")
    tick(exp, 10.01)
    assert exp.phase == "FAILED" and "deadline" in exp.failure
    reason, count = exp.failure, len(exp.transitions)
    exp.fail("secondary failure")
    assert exp.failure == reason and len(exp.transitions) == count
    assert not exp.report()["success"]


@pytest.mark.parametrize("profile", ["baseline_40mm_100g", "sleek_330ml_approx_full"])
def test_real_cpu_scene_diagnostics_never_replay_free_object_state(profile):
    """Two 10 ms binding checks only; not a grasp or trajectory sweep."""
    exp = TopGraspExperiment(profile=profile, candidate=TopGraspCandidate(depth_m=0), keep_trace=False)
    assert not exp.done
    for _ in range(10):
        exp.step()
    assert exp.data.time == pytest.approx(.01) and exp.phase == "READY"
    before = {key: getattr(exp.data, key).copy() for key in
              ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "xfrc_applied", "qfrc_applied")}
    physics_time = exp.data.time
    for _ in range(3):
        exp.sync()
    assert exp.data.time == physics_time
    for key, value in before.items():
        np.testing.assert_array_equal(getattr(exp.data, key), value)
    assert not np.any(exp.data.xfrc_applied) and not np.any(exp.data.qfrc_applied)
    assert exp.model.nu == 12 and exp.model.joint("cylinder_free").dofadr[0] >= 0
    assert not exp.grasp_verified and not exp.release_commanded
    assert exp.report()["object_free"] and not exp.report()["object_pose_replay"]
