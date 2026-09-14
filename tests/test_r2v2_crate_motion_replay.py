"""Pure replay/FSM contracts; no policy loading or simulated grasp claims."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import common.r2v2_crate_motion_replay as replay
from common.path_config import PROJECT_ROOT
from common.r2v2_crate_lift import load_lift_config
from common.r2v2_crate_motion_recording import load_crate_motion, transform_from_pose


@pytest.fixture(scope="module")
def motion():
    return load_crate_motion(PROJECT_ROOT / "reference_motion_bank/r2v2_crate/down20_yaw15_60mm")


@pytest.fixture
def experiment(motion):
    exp = replay.CrateMotionReplayExperiment.__new__(replay.CrateMotionReplayExperiment)
    exp.motion = motion
    exp.source_boundaries = {event["state"]: event["time_s"] for event in motion.manifest["transitions"]}
    exp.phase, exp.phase_start, exp.failure, exp.failure_phase = "RESET_SETTLE", 0., None, None
    exp.data = SimpleNamespace(time=0., qpos=np.arange(12, dtype=float), qvel=np.zeros(12),
                               warning=SimpleNamespace(number=np.zeros(1, dtype=int)))
    exp.scratch = SimpleNamespace(xpos=np.array([[.2, .2, 1.1], [.2, -.2, 1.1], [.38, 0., 1.011]]),
                                  xmat=np.tile(np.eye(3).ravel(), (3, 1)), qvel=np.zeros(6))
    ids = {"left_hand_roll_link": 0, "right_hand_roll_link": 1, "cargo_crate": 2}
    exp.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=ids[name]))
    exp.policy = SimpleNamespace(set_target_world=Mock(), follow_current=Mock(), act=Mock(), apply=Mock())
    exp.hands = SimpleNamespace(command=Mock(), update=Mock(), apply=Mock())
    exp.params = load_lift_config()
    exp.fullbody_cfg = dict(phase_timeout_s=10., prealign_position_m=.02,
        prealign_orientation_deg=10., insertion_position_m=.005, insertion_orientation_deg=3.,
        motion_speed_mps=.02, motion_stable_s=.3, reset_settle_s=.4, standing_minimum_s=2.)
    exp.current_metrics = dict(base_tilt_deg=0., clearance_m=.1, crate_tilt_deg=0.,
        grasp_slip_m=0., grasp_rotation_slip_deg=0., crate_linear_speed_m_s=0., crate_angular_speed_rad_s=0.,
        table_vertical_force_N=0., hands={side: dict(T_wrist_crate=np.eye(4), finger_normal_force_N=0.,
            finger_handle_vertical_force_N=0., vertical_force_N=0.) for side in replay.SIDES})
    exp.stable_since = exp.bad_since = None
    exp.baseline_relations = None
    exp.trial_confirmed = exp.prealign_passed = False
    exp.source_time_s = 0.
    exp.world_anchor = exp.desired_crate_world = None
    exp.steps = 0
    exp.transitions, exp.targets, exp.hold_samples, exp.samples = [], [], [], []
    exp.goal_wrist_transforms = {side: transform_from_pose(exp.scratch.xpos[index], [1, 0, 0, 0])
                                for index, side in enumerate(replay.SIDES)}
    exp._at_wrist_targets = Mock(return_value=False)
    return exp


def advance_gate(exp, elapsed):
    exp.data.time = exp.phase_start+elapsed
    exp._update_source_clock()
    exp._gate()


def test_relocated_targets_preserve_measured_source_lift_and_se3_rotation(motion):
    anchor = transform_from_pose([.38, -.03, 1.011], [np.cos(.2), 0, 0, np.sin(.2)])
    start = motion.sample(0.)
    end = motion.sample(motion.duration_s)
    start_crate, start_wrists = replay.relocated_targets(start, anchor)
    end_crate, end_wrists = replay.relocated_targets(end, anchor)
    np.testing.assert_allclose(start_wrists, anchor @ start["T_anchor_wrist"], atol=1e-12)
    np.testing.assert_allclose(end_wrists, anchor @ end["T_anchor_wrist"], atol=1e-12)
    assert end_crate[2, 3]-start_crate[2, 3] > .1
    np.testing.assert_allclose(end_wrists[:, 2, 3]-start_wrists[:, 2, 3],
        motion.arrays["T_world_wrist"][-1, :, 2, 3]-motion.arrays["T_world_wrist"][0, :, 2, 3], atol=1e-12)
    # There is no live-crate input: commanded object motion comes from source.
    wrong_live_crate = transform_from_pose([9., 8., 7.], [1, 0, 0, 0])
    assert not np.allclose(end_wrists, wrong_live_crate @ end["T_crate_wrist"])


def test_streaming_never_chases_a_slipping_live_crate(experiment):
    exp = experiment
    exp.world_anchor = transform_from_pose([.38, 0., 1.011], [1, 0, 0, 0])
    exp.source_time_s = 9.
    exp._stream_targets()
    original = {side: pose.copy() for side, pose in exp.goal_wrist_transforms.items()}
    exp.scratch.xpos[2] += [1., 2., -3.]
    exp._stream_targets()
    for side in replay.SIDES:
        np.testing.assert_array_equal(exp.goal_wrist_transforms[side], original[side])
    assert len(exp.targets) == 2


@pytest.mark.parametrize("key,value", [
    ("grasp_slip_m", None), ("grasp_slip_m", float("nan")),
    ("grasp_rotation_slip_deg", None), ("grasp_rotation_slip_deg", float("inf")),
])
def test_success_gate_rejects_missing_or_nonfinite_slip(experiment, key, value):
    experiment.current_metrics[key] = value
    assert not experiment._lift_good()


@pytest.mark.parametrize("phase", replay.SOURCE_PHASES)
def test_source_clock_caps_at_recorded_segment_end(experiment, phase):
    exp = experiment
    exp.data.time = 20.
    exp.enter(phase)
    start = exp.source_boundaries[phase]
    next_phase = (*replay.SOURCE_PHASES, "COMPLETE")[replay.SOURCE_PHASES.index(phase)+1]
    end = exp.source_boundaries[next_phase]
    exp.data.time = exp.phase_start+(end-start)/2
    exp._update_source_clock()
    assert exp.source_time_s == pytest.approx((start+end)/2)
    assert not exp._source_segment_finished()
    exp.data.time = exp.phase_start+100.
    exp._update_source_clock()
    assert exp.source_time_s == end
    assert exp._source_segment_finished()
    exp._update_source_clock()
    assert exp.source_time_s == end


def test_insert_settle_requires_actual_pose_before_closing(experiment):
    exp = experiment
    exp.enter("INSERT_SETTLE")
    for elapsed in (.6, 1., 2., 5., 9.99):
        advance_gate(exp, elapsed)
        assert exp.phase == "INSERT_SETTLE"
        exp.hands.command.assert_not_called()
    assert not exp.trial_confirmed
    advance_gate(exp, 10.)
    assert exp.phase == "FAILED" and exp.failure_phase == "INSERT_SETTLE"
    exp.hands.command.assert_not_called()


def test_insert_settle_requires_completed_source_segment_and_continuous_hold(experiment):
    exp = experiment
    exp._at_wrist_targets.return_value = True
    exp.enter("INSERT_SETTLE")
    advance_gate(exp, .1)
    assert exp.phase == "INSERT_SETTLE" and exp.stable_since is None
    segment = exp.source_boundaries["CLOSE"]-exp.source_boundaries["INSERT_SETTLE"]
    advance_gate(exp, segment)
    advance_gate(exp, segment+.2)
    assert exp.phase == "INSERT_SETTLE"
    exp._at_wrist_targets.return_value = False
    advance_gate(exp, segment+.25)
    exp._at_wrist_targets.return_value = True
    advance_gate(exp, segment+.3)
    advance_gate(exp, segment+.59)
    assert exp.phase == "INSERT_SETTLE"
    advance_gate(exp, segment+.6)
    assert exp.phase == "CLOSE"
    assert exp.hands.command.call_args_list == [(('left', 1),), (('right', 1),)]


def test_close_without_actual_bilateral_contact_never_lifts_and_timeout_keeps_closed(experiment):
    exp = experiment
    exp.enter("CLOSE")
    initial_calls = list(exp.hands.command.call_args_list)
    for elapsed in (1., 3., 5., 9.99):
        advance_gate(exp, elapsed)
        assert exp.phase == "CLOSE"
    advance_gate(exp, 10.)
    assert exp.phase == "FAILED" and exp.failure_phase == "CLOSE"
    assert "timeout" in exp.failure
    assert exp.hands.command.call_args_list == initial_calls
    assert all(call.args[1] == 1 for call in initial_calls)
    assert exp.baseline_relations is None


def test_one_hand_contact_does_not_count_as_bilateral_grasp(experiment):
    exp = experiment
    exp.enter("CLOSE")
    exp.current_metrics["hands"]["left"]["finger_normal_force_N"] = 10.
    advance_gate(exp, 3.)
    advance_gate(exp, 4.)
    assert exp.phase == "CLOSE"
    exp.current_metrics["hands"]["right"]["finger_normal_force_N"] = .1
    advance_gate(exp, 4.1)
    advance_gate(exp, 4.4)
    assert exp.phase == "PROBE_LIFT"
    assert set(exp.baseline_relations) == {"left", "right"}


def test_failure_is_terminal_and_never_resumes_physics(experiment, monkeypatch):
    exp = experiment
    exp.enter("CLOSE")
    exp.fail("synthetic slip")
    assert exp.failure_phase == "CLOSE" and exp.phase == "FAILED"
    step = Mock()
    monkeypatch.setattr(replay.mujoco, "mj_step", step)
    exp.step()
    step.assert_not_called()
    exp.policy.apply.assert_not_called()
    exp.hands.apply.assert_not_called()
    assert all(call.args[1] == 1 for call in exp.hands.command.call_args_list)


def test_reset_settle_follows_measured_wrist_without_teleport_or_crate_targets(experiment, monkeypatch):
    exp = experiment
    exp.sync = Mock()
    exp._safety = Mock()
    exp.record = Mock()
    monkeypatch.setattr(replay.mujoco, "mj_step", Mock())
    qpos, qvel = exp.data.qpos.copy(), exp.data.qvel.copy()
    exp.step()
    exp.policy.follow_current.assert_called_once_with(exp.scratch)
    exp.policy.set_target_world.assert_not_called()
    assert exp.world_anchor is None and not exp.targets
    for index, side in enumerate(replay.SIDES):
        np.testing.assert_array_equal(exp.goal_wrist_transforms[side][:3, 3], exp.scratch.xpos[index])
    np.testing.assert_array_equal(exp.data.qpos, qpos)
    np.testing.assert_array_equal(exp.data.qvel, qvel)


def test_stand_locks_current_world_targets_and_does_not_follow_subsequent_drift(experiment, monkeypatch):
    exp = experiment
    qpos, qvel = exp.data.qpos.copy(), exp.data.qvel.copy()
    exp.data.time = .4
    exp.enter("STAND")
    assert exp.policy.set_target_world.call_count == 2
    locked = {side: goal.copy() for side, goal in exp.goal_wrist_transforms.items()}
    np.testing.assert_array_equal(exp.data.qpos, qpos)
    np.testing.assert_array_equal(exp.data.qvel, qvel)
    exp.policy.set_target_world.reset_mock()
    exp.scratch.xpos[:2] += [.1, .2, -.2]
    exp.sync, exp._safety, exp.record = Mock(), Mock(), Mock()
    monkeypatch.setattr(replay.mujoco, "mj_step", Mock())
    exp.step()
    exp.policy.follow_current.assert_not_called()
    exp.policy.set_target_world.assert_not_called()
    for side in replay.SIDES:
        np.testing.assert_array_equal(exp.goal_wrist_transforms[side], locked[side])
    assert exp.world_anchor is None and not exp.targets


def test_prealign_anchor_is_set_from_actual_crate_without_physics_pose_mutation(experiment):
    exp = experiment
    before = exp.data.qpos.copy()
    crate_actual = transform_from_pose(exp.scratch.xpos[2], [1, 0, 0, 0])
    exp.enter("PREALIGN")
    settled = exp.motion.sample(exp.source_boundaries["INSERT"])
    np.testing.assert_allclose(exp.world_anchor @ settled["T_anchor_crate"], crate_actual, atol=1e-12)
    np.testing.assert_array_equal(exp.data.qpos, before)
    assert exp.source_time_s == 0.
    assert len(exp.targets) == 1 and exp.policy.set_target_world.call_count == 2
