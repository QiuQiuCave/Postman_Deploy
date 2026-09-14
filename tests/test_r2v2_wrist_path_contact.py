"""CPU-only contracts for the timed physical contact experiment.

No policy session, robot stepping or rendered grasp is invoked by these tests.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_lift import load_lift_config
from common.r2v2_crate_reach import CrateReachExperiment
from common.r2v2_crate_motion_replay_scene import build_motion_replay_model
from common.r2v2_reach_sim import load_reach_config
from common.r2v2_wrist_path_contact import (
    ArchivedContactPath, HOME, PREPARATION_S, SAFE, SIDES,
    WristPathContactExperiment, contact_lift_conditions,
)


MANIFEST = Path('/root/code/amo/AMO_R2/src/r2v2_loco/assets/reach_paths/crate_closer5_down5_lift2_rear3_v1/manifest.json')


@pytest.fixture(scope='module')
def path():
    if not MANIFEST.is_file():
        pytest.skip('Versioned training path asset is not installed alongside deployment repo')
    return ArchivedContactPath(MANIFEST)


def metrics():
    return dict(grasp_slip_m=0., grasp_rotation_slip_deg=0., clearance_m=.02,
        table_contact=False, floor_contact=False, floor_contacts=[], nonhand_crate_contact_count=0,
        robot_table_contact_count=0, crate_tilt_deg=0., crate_linear_speed_m_s=0.,
        crate_angular_speed_rad_s=0., finite_state=True, base_tilt_deg=0.,
        max_hand_crate_penetration_m=0., max_hand_self_penetration_m=0.,
        wrist_tracking_error_m=0., wrist_tracking_error_deg=0.,
        hands={s:dict(finger_handle_vertical_force_N=2., vertical_force_N=2.,
                      finger_normal_force_N=1., thumb_normal_force_N=0.,
                      T_wrist_crate=np.eye(4)) for s in SIDES},
        wrist_errors={s:dict(position_m=0., orientation_deg=0., linear_speed_mps=0.) for s in SIDES})


def test_archived_preparation_matches_exact_training_home_to_safe(path):
    assert path.phase_names[:4] == ('STAND', 'HOME', 'SAFE_WAIT', 'START_HOLD')
    np.testing.assert_array_equal(path.phase_start[:4], [0., .4, 5.4, 13.4])
    for t in (0., .1, .4, 3., 5.4, 7.3, 9.4, 12.5, 13.4, 15.399):
        _, wrists, _ = path.sample(t)
        fraction = np.clip((t-5.4)/8., 0., 1.)
        expected = HOME+(SAFE-HOME)*fraction*fraction*(3.-2.*fraction)
        np.testing.assert_allclose(wrists[:, :3, 3], expected, atol=1e-14)
        np.testing.assert_allclose(wrists[:, :3, :3], np.tile(np.eye(3), (2, 1, 1)), atol=1e-14)
    np.testing.assert_allclose(path.sample(PREPARATION_S)[1][:, :3, 3], SAFE, atol=1e-14)


def test_archived_source_keyframes_and_revised_rigid_lift_are_exact(path):
    for index in np.unique(np.r_[0, np.arange(0, len(path.time), 31), len(path.time)-1]):
        _, wrists, crate = path.sample(PREPARATION_S+path.time[index])
        np.testing.assert_allclose(wrists, path.wrists[index], atol=1e-12)
        np.testing.assert_allclose(crate, path.crate[index], atol=1e-12)
    close_time = path.phase_start[path.phase_names.index('CLOSE')]
    lift_time = path.phase_start[path.phase_names.index('PROBE_LIFT')]
    _, inserted, crate_inserted = path.sample(close_time)
    np.testing.assert_allclose(path.sample(lift_time-1e-6)[1], inserted, atol=1e-12)
    for fraction in np.linspace(0., 1., 11):
        _, wrists, crate = path.sample(lift_time+4.01*fraction)
        np.testing.assert_allclose(np.linalg.inv(crate)@wrists,
                                   np.linalg.inv(crate_inserted)@inserted, atol=1e-12)
        np.testing.assert_allclose(wrists[:, :3, :3], inserted[:, :3, :3], atol=1e-12)
    final = path.sample(path.duration_s+2.)[1]
    np.testing.assert_allclose(final[:, :3, 3]-inserted[:, :3, 3], [[-.03, 0., .02]]*2, atol=1e-12)


@pytest.mark.parametrize('bad_time', [-1., np.nan, np.inf, -np.inf])
def test_archived_time_rejects_invalid_values(path, bad_time):
    with pytest.raises(ValueError, match='simulation time'):
        path.sample(bad_time)


def test_archived_path_integrity_is_checked(path, tmp_path):
    info = deepcopy(path.manifest)
    info['trajectory_file'] = str(path.manifest_path.parent/info['trajectory_file'])
    info['trajectory_sha256'] = '0'*64
    bad = tmp_path/'manifest.json'
    bad.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='hash mismatch'):
        ArchivedContactPath(bad)


def test_archived_numpy_path_matches_native_training_sampler_cpu(path):
    native_python = Path('/root/code/amo/AMO_R2/.venv/bin/python')
    if not native_python.is_file():
        pytest.skip('Optional native training environment is not installed')
    times = np.unique(np.r_[np.arange(0., path.duration_s+2., .01),
        np.random.default_rng(7).uniform(0., path.duration_s+2., 1000), path.phase_start,
        path.phase_start[1:]-1e-7, path.phase_start[1:]+1e-7])
    program = '''
import json, sys, torch
from r2v2_loco.tasks.reach.wrist_path_data import WristPathData
request = json.load(sys.stdin)
p = WristPathData(manifest_path=request['manifest'], device='cpu', dtype=torch.float64)
position, quaternion, phase = p.sample(torch.tensor(request['times'], dtype=torch.float64))
print(json.dumps(dict(position=position.tolist(), quaternion=quaternion.tolist(),
                     phase=phase.tolist(), phase_names=p.phase_names,
                     phase_start=p.phase_start_s.tolist())))
'''
    result = subprocess.run([str(native_python), '-c', program],
        input=json.dumps(dict(manifest=str(path.manifest_path), times=times.tolist())),
        text=True, capture_output=True, timeout=90,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, cwd='/root/code/amo/AMO_R2')
    assert result.returncode == 0, result.stderr
    native = json.loads(result.stdout.strip().splitlines()[-1])
    samples = [path.sample(float(t)) for t in times]
    desired = np.array([sample[1] for sample in samples])
    quaternions = np.array(native['quaternion'])
    rotations = Rotation.from_quat(quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(len(times), 2, 3, 3)
    np.testing.assert_allclose(desired[:, :, :3, 3], native['position'], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(desired[:, :, :3, :3], rotations, atol=1e-9, rtol=0.)
    mismatches = [(i, sample[0], native['phase_names'][native['phase'][i]])
                  for i, sample in enumerate(samples)
                  if sample[0] != native['phase_names'][native['phase'][i]]]
    # Deployment intentionally adds 1 ns for decimal transition timestamps;
    # native float64 keeps the raw boundary. No mismatch away from it is valid.
    for index, deployed_phase, native_phase in mismatches:
        native_index = native['phase_names'].index(native_phase)
        deployed_index = native['phase_names'].index(deployed_phase)
        assert deployed_index == native_index+1
        assert abs(times[index]-native['phase_start'][deployed_index]) <= 1e-9
    print(dict(native_path_samples=len(times),
        position_max_abs_m=float(np.max(np.abs(desired[:, :, :3, 3]-native['position']))),
        rotation_matrix_max_abs=float(np.max(np.abs(desired[:, :, :3, :3]-rotations))),
        sub_nanosecond_boundary_aliases=len(mismatches)))


def test_physical_pickup_and_strict_tracking_are_distinct():
    m = metrics()
    assert contact_lift_conditions(m, baseline_contact_verified=True, weight_N=3.924) == (True, True)
    m['wrist_errors']['left']['position_m'] = .10
    assert contact_lift_conditions(m, baseline_contact_verified=True, weight_N=3.924) == (True, False)


@pytest.mark.parametrize('mutation', [
    lambda m: m.update(clearance_m=0.),  # only hands move, crate stays on table
    lambda m: m.update(table_contact=True),
    lambda m: m.update(floor_contact=True),
    lambda m: m.update(nonhand_crate_contact_count=1),
    lambda m: m.update(robot_table_contact_count=1),
    lambda m: m.update(grasp_slip_m=.015),
    lambda m: m.update(grasp_slip_m=None),
    lambda m: m.update(grasp_slip_m=np.nan),
    lambda m: m.update(grasp_rotation_slip_deg=5.),
    lambda m: m.update(grasp_rotation_slip_deg=None),
    lambda m: m.update(grasp_rotation_slip_deg=np.inf),
    lambda m: m.update(crate_tilt_deg=8.01),
    lambda m: m.update(crate_linear_speed_m_s=.02),
    lambda m: m.update(crate_angular_speed_rad_s=.1),
    lambda m: m['hands']['right'].update(finger_handle_vertical_force_N=0.),
    lambda m: [m['hands'][s].update(finger_handle_vertical_force_N=0.,
                 thumb_normal_force_N=100., vertical_force_N=10.) for s in SIDES],
    lambda m: [m['hands'][s].update(vertical_force_N=.2) for s in SIDES],
])
def test_grasp_claim_rejects_nonphysical_cases(mutation):
    m = metrics()
    mutation(m)
    assert contact_lift_conditions(m, baseline_contact_verified=True, weight_N=3.924) == (False, False)


def test_grasp_claim_requires_preexisting_verified_baseline():
    assert contact_lift_conditions(metrics(), baseline_contact_verified=False, weight_N=3.924) == (False, False)


@pytest.fixture
def mock_experiment(path):
    e = WristPathContactExperiment.__new__(WristPathContactExperiment)
    e.path = path
    e.phase, e.phase_start, e.failure, e.failure_phase = 'STAND', 0., None, None
    e.data = SimpleNamespace(time=0., qpos=np.zeros(3), qvel=np.zeros(3),
        warning=SimpleNamespace(number=np.zeros(1, dtype=int)), xfrc_applied=np.zeros((1, 6)),
        qfrc_applied=np.zeros(3), actuator_force=np.zeros(1))
    e.model = SimpleNamespace(nmocap=0, nq=64, nv=62, nu=40, neq=10, eq_type=np.array([mujoco.mjtEq.mjEQ_JOINT]),
        jnt_range=np.array([[-1., 1.]]*3), jnt_qposadr=np.arange(3))
    e.cfg = {'gates':dict(max_joint_violation_rad=.05, max_self_penetration_m=.005,
                         max_base_tilt_deg=35., min_base_height_m=.55)}
    e.params = load_lift_config()
    e.body_map = SimpleNamespace(qpos=np.array([0]), joints=np.array([0]))
    e.joints, e.mimics = np.array([1, 2]), [(2, 1, 1., 0.)]
    e.scratch = SimpleNamespace(contact=[], xpos=np.array([[0., 0., .82]]))
    e.base, e.floor = 0, 0
    e.robot_geoms, e.foot_geoms, e.hand_geoms = {1, 2, 3, 4}, {1}, {3, 4}
    e.crate_geoms, e.table_geoms = {5}, {6}
    e.hands = SimpleNamespace(command=Mock(), maps={s:SimpleNamespace(actuators=np.array([0])) for s in SIDES})
    e.current_metrics, e.peaks = metrics(), defaultdict(float)
    e.constraint_events, e.constraint_active, e.constraint_violations_seen = [], set(), set()
    e.source_time_s = 0.
    e.transitions, e.targets, e.hold_samples = [], [], []
    e.physical_hold_s = e.strict_hold_s = e.contact_hold_s = 0.
    e.max_physical_hold_s = e.max_strict_hold_s = 0.
    e._hold_since = dict(contact=None, physical=None, strict=None)
    e.baseline_contact_verified, e.baseline_relations = False, None
    e.crate_params = replace(load_crate_config(), width=.26)
    e.samples = []
    e.initial_metrics = deepcopy(e.current_metrics)
    e.hand_cfg, e.layout = {}, {}
    e.parity_evidence = dict(checkpoint_sha256='checkpoint', onnx_sha256='onnx')
    e.parity_path = Path('/tmp/not-loaded-parity.json')
    return e


def contact(a, b, penetration):
    return SimpleNamespace(geom=np.array([a, b]), dist=-penetration)


def test_safety_logs_prop_collisions_without_stopping(mock_experiment):
    e = mock_experiment
    e.scratch.contact = [contact(3, 5, .1), contact(2, 5, .05), contact(2, 6, .05), contact(1, 0, .002)]
    e.current_metrics.update(max_hand_crate_penetration_m=.1, crate_tilt_deg=90.,
                             clearance_m=-1., grasp_slip_m=1., wrist_tracking_error_m=1.)
    e._safety()
    assert not e.done and e.failure is None
    assert e.current_metrics['robot_table_contact_count'] == 1
    assert e.current_metrics['nonhand_crate_contact_count'] == 1
    assert e.peaks['hand_crate_penetration_m'] == .1


@pytest.mark.parametrize('mutation,reason', [
    (lambda e: e.scratch.contact.append(contact(2, 0, .001)), 'non-foot ground contact'),
    (lambda e: e.current_metrics.update(base_tilt_deg=35.1), 'base tilt/height'),
    (lambda e: e.scratch.xpos.__setitem__((0, 2), .54), 'base tilt/height'),
    (lambda e: e.current_metrics.update(finite_state=False), 'Nonfinite'),
    (lambda e: e.data.warning.number.__setitem__(0, 1), 'MuJoCo warning'),
    (lambda e: e.data.xfrc_applied.__setitem__((0, 0), 1.), 'assistance'),
    (lambda e: e.model.eq_type.__setitem__(0, mujoco.mjtEq.mjEQ_WELD), 'fixture'),
])
def test_safety_retains_robot_and_numerical_stops(mock_experiment, mutation, reason):
    e = mock_experiment
    mutation(e)
    e._safety()
    assert e.done and e.phase == 'FAILED'
    assert reason in e.failure


@pytest.mark.parametrize('mutation,reason', [
    (lambda e: e.scratch.contact.append(contact(2, 3, .006)), 'robot self penetration'),
    (lambda e: e.data.qpos.__setitem__(0, 1.06), 'body joint limit'),
    (lambda e: e.data.qpos.__setitem__(slice(1, 3), 1.010032), 'hand joint limit'),
    (lambda e: e.data.qpos.__setitem__(2, .02), 'hand mimic error'),
])
def test_exploratory_constraints_are_logged_and_legacy_safety_still_stops(mock_experiment, mutation, reason):
    e = mock_experiment
    e.data.time = .04
    mutation(e)
    original_ranges = e.model.jnt_range.copy()
    original_q = e.data.qpos.copy()
    e._safety()
    assert not e.done and e.failure is None
    assert reason in e.constraint_violations_seen
    assert reason in e.current_metrics['constraint_deviations']
    assert len(e.constraint_events) == 1
    assert e.constraint_events[0]['time_s'] == .04
    np.testing.assert_array_equal(e.model.jnt_range, original_ranges)
    np.testing.assert_array_equal(e.data.qpos, original_q)
    e._safety()
    assert len(e.constraint_events) == 1  # one state change, not per-frame spam
    CrateReachExperiment._safety(e)
    assert e.done and reason in e.failure  # old strict class behavior preserved


def test_initial_contact_hand_overrun_does_not_stop_timed_trial(mock_experiment):
    e = mock_experiment
    e.data.time = .04
    e.data.qpos[1:] = 1.010032  # regression: first real trial stopped at this amount
    e.scratch.contact = [contact(3, 5, .00987)]
    e.current_metrics['max_hand_crate_penetration_m'] = .00987
    e._safety(); e._gate()
    assert not e.done and e.phase == 'STAND'
    assert e.peaks['joint_violation_rad'] == pytest.approx(.010032)
    assert 'hand joint limit' in e.constraint_violations_seen


def test_report_does_not_turn_earlier_pickup_or_constraint_failure_into_strict_success(mock_experiment):
    e = mock_experiment
    e.phase = 'COMPLETE'
    e.max_physical_hold_s = e.max_strict_hold_s = 3.
    report = e.report()
    assert report['pickup_observed']
    assert not report['physical_pickup_verified'] and not report['strict_success']
    e.physical_hold_s = e.strict_hold_s = 3.
    e.constraint_violations_seen.add('hand joint limit')
    report = e.report()
    assert report['physical_pickup_verified']
    assert not report['strict_success'] and not report['constraints_passed']


def test_clock_advances_full_schedule_despite_all_grasp_and_pose_failures(mock_experiment):
    e = mock_experiment
    e._at_wrist_targets = Mock(side_effect=AssertionError('No pose gate is permitted'))
    e._stable = Mock(side_effect=AssertionError('No stable gate is permitted'))
    e._bilateral_contact = Mock(return_value=False)
    e.current_metrics.update(grasp_slip_m=.5, table_contact=True, clearance_m=-.1,
                             wrist_tracking_error_m=1.)
    for side in SIDES:
        e.current_metrics['wrist_errors'][side].update(position_m=1., orientation_deg=90., linear_speed_mps=1.)
    for phase, time_s in zip(e.path.phase_names, e.path.phase_start):
        e.data.time = float(time_s+1e-6)
        e._update_source_clock(); e._gate()
        assert e.phase == phase and e.failure is None
    assert e.hands.command.call_count == 2
    assert not e.baseline_contact_verified and e.max_physical_hold_s == 0.
    e.data.time = e.path.duration_s+2.
    e._gate()
    assert e.phase == 'COMPLETE' and e.failure is None


def test_close_timer_requires_continuous_contact_but_does_not_gate_lift(mock_experiment):
    e = mock_experiment
    e.data.time = e.path.close_time
    e._gate()
    for _ in range(29):
        e.data.time += .01
        e._gate()
    assert e.contact_hold_s == pytest.approx(.29)
    e.data.time += .01
    e._gate()
    assert e.contact_hold_s == pytest.approx(.3)
    e.current_metrics['hands']['right']['finger_normal_force_N'] = 0.
    e.data.time += .01; e._gate()
    assert e.contact_hold_s == 0.
    e.data.time = e.path.phase_start[e.path.phase_names.index('PROBE_LIFT')]
    e._gate()
    assert e.phase == 'PROBE_LIFT' and not e.baseline_contact_verified


def test_hold_timer_uses_real_elapsed_time_not_number_of_gate_calls(mock_experiment):
    e = mock_experiment
    e.phase = 'HOLD'
    e.baseline_contact_verified = True
    started = float(e.path.phase_start[e.path.phase_names.index('HOLD')])
    e.data.time = started
    for _ in range(5):
        e._gate()
    assert e.physical_hold_s == 0. and e.strict_hold_s == 0.
    e.data.time = started+1.99
    e._gate()
    assert e.physical_hold_s == pytest.approx(1.99)
    assert e.physical_hold_s < 2.
    e.data.time = started+2.
    e._gate()
    assert e.physical_hold_s == pytest.approx(2.)
    assert e.strict_hold_s == pytest.approx(2.)


def test_final_tick_loss_of_contact_invalidates_completed_lift(mock_experiment):
    e = mock_experiment
    e.phase = 'HOLD'
    e.baseline_contact_verified = True
    e.data.time = e.path.duration_s-.1
    e._gate()
    e.data.time += 2.
    e._gate()
    assert e.physical_hold_s == pytest.approx(2.)
    assert not e.done
    e.current_metrics['table_contact'] = True
    e.data.time = e.path.duration_s+2.
    e._gate()
    assert e.phase == 'COMPLETE'
    assert e.physical_hold_s == 0. and e.strict_hold_s == 0.
    report = e.report()
    assert report['pickup_observed']
    assert not report['physical_pickup_verified'] and not report['strict_success']


def test_hand_box_penetration_rejects_only_strict_success_not_timed_contact_experiment(mock_experiment):
    e = mock_experiment
    e.current_metrics['max_hand_crate_penetration_m'] = .003001
    physical, accurate_pickup = contact_lift_conditions(e.current_metrics,
        baseline_contact_verified=True, weight_N=3.924)
    assert physical and accurate_pickup  # instantaneous load/pose metrics only
    e._safety()
    assert not e.done and e.failure is None
    assert e.peaks['hand_crate_penetration_m'] == .003001
    assert 'hand-crate penetration' in e.constraint_violations_seen
    e.phase = 'COMPLETE'
    e.physical_hold_s = e.strict_hold_s = 3.
    report = e.report()
    assert report['physical_pickup_verified']
    assert not report['strict_success'] and not report['constraints_passed']


def test_near_table_override_is_explicit_and_keeps_real_contacts():
    cfg = load_reach_config('deploy_mujoco/config/r2v2_reach_wrist_v2.yaml')
    params = replace(load_crate_config(), width=.26)
    layout = dict(table_top_m=.9609189696536513, table_center_xy=(.43, 0.), crate_center_xy=(.33, 0.))
    with pytest.raises(ValueError, match='Table front'):
        build_motion_replay_model(cfg, params, **layout)
    with pytest.raises(TypeError, match='explicit boolean'):
        build_motion_replay_model(cfg, params, **layout, allow_near_table=1)
    model, _, actual = build_motion_replay_model(cfg, params, **layout, allow_near_table=True)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (64, 62, 40, 10, 0)
    assert model.geom('lift_table_geom').contype[0] != 0
    assert model.geom('crate_left_handle_beam').contype[0] != 0
    np.testing.assert_allclose(actual['table_center_xy'], [.43, 0.])
    np.testing.assert_allclose(actual['crate_center_xy'], [.33, 0.])
    with pytest.raises(ValueError, match='footprint'):
        build_motion_replay_model(cfg, params, **dict(layout, crate_center_xy=(.55, 0.)), allow_near_table=True)


@pytest.fixture(scope='module')
def payload_path():
    manifest = MANIFEST.parent.parent/'crate_closer5_down5_payload_v2/manifest.json'
    if not manifest.is_file():
        pytest.skip('Versioned payload path asset is not installed')
    return ArchivedContactPath(manifest)


def test_payload_archive_retains_new_phase_names_and_native_poses(payload_path):
    assert payload_path.task_id == 'R2V2-Reach-CrateWristPayload-v2-28DoF'
    assert payload_path.close_phase == 'CLOSE_SEAT'
    assert payload_path.phase_names[-5:] == (
        'CLOSE_SEAT', 'PROBE_LIFT', 'HOLD_PROBE', 'LIFT_HIGHER', 'HOLD_HIGHER')
    assert payload_path.duration_s == pytest.approx(62.21)
    test_archived_numpy_path_matches_native_training_sampler_cpu(payload_path)


def test_payload_policy_identity_and_path_hash_cannot_be_relabeled(payload_path):
    from common.r2v2_wrist_path_contact import validate_contact_policy_path
    parity = dict(task=payload_path.task_id)
    metadata = dict(task_id=payload_path.task_id, path_contract='wrist_payload_path_v2',
                    path_trajectory_sha256=payload_path.sha256)
    validate_contact_policy_path(payload_path, parity, metadata)
    with pytest.raises(ValueError, match='path task'):
        validate_contact_policy_path(payload_path, dict(task='R2V2-Reach-CrateWristPath-v1-28DoF'), metadata)
    for key, value in (('task_id', 'R2V2-Reach-CrateWristPath-v1-28DoF'),
                       ('path_contract', 'wrist_path_v1'), ('path_trajectory_sha256', '0'*64)):
        with pytest.raises(ValueError):
            validate_contact_policy_path(payload_path, parity, {**metadata, key: value})


def test_payload_manifest_requires_explicit_supported_contract(payload_path, tmp_path):
    manifest = deepcopy(payload_path.manifest)
    manifest['trajectory_file'] = str(payload_path.manifest_path.parent/manifest['trajectory_file'])
    manifest['payload_training']['contract'] = 'unknown'
    filename = tmp_path/'manifest.json'
    filename.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contract'):
        ArchivedContactPath(filename)
    del manifest['payload_training']
    filename.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='phase sequence'):
        ArchivedContactPath(filename)


def test_payload_timed_trial_closes_once_and_continues_high_goals_without_fake_grasp(
        mock_experiment, payload_path):
    e = mock_experiment
    e.path = payload_path
    e._bilateral_contact = Mock(return_value=False)
    e.current_metrics.update(table_contact=True, clearance_m=0.)
    for phase, time_s in zip(e.path.phase_names, e.path.phase_start):
        e.data.time = float(time_s+1e-6)
        e._update_source_clock(); e._gate()
        assert e.phase == phase and not e.done
        e._gate()  # repeated ticks must not send another binary command
    assert e.hands.command.call_count == 2
    assert not e.baseline_contact_verified
    e.data.time = e.path.duration_s+2.
    e._gate()
    report = e.report()
    assert report['schedule_completed']
    assert not report['physical_pickup_verified']
    assert not report['payload_training_external_force_replayed']


@pytest.mark.parametrize('phase', ['HOLD_PROBE', 'LIFT_HIGHER', 'HOLD_HIGHER'])
def test_payload_high_phases_measure_real_load_and_reject_final_loss(
        mock_experiment, payload_path, phase):
    e = mock_experiment
    e.path = payload_path
    e.phase = phase
    e.baseline_contact_verified = True
    e.data.time = float(payload_path.phase_start[payload_path.phase_names.index(phase)])
    e._gate()
    e.data.time += 1.
    e._gate()
    assert e.physical_hold_s == pytest.approx(1.)
    e.current_metrics['table_contact'] = True
    e.data.time += .01
    e._gate()
    assert e.physical_hold_s == 0.
