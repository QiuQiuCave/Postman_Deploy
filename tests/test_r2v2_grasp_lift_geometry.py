"""CPU invariants for diagnostic grasp-FSM geometry; no IK drives playback."""
import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_wrist_path_contact import ArchivedContactPath
from r2v2_description.model import JointMap, SIDES, build_model, hand_names, urdf_hand_joints
from tools.check_r2v2_crate_height_ik import DEFAULT_BANK
from tools.check_r2v2_grasp_lift_geometry import (
    DEFAULT_PATH, common_transform, main, set_static_hand_fraction, write_candidate,
)


def test_shared_rotation_preserves_inter_wrist_and_seated_object_relationship():
    crate = np.eye(4)
    crate[:3, 3] = [.33, 0., .96]
    wrists = np.tile(np.eye(4), (2, 1, 1))
    wrists[:, :3, 3] = [[.29, .263, 1.12], [.29, -.263, 1.12]]
    seated = common_transform(crate[:3, 3], [0., 0., .005])
    relation = np.linalg.inv(crate)@(seated@wrists)
    for fraction in np.linspace(0., 1., 21):
        delta = common_transform(crate[:3, 3], [-.05*fraction, 0., .05*fraction], -5*fraction)
        moved_wrists = delta@wrists
        moved_crate = delta@np.linalg.inv(seated)@crate
        np.testing.assert_allclose(np.linalg.inv(moved_wrists[0])@moved_wrists[1],
            np.linalg.inv(wrists[0])@wrists[1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(moved_crate)@moved_wrists, relation, atol=1e-12)


def test_diagnostic_hand_fraction_keeps_model_and_reference_config_unchanged():
    motion = load_crate_motion(DEFAULT_BANK)
    cfg = motion.manifest['hand_configuration']
    original_cfg = copy.deepcopy(cfg)
    model = build_model(cfg)
    screen = SimpleNamespace(model=model, data=mujoco.MjData(model))
    ranges, inertia, equality = model.jnt_range.copy(), model.body_inertia.copy(), model.eq_data.copy()
    for fraction in (0., .5, 1.):
        set_static_hand_fraction(screen, cfg, fraction)
        for side in SIDES:
            mapping = JointMap.create(model, hand_names(side))
            poses = cfg['hands'][side]
            expected = np.asarray(poses['open'])+fraction*(np.asarray(poses['closed'])-poses['open'])
            np.testing.assert_allclose(screen.data.qpos[mapping.qpos], expected)
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find('mimic')
            if mimic is not None:
                source = screen.data.qpos[model.joint(mimic.get('joint')).qposadr[0]]
                expected = source*float(mimic.get('multiplier', '1'))+float(mimic.get('offset', '0'))
                assert screen.data.qpos[model.joint(name).qposadr[0]] == pytest.approx(expected)
    assert cfg == original_cfg
    np.testing.assert_array_equal(model.jnt_range, ranges)
    np.testing.assert_array_equal(model.body_inertia, inertia)
    np.testing.assert_array_equal(model.eq_data, equality)
    with pytest.raises(ValueError):
        set_static_hand_fraction(screen, cfg, 1.01)


@pytest.mark.skipif(not (DEFAULT_PATH/'manifest.json').exists(), reason='local archived selected path required')
def test_revised_insert_anchor_is_exact_deployed_close_target():
    path = ArchivedContactPath(DEFAULT_PATH/'manifest.json')
    with np.load(DEFAULT_PATH/path.manifest['trajectory_file'], allow_pickle=False) as arrays:
        index = np.flatnonzero(arrays['phase'] == 'INSERT_SETTLE')[-1]
        wrists, crate = arrays['T_world_wrist_goal'][index], arrays['T_world_crate_desired'][index]
    phase, actual_wrists, actual_crate = path.sample(path.close_time)
    assert phase == 'CLOSE'
    assert path.sha256 == '5cf6ff6794fa58ffc114db55194809ea410de5033fb8f129e47a7ac474dcc34b'
    np.testing.assert_allclose(actual_wrists, wrists, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(actual_crate, crate, atol=1e-12, rtol=0.)


def test_candidate_never_mislabels_props_ignored_pass_as_collision_free(tmp_path):
    report = dict(candidate_id='diagnostic', training_path_trajectory_sha256='abc',
        targets={}, source_bindings={}, scope='sampled_static_kinematics_not_dynamic_contact_success',
        configurations=[dict(points=[dict(strict_static_candidate=True,
            max_adjacent_body_q_step_rad=.3,
            contacts=dict(robot_table=[], nonhand_crate=[dict(penetration_m=.1)]))])])
    path = tmp_path/'report.json'
    path.write_text(json.dumps(report))
    candidate = write_candidate(path)
    assert candidate['nominal_sampled_geometry_passed'] is True
    assert candidate['nonhand_prop_collision_free_samples'] is False
    assert candidate['max_nonhand_prop_penetration_m'] == .1
    assert candidate['max_adjacent_body_q_step_rad'] == .3
    assert candidate['grasp_seating_contact_verified'] is False
    assert candidate['runtime_actual_pose_envelope_verified'] is False


@pytest.mark.parametrize('option,value', [('--probe-lift-cm', '.4'), ('--probe-forward-cm', 'nan')])
def test_invalid_probe_parameters_fail_before_creating_artifacts(monkeypatch, tmp_path, option, value):
    destination = tmp_path/'not-created'
    monkeypatch.setattr('sys.argv', ['geometry', '--output', str(destination), option, value])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert not destination.exists()
