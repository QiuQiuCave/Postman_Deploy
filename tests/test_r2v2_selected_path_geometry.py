"""Invariants for exact-world-target offline geometry screening."""

import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_reach_sim import load_reach_config
from tools.check_r2v2_selected_path_geometry import SelectedPathScreen, path_samples
from tools.export_r2v2_revised_lift_path import revise_lift


def test_selected_screen_preserves_robot_and_shifts_only_table():
    cfg = load_reach_config(PROJECT_ROOT/'deploy_mujoco/config/r2v2_reach_wrist_v2.yaml')
    motion = load_crate_motion(PROJECT_ROOT/'reference_motion_bank/r2v2_crate/down20_yaw15_60mm')
    manifest = dict(delta_x_m=-.05, path_metadata=dict(table_top_m=.9609189696536513))
    screen = SelectedPathScreen(cfg, motion, manifest, .05)
    assert np.isclose(screen.model.body('tabletop').pos[0], .43)
    np.testing.assert_array_equal(screen.model.jnt_range, screen.original_ranges)
    np.testing.assert_array_equal(screen.model.body_inertia, screen.original_inertia)
    limits = screen.model.jnt_range[screen.mapping.joints]
    np.testing.assert_allclose(screen.lower[6:], limits[:, 0]+.05*np.diff(limits).ravel())
    assert screen.model.nmocap == 0


def test_sample_selection_covers_motion_not_only_settled_duplicates():
    phases = ['OUTSIDE', 'TURN_WRISTS', 'PREALIGN', 'READY', 'INSERT',
              'INSERT_SETTLE', 'CLOSE', 'PROBE_LIFT', 'HOLD']
    durations = [3., 4., 3., 1., 2.51, .51, 2.8, 4.01, 2.01]
    times, labels = [], []
    start = 0.
    for phase, duration in zip(phases, durations):
        t = np.arange(start, start+duration+2., .01)
        times.extend(t)
        labels.extend([phase]*len(t))
        start = t[-1]+.01
    arrays = dict(time_s=np.asarray(times), phase=np.asarray(labels))
    samples = path_samples(arrays)
    assert len(samples) == 14
    assert {x[0] for x in samples} == set(phases)
    for phase, fraction, index in samples:
        phase_start = arrays['time_s'][np.flatnonzero(arrays['phase'] == phase)[0]]
        assert abs(arrays['time_s'][index]-phase_start-durations[phases.index(phase)]*fraction) < .011


def test_revised_lift_preserves_prefix_and_shared_crate_relationship():
    phases = np.array(['INSERT_SETTLE']*2+['CLOSE']*2+['PROBE_LIFT']*6+['HOLD']*2+['COMPLETE'])
    count = len(phases)
    wrists = np.tile(np.eye(4), (count, 2, 1, 1))
    wrists[:, :, :3, 3] = [[.29, .263, 1.12], [.29, -.263, 1.12]]
    crate = np.tile(np.eye(4), (count, 1, 1))
    crate[:, :3, 3] = [.33, 0., .96]
    arrays = dict(time_s=np.arange(count, dtype=float), phase=phases,
        T_world_wrist_goal=wrists, T_world_crate_desired=crate,
        T_crate_wrist_goal=np.linalg.inv(crate)[:, None]@wrists,
        source_time_s=np.arange(count, dtype=float),
        source_hand_command=np.zeros((count, 2), dtype=np.int8),
        screened_hand_command=np.zeros((count, 2), dtype=np.int8))
    output, inserted, close = revise_lift(arrays, translation_m=[-.03, 0., .02])
    assert (inserted, close) == (1, 2)
    for key in arrays:
        np.testing.assert_array_equal(output[key][:close], arrays[key][:close])
    np.testing.assert_allclose(output['T_world_wrist_goal'][close], wrists[inserted])
    np.testing.assert_allclose(output['T_world_wrist_goal'][-1, :, :3, 3]-wrists[inserted, :, :3, 3],
                               [[-.03, 0., .02]]*2)
    np.testing.assert_allclose(output['T_crate_wrist_goal'], arrays['T_crate_wrist_goal'])
    np.testing.assert_array_equal(output['T_world_wrist_goal'][:, :, :3, :3], wrists[:, :, :3, :3])
    assert np.all(np.isnan(output['source_time_s'][close:]))
