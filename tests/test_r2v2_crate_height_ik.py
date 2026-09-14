"""Offline geometry screen invariants; no inference or policy success claims."""

import numpy as np
import pytest

from common.path_config import PROJECT_ROOT
from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_reach_sim import load_reach_config
from tools.check_r2v2_crate_height_ik import BASE_TABLE_HEIGHT, GeometryScreen


@pytest.fixture(scope='module')
def screens():
    motion = load_crate_motion(PROJECT_ROOT / 'reference_motion_bank/r2v2_crate/down20_yaw15_60mm')
    cfg = load_reach_config(PROJECT_ROOT / 'deploy_mujoco/config/r2v2_reach_wrist_v2.yaml')
    return [GeometryScreen(cfg, motion, BASE_TABLE_HEIGHT-drop) for drop in (0., .2)]


def test_height_translation_preserves_relative_path_and_recorded_lift(screens):
    first, lowered = screens
    for time_s in (1., 3.51, 6.82, 10.83):
        crate_a, wrists_a = first.targets(time_s)
        crate_b, wrists_b = lowered.targets(time_s)
        np.testing.assert_allclose(crate_b[:3, 3]-crate_a[:3, 3], [0., 0., -.2], atol=1e-12)
        np.testing.assert_allclose(wrists_b[:, :3, 3]-wrists_a[:, :3, 3],
            np.tile([0., 0., -.2], (2, 1)), atol=1e-12)
        np.testing.assert_allclose(wrists_a[:, :3, :3], wrists_b[:, :3, :3], atol=1e-12)
    start, _ = first.targets(1.)
    lifted, _ = first.targets(first.motion.duration_s)
    assert lifted[2, 3]-start[2, 3] > .1


def test_ik_only_updates_base_and_body_not_fingers_or_model(screens):
    screen = screens[0]
    screen.targets(3.51)
    original = screen.data.qpos.copy()
    movable = set(range(screen.root_qadr, screen.root_qadr+7)) | set(screen.mapping.qpos)
    protected = [i for i in range(screen.model.nq) if i not in movable]
    q = screen.home.copy()
    q[6:] = np.clip(q[6:]+.01, screen.lower[6:], screen.upper[6:])
    screen.write(q)
    np.testing.assert_array_equal(screen.data.qpos[protected], original[protected])
    np.testing.assert_array_equal(screen.model.jnt_range, screen.original_ranges)
    np.testing.assert_array_equal(screen.model.body_inertia, screen.original_inertia)
    assert screen.model.nmocap == 0
    assert screen.model.neq == 10


def test_bounds_are_original_body_hard_limits(screens):
    screen = screens[0]
    expected = screen.model.jnt_range[screen.mapping.joints]
    np.testing.assert_allclose(screen.lower[6:], expected[:, 0]+1e-6)
    np.testing.assert_allclose(screen.upper[6:], expected[:, 1]-1e-6)
    assert np.all(screen.home >= screen.lower)
    assert np.all(screen.home <= screen.upper)
