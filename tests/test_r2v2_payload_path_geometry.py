"""Pure CPU desired-asset and shared-neighborhood invariants."""
import json

import numpy as np

from tools.build_r2v2_payload_path_asset import LOADED_PHASES, build_payload_arrays
from tools.check_r2v2_grasp_lift_geometry import DEFAULT_PATH
from tools.check_r2v2_payload_path_geometry import perturb_pair


def test_payload_archive_preserves_all_original_prefix_arrays_and_has_no_finger_commands():
    manifest = json.loads((DEFAULT_PATH/'manifest.json').read_text())
    with np.load(DEFAULT_PATH/manifest['trajectory_file'], allow_pickle=False) as source:
        original = {name: source[name].copy() for name in source.files}
    arrays, count, stages = build_payload_arrays(original)
    assert count == 2602
    for name in original:
        np.testing.assert_array_equal(arrays[name][:count], original[name][:count])
    assert len(dict.fromkeys(arrays['phase'].tolist())) == 11
    assert 'COMPLETE' not in arrays['phase']
    assert arrays['time_s'][-1] < 50.
    assert len(stages) == 5
    np.testing.assert_array_equal(arrays['payload_load_mask'], np.isin(arrays['phase'], LOADED_PHASES))
    assert not arrays['screened_hand_command'].any()
    loaded = arrays['payload_load_mask']
    relation = arrays['T_crate_wrist_goal'][loaded]
    np.testing.assert_allclose(relation, np.broadcast_to(relation[0], relation.shape), atol=1e-12, rtol=0.)


def test_payload_neighborhood_is_one_rigid_transform_and_ramps_from_identity():
    pair = np.tile(np.eye(4), (2, 1, 1))
    pair[:, :3, 3] = [[.3, .25, 1.15], [.3, -.25, 1.15]]
    crate = np.eye(4)
    crate[:3, 3] = [.33, 0., .99]
    xyz, rpy = np.array([.002, -.002, .002]), np.deg2rad([.5, -.5, .5])
    anchor = np.array([.33, 0., .9619360130597783])
    p, c = perturb_pair(pair, crate, xyz, rpy, 0., anchor)
    np.testing.assert_array_equal(p, pair)
    np.testing.assert_array_equal(c, crate)
    for weight in (.1, .5, 1.):
        p, c = perturb_pair(pair, crate, xyz, rpy, weight, anchor)
        np.testing.assert_allclose(np.linalg.inv(c)@p, np.linalg.inv(crate)@pair, atol=1e-12, rtol=0.)
        np.testing.assert_allclose(np.linalg.inv(p[0])@p[1], np.linalg.inv(pair[0])@pair[1], atol=1e-12, rtol=0.)
