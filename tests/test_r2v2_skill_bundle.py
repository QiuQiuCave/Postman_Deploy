"""Published weights are usable files, not a claim of physical acceptance."""
import hashlib
import json
from pathlib import Path

import pytest

from tools.verify_r2v2_skill_bundle import DEFAULT, checked_file, verify


def test_published_weights_hashes_interfaces_and_finite_inference():
    report = verify(DEFAULT)
    assert report['passed']
    assert report['models']['loco_9999']['output_shape'] == [1, 14]
    assert report['models']['dual_wrist_vertical_grip_stop4']['status'] == \
        'experimental_failed_standing_gate'


def test_rejects_changed_file_before_loading_model(tmp_path):
    (tmp_path/'example.bin').write_bytes(b'changed')
    manifest = dict(files={'example.bin': dict(bytes=8,
        sha256=hashlib.sha256(b'original').hexdigest())}, models={})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='integrity'):
        verify(tmp_path)


def test_rejects_path_outside_bundle(tmp_path):
    inside = tmp_path/'bundle'
    inside.mkdir()
    (tmp_path/'outside').write_text('not part of bundle')
    with pytest.raises(ValueError, match='out-of-bundle'):
        checked_file(inside, '../outside')


def test_shared_cola_config_changes_only_weight_paths():
    root = Path(__file__).resolve().parents[1]
    shared = json.loads((root/'deploy_mujoco/config/r2v2_loco_cola_shared_weights.json').read_text())
    old = json.loads((root/'deploy_mujoco/config/r2v2_loco_cola_straight.json').read_text())
    for key in ('reach_root', 'loco_root'):
        assert (root/shared.pop(key)).is_dir()
        old.pop(key)
    assert shared == old
