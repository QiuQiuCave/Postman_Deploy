"""Assemble a hash-bound, explicitly scoped sampled-static payload proof.

Unchanged virtual-prop prefix coverage and new positive-clearance suffix
coverage remain distinguishable. This is never full contact/grasp evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_r2v2_grasp_lift_geometry import DEFAULT_PATH
from tools.check_r2v2_payload_path_geometry import NEW_PHASES, sample_indices, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    path = args.path.resolve()
    manifest = json.loads((path/'manifest.json').read_text())
    digest = sha(path/manifest['trajectory_file'])
    assert digest == manifest['trajectory_sha256']
    old_manifest = json.loads((DEFAULT_PATH/'manifest.json').read_text())
    old_certificate_file = DEFAULT_PATH/'training_feasibility.json'
    old_certificate = json.loads(old_certificate_file.read_text())
    assert old_certificate['full_path_verified'] is True
    assert old_certificate['trajectory_sha256'] == sha(DEFAULT_PATH/old_manifest['trajectory_file'])
    with np.load(path/manifest['trajectory_file'], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    count = manifest['revision']['prefix_unchanged_samples']
    prefix_hashes = {}
    with np.load(DEFAULT_PATH/old_manifest['trajectory_file'], allow_pickle=False) as old:
        for key in old.files:
            original = old[key][:count]
            np.testing.assert_array_equal(arrays[key][:count], original)
            current = arrays[key][:count].astype(original.dtype)
            np.testing.assert_array_equal(current, arrays[key][:count])
            old_hash = hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest()
            assert old_hash == hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest()
            prefix_hashes[key] = dict(sha256=old_hash, shape=list(original.shape),
                                     canonical_dtype=str(original.dtype))
    tree = Path(old_certificate['model_asset_tree']['path'])
    tree_files = sorted(p for p in tree.rglob('*') if p.is_file())
    tree_digest = hashlib.sha256(''.join(f'{p.relative_to(tree).as_posix()} {sha(p)}\n'
                                       for p in tree_files).encode()).hexdigest()
    assert tree_digest == old_certificate['model_asset_tree']['sha256']
    # The only inherited scene-builder change added an opt-in layout guard
    # waiver. Exact reverse-text hash proves all former geometry code remains.
    builder = ROOT/'common/r2v2_crate_motion_replay_scene.py'
    text = builder.read_text()
    text = text.replace(', virtual_props=False, allow_near_table=False,', ', virtual_props=False,')
    text = text.replace('    if not isinstance(allow_near_table, bool):\n        raise TypeError("allow_near_table must be an explicit boolean")\n', '')
    text = text.replace('not virtual_props and not allow_near_table and', 'not virtual_props and')
    text = text.replace('        "allow_near_table": allow_near_table,\n', '')
    text = text.replace('        "near_table_note": "Explicit exploratory layout waiver only; all contact physics remain active" if allow_near_table else None,\n', '')
    historical_builder_sha = hashlib.sha256(text.encode()).hexdigest()
    for entry in old_certificate['model_sources']:
        actual = historical_builder_sha if Path(entry['path']) == builder else sha(entry['path'])
        assert actual == entry['sha256'], entry['path']
    center_file = evidence_root/'lift4_open_center/report.json'
    center = json.loads(center_file.read_text())
    assert center['trajectory_sha256'] == digest
    assert center['manifest_sha256'] == sha(path/'manifest.json')
    suffix_summaries, failed_physical_prefix = {}, []
    for cfg in center['configurations']:
        new = [p for p in cfg['points'] if p['phase'] in NEW_PHASES]
        assert len(new) == 37 and all(p['strict_static_candidate'] for p in new)
        suffix_summaries[cfg['feet']] = summarize(new)
        failed_physical_prefix += [dict(feet=cfg['feet'], phase=p['phase'], index=p['index'],
            wrist_errors=p['wrist_errors']) for p in cfg['points']
            if p['phase'] not in NEW_PHASES and not p['strict_static_candidate']]
    corners_files = [evidence_root/f'lift4_corners_{feet}/report.json' for feet in ('nominal', 'prepared')]
    expected_indices = {index for _, _, index in sample_indices(arrays, neighborhood=True)}
    corners_summary = {}
    for feet, file in zip(('nominal', 'prepared'), corners_files):
        report = json.loads(file.read_text())
        assert report['trajectory_sha256'] == digest
        cfg, = report['configurations']
        assert cfg['feet'] == feet
        points = cfg['points']
        assert len(points) == 512 and all(p['strict_static_candidate'] for p in points)
        assert {p['index'] for p in points} == expected_indices
        for index in expected_indices:
            selected = [p for p in points if p['index'] == index]
            assert len(selected) == 64
            signs = {tuple(np.sign(np.r_[p['xyz_m'], p['rpy_rad']]).astype(int)) for p in selected}
            assert len(signs) == 64
            for p in selected:
                np.testing.assert_allclose(np.abs(p['xyz_m']), [.002]*3, atol=1e-12, rtol=0.)
                np.testing.assert_allclose(np.abs(p['rpy_rad']), [np.deg2rad(.5)]*3, atol=1e-12, rtol=0.)
        corners_summary[feet] = summarize(points)
    parity_file, native_file = evidence_root/'asset_parity/report.json', evidence_root/'native_sampling/report.json'
    parity, native = json.loads(parity_file.read_text()), json.loads(native_file.read_text())
    assert parity['passed'] and native['passed'] and native['trajectory_sha256'] == digest
    model_sources = {str(p): sha(p) for p in tree_files}
    model_sources.update({entry['path']: sha(entry['path']) for entry in old_certificate['model_sources']})
    for filename in ('r2v2_with_hand_constants.py', 'r2v2_wrist_constants.py'):
        source = ROOT.parent/'AMO_R2/src/r2v2_loco/robots'/filename
        model_sources[str(source)] = sha(source)
    source_files = [old_certificate_file, DEFAULT_PATH/'manifest.json', DEFAULT_PATH/old_manifest['trajectory_file'],
        *[Path(entry['path']) for entry in old_certificate['evidence_reports']],
        center_file, *corners_files, parity_file, native_file]
    certificate = dict(schema_version=2, contract='wrist_payload_path_v2',
        scope='sampled_static_kinematics_not_dynamic_contact_success',
        trajectory_sha256=digest, manifest_sha256=sha(path/'manifest.json'),
        center_verified=True, prefix_verified=True, loaded_suffix_verified=True, full_path_verified=True,
        full_path_verification_meaning='sampled kinematic coverage composed from exact inherited virtual prefix and new positive-clearance suffix; NOT continuous configuration or full physical-contact clearance',
        full_physical_prop_clearance_verified=False,
        prefix_scope='inherited_virtual_prop_kinematics', suffix_scope='new_positive_nonhand_prop_clearance',
        prefix_phase_names=['STAND', 'HOME', 'SAFE_WAIT', 'START_HOLD', 'OUTSIDE', 'TURN_WRISTS',
                            'PREALIGN', 'READY', 'INSERT', 'INSERT_SETTLE'],
        verified_phase_names=['STAND', 'HOME', 'SAFE_WAIT', 'START_HOLD', *dict.fromkeys(arrays['phase'].tolist())],
        prefix_preserved_samples=count, prefix_array_hashes=prefix_hashes,
        prefix_array_hash_recipe='Raw C-order bytes after lossless cast to original dtype; phase width may grow for new names',
        inherited_model_asset_tree_sha256=tree_digest,
        inherited_scene_builder_proof=dict(current_sha256=sha(builder), original_sha256=historical_builder_sha,
            changes='Only opt-in allow_near_table layout guard/metadata; defaultfalse and all robot geometry unchanged'),
        validated_envelope=dict(position_xyz_m=[.002]*3, rpy_rad=[float(np.deg2rad(.5))]*3,
            global_scene_jitter_enabled=False),
        neighborhood_scope='64 simultaneous XYZ/RPY corners at 8 distinct new-phase time samples per fixed-foot reference; CLOSE ramp explicitly sampled; stationary holds reuse identical endpoint transforms',
        neighborhood_not_continuous_in_time_or_space_guarantee=True,
        suffix_center_summary=suffix_summaries, neighborhood_summary=corners_summary,
        failed_full_physical_prefix_samples=failed_physical_prefix,
        failed_physical_prefix_interpretation='Retained transparently. These stronger solid-prop checks failed; original prefix remains virtual-prop kinematic curriculum only.',
        measured_dynamic_grasp_verified=False, rigid_open_hand_model_equivalence_verified=True,
        load_model_scope='Only a static combined-COM support proxy checked here; training uses wrist gravity-equivalent forces, not actual contact or payload inertia',
        evidence_reports=[dict(path=str(file.resolve()), sha256=sha(file)) for file in source_files],
        model_sources=model_sources)
    destination = path/'training_feasibility.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(certificate, indent=2, allow_nan=False))
    print(json.dumps(dict(certificate=str(destination), sha256=sha(destination),
        center_samples=74, neighborhood_samples=1024, prefix_preserved_samples=count)), flush=True)


if __name__ == '__main__':
    main()
