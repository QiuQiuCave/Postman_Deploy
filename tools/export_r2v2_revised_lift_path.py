"""Version a desired path with a coherent small rigid-object lift suffix.

No recorded motion is overwritten; prefix samples remain byte-identical.
This exports targets, not successful control or grasp demonstrations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_r2v2_selected_path_geometry import DEFAULT_PATH


def revise_lift(arrays, *, translation_m):
    output = {key: value.copy() for key, value in arrays.items()}
    delta_xyz = np.asarray(translation_m, dtype=float)
    if delta_xyz.shape != (3,) or not np.all(np.isfinite(delta_xyz)) or delta_xyz[2] <= 0.:
        raise ValueError('A finite shared translation with positive lift is required')
    inserted = int(np.flatnonzero(arrays['phase'] == 'INSERT_SETTLE')[-1])
    close_start = int(np.flatnonzero(arrays['phase'] == 'CLOSE')[0])
    lift_ids = np.flatnonzero(arrays['phase'] == 'PROBE_LIFT')
    lift_start = arrays['time_s'][lift_ids[0]]
    lift_duration = 4.01
    wrists = arrays['T_world_wrist_goal'][inserted]
    crate = arrays['T_world_crate_desired'][inserted]
    relative = np.linalg.inv(crate)@wrists
    for index in range(close_start, len(arrays['time_s'])):
        fraction = np.clip((arrays['time_s'][index]-lift_start)/lift_duration, 0., 1.)
        fraction = fraction**3*(10.-15.*fraction+6.*fraction**2)
        delta = np.eye(4)
        delta[:3, 3] = delta_xyz*fraction
        output['T_world_wrist_goal'][index] = delta@wrists
        output['T_world_crate_desired'][index] = delta@crate
        output['T_crate_wrist_goal'][index] = relative
        output['source_time_s'][index] = np.nan  # Newly planned, not source motion samples.
    for key in arrays:
        np.testing.assert_array_equal(output[key][:close_start], arrays[key][:close_start])
    np.testing.assert_allclose(output['T_world_crate_desired'][close_start:, None]
                               @output['T_crate_wrist_goal'][close_start:],
                               output['T_world_wrist_goal'][close_start:], atol=1e-12)
    return output, inserted, close_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forward-cm', type=float, default=-3.)
    parser.add_argument('--lift-cm', type=float, required=True)
    args = parser.parse_args()
    source_manifest = json.loads((args.source/'manifest.json').read_text())
    source_archive = args.source/source_manifest['trajectory_file']
    source_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if source_sha != source_manifest['trajectory_sha256']:
        raise ValueError('Source trajectory hash mismatch')
    with np.load(source_archive, allow_pickle=False) as payload:
        original = {key: payload[key].copy() for key in payload.files}
    translation = [args.forward_cm*.01, 0., args.lift_cm*.01]
    arrays, inserted, close_start = revise_lift(original, translation_m=translation)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output/'planned_path.npz'
    np.savez_compressed(archive, **arrays)
    manifest = copy.deepcopy(source_manifest)
    manifest['schema_version'] = 2
    manifest['scope'] = 'DESIRED revised complete virtual-prop empty-hand path; not a successful control/grasp demonstration'
    manifest['trajectory_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest['revision'] = dict(kind='shared_rigid_lift_suffix_v1', source_path=str(args.source),
        source_trajectory_sha256=source_sha, prefix_unchanged_through_index=inserted,
        prefix_unchanged_until_time_s=float(arrays['time_s'][close_start]),
        close_behavior='hold insertion-end wrist/crate transforms; remove former 3cm seating lift',
        lift_translation_world_m=translation, lift_timing='quintic smoothstep over existing 4.01s PROBE_LIFT move then existing holds',
        grasp_orientations_unchanged=True, original_phase_timing_unchanged=True,
        hand_commands='screened commands remain 0; source_hand_command metadata inherited for independent future FSM only')
    manifest['source_time_nan_meaning'] = 'added approach or revised suffix; no corresponding recorded source-time sample'
    manifest['path_metadata']['revision'] = manifest['revision']
    for field in ('trial_report', 'trial_report_sha256', 'trial_completed', 'trial_stopped_at_s', 'trial_failure'):
        manifest.pop(field, None)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(dict(path=str(args.output), sha256=manifest['trajectory_sha256'],
                         prefix_samples=close_start, samples=len(arrays['time_s']), translation_m=translation)), flush=True)


if __name__ == '__main__':
    main()
