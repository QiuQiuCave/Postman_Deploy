"""Version desired payload-training paths; never overwrite source trajectories.

The retained prefix is byte-identical by array value through INSERT_SETTLE.
The new suffix is only a whole-body wrist-goal curriculum, NOT a recorded or
verified finger grasp. Payload masks are curriculum labels, not grasp labels.
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

from tools.check_r2v2_grasp_lift_geometry import DEFAULT_PATH, common_transform


LOADED_PHASES = ('PROBE_LIFT', 'HOLD_PROBE', 'LIFT_HIGHER', 'HOLD_HIGHER')


def build_payload_arrays(original, *, close_pitch_deg=-3., probe_pitch_deg=-7.,
                         higher_pitch_deg=-7., probe_lift_m=.03, higher_lift_m=.04):
    index = int(np.flatnonzero(original['phase'] == 'INSERT_SETTLE')[-1])
    count = index+1
    wrists, crate = original['T_world_wrist_goal'][index], original['T_world_crate_desired'][index]
    pivot = crate[:3, 3]
    stages = [
        ('CLOSE_SEAT', 4.8, np.array([0., 0., .005]), close_pitch_deg),
        ('PROBE_LIFT', 6., np.array([0., 0., probe_lift_m]), probe_pitch_deg),
        ('HOLD_PROBE', 2., np.array([0., 0., probe_lift_m]), probe_pitch_deg),
        ('LIFT_HIGHER', 4., np.array([0., 0., higher_lift_m]), higher_pitch_deg),
        ('HOLD_HIGHER', 4., np.array([0., 0., higher_lift_m]), higher_pitch_deg),
    ]
    seat_delta = common_transform(pivot, stages[0][2], close_pitch_deg)
    parts = {key: [value[:count].copy()] for key, value in original.items()}
    parts['payload_load_mask'] = [np.zeros(count, dtype=bool)]
    previous_shift, previous_pitch = np.zeros(3), 0.
    time_s = float(original['time_s'][index])
    stage_metadata = []
    for phase, duration, shift, pitch in stages:
        n = int(round(duration/.01))
        times = time_s+np.arange(1, n+1)*.01
        fraction = np.linspace(0., 1., n)
        smooth = fraction**3*(10.-15.*fraction+6.*fraction**2)
        goal_wrists, goal_crates = [], []
        for weight in smooth:
            delta = common_transform(pivot, previous_shift+(shift-previous_shift)*weight,
                                     previous_pitch+(pitch-previous_pitch)*weight)
            goal_wrists.append(delta@wrists)
            goal_crates.append(crate.copy() if phase == 'CLOSE_SEAT'
                               else delta@np.linalg.inv(seat_delta)@crate)
        goal_wrists, goal_crates = np.asarray(goal_wrists), np.asarray(goal_crates)
        values = dict(time_s=times, phase=np.full(n, phase, dtype='U32'),
            T_world_wrist_goal=goal_wrists, T_world_crate_desired=goal_crates,
            T_crate_wrist_goal=np.linalg.inv(goal_crates)[:, None]@goal_wrists,
            source_time_s=np.full(n, np.nan), source_hand_command=np.zeros((n, 2), dtype=np.int8),
            screened_hand_command=np.zeros((n, 2), dtype=np.int8),
            payload_load_mask=np.full(n, phase in LOADED_PHASES, dtype=bool))
        for key in parts:
            parts[key].append(values[key])
        stage_metadata.append(dict(name=phase, start_s=float(times[0]),
            next_start_s=float(times[-1]+.01), samples=n, motion_duration_s=duration,
            common_translation_from_insert_world_m=shift.tolist(), common_world_y_pitch_deg=pitch))
        time_s = float(times[-1])
        previous_shift, previous_pitch = shift, pitch
    result = {key: np.concatenate(value) for key, value in parts.items()}
    for key in original:
        np.testing.assert_array_equal(result[key][:count], original[key][:count])
    np.testing.assert_allclose(result['T_world_crate_desired'][:, None]@result['T_crate_wrist_goal'],
                               result['T_world_wrist_goal'], atol=1e-12, rtol=0.)
    assert np.all(np.diff(result['time_s']) > 0.)
    assert not result['screened_hand_command'].any()
    loaded = result['payload_load_mask']
    np.testing.assert_allclose(result['T_crate_wrist_goal'][loaded],
        np.broadcast_to(result['T_crate_wrist_goal'][np.flatnonzero(loaded)[0]], result['T_crate_wrist_goal'][loaded].shape),
        atol=1e-12, rtol=0.)
    return result, count, stage_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--close-pitch-deg', type=float, default=-3.)
    parser.add_argument('--probe-pitch-deg', type=float, default=-7.)
    parser.add_argument('--higher-pitch-deg', type=float, default=-7.)
    parser.add_argument('--probe-lift-m', type=float, default=.03)
    parser.add_argument('--higher-lift-m', type=float, default=.04)
    args = parser.parse_args()
    source = json.loads((args.source/'manifest.json').read_text())
    source_archive = args.source/source['trajectory_file']
    digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if digest != source['trajectory_sha256']:
        raise ValueError('Source trajectory hash mismatch')
    with np.load(source_archive, allow_pickle=False) as payload:
        original = {k: payload[k].copy() for k in payload.files}
    arrays, retained, stages = build_payload_arrays(original,
        close_pitch_deg=args.close_pitch_deg, probe_pitch_deg=args.probe_pitch_deg,
        higher_pitch_deg=args.higher_pitch_deg, probe_lift_m=args.probe_lift_m,
        higher_lift_m=args.higher_lift_m)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output/'planned_path.npz'
    np.savez_compressed(archive, **arrays)
    manifest = copy.deepcopy(source)
    for key in ('checkpoint_sha256', 'prepared_state_sha256'):
        manifest.pop(key, None)
    manifest.update(schema_version=3,
        scope='DESIRED wrist payload-training curriculum, NOT verified dynamic grasp or collision-free motion',
        samples=len(arrays['time_s']), duration_s=float(arrays['time_s'][-1]),
        trajectory_file=archive.name, trajectory_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    manifest['revision'] = dict(kind='wrist_payload_path_v2', source_path=str(args.source.resolve()),
        source_trajectory_sha256=digest, prefix_unchanged_samples=retained,
        prefix_unchanged_through='INSERT_SETTLE', prefix_end_s=float(arrays['time_s'][retained-1]),
        changes='Common wrist-pair pitch during seating, then coherent object lift with no center-X retraction.',
        prior_finger_grasp_demonstration_preserved=False,
        source_hand_command_added_suffix='zero placeholder, not a recorded finger-control sample',
        interpolant='100Hz whole-pair rigid transforms, quintic stage motion; consumer linear position + quaternion SLERP')
    manifest['payload_training'] = dict(contract='wrist_payload_path_v2', loaded_phases=list(LOADED_PHASES),
        neighborhood=dict(position_m=[.002]*3, rpy_rad=[float(np.deg2rad(.5))]*3),
        ramp_start_phase='CLOSE_SEAT', full_phase='PROBE_LIFT',
        ramp='cubic smoothstep of phase elapsed divided by full CLOSE_SEAT duration',
        rotation_convention='one active extrinsic XYZ rotation Rz(yaw)Ry(pitch)Rx(roll) for both wrists/object',
        anchor_world_m=np.asarray(manifest['path_metadata']['world_anchor'])[:3, 3].tolist(),
        global_scene_jitter_enabled=False, load_mask_key='payload_load_mask',
        mask_meaning='curriculum force enable only; NOT measured or assumed real grasp success',
        planned_neighborhood_requires_external_geometry_certificate=True)
    manifest['path_metadata']['segments'] = [s for s in source['path_metadata']['segments']
        if s['name'] in set(arrays['phase'][:retained])]+stages
    manifest['path_metadata']['revision'] = manifest['revision']
    manifest['source_time_nan_meaning'] = 'New planned payload suffix; no corresponding successful source hand experiment'
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(dict(path=str(args.output), sha256=manifest['trajectory_sha256'],
        samples=manifest['samples'], retained_prefix_samples=retained,
        phases=list(dict.fromkeys(arrays['phase'].tolist())))), flush=True)


if __name__ == '__main__':
    main()
