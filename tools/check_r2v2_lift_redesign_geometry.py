"""Search coherent rigid-object lift translations without changing the prefix.

Offline static IK only. None of these candidates drives simulation or a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.r2v2_crate_height_state import load_common_start
from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_reach_sim import load_reach_config
from tools.check_r2v2_crate_height_ik import DEFAULT_BANK, DEFAULT_CONFIG, _jsonable
from tools.check_r2v2_selected_path_geometry import (
    DEFAULT_PATH, DEFAULT_PARITY, DEFAULT_PREPARED, SelectedPathScreen,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--lift-cm', nargs='+', type=float, default=[2., 3., 5.])
    parser.add_argument('--forward-cm', nargs='+', type=float, default=[-3., 0., 3., 5.])
    parser.add_argument('--feet', nargs='+', choices=['nominal', 'prepared'], default=['nominal', 'prepared'])
    parser.add_argument('--max-nfev', type=int, default=180)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--fractions', nargs='+', type=float, default=[1.])
    parser.add_argument('--support-penalty', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.path/'manifest.json').read_text())
    archive = args.path/manifest['trajectory_file']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['trajectory_sha256']:
        raise ValueError('Source path hash mismatch')
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    index = int(np.flatnonzero(arrays['phase'] == 'INSERT_SETTLE')[-1])
    wrists = arrays['T_world_wrist_goal'][index]
    crate = arrays['T_world_crate_desired'][index]
    config = load_reach_config(DEFAULT_CONFIG)
    motion = load_crate_motion(DEFAULT_BANK)
    prepared = load_common_start(DEFAULT_PREPARED, config, DEFAULT_PARITY)
    evidence = Path('/root/autodl-tmp/Postman_Deploy/selected_path_posttrain_preflight_20260911')
    report = dict(scope='static_IK_coherent_shared_object_translation_no_dynamics',
        source_path=str(args.path), source_trajectory_sha256=digest, insert_end_index=index,
        insert_end_time_s=float(arrays['time_s'][index]), insert_wrists_world=wrists,
        insert_crate_world=crate, joint_margin_fraction=.05,
        root_search='expanded numerical bounds; final tilt<35deg gate',
        support_penalty=args.support_penalty,
        caveats=['Sampled IK only, not dynamic or continuous path proof.',
                 'Real model limits/inertia unchanged; props ignored in success gate.',
                 'Failed local solves do not prove unreachable.'], configurations=[])
    begun = time.time()
    for feet in args.feet:
        screen = SelectedPathScreen(config, motion, manifest, .05,
                                    prepared if feet == 'prepared' else None, True)
        if args.support_penalty:
            original_residual = screen.residual
            def supported_residual(x, target, original=original_residual, model_screen=screen):
                residual = original(x, target)
                com = model_screen.data.subtree_com[model_screen.base, :2]
                violation = np.maximum(model_screen.support_min+.002-com, 0.)
                violation += np.maximum(com-(model_screen.support_max-.002), 0.)
                return np.r_[residual, violation/.001]
            screen.residual = supported_residual
        initial_file = evidence/('expanded_root/report.json' if feet == 'prepared' else 'report.json')
        initial_report = json.loads(initial_file.read_text())
        initial_cfg = next(c for c in initial_report['configurations']
                           if c['feet'] == feet and c['joint_margin_fraction'] == .05)
        initial = next(p for p in initial_cfg['points'] if p['phase'] == 'INSERT_SETTLE')
        seed = np.r_[initial['base_position_m'],
            Rotation.from_euler('xyz', initial['base_rpy_deg'], degrees=True).as_rotvec(),
            initial['body_q_rad']]
        config_result = dict(feet=feet, foot_frames=screen.foot_frames, candidates=[])
        report['configurations'].append(config_result)
        for lift_cm in args.lift_cm:
            for forward_cm in args.forward_cm:
                previous = seed.copy()
                candidate = dict(lift_cm=lift_cm, forward_cm=forward_cm, points=[])
                config_result['candidates'].append(candidate)
                for fraction in args.fractions:
                    delta = np.eye(4)
                    delta[:3, 3] = [forward_cm*.01*fraction, 0., lift_cm*.01*fraction]
                    previous, metrics = screen.solve(0., previous, args.max_nfev, args.seeds,
                        target_override=(delta@crate, delta@wrists))
                    metrics['fraction'] = fraction
                    candidate['points'].append(metrics)
                    print(f'{feet} lift={lift_cm:g}cm forward={forward_cm:g}cm f={fraction:g}: '
                          f"pass={metrics['strict_static_candidate']} "
                          f"wrist_mm={max(e['position_m'] for e in metrics['wrist_errors'].values())*1000:.3f} "
                          f"wrist_deg={max(e['orientation_deg'] for e in metrics['wrist_errors'].values()):.3f} "
                          f"margin_deg={np.rad2deg(metrics['min_joint_margin_rad']):.2f} "
                          f"tilt_deg={metrics['base_tilt_deg']:.2f} "
                          f"self_mm={metrics['max_forbidden_penetration_m']*1000:.3f} "
                          f"com={metrics['com_in_foot_aabb']}", flush=True)
                    report['elapsed_s'] = time.time()-begun
                    (args.output/'report.json').write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False))
    print(f'Report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
