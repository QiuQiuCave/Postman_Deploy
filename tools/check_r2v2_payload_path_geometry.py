"""CPU static preflight of exact payload NPZ and jointly perturbed wrist goals.

The robot is fixed at the training open-hand FK. No IK state is ever used as
simulation control. Both planted-foot references are checked independently.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
from tools.check_r2v2_grasp_lift_geometry import add_prop_avoidance_objective, initial_seed, recover_actual_seed
from tools.check_r2v2_selected_path_geometry import DEFAULT_PARITY, DEFAULT_PREPARED, SelectedPathScreen

NEW_PHASES = ('CLOSE_SEAT', 'PROBE_LIFT', 'HOLD_PROBE', 'LIFT_HIGHER', 'HOLD_HIGHER')


def perturb_pair(wrists, crate, xyz, rpy, weight, anchor):
    """Exact NumPy counterpart of rigid_payload_neighborhood in AMO."""
    rotation = Rotation.from_euler('xyz', np.asarray(rpy)*weight).as_matrix()
    delta = np.eye(4)
    delta[:3, :3] = rotation
    anchor = np.asarray(anchor)
    delta[:3, 3] = anchor+np.asarray(xyz)*weight-rotation@anchor
    return delta@wrists, delta@crate


def sample_indices(arrays, *, prefix=False, neighborhood=False):
    samples = []
    phases = list(dict.fromkeys(arrays['phase'].tolist()))
    for phase in phases:
        if not prefix and phase not in NEW_PHASES:
            continue
        indices = np.flatnonzero(arrays['phase'] == phase)
        if neighborhood:
            if phase in ('HOLD_PROBE', 'HOLD_HIGHER'):
                continue
            fractions = [.25, .5, .75, 1.] if phase == 'CLOSE_SEAT' else [.5, 1.]
        else:
            fractions = np.linspace(0., 1., 11) if phase in NEW_PHASES else [0., .25, .5, .75, 1.]
            if phase in ('READY', 'INSERT_SETTLE', 'HOLD_PROBE', 'HOLD_HIGHER'):
                fractions = [0., 1.]
        for fraction in fractions:
            index = int(indices[round(float(fraction)*(len(indices)-1))])
            samples.append((phase, float(fraction), index))
    return samples


def pose_seed(point):
    return np.r_[point['base_position_m'],
        Rotation.from_euler('xyz', point['base_rpy_deg'], degrees=True).as_rotvec(), point['body_q_rad']]


def summarize(points):
    return dict(samples=len(points), passed=sum(p['strict_static_candidate'] for p in points),
        max_wrist_position_m=max(e['position_m'] for p in points for e in p['wrist_errors'].values()),
        max_wrist_orientation_deg=max(e['orientation_deg'] for p in points for e in p['wrist_errors'].values()),
        min_joint_margin_rad=min(p['min_joint_margin_rad'] for p in points),
        min_prop_clearance_m=min(min(p['nonhand_prop_clearance_m'].values()) for p in points),
        max_self_or_nonfoot_penetration_m=max(max((c['penetration_m'] for k in ('robot_self', 'nonfoot_ground')
            for c in p['contacts'][k]), default=0.) for p in points),
        max_base_tilt_deg=max(p['base_tilt_deg'] for p in points),
        max_adjacent_body_q_step_rad=max((p['adjacent_body_q_step_rad'] for p in points
            if p['adjacent_body_q_step_rad'] is not None), default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-prefix', action='store_true')
    parser.add_argument('--neighborhood', action='store_true')
    parser.add_argument('--center-report', type=Path)
    parser.add_argument('--neighborhood-kind', choices=('axes', 'corners'), default='axes')
    parser.add_argument('--position-m', type=float, default=.002)
    parser.add_argument('--angle-deg', type=float, default=.5)
    parser.add_argument('--max-nfev', type=int, default=180)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--minimum-clearance-m', type=float, default=.002)
    parser.add_argument('--actual-seed-trace', type=Path)
    parser.add_argument('--feet', nargs='+', choices=('nominal', 'prepared'), default=['nominal', 'prepared'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.path/'manifest.json').read_text())
    archive = args.path/manifest['trajectory_file']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == manifest['trajectory_sha256']
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    center = json.loads(args.center_report.read_text()) if args.center_report else None
    if args.neighborhood and (center is None or center['trajectory_sha256'] != digest):
        raise ValueError('Neighborhood solves require a matching center report')
    config, motion = load_reach_config(DEFAULT_CONFIG), load_crate_motion(DEFAULT_BANK)
    prepared = load_common_start(DEFAULT_PREPARED, config, DEFAULT_PARITY)
    actual = None
    if args.actual_seed_trace:
        trace = json.loads(args.actual_seed_trace.read_text())
        actual = next(x for x in reversed(trace) if x['phase'] == 'CLOSE')
        del trace
    close_ids = np.flatnonzero(arrays['phase'] == 'CLOSE_SEAT')
    close_start = arrays['time_s'][close_ids[0]]
    close_end = arrays['time_s'][close_ids[-1]+1]
    anchor = np.asarray(manifest['payload_training']['anchor_world_m'])
    report = dict(schema_version=2, contract='wrist_payload_path_v2',
        scope='sampled_static_kinematics_not_dynamic_contact_success',
        path=str(args.path.resolve()), trajectory_sha256=digest,
        manifest_sha256=hashlib.sha256((args.path/'manifest.json').read_bytes()).hexdigest(),
        hand_state='all fingers fixed at training open FK; deployment articulated geometry conservatively retained',
        fixed_open_rad=motion.manifest['hand_configuration']['hands']['left']['open'],
        loaded_stage_support='robot plus 0.4kg crate COM inside planted foot collision AABB, static proxy only',
        thresholds=dict(wrist_position_m=.005, wrist_orientation_deg=3., foot_position_m=.003,
            foot_orientation_deg=1., minimum_nonhand_box_table_clearance_m=args.minimum_clearance_m,
            self_nonfoot_penetration_m=.002, root_tilt_deg_exclusive=35.,
            body_joint_bounds='central90pct of unchanged real joint ranges'),
        neighborhood=dict(enabled=args.neighborhood, kind=args.neighborhood_kind,
            position_xyz_m=[args.position_m]*3, rpy_rad=[float(np.deg2rad(args.angle_deg))]*3,
            anchor_world_m=anchor, global_scene_jitter_enabled=False,
            ramp='RzRyRx(alpha*rpy), alpha*xyz; alpha=cubic CLOSE_SEAT time fraction then1'),
        no_dynamic_force_or_grasp_success_claim=True, no_continuous_configuration_path_claim=True,
        configurations=[])
    samples = sample_indices(arrays, prefix=args.include_prefix, neighborhood=args.neighborhood)
    perturbations = [(np.zeros(3), np.zeros(3))]
    if args.neighborhood:
        if args.neighborhood_kind == 'corners':
            perturbations = [(np.asarray(sign[:3])*args.position_m, np.asarray(sign[3:])*np.deg2rad(args.angle_deg))
                             for sign in itertools.product((-1., 1.), repeat=6)]
        else:
            perturbations = [(np.eye(6)[axis]*sign) for axis in range(6) for sign in (-1., 1.)]
            perturbations = [(p[:3]*args.position_m, p[3:]*np.deg2rad(args.angle_deg)) for p in perturbations]
    started = time.monotonic()
    for feet in args.feet:
        screen = SelectedPathScreen(config, motion, manifest, .05,
            prepared if feet == 'prepared' else None, True)
        screen.support_penalty = True
        add_prop_avoidance_objective(screen, args.minimum_clearance_m)
        previous = initial_seed(feet)
        if actual is not None:
            previous = recover_actual_seed(screen, actual)
        result = dict(feet=feet, foot_frames=screen.foot_frames, points=[])
        report['configurations'].append(result)
        previous_q = None
        for phase, fraction, index in samples:
            nominal = None
            if center is not None:
                center_points = next(c['points'] for c in center['configurations'] if c['feet'] == feet)
                nominal = min((p for p in center_points if p['phase'] == phase), key=lambda p: abs(p['index']-index))
                if not nominal['strict_static_candidate']:
                    raise ValueError(f'Cannot certify neighborhood from failed center {feet} {phase}')
            for xyz, rpy in perturbations:
                if nominal is not None:
                    previous = pose_seed(nominal)
                t = arrays['time_s'][index]
                u = float(np.clip((t-close_start)/(close_end-close_start), 0., 1.))
                weight = u*u*(3.-2.*u)
                wrists, crate = perturb_pair(arrays['T_world_wrist_goal'][index],
                    arrays['T_world_crate_desired'][index], xyz, rpy, weight, anchor)
                previous, metrics = screen.solve(0., previous, args.max_nfev, args.seeds,
                    target_override=(crate, wrists))
                loaded = bool(arrays['payload_load_mask'][index])
                robot_mass = screen.model.body_subtreemass[screen.base]
                crate_id = screen.model.body('cargo_crate').id
                payload_mass = screen.model.body_mass[crate_id] if loaded else 0.
                combined_com = (robot_mass*screen.data.subtree_com[screen.base]
                    +payload_mass*screen.data.xipos[crate_id])/(robot_mass+payload_mass)
                supported = bool(np.all(combined_com[:2] >= screen.support_min)
                                 and np.all(combined_com[:2] <= screen.support_max))
                metrics['strict_static_candidate'] = bool(metrics['strict_static_candidate'] and supported)
                metrics.update(phase=phase, fraction=fraction, index=index, path_time_s=float(t),
                    xyz_m=xyz, rpy_rad=rpy, neighborhood_weight=weight, payload_load_mask=loaded,
                    combined_body_payload_com_world_m=combined_com, combined_com_support_proxy=supported,
                    adjacent_body_q_step_rad=None if previous_q is None or args.neighborhood else
                        float(np.max(np.abs(previous[6:]-previous_q))))
                previous_q = previous[6:].copy()
                result['points'].append(metrics)
                result['summary'] = summarize(result['points'])
                report['elapsed_s'] = time.monotonic()-started
                (args.output/'report.json').write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False))
                print(f'{feet} {phase}@{fraction:g} xyz={xyz.tolist()} rpydeg={np.rad2deg(rpy).tolist()} '
                    f"pass={metrics['strict_static_candidate']} "
                    f"wrist_mm={max(e['position_m'] for e in metrics['wrist_errors'].values())*1000:.2f} "
                    f"angle={max(e['orientation_deg'] for e in metrics['wrist_errors'].values()):.2f} "
                    f"gap_mm={min(metrics['nonhand_prop_clearance_m'].values())*1000:.2f}", flush=True)
    print(args.output/'report.json', flush=True)


if __name__ == '__main__':
    main()
