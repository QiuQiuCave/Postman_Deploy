"""Bounded CPU-only screening of nominal grasp-FSM wrist goals.

IK states are diagnostic only: they never drive a simulation or a policy.
No robot joint ranges, inertia, collisions, or equality constraints are edited.
The static hand reference is interpolated only to inspect open/closing/closed
geometry, not to certify a physically supported grasp.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.r2v2_crate_height_state import load_common_start
from common.r2v2_crate_motion_recording import interpolate_transform, load_crate_motion
from common.r2v2_reach_sim import load_reach_config
from r2v2_description.model import SIDES, SOURCE, initialize_hands
from tools.check_r2v2_crate_height_ik import DEFAULT_BANK, DEFAULT_CONFIG, _jsonable
from tools.check_r2v2_selected_path_geometry import (
    DEFAULT_PARITY, DEFAULT_PREPARED, SelectedPathScreen,
)

EVIDENCE = Path('/root/autodl-tmp/Postman_Deploy/selected_path_posttrain_preflight_20260911')
DEFAULT_PATH = EVIDENCE/'revised_lift2_rear3_v1'


def common_transform(center, translation, pitch_deg=0.):
    """A single world rigid transform preserves both wrists' relative SE3."""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('y', pitch_deg, degrees=True).as_matrix()
    center = np.asarray(center, dtype=float)
    transform[:3, 3] = center+np.asarray(translation)-transform[:3, :3]@center
    return transform


def set_static_hand_fraction(screen, hand_config, fraction):
    """Assign only diagnostic joint state, with exact URDF mimic relations."""
    if not 0. <= fraction <= 1.:
        raise ValueError('Hand fraction must be in [0, 1]')
    cfg = copy.deepcopy(hand_config)
    for side in SIDES:
        poses = cfg['hands'][side]
        q = np.asarray(poses['open'])
        poses['open'] = (q+fraction*(np.asarray(poses['closed'])-q)).tolist()
    initialize_hands(screen.model, screen.data, cfg)


def initial_seed(feet):
    source = EVIDENCE/('expanded_root/report.json' if feet == 'prepared' else 'report.json')
    report = json.loads(source.read_text())
    cfg = next(c for c in report['configurations']
               if c['feet'] == feet and c['joint_margin_fraction'] == .05)
    p = next(p for p in cfg['points'] if p['phase'] == 'INSERT_SETTLE')
    return np.r_[p['base_position_m'],
        Rotation.from_euler('xyz', p['base_rpy_deg'], degrees=True).as_rotvec(), p['body_q_rad']]


def static_summary(points):
    return dict(sample_count=len(points), passed=sum(p['strict_static_candidate'] for p in points),
        max_wrist_position_m=max(e['position_m'] for p in points for e in p['wrist_errors'].values()),
        max_wrist_orientation_deg=max(e['orientation_deg'] for p in points for e in p['wrist_errors'].values()),
        min_hard_joint_margin_rad=min(p['min_joint_margin_rad'] for p in points),
        max_self_or_nonfoot_penetration_m=max(p['max_forbidden_penetration_m'] for p in points),
        max_base_tilt_deg=max(p['base_tilt_deg'] for p in points),
        com_inside_foot_proxy=all(p['com_in_foot_aabb'] for p in points),
        max_adjacent_body_q_step_rad=max((p['max_adjacent_body_q_step_rad']
            for p in points if p['max_adjacent_body_q_step_rad'] is not None), default=0.))


def write_candidate(report_path, output_path=None):
    """Bind targets to inspectable evidence without claiming physical safety."""
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    points = [p for c in report['configurations'] for p in c['points']]
    prop_penetration = max((c['penetration_m'] for p in points
        for group in ('robot_table', 'nonhand_crate') for c in p['contacts'][group]), default=0.)
    candidate = dict(schema_version=1, candidate_id=report['candidate_id'], goal_frame='world',
        training_path_trajectory_sha256=report['training_path_trajectory_sha256'],
        targets=report['targets'], bindings=report['source_bindings'],
        nominal_sampled_geometry_passed=all(p['strict_static_candidate'] for p in points),
        geometry_scope=report['scope'], runtime_actual_pose_envelope_verified=False,
        target_interpolation=report.get('target_interpolation', 'common_rigid_rotation_about_initial_crate_center'),
        nonhand_prop_collision_free_samples=bool(prop_penetration <= 1e-6),
        max_nonhand_prop_penetration_m=prop_penetration,
        max_adjacent_body_q_step_rad=max((p['max_adjacent_body_q_step_rad'] for p in points
            if p['max_adjacent_body_q_step_rad'] is not None), default=0.),
        grasp_seating_contact_verified=False,
        higher_goal_is_outside_the_frozen_training_path=True,
        offsets_reference='all absolute targets derive from archived nominal INSERT; not actual wrists',
        safety_scope='NOT a collision-free grasp/carry or RL-success certificate; nonhand contact must not support pickup',
        evidence=dict(file=str(report_path), sha256=hashlib.sha256(report_path.read_bytes()).hexdigest()))
    output_path = Path(output_path) if output_path else report_path.parent/'candidate.json'
    output_path.write_text(json.dumps(candidate, default=_jsonable, indent=2, allow_nan=False))
    return candidate


def recover_actual_seed(screen, trace_row):
    """Recover root SE3 from logged true wrist SE3 and all body joint angles."""
    from common.r2v2_grasp_recording import body_transform
    x = np.r_[np.zeros(6), trace_row['body_q_rad']]
    screen.write(x)
    local_wrist = body_transform(screen.data, screen.wrists[0])
    actual = np.asarray(trace_row['metrics']['hands']['left']['T_world_wrist'])
    root = actual@np.linalg.inv(local_wrist)
    x[:3] = root[:3, 3]
    x[3:6] = Rotation.from_matrix(root[:3, :3]).as_rotvec()
    screen.write(x)
    for side, body in zip(SIDES, screen.wrists):
        np.testing.assert_allclose(body_transform(screen.data, body),
            trace_row['metrics']['hands'][side]['T_world_wrist'], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(x[:3], trace_row['base_position_world_m'], atol=1e-10, rtol=0.)
    return x


def add_prop_avoidance_objective(screen, minimum_clearance_m=0.):
    """Offline-only fixed-dimension penalties; no collision model edits."""
    old_residual, old_metrics = screen.residual, screen.metrics
    groups = {
        'robot_table': [(a, b) for a in screen.robot_geoms for b in screen.table_geoms],
        'nonhand_crate': [(a, b) for a in screen.robot_geoms-screen.hand_geoms for b in screen.crate_geoms],
    }
    groups = {key: np.asarray([(a, b) for a, b in pairs
        if (screen.model.geom_contype[a]&screen.model.geom_conaffinity[b])
        or (screen.model.geom_contype[b]&screen.model.geom_conaffinity[a])], dtype=int).reshape(-1, 2)
        for key, pairs in groups.items()}

    def clearances():
        values = {}
        for name, pairs in groups.items():
            bounds = (np.linalg.norm(screen.data.geom_xpos[pairs[:, 0]]-screen.data.geom_xpos[pairs[:, 1]], axis=1)
                      -screen.model.geom_rbound[pairs[:, 0]]-screen.model.geom_rbound[pairs[:, 1]])
            near = pairs[bounds < .05]
            values[name] = min((float(mujoco.mj_geomDistance(screen.model, screen.data, int(a), int(b), .05, None))
                                for a, b in near), default=.05)
        return values

    def residual(x, target):
        original = old_residual(x, target)
        if minimum_clearance_m > 0.:
            distances = clearances()
            penalties = [max(minimum_clearance_m+.001-distances[k], 0.)/.0002
                         for k in ('robot_table', 'nonhand_crate')]
        else:
            mujoco.mj_collision(screen.model, screen.data)
            contacts = screen.contacts()
            penalties = [max((c['penetration_m'] for c in contacts[k]), default=0.)/.0002
                         for k in ('robot_table', 'nonhand_crate')]
        return np.r_[original, penalties]

    def metrics(x, target):
        result = old_metrics(x, target)
        result['props_ignored_static_candidate'] = result['strict_static_candidate']
        penetration = max((c['penetration_m'] for k in ('robot_table', 'nonhand_crate')
                           for c in result['contacts'][k]), default=0.)
        result['max_nonhand_prop_penetration_m'] = penetration
        result['nonhand_prop_clearance_m'] = clearances()
        result['strict_static_candidate'] = bool(result['strict_static_candidate'] and penetration <= 1e-6
            and min(result['nonhand_prop_clearance_m'].values()) >= minimum_clearance_m-1e-9)
        result['max_forbidden_penetration_m'] = max(result['max_forbidden_penetration_m'], penetration)
        result['gate_scope'] += '_and_no_detected_nonhand_prop_overlap'
        return result

    screen.residual, screen.metrics = residual, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--seat-cm', type=float, default=.5)
    parser.add_argument('--probe-lift-cm', type=float, default=2.)
    parser.add_argument('--probe-forward-cm', type=float, default=-3.)
    parser.add_argument('--higher-lift-cm', type=float, default=3.)
    parser.add_argument('--higher-forward-cm', type=float, default=-5.)
    parser.add_argument('--higher-pitch-deg', type=float, default=0.)
    parser.add_argument('--endpoint-grid', action='store_true')
    parser.add_argument('--grid-lift-cm', nargs='+', type=float, default=[4., 5., 6.])
    parser.add_argument('--grid-forward-cm', nargs='+', type=float, default=[-5., -8.])
    parser.add_argument('--grid-pitch-deg', nargs='+', type=float, default=[-5., 5.])
    parser.add_argument('--samples', type=int, default=11)
    parser.add_argument('--max-nfev', type=int, default=180)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--waypoints-only', action='store_true')
    parser.add_argument('--avoid-nonhand-props', action='store_true')
    parser.add_argument('--minimum-prop-clearance-m', type=float, default=0.)
    parser.add_argument('--actual-seed-trace', type=Path)
    parser.add_argument('--linear-wrist-interpolation', action='store_true')
    args = parser.parse_args()
    if args.samples < 2 or not 0 <= args.seat_cm < 2:
        parser.error('At least two samples and a seat displacement in [0, 2) cm required')
    if (not np.all(np.isfinite([args.probe_lift_cm, args.probe_forward_cm]))
            or args.probe_lift_cm <= args.seat_cm):
        parser.error('Probe displacement must be finite and higher than the seated target')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.path/'manifest.json').read_text())
    archive = args.path/manifest['trajectory_file']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['trajectory_sha256']:
        raise ValueError('Trajectory hash mismatch')
    with np.load(archive, allow_pickle=False) as arrays:
        inserted = int(np.flatnonzero(arrays['phase'] == 'INSERT_SETTLE')[-1])
        wrists = arrays['T_world_wrist_goal'][inserted].copy()
        crate = arrays['T_world_crate_desired'][inserted].copy()
    from common.r2v2_wrist_path_contact import ArchivedContactPath
    archived_path = ArchivedContactPath(args.path/'manifest.json')
    _, deployed_close_wrists, deployed_close_crate = archived_path.sample(archived_path.close_time)
    np.testing.assert_allclose(wrists, deployed_close_wrists, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(crate, deployed_close_crate, atol=1e-12, rtol=0.)
    center = crate[:3, 3]
    translations = dict(insert=np.zeros(3), close_seat=np.array([0., 0., args.seat_cm*.01]),
        probe=np.array([args.probe_forward_cm*.01, 0., args.probe_lift_cm*.01]),
        higher=np.array([args.higher_forward_cm*.01, 0., args.higher_lift_cm*.01]))
    pitches = dict(insert=0., close_seat=0., probe=0., higher=args.higher_pitch_deg)
    goals = {stage: common_transform(center, shift, pitches[stage])@wrists
             for stage, shift in translations.items()}
    cfg, motion = load_reach_config(DEFAULT_CONFIG), load_crate_motion(DEFAULT_BANK)
    prepared = load_common_start(DEFAULT_PREPARED, cfg, DEFAULT_PARITY)
    actual_row = None
    if args.actual_seed_trace:
        trace = json.loads(args.actual_seed_trace.read_text())
        actual_row = next(p for p in reversed(trace) if p['phase'] == 'CLOSE')
        del trace
    candidate_id = (f'seat{args.seat_cm:g}cm_probe{args.probe_lift_cm:g}'
                    f'rear{-args.probe_forward_cm:g}_higher'
                    f'{args.higher_lift_cm:g}rear{-args.higher_forward_cm:g}'
                    f'_pitch{args.higher_pitch_deg:g}')
    model_paths = [p for p in sorted(SOURCE.rglob('*')) if p.is_file()]+[
        ROOT/'r2v2_description/model.py', ROOT/'common/r2v2_reach_sim.py',
        ROOT/'common/r2v2_crate_motion_replay_scene.py']
    model_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in model_paths if p.is_file()}
    report = dict(schema_version=1, candidate_id=candidate_id,
        scope='sampled_static_kinematics_not_dynamic_contact_success',
        training_path_trajectory_sha256=digest,
        targets={k:v for k,v in goals.items() if k != 'insert'},
        T_world_crate_initial=crate, T_world_wrist_insert=wrists,
        criterion=dict(wrist_position_m=.005, wrist_orientation_deg=3., foot_position_m=.003,
            foot_orientation_deg=1., self_or_nonfoot_penetration_m=.002,
            base_tilt_deg_exclusive=35., body_joint_bounds='central90pct of unmodified real limits',
            support='COM inside fixed feet collision AABB; geometric proxy only',
            prop_contact_stops=False),
        offline_nonhand_prop_avoidance=args.avoid_nonhand_props,
        actual_seed_trace=str(args.actual_seed_trace) if args.actual_seed_trace else None,
        target_interpolation=('per_wrist_linear_position_slerp_orientation' if args.linear_wrist_interpolation
                              else 'common_rigid_rotation_about_initial_crate_center'),
        hand_posture='open-to-closed linear FK during seat; closed after seat; exact URDF mimic',
        source_bindings=dict(path=str(args.path),
            path_manifest_sha256=hashlib.sha256((args.path/'manifest.json').read_bytes()).hexdigest(),
            model_sources=model_hashes,
            hand_reference_manifest_sha256=hashlib.sha256((DEFAULT_BANK/'manifest.json').read_bytes()).hexdigest()),
        archived_contact_path_close_anchor_verified=True,
        archived_contact_path_close_time_s=float(archived_path.close_time),
        interpretation=['Targets are absolute nominal world wrist-link SE3, not TCP offset.',
            'No guarantee for actual wrist orientation/position deviations or RL tracking.',
            'IK output must never become deployed actuator targets.',
            'Seat keeps the crate on the table; later crate pose is only a hypothetical rigid hold.',
            'No physical grasp, force balance, contact clearance, or continuous-path certificate.',
            'Failed local solves do not prove unreachability.'], configurations=[])
    started = time.monotonic()
    for feet in ('nominal', 'prepared'):
        screen = SelectedPathScreen(cfg, motion, manifest, .05,
                                    prepared if feet == 'prepared' else None, True)
        screen.support_penalty = True
        if args.avoid_nonhand_props:
            add_prop_avoidance_objective(screen, args.minimum_prop_clearance_m)
        result = dict(feet=feet, foot_frames=screen.foot_frames, points=[])
        report['configurations'].append(result)
        previous = initial_seed(feet)
        if actual_row is not None:
            previous = recover_actual_seed(screen, actual_row)
            result['actual_seed_time_s'] = actual_row['time_s']
            result['actual_seed_x'] = previous.copy()
        endpoint_seed = previous.copy()
        samples = []
        if args.endpoint_grid:
            for z in args.grid_lift_cm:
                for x in args.grid_forward_cm:
                    for pitch in args.grid_pitch_deg:
                        samples.append(('higher_endpoint', 1., np.array([x*.01, 0., z*.01]), pitch, 1.))
        else:
            for begin, end in [('insert', 'close_seat'), ('close_seat', 'probe'), ('probe', 'higher')]:
                for f in ([1.] if args.waypoints_only else np.linspace(0., 1., args.samples)):
                    shift = (1.-f)*translations[begin]+f*translations[end]
                    pitch = (1.-f)*pitches[begin]+f*pitches[end]
                    samples.append((end, float(f), shift, pitch, float(f) if end == 'close_seat' else 1.))
        previous_q = None
        for stage, f, shift, pitch, hand_fraction in samples:
            if args.endpoint_grid:
                previous = endpoint_seed.copy()
            set_static_hand_fraction(screen, motion.manifest['hand_configuration'], hand_fraction)
            delta = common_transform(center, shift, pitch)
            seat_delta = common_transform(center, translations['close_seat'])
            desired_wrists = delta@wrists
            if args.linear_wrist_interpolation and not args.endpoint_grid:
                begin = dict(close_seat='insert', probe='close_seat', higher='probe')[stage]
                desired_wrists = interpolate_transform(goals[begin], goals[stage], f)
            desired_crate = (crate.copy() if stage == 'close_seat'
                else desired_wrists[0]@np.linalg.inv(goals['close_seat'][0])@crate)
            relative_error = 0.
            if stage != 'close_seat':
                relative_error = float(np.max(np.abs(np.linalg.inv(desired_crate)@desired_wrists
                    -np.linalg.inv(crate)@goals['close_seat'])))
                if not args.linear_wrist_interpolation:
                    assert relative_error < 1e-12
            previous, metrics = screen.solve(0., previous, args.max_nfev, args.seeds,
                                             target_override=(desired_crate, desired_wrists))
            metrics.update(stage=stage, fraction=f, translation_from_insert_world_m=shift,
                           common_pitch_deg=pitch, static_hand_closed_fraction=hand_fraction,
                           hypothetical_right_hand_object_relation_max_SE3_error=relative_error)
            metrics['max_adjacent_body_q_step_rad'] = (None if previous_q is None else
                float(np.max(np.abs(previous[6:]-previous_q))))
            previous_q = previous[6:].copy()
            result['points'].append(metrics)
            print(f'{feet} {stage}@{f:g} xyz={shift.tolist()} pitch={pitch:g}: '
                  f"pass={metrics['strict_static_candidate']} "
                  f"wrist_mm={max(e['position_m'] for e in metrics['wrist_errors'].values())*1000:.3f} "
                  f"wrist_deg={max(e['orientation_deg'] for e in metrics['wrist_errors'].values()):.3f} "
                  f"self_mm={metrics['max_forbidden_penetration_m']*1000:.3f}", flush=True)
            result['summary'] = static_summary(result['points'])
            report['elapsed_s'] = time.monotonic()-started
            (args.output/'report.json').write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False))
    if not args.endpoint_grid:
        write_candidate(args.output/'report.json')
    print(f'Report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
