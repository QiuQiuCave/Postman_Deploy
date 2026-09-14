"""Static preflight of an exact archived path; never a robot controller.

Props are ignored by the reach gate. Their post-FK overlaps are reported only.
A failed bounded local solve is not proof that a pose is unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_grasp_recording import body_transform
from common.r2v2_reach_sim import load_reach_config
from common.r2v2_tabletop_demo import copy_robot_initial_state
from tools.check_r2v2_crate_height_ik import DEFAULT_BANK, DEFAULT_CONFIG, GeometryScreen, _jsonable


DEFAULT_PATH = Path('/root/autodl-tmp/Postman_Deploy/crate_height_sweep_closer5_virtual_20260911/planned_paths/down_05cm')
DEFAULT_PREPARED = Path('/root/autodl-tmp/Postman_Deploy/crate_height_sweep_20260911/preparation/prepared_state.json')
DEFAULT_PARITY = Path('/root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/parity/report.json')


def path_samples(arrays, dense=False):
    """Select moving phase fractions, excluding appended settle duplicates."""
    duration = dict(OUTSIDE=3., TURN_WRISTS=4., PREALIGN=3., READY=1.,
                   INSERT=2.51, INSERT_SETTLE=.51, CLOSE=2.8, PROBE_LIFT=4.01, HOLD=2.01)
    selected = []
    for phase, moving_s in duration.items():
        ids = np.flatnonzero(arrays['phase'] == phase)
        if not len(ids):
            raise ValueError(f'Missing phase {phase}')
        start = arrays['time_s'][ids[0]]
        fractions = [0., .25, .5, .75, 1.] if dense else [.5, 1.]
        if phase in ('READY', 'INSERT_SETTLE', 'CLOSE', 'HOLD') and not dense:
            fractions = [1.]
        for fraction in fractions:
            index = int(ids[np.argmin(np.abs(arrays['time_s'][ids]-(start+moving_s*fraction)))])
            selected.append((phase, fraction, index))
    return selected


class SelectedPathScreen(GeometryScreen):
    def __init__(self, config, motion, manifest, margin, prepared=None, expanded_root=False):
        super().__init__(config, motion, manifest['path_metadata']['table_top_m'], margin)
        # Only offline prop geometry is shifted. Robot assets remain unmodified.
        self.model.body('tabletop').pos[0] += manifest['delta_x_m']
        if prepared is not None:
            _, _, source_model, source_data = prepared
            copy_robot_initial_state(source_model, source_data, self.model, self.data)
            self.initial_qpos = self.data.qpos.copy()
            self.foot_frames = [body_transform(self.data, i) for i in self.feet]
            self.com_target = self.data.subtree_com[self.base, :2].copy()
            foot_points = self.data.geom_xpos[list(self.foot_ids), :2]
            radii = self.model.geom_size[list(self.foot_ids), 0, None]
            self.support_min = (foot_points-radii).min(axis=0)
            self.support_max = (foot_points+radii).max(axis=0)
            self.home = np.clip(self.pack(), self.lower+1e-7, self.upper-1e-7)
            self.write(self.home)
        if expanded_root:
            # Search bounds are numerical, not robot joint limits. This control
            # checks sensitivity without altering any articulated-body range.
            self.lower[:6] = [-.25, -.25, .55, -.65, -.65, -.8]
            self.upper[:6] = [.25, .25, 1.05, .65, .65, .8]
            self.home = np.clip(self.home, self.lower+1e-7, self.upper-1e-7)
        self.support_penalty = False

    def residual(self, x, targets):
        result = super().residual(x, targets)
        if not self.support_penalty:
            return result
        com = self.data.subtree_com[self.base, :2]
        violation = np.maximum(self.support_min+.002-com, 0.)
        violation += np.maximum(com-(self.support_max-.002), 0.)
        return np.r_[result, violation/.001]

    def metrics(self, x, targets):
        result = super().metrics(x, targets)
        result['prop_including_original_strict_static_candidate'] = result['strict_static_candidate']
        result['prop_including_max_forbidden_penetration_m'] = result['max_forbidden_penetration_m']
        contacts = result['contacts']
        bad = max((c['penetration_m'] for key in ('robot_self', 'nonfoot_ground')
                   for c in contacts[key]), default=0.)
        result['max_forbidden_penetration_m'] = bad
        result['strict_static_candidate'] = bool(result['kinematic_pose_gate_passed']
            and bad <= .002 and result['com_in_foot_aabb'] and result['base_tilt_deg'] < 35.)
        result['gate_scope'] = 'pose_feet_hardlimits_self_ground_support_proxy_tilt35_only_props_ignored'
        result['root_search_lower'] = self.lower[:6].copy()
        result['root_search_upper'] = self.upper[:6].copy()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=DEFAULT_PATH)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--parity', type=Path, default=DEFAULT_PARITY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feet', nargs='+', choices=['nominal', 'prepared'], default=['nominal', 'prepared'])
    parser.add_argument('--margins', nargs='+', type=float, default=[0., .05])
    parser.add_argument('--max-nfev', type=int, default=120)
    parser.add_argument('--seeds', type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--expanded-root', action='store_true')
    parser.add_argument('--prefix-randomization', action='store_true',
                        help='Diagnostic +/-1cm XYZ and +/-2deg yaw around center at prefix endpoints')
    parser.add_argument('--lift-scan', action='store_true',
                        help='Sample the exact recorded CLOSE/lift prefix at each additional cm')
    parser.add_argument('--support-penalty', action='store_true')
    parser.add_argument('--randomization-distance-m', type=float, default=.01)
    parser.add_argument('--randomization-yaw-deg', type=float, default=2.)
    parser.add_argument('--randomization-corners', action='store_true')
    parser.add_argument('--nominal-seed-report', type=Path,
                        help='Independent jitter solves start from matched certified center IK state')
    parser.add_argument('--preparation-only', action='store_true',
                        help='11 exact HOME-to-SAFE_WAIT samples including endpoints; no archive stages')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.path/'manifest.json').read_text())
    archive = args.path/manifest['trajectory_file']
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest['trajectory_sha256']:
        raise ValueError('Path archive hash mismatch')
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    if args.preparation_only:
        fraction = np.linspace(0., 1., 11)
        smooth = fraction**2*(3.-2.*fraction)
        home = np.array([[.246, .19698, 1.087902013081883],
                         [.246, -.19643, 1.087902013081883]])
        safe = np.array([[.18, .27, 1.13], [.18, -.27, 1.13]])
        wrists = np.tile(np.eye(4), (11, 2, 1, 1))
        wrists[:, :, :3, 3] = home[None]+smooth[:, None, None]*(safe-home)[None]
        arrays = dict(time_s=5.4+8.*fraction, phase=np.full(11, 'SAFE_WAIT'),
            T_world_wrist_goal=wrists, T_world_crate_desired=np.tile(arrays['T_world_crate_desired'][0], (11, 1, 1)),
            source_time_s=np.full(11, np.nan))
    config = load_reach_config(args.config)
    nominal_seeds = None
    if args.nominal_seed_report:
        nominal_seeds = json.loads(args.nominal_seed_report.read_text())
        if nominal_seeds['path_sha256'] != digest:
            raise ValueError('Nominal IK seeds belong to a different target path')
    motion = load_crate_motion(DEFAULT_BANK)
    prepared = load_common_start(args.prepared, config, args.parity) if 'prepared' in args.feet else None
    report = dict(scope='offline_static_IK_only_no_dynamics_no_contact_certificate',
        path=str(args.path), path_sha256=digest, prepared=str(args.prepared),
        selected_scene=dict(delta_x_m=manifest['delta_x_m'], delta_z_m=manifest['delta_z_m']),
        thresholds=dict(wrist_position_m=.005, wrist_angle_deg=3., foot_position_m=.003,
                        foot_angle_deg=1., self_ground_penetration_m=.002),
        caveats=['Failure to find local bounded solution is not proof of unreachability.',
                 'COM in foot AABB is only a geometric support proxy.',
                 'Contacts are checked after IK, not optimized as avoidance costs.',
                 'No physical table/crate gate; all finger joints remain fixed open.'],
        configurations=[])
    if args.nominal_seed_report:
        report['nominal_seed_report'] = str(args.nominal_seed_report)
        report['nominal_seed_report_sha256'] = hashlib.sha256(args.nominal_seed_report.read_bytes()).hexdigest()
    begun = time.time()
    for feet in args.feet:
        for margin in args.margins:
            screen = SelectedPathScreen(config, motion, manifest, margin,
                                        prepared if feet == 'prepared' else None, args.expanded_root)
            screen.support_penalty = args.support_penalty
            item = dict(feet=feet, joint_margin_fraction=margin, foot_frames=screen.foot_frames,
                        initial_body_qpos=screen.initial_qpos, points=[])
            report['configurations'].append(item)
            previous = screen.home.copy()
            if args.preparation_only:
                samples = [('SAFE_WAIT', float(f), i, np.zeros(3), 0.) for i, f in enumerate(np.linspace(0., 1., 11))]
            else:
                samples = [(p, f, i, np.zeros(3), 0.) for p, f, i in path_samples(arrays, args.dense)]
            if args.prefix_randomization:
                distance, angle = args.randomization_distance_m, args.randomization_yaw_deg
                perturbations = [(np.eye(3)[axis]*sign*distance, 0.) for axis in range(3) for sign in (-1, 1)]
                perturbations += [(np.zeros(3), yaw) for yaw in (-angle, angle)]
                if args.randomization_corners:
                    perturbations = [(np.asarray(signs[:3])*distance, signs[3]*angle)
                                     for signs in itertools.product((-1., 1.), repeat=4)]
                if args.preparation_only:
                    samples = [('SAFE_WAIT', float(f), i, shift*f*f*(3.-2.*f), yaw*f*f*(3.-2.*f))
                               for i, f in enumerate(np.linspace(0., 1., 11)) for shift, yaw in perturbations]
                else:
                    samples = [(p, f, i, shift, yaw) for p, f, i in path_samples(arrays)
                               if f == 1. and p in ('OUTSIDE', 'TURN_WRISTS', 'PREALIGN', 'INSERT', 'PROBE_LIFT')
                               for shift, yaw in perturbations]
            if args.lift_scan:
                ids = np.flatnonzero(np.isin(arrays['phase'], ['CLOSE', 'PROBE_LIFT']))
                baseline_z = arrays['T_world_wrist_goal'][ids[0], 0, 2, 3]
                samples = []
                for lift in np.arange(0., .131, .01):
                    index = int(ids[np.argmin(np.abs(
                        arrays['T_world_wrist_goal'][ids, 0, 2, 3]-(baseline_z+lift)))])
                    samples.append((str(arrays['phase'][index]), float(lift), index, np.zeros(3), 0.))
            for phase, fraction, index, shift, yaw in samples:
                if nominal_seeds is not None:
                    seed_cfg = next(c for c in nominal_seeds['configurations']
                                    if c['feet'] == feet and c['joint_margin_fraction'] == margin)
                    seed = min((p for p in seed_cfg['points'] if p['phase'] == phase),
                               key=lambda p: abs(p['phase_fraction']-fraction))
                    if not seed['strict_static_candidate']:
                        raise ValueError('Refusing failed nominal seed')
                    previous = np.r_[seed['base_position_m'],
                        Rotation.from_euler('xyz', seed['base_rpy_deg'], degrees=True).as_rotvec(),
                        seed['body_q_rad']]
                desired_crate = arrays['T_world_crate_desired'][index].copy()
                desired_wrists = arrays['T_world_wrist_goal'][index].copy()
                perturb = np.eye(4)
                perturb[:3, :3] = Rotation.from_euler('z', yaw, degrees=True).as_matrix()
                center = np.r_[manifest['path_metadata']['crate_center_xy_m'],
                               manifest['path_metadata']['table_top_m']]
                perturb[:3, 3] = center+shift-perturb[:3, :3]@center
                desired_crate = perturb@desired_crate
                desired_wrists = perturb@desired_wrists
                previous, metrics = screen.solve(float(arrays['source_time_s'][index]), previous,
                    args.max_nfev, args.seeds, target_override=(
                        desired_crate, desired_wrists))
                metrics.update(phase=phase, phase_fraction=fraction, index=index,
                               path_time_s=float(arrays['time_s'][index]),
                               perturb_translation_m=shift, perturb_yaw_deg=yaw)
                item['points'].append(metrics)
                # JSON deliberately uses null rather than NaN for added phases.
                if not np.isfinite(metrics['source_time_s']):
                    metrics['source_time_s'] = None
                print(f'{feet} margin={margin:g} {phase}@{fraction:g} shift={shift.tolist()} yaw={yaw}: '
                      f"pass={metrics['strict_static_candidate']} "
                      f"wrist_mm={max(x['position_m'] for x in metrics['wrist_errors'].values())*1000:.2f} "
                      f"wrist_deg={max(x['orientation_deg'] for x in metrics['wrist_errors'].values()):.2f} "
                      f"hardmargin_deg={np.rad2deg(metrics['min_joint_margin_rad']):.2f}", flush=True)
                report['elapsed_s'] = time.time()-begun
                (args.output/'report.json').write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False))
    print(f'Report: {args.output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
