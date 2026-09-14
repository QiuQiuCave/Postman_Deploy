"""Offline geometry screen; NEVER a policy driver or a grasp-success test.

The original complete model, limits and collision shapes are preserved. Only
static candidate states are assigned for FK/IK. Both foot link frames remain
near their initialized world poses. Fingers stay at the recorded open pose;
therefore late-phase hand/crate contacts do not certify a closed-hand grasp.
Failure to find a solution is not proof of geometric unreachability.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.r2v2_crate import CrateParameters
from common.r2v2_crate_lift_metrics import _descendant
from common.r2v2_crate_motion_recording import load_crate_motion, pose_from_transform
from common.r2v2_crate_motion_replay_scene import build_motion_replay_model
from common.r2v2_grasp_recording import body_transform
from common.r2v2_reach_sim import initialize_robot, foot_collision_ids, load_reach_config
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES


DEFAULT_CONFIG = Path('/root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/policy/reach_wrist_v2.yaml')
DEFAULT_BANK = ROOT / 'reference_motion_bank/r2v2_crate/down20_yaw15_60mm'
BASE_TABLE_HEIGHT = 1.0109189696536514


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _rot_error(actual, desired):
    return Rotation.from_matrix(desired.T @ actual).as_rotvec()


class GeometryScreen:
    """Independent bounded 34-variable static IK, with immutable robot model."""

    def __init__(self, config, motion, table_height, joint_margin_fraction=0.):
        self.motion = motion
        self.model, _, self.layout = build_motion_replay_model(
            config, CrateParameters(**motion.manifest['crate_parameters']),
            table_top_m=table_height,
        )
        self.data = mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, motion.manifest['hand_configuration'])
        self.mapping = JointMap.create(self.model, BODY_JOINTS)
        self.original_ranges = self.model.jnt_range.copy()
        self.original_inertia = self.model.body_inertia.copy()
        self.root_qadr = int(self.model.joint('floating_base_joint').qposadr[0])
        self.crate_qadr = int(self.model.joint('crate_free').qposadr[0])
        self.base = self.model.body('base_link').id
        self.wrists = [self.model.body(f'{s}_hand_roll_link').id for s in SIDES]
        self.feet = [self.model.body(f'{s}_ankle_roll_link').id for s in SIDES]
        self.foot_frames = [body_transform(self.data, i) for i in self.feet]
        self.foot_ids = set(foot_collision_ids(self.model))
        self.robot_geoms = {g for g in range(self.model.ngeom)
            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.hand_geoms = {g for g in self.robot_geoms
            if any(_descendant(self.model, int(self.model.geom_bodyid[g]), i) for i in self.wrists)}
        self.table_geoms = {g for g in range(self.model.ngeom)
            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.model.body('tabletop').id)}
        self.crate_geoms = {g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) == self.model.body('cargo_crate').id}
        self.floor = self.model.geom('floor').id
        self.com_target = self.data.subtree_com[self.base, :2].copy()
        # This footprint is a geometric support proxy, not a dynamics certificate.
        foot_points = self.data.geom_xpos[list(self.foot_ids), :2]
        foot_radii = self.model.geom_size[list(self.foot_ids), 0, None]
        self.support_min = (foot_points-foot_radii).min(axis=0)
        self.support_max = (foot_points+foot_radii).max(axis=0)
        self.initial_qpos = self.data.qpos.copy()
        self.home = self.pack()
        if not 0. <= joint_margin_fraction < .5:
            raise ValueError('Joint margin fraction must be in [0, 0.5)')
        joint_ranges = self.model.jnt_range[self.mapping.joints]
        joint_margin = np.maximum(1e-6, joint_margin_fraction*(joint_ranges[:, 1]-joint_ranges[:, 0]))
        self.lower = np.r_[[-.15, -.12, .52], [-.60, -.60, -.40], joint_ranges[:, 0]+joint_margin]
        self.upper = np.r_[[.15, .12, .92], [.60, .60, .40], joint_ranges[:, 1]-joint_margin]
        self.home = np.clip(self.home, self.lower+1e-7, self.upper-1e-7)
        self.write(self.home)
        # Set the initial settled source crate exactly onto the new tabletop.
        boundaries = {e['state']: e['time_s'] for e in motion.manifest['transitions']}
        settled_source = motion.sample(boundaries['INSERT'])['T_anchor_crate']
        settled_world = np.eye(4)
        settled_world[:3, 3] = [.38, 0., table_height]
        self.world_anchor = settled_world @ np.linalg.inv(settled_source)

    def pack(self):
        root = self.data.qpos[self.root_qadr:self.root_qadr+7]
        rv = Rotation.from_quat(root[3:7][[1, 2, 3, 0]]).as_rotvec()
        return np.r_[root[:3], rv, self.data.qpos[self.mapping.qpos]]

    def write(self, x):
        self.data.qpos[self.root_qadr:self.root_qadr+3] = x[:3]
        quat = Rotation.from_rotvec(x[3:6]).as_quat()
        self.data.qpos[self.root_qadr+3:self.root_qadr+7] = quat[[3, 0, 1, 2]]
        self.data.qpos[self.mapping.qpos] = x[6:]
        # FK only during optimization: no artificial simulation/control forces.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def targets(self, source_time):
        sample = self.motion.sample(source_time)
        desired_crate = self.world_anchor @ sample['T_anchor_crate']
        desired_wrists = desired_crate @ sample['T_crate_wrist']
        pos, quat = pose_from_transform(desired_crate)
        self.data.qpos[self.crate_qadr:self.crate_qadr+7] = np.r_[pos, quat]
        return desired_crate, desired_wrists

    def residual(self, x, targets):
        self.write(x)
        residual = []
        for body, desired in zip(self.wrists, targets):
            residual.extend((self.data.xpos[body]-desired[:3, 3])/.005)
            residual.extend(_rot_error(self.data.xmat[body].reshape(3, 3), desired[:3, :3])/.05)
        for body, desired in zip(self.feet, self.foot_frames):
            residual.extend((self.data.xpos[body]-desired[:3, 3])/.001)
            residual.extend(_rot_error(self.data.xmat[body].reshape(3, 3), desired[:3, :3])/.01)
        residual.extend(.15*(self.data.subtree_com[self.base, :2]-self.com_target)/.03)
        # Mild preference only, allowing waist, knees, hips and base to cooperate.
        weights = np.r_[[.5, .5, .3], [.1, .1, .1], np.full(28, .06)]
        residual.extend(weights*(x-self.home))
        return np.asarray(residual)

    def contacts(self):
        groups = {name: [] for name in ('robot_self', 'robot_table', 'nonhand_crate',
            'hand_crate_open_pose', 'nonfoot_ground')}
        for contact in self.data.contact:
            if contact.dist >= -1e-6:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            pair = {a, b}
            name = None
            if pair <= self.robot_geoms:
                name = 'robot_self'
            elif pair & self.robot_geoms and pair & self.table_geoms:
                name = 'robot_table'
            elif pair & self.robot_geoms and pair & self.crate_geoms:
                name = 'hand_crate_open_pose' if pair & self.hand_geoms else 'nonhand_crate'
            elif self.floor in pair and pair & self.robot_geoms and not pair & self.foot_ids:
                name = 'nonfoot_ground'
            if name:
                groups[name].append({'geom1': self.model.geom(a).name,
                    'geom2': self.model.geom(b).name, 'penetration_m': float(-contact.dist)})
        for group in groups.values():
            group.sort(key=lambda item: item['penetration_m'], reverse=True)
        return groups

    def metrics(self, x, targets):
        self.write(x)
        mujoco.mj_forward(self.model, self.data)
        wrist_errors, foot_errors = {}, {}
        for sides, bodies, goals, output in (
            (SIDES, self.wrists, targets, wrist_errors),
            (SIDES, self.feet, self.foot_frames, foot_errors),
        ):
            for side, body, goal in zip(sides, bodies, goals):
                output[side] = {'position_m': float(np.linalg.norm(self.data.xpos[body]-goal[:3, 3])),
                    'orientation_deg': float(np.rad2deg(np.linalg.norm(
                        _rot_error(self.data.xmat[body].reshape(3, 3), goal[:3, :3]))))}
        limits = self.model.jnt_range[self.mapping.joints]
        margins = np.minimum(x[6:]-limits[:, 0], limits[:, 1]-x[6:])
        com = self.data.subtree_com[self.base].copy()
        base_r = self.data.xmat[self.base].reshape(3, 3)
        contacts = self.contacts()
        strict_groups = ('robot_self', 'robot_table', 'nonhand_crate', 'nonfoot_ground')
        max_bad = max([c['penetration_m'] for k in strict_groups for c in contacts[k]], default=0.)
        pose_ok = (max(e['position_m'] for e in wrist_errors.values()) <= .005
            and max(e['orientation_deg'] for e in wrist_errors.values()) <= 3.
            and max(e['position_m'] for e in foot_errors.values()) <= .003
            and max(e['orientation_deg'] for e in foot_errors.values()) <= 1.)
        # A warning for soft-limit proximity is not a hard-limit violation.
        soft_margin = margins-.05*(limits[:, 1]-limits[:, 0])
        result = {'wrist_errors': wrist_errors, 'foot_errors': foot_errors,
            'body_joint_names': BODY_JOINTS, 'body_q_rad': x[6:].copy(),
            'joint_limit_margin_rad': margins, 'min_joint_margin_rad': float(margins.min()),
            'closest_limit_joint': BODY_JOINTS[int(margins.argmin())],
            'min_90pct_soft_range_margin_rad': float(soft_margin.min()),
            'joint_use_deg': {n: float(np.rad2deg(x[6+i])) for i, n in enumerate(BODY_JOINTS)
                if any(part in n for part in ('ankle', 'waist', 'knee', 'hip'))},
            'base_position_m': self.data.xpos[self.base].copy(),
            'base_rpy_deg': Rotation.from_matrix(base_r).as_euler('xyz', degrees=True),
            'base_tilt_deg': float(np.rad2deg(np.arccos(np.clip(base_r[2, 2], -1., 1.)))),
            'robot_com_world_m': com, 'com_in_foot_aabb': bool(np.all(com[:2] >= self.support_min)
                and np.all(com[:2] <= self.support_max)),
            'contacts': contacts, 'max_forbidden_penetration_m': max_bad,
            'kinematic_pose_gate_passed': pose_ok,
            'strict_static_candidate': bool(pose_ok and max_bad <= .002
                and np.all(com[:2] >= self.support_min) and np.all(com[:2] <= self.support_max)),
            'state_kind': 'offline_static_IK_only_not_policy_not_grasp'}
        assert np.array_equal(self.model.jnt_range, self.original_ranges)
        assert np.array_equal(self.model.body_inertia, self.original_inertia)
        return result

    def solve(self, source_time, previous, max_nfev, seeds, target_override=None):
        if target_override is None:
            desired_crate, targets = self.targets(source_time)
        else:
            desired_crate, targets = target_override
            pos, quat = pose_from_transform(desired_crate)
            self.data.qpos[self.crate_qadr:self.crate_qadr+7] = np.r_[pos, quat]
        candidates = []
        starts = [previous.copy(), self.home.copy()]
        for knee in (.6, 1.):
            start = self.home.copy()
            for side in SIDES:
                start[6+BODY_JOINTS.index(f'{side}_knee_joint')] = knee
                start[6+BODY_JOINTS.index(f'{side}_hip_pitch_joint')] = -knee/2
                start[6+BODY_JOINTS.index(f'{side}_ankle_pitch_joint')] = -knee/2
            start[2] -= .06 if knee == .6 else .14
            start[6+BODY_JOINTS.index('waist_pitch_joint')] = .25
            starts.append(start)
        for seed_index, start in enumerate(starts[:seeds]):
            start = np.clip(start, self.lower+1e-7, self.upper-1e-7)
            solved = least_squares(self.residual, start, args=(targets,), bounds=(self.lower, self.upper),
                x_scale='jac', max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
            metrics = self.metrics(solved.x, targets)
            pose_cost = max(e['position_m']/.005+e['orientation_deg']/3
                for e in metrics['wrist_errors'].values())
            score = pose_cost + 1000*metrics['max_forbidden_penetration_m']
            candidates.append((score, solved.x.copy(), metrics, {'seed_index': seed_index,
                'success': bool(solved.success), 'message': solved.message,
                'nfev': int(solved.nfev), 'cost': float(solved.cost), 'ranking_score': score}))
        best = min(candidates, key=lambda c: c[0])
        self.write(best[1])
        mujoco.mj_forward(self.model, self.data)
        return best[1], dict(best[2], solver=best[3],
            all_solver_attempts=[c[3] for c in candidates], source_time_s=source_time,
            desired_crate_world=desired_crate, desired_wrist_world=targets,
            offline_qpos=self.data.qpos.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--motion', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--drops-cm', type=float, nargs='+', default=[0, 5, 10, 15, 20])
    parser.add_argument('--max-nfev', type=int, default=120)
    parser.add_argument('--seeds', type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument('--joint-margin-fraction', type=float, default=0.,
        help='Restrict optimizer only; 0.05 keeps central 90%% of each original hard range. Model unchanged.')
    parser.add_argument('--approach-only', action='store_true',
        help='Check added outside/turn approach keyframes instead of four source keyframes.')
    parser.add_argument('--samples-per-segment', type=int, default=0,
        help='Positive value >=2 samples every HeightPath phase instead of only keyframes. Not continuous certification.')
    args = parser.parse_args()
    if args.samples_per_segment and (args.samples_per_segment < 2 or args.approach_only):
        parser.error('--samples-per-segment must be >=2 and cannot accompany --approach-only')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    config, motion = load_reach_config(args.config), load_crate_motion(args.motion)
    phases = {e['state']: e['time_s'] for e in motion.manifest['transitions']}
    keyframes = [('READY_END', phases['INSERT']), ('INSERT_SETTLE', phases['INSERT_SETTLE']),
        ('PROBE_LIFT_ENTRY', phases['PROBE_LIFT']), ('HOLD_ENTRY', phases['HOLD'])]
    rows, started = [], time.monotonic()
    for drop in args.drops_cm:
        screen = GeometryScreen(config, motion, BASE_TABLE_HEIGHT-drop/100, args.joint_margin_fraction)
        previous = screen.home.copy()
        results = []
        evaluated = [(name, source_time, None) for name, source_time in keyframes]
        if args.approach_only:
            from common.r2v2_crate_height_path import HeightPath
            path = HeightPath(-drop/100, motion_path=args.motion)
            evaluated = []
            for name, segment, fraction in (
                ('APPROACH_START', 'OUTSIDE', 0.), ('OUTSIDE_END', 'OUTSIDE', 1.),
                ('TURN_MIDPOINT', 'TURN_WRISTS', .5), ('TURN_END', 'TURN_WRISTS', 1.),
                ('PREALIGN_MIDPOINT', 'PREALIGN', .5),
            ):
                sample = path.sample(segment, fraction*path._by_name[segment]['duration_s'])
                evaluated.append((name, None, (sample['T_world_crate'], sample['T_world_wrist'])))
        if args.samples_per_segment:
            from common.r2v2_crate_height_path import HeightPath
            path = HeightPath(-drop/100, motion_path=args.motion)
            evaluated = []
            for segment in path.segments:
                for sample_index, elapsed in enumerate(np.linspace(0., segment['duration_s'], args.samples_per_segment)):
                    sample = path.sample(segment['name'], elapsed)
                    name = f"{segment['name']}_{sample_index:03d}"
                    evaluated.append((name, sample['source_time_s'],
                        (sample['T_world_crate'], sample['T_world_wrist'])))
        for name, source_time, override in evaluated:
            previous, result = screen.solve(source_time, previous, args.max_nfev, args.seeds, override)
            result['keyframe'] = name
            results.append(result)
            print(json.dumps({'drop_cm': drop, 'keyframe': name,
                'wrist_errors': result['wrist_errors'],
                'min_joint_margin_rad': result['min_joint_margin_rad'],
                'forbidden_penetration_m': result['max_forbidden_penetration_m'],
                'static_candidate': result['strict_static_candidate'],
                'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
        row = {'drop_cm': drop, 'table_height_m': screen.layout['table_top_m'], 'keyframes': results,
            'static_candidates_found': sum(r['strict_static_candidate'] for r in results),
            'pose_gate_passed_frames': sum(r['kinematic_pose_gate_passed'] for r in results)}
        rows.append(row)
        destination = args.output / f'down_{int(drop):02d}cm.json'
        destination.write_text(json.dumps(row, default=_jsonable, indent=2, allow_nan=False)+'\n')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'method': __doc__,
        'scope': 'Offline geometry screening ONLY. Not a dynamics, policy, collision-free path or grasp certificate.',
        'interpretation': 'No solution found means solver did not find one, not proof of unreachability.',
        'hands': 'Fixed recorded OPEN hand pose throughout; hand/crate penetration listed separately.',
        'crate': 'Placed at source desired moving-crate transform in each isolated static candidate.',
        'constraints': {'wrist_position_m': .005, 'wrist_orientation_deg': 3.,
            'foot_position_m': .003, 'foot_orientation_deg': 1., 'forbidden_penetration_m': .002,
            'optimizer_joint_bounds': 'Original hard limits with optimizer-only interior margin; model unchanged',
            'optimizer_joint_margin_fraction': args.joint_margin_fraction,
            'COM': 'Weak target term and bounding-box-only final screen, no force/friction calculation'},
        'motion_sha256': motion.manifest['trajectory_sha256'], 'config': str(args.config.resolve()),
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'solver': {'max_nfev': args.max_nfev, 'seeds': args.seeds}, 'results': rows,
        'keyframe_scope': ('every HeightPath segment, discrete static samples' if args.samples_per_segment else
            'added outside/turn approach' if args.approach_only else 'four source keyframes'),
        'samples_per_segment': args.samples_per_segment,
        'elapsed_s': time.monotonic()-started}
    (args.output/'report.json').write_text(json.dumps(report, default=_jsonable, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
