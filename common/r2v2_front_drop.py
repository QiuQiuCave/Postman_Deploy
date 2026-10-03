"""Frozen whole-body Reach plus independent binary fingers: tabletop -> crate.

Only desired wrist poses are interpolated. No robot/object state is replayed,
no object is attached to the hand, and an airborne release is authorized only
over the measured crate opening after a physically verified pickup.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_cylinder_test import load_cylinder_profile
from common.r2v2_front_drop_scene import build_front_drop_model, build_front_drop_collision_diagnostic
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import initialize_robot, foot_collision_ids, load_reach_config, resolve_asset, require_parity, sha256
from common.r2v2_tabletop_scene import _descendant
from common.r2v2_top_grasp import TopGraspExperiment, pose_quaternion, rotation_error_deg, unsupported_load_conditions
from r2v2_description.model import BODY_JOINTS, SIDES, JointMap, SOURCE, urdf_hand_joints

TASK = 'R2V2-Reach-FrontManipulation-v1-28DoF'
SCOPE = 'tabletop_to_crate_drop_v1'
PHASES = ('STAND', 'HOME', 'SAFE_OUT', 'APPROACH', 'GRASP', 'PROBE', 'LIFT_CLEAR',
          'TRANSFER_ABOVE', 'DROP_RELEASE', 'RETREAT_HIGH', 'RETURN_HOME')
ROOT = Path('/root/autodl-tmp/Postman_Deploy/front_drop_20260927')
DEFAULT_CONFIG = dict(schema_version=1, reach_config=str(ROOT/'policy/reach_config.yaml'),
    parity_report=str(ROOT/'parity/report.json'), path_manifest=str(ROOT/'policy/path_manifest.json'),
    air_evidence=None, stage_timeout_s=10., startup_mode='hold_home')


def canonical_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class FrontDropPath:
    """NumPy mirror of the archived target interpolation, not a qpos record."""
    def __init__(self, filename):
        self.document = json.loads(Path(filename).read_text())
        d = self.document
        self.sha256 = canonical_digest({k: v for k, v in d.items() if k != 'content_sha256'})
        if (d.get('content_sha256') != self.sha256 or d.get('schema_version') != 1
                or d.get('endpoint_contract') != 'wrist_world_v2'
                or d.get('path_contract') != 'front_manipulation_path_v1' or d.get('task_scope') != SCOPE
                or d.get('geometry_validation', {}).get('passed') is not True):
            raise ValueError('Invalid certified front-drop path contract/hash')
        self.phases = d['phases']
        if tuple(p['name'] for p in self.phases) != PHASES:
            raise ValueError('Unexpected front-drop phase sequence')
        self.indices = {p['name']: i for i, p in enumerate(self.phases)}
        self.positions = np.asarray([p['position_m'] for p in self.phases], dtype=float)
        self.quaternions = np.asarray([p['quaternion_wxyz'] for p in self.phases], dtype=float)
        if (self.positions.shape != (len(PHASES), 2, 3) or self.quaternions.shape != (len(PHASES), 2, 4)
                or not np.isfinite(self.positions).all() or not np.isfinite(self.quaternions).all()
                or not np.allclose(np.linalg.norm(self.quaternions, axis=-1), 1., atol=1e-6)):
            raise ValueError('Invalid wrist target arrays')
        if not np.allclose(self.positions[:, 1], self.positions[0, 1], atol=1e-9, rtol=0):
            raise ValueError('Inactive wrist must remain world-fixed')
        if not np.allclose(np.abs(self.quaternions[:, 1] @ self.quaternions[0, 1]), 1., atol=1e-9, rtol=0):
            raise ValueError('Inactive wrist orientation must remain world-fixed')
        for p in self.phases:
            if not 0. <= p['motion_s'] <= p['duration_s'] <= 10. or p['duration_s'] <= 0.:
                raise ValueError('Invalid path duration')
        self.scene = copy.deepcopy(d['provenance']['scene'])
        # Check that the wrist goal really derives from the recorded can/wrist
        # relation, not the cylinder centre mistaken for a wrist target.
        scene = self.scene
        can = pose(scene['can_initial_world_m'], scene['can_initial_quaternion_wxyz'])
        relative = pose(scene['grasp_can_in_wrist_position_m'], scene['grasp_can_in_wrist_quaternion_wxyz'])
        expected = can @ np.linalg.inv(relative)
        if not np.allclose(expected, self.target('GRASP')[0], atol=1e-7):
            raise ValueError('Archived grasp goal differs from wrist/object calibration')

    def target(self, name):
        i = self.indices[name]
        return np.array([pose(p, q) for p, q in zip(self.positions[i], self.quaternions[i])])

    def sample_phase(self, name, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0.:
            raise ValueError('Expected nonnegative finite path time')
        i = self.indices[name]
        duration = self.phases[i]['motion_s']
        f = 1. if duration == 0. else float(np.clip(elapsed/duration, 0., 1.))
        f = f*f*(3.-2.*f)
        start = max(0, i-1)
        position = (1.-f)*self.positions[start]+f*self.positions[i]
        result = []
        for p, q1, q2 in zip(position, self.quaternions[start], self.quaternions[i]):
            if q1 @ q2 < 0.: q2 = -q2
            dot = float(np.clip(q1 @ q2, -1., 1.))
            if dot > .9995:
                q = (1.-f)*q1+f*q2
            else:
                angle = np.arccos(dot)
                q = (np.sin((1.-f)*angle)*q1+np.sin(f*angle)*q2)/np.sin(angle)
            result.append(pose(p, q/np.linalg.norm(q)))
        return np.asarray(result)


def pose(position, quaternion):
    result = np.eye(4)
    result[:3, 3] = position
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    result[:3, :3] = rotation.reshape(3, 3)
    return result


def load_config(source=None):
    values = {} if source is None else (json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else copy.deepcopy(source))
    if not isinstance(values, dict) or set(values)-set(DEFAULT_CONFIG):
        raise ValueError('Unknown front-drop configuration fields')
    result = {**DEFAULT_CONFIG, **values}
    if type(result['schema_version']) is not int or result['schema_version'] != 1:
        raise ValueError('Expected front-drop schema_version 1')
    if result['startup_mode'] not in ('follow_current', 'hold_home'):
        raise ValueError('Expected explicit follow_current or hold_home startup')
    timeout = result['stage_timeout_s']
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not np.isfinite(timeout) or not 0. < timeout <= 10.:
        raise ValueError('Stage deadline must be positive and at most 10s')
    for key in ('reach_config', 'parity_report', 'path_manifest'):
        result[key] = str(resolve_asset(result[key]))
    if result['air_evidence'] is not None:
        result['air_evidence'] = str(resolve_asset(result['air_evidence']))
    return result


def cylinder_envelope_in_crate(object_transform, crate_transform, radius, half_height, dimensions, wall, opening_bounds=None):
    """Exact solid-cylinder support along crate axes, including held tilt."""
    local = np.linalg.inv(crate_transform) @ object_transform
    axial = np.clip(local[:3, 2], -1., 1.)
    extent = half_height*np.abs(axial)+radius*np.sqrt(np.maximum(0., 1.-axial**2))
    low, high = local[:3, 3]-extent, local[:3, 3]+extent
    inner = np.asarray(dimensions[:2])/2.-wall
    xy_margin = float(np.min(np.r_[low[:2]+inner, inner-high[:2]]))
    if opening_bounds is None:
        opening = inner.copy(); opening[1] = dimensions[1]/2.-max(wall, .010)
        opening_bounds = dict(min=-opening, max=opening)
    opening_margin = float(np.min(np.r_[low[:2]-np.asarray(opening_bounds['min'])[:2],
                                       np.asarray(opening_bounds['max'])[:2]-high[:2]]))
    return dict(local_center_m=local[:3, 3].tolist(), local_min_m=low.tolist(), local_max_m=high.tolist(),
                opening_xy_margin_m=opening_margin, interior_xy_margin_m=xy_margin,
                bottom_above_rim_m=float(low[2]-dimensions[2]))


def measured_pickup(metrics, weight_N):
    """Allow the baseline's tilted grip: dropping does not require upright placement."""
    m = metrics
    return bool(unsupported_load_conditions(m, weight_N) and not m['crate_object_contact']
        and m.get('grasp_slip_m') is not None and m['grasp_slip_m'] < .015
        and m['grasp_rotation_slip_deg'] < 5.)


def valid_drop_location(metrics):
    return bool(metrics['finite_state'] and metrics['crate_table_contact']
        and metrics['opening_xy_margin_m'] >= .005 and metrics['bottom_above_rim_m'] >= .010
        and metrics['hand_above_rim_m'] >= .010 and metrics['crate_tilt_deg'] < 5.
        and metrics['crate_speed_mps'] < .02 and metrics['crate_angular_speed_radps'] < .15)


def deposited(metrics, weight_N):
    return bool(metrics['finite_state'] and metrics['crate_table_contact']
        and metrics['crate_tilt_deg'] < 5. and metrics['crate_speed_mps'] < .02 and metrics['crate_angular_speed_radps'] < .15
        and metrics['interior_xy_margin_m'] >= 0. and metrics['object_inside_crate']
        and metrics['crate_object_contact'] and metrics['crate_vertical_force_N'] >= .8*weight_N
        and not metrics['floor_contact'] and not metrics['table_contact']
        and metrics['active_hand_object_contact_count'] == 0 and metrics['parked_hand_contact_count'] == 0
        and metrics['object_linear_speed_mps'] < .02 and metrics['object_angular_speed_radps'] < .15)


class FrontDropExperiment(TopGraspExperiment):
    """Standalone task; inherit only contact measurement/recording helpers."""
    def __init__(self, config=None, mode='air', keep_trace=True):
        if mode not in ('air', 'contact'):
            raise ValueError('Mode must be air or contact')
        self.config, self.mode = load_config(config), mode
        self.cfg = load_reach_config(self.config['reach_config'])
        self.parity = require_parity(self.config['parity_report'], self.cfg)
        self.path = FrontDropPath(self.config['path_manifest'])
        self.policy_sha = sha256(resolve_asset(self.cfg['policy_path']))
        if self.parity.get('task') != TASK:
            raise ValueError('Numerical observation parity must belong to the front task')
        self.model, self.hand_cfg, self.layout = build_front_drop_model(self.cfg, self.path.scene, mode)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, self.hand_cfg)
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        # Match the training reset's existing 90% soft-limit clipping, not a
        # new physical limit or a scripted standing controller.
        limits = self.model.jnt_range[self.body_map.joints]
        middle, half = limits.mean(1), .45*(limits[:, 1]-limits[:, 0])
        self.data.qpos[self.body_map.qpos] = np.clip(self.data.qpos[self.body_map.qpos], middle-half, middle+half)
        mujoco.mj_forward(self.model, self.data)
        self.policy = ReachPolicy(self.model, self.data, resolve_asset(self.cfg['policy_path']), expected_endpoint_contract='wrist_world_v2')
        for key, expected in (('task_id', TASK), ('task_scope', SCOPE), ('path_sha256', self.path.sha256)):
            if self.policy.metadata.get(key) != expected:
                raise ValueError('Policy/path metadata mismatch: '+key)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        assert not set(self.body_map.actuators).intersection(np.concatenate([x.actuators for x in self.hands.maps.values()]))
        self.side, self.profile = 'left', load_cylinder_profile('baseline_40mm_100g')
        self.object_body, self.object_geom = self.model.body('test_cylinder').id, self.model.geom('cylinder_geom').id
        self.table_geom, self.floor_geom = self.model.geom('tabletop_geom').id, self.model.geom('floor').id
        self.base = self.model.body('base_link').id
        self.crate_body = self.model.body('cargo_crate').id
        self.wrist_bodies = {s: self.model.body(s+'_hand_roll_link').id for s in SIDES}
        self.robot_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.crate_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.crate_body)}
        self.table_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.model.body('tabletop').id)}
        # AIR needs its own compiled contact BVHs, not runtime mask changes:
        # the non-colliding live props have no broad-phase collision tree.
        self.diagnostic_model = (build_front_drop_collision_diagnostic(self.cfg, self.path.scene)
                                 if mode == 'air' else None)
        self.diagnostic_data = (mujoco.MjData(self.diagnostic_model)
                                if self.diagnostic_model is not None else None)
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.geom_sides = {g: s for g in self.robot_geoms for s, b in self.wrist_bodies.items()
                           if _descendant(self.model, int(self.model.geom_bodyid[g]), b)}
        self.hand_joints = np.array([self.model.joint(n).id for n in urdf_hand_joints()])
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find('mimic')
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0], self.model.joint(mimic.get('joint')).qposadr[0],
                    float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))))
        self.weight_N, self.table_height = self.profile['mass_kg']*9.81, self.path.scene['tabletop_height_m']
        self.phase, self.phase_start, self.steps = 'STAND', 0., 0
        self.failure = self.failure_phase = None
        self.grasp_verified = self.release_commanded = self.completed = False
        self.baseline_relation = self.alignment_relation = None
        self.upright_attempted, self.max_verified_hold_s = False, 0.
        self.stable_since = self.bad_since = None
        self.keep_trace, self.samples, self.targets = bool(keep_trace), [], []
        self.transitions = [dict(time_s=0., state='STAND')]
        self.stage_results, self.safety_events, self.contact_events = [], [], []
        self.current_metrics = {}
        self.goal_wrist_transforms = {s: body_transform(self.data, b) for s, b in self.wrist_bodies.items()}
        self.active_goal = self.goal_wrist_transforms['left'].copy()
        self.initial_object_position = self.data.xpos[self.object_body].copy()
        self.foot_anchor = self._feet(self.data)
        self.stage_collision = False
        self.peaks = dict.fromkeys(('foot_drift_m', 'base_tilt_deg', 'hand_object_penetration_m',
            'hand_self_penetration_m', 'hand_joint_violation_rad', 'mimic_error_rad', 'grasp_slip_m',
            'grasp_rotation_slip_deg', 'clearance_m', 'body_joint_violation_rad', 'robot_self_penetration_m'), 0.)
        self.bindings = self._bindings()
        if mode == 'contact':
            self._require_air()
        scene = self.path.scene
        dims = np.asarray(scene['crate_dimensions_depth_width_height_m'])
        self.render_wireframes = [dict(kind='box', position=scene['table_center_world_m'], size=scene['table_half_size_m'], color=(1., .73, .32, 1.)),
            dict(kind='box', position=np.asarray(scene['crate_base_world_m'])+[0., 0., dims[2]/2.], size=dims/2., color=(.25, .85, 1., 1.)),
            dict(kind='cylinder', position=scene['can_initial_world_m'], size=[.02, .06], color=(1., .2, .15, 1.))]
        self.sync()
        self.initial_metrics = copy.deepcopy(self.current_metrics)
        self._safety(); self.record()

    def _bindings(self):
        source = {str(p.relative_to(SOURCE)): sha256(p) for p in sorted(SOURCE.rglob('*')) if p.is_file()}
        return dict(path_sha256=self.path.sha256, onnx_sha256=self.policy_sha,
            task_config_sha256=canonical_digest({k: v for k, v in self.config.items() if k != 'air_evidence'}),
            checkpoint_sha256=self.parity['checkpoint_sha256'], robot_source_sha256=canonical_digest(source),
            hand_config_sha256=canonical_digest(self.hand_cfg), controller_sha256=sha256(__file__),
            hand_controller_sha256=sha256(PROJECT_ROOT/'common/r2v2_hand_control.py'),
            adapter_sha256=sha256(PROJECT_ROOT/'common/r2v2_reach_policy.py'),
            scene_code_sha256=sha256(PROJECT_ROOT/'common/r2v2_front_drop_scene.py'),
            metric_code_sha256=sha256(PROJECT_ROOT/'common/r2v2_top_grasp.py'), mujoco_version=mujoco.__version__)

    def _require_air(self):
        if self.config['air_evidence'] is None:
            raise ValueError('Physical contact requires a matching successful deployment AIR report')
        report = json.loads(Path(self.config['air_evidence']).read_text())
        if (report.get('schema_version') != 1 or report.get('mode') != 'air' or report.get('air_passed') is not True
                or report.get('success') is not True or report.get('failure') is not None
                or report.get('phase') != 'COMPLETE' or report.get('experiment_completed') is not True
                or report.get('grasp_verified') is not False or report.get('release_commanded') is not False
                or report.get('bindings') != self.bindings
                or [p['name'] for p in report.get('stage_results', [])] != [*PHASES[1:], 'FINAL_HOLD']
                or not all(p.get('passed') is True for p in report['stage_results'])):
            raise ValueError('Mismatched or failed deployment AIR evidence')

    def _feet(self, data):
        return np.array([data.xpos[self.model.body(s+'_ankle_roll_link').id] for s in SIDES])

    def _set_goals(self, targets):
        for side, target in zip(SIDES, targets):
            self.goal_wrist_transforms[side] = target.copy()
            self.policy.set_target_world(side, target[:3, 3], pose_quaternion(target))
        self.active_goal = self.goal_wrist_transforms['left'].copy()

    def enter(self, phase, target=None, duration=0.):
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since = self.bad_since = None
        self.stage_collision = False
        self.transitions.append(dict(time_s=self.phase_start, state=phase))
        if phase == 'HOME':
            self.foot_anchor = self._feet(self.scratch)
        if phase == 'CLOSE':
            if self.mode != 'contact': raise RuntimeError('AIR cannot close fingers')
            self.hands.command('left', 1)
        elif phase == 'OPEN':
            if self.mode != 'contact' or not self.grasp_verified or not valid_drop_location(self.current_metrics):
                self.fail('Release forbidden without verified grasp and measured drop clearance'); return
            self.hands.command('left', 0)
            self.release_commanded = True
            self.release_snapshot = copy.deepcopy(self.current_metrics)

    def fail(self, reason):
        if self.done: return
        self.safety_events.append(dict(time_s=float(self.data.time), phase=self.phase, reason=str(reason)))
        TopGraspExperiment.fail(self, reason)

    def _pose_good(self):
        m = self.current_metrics
        arms = all(e['position_m'] < .005 and e['orientation_deg'] < 3.
            and e['linear_speed_mps'] < .02 and e['angular_speed_radps'] < .15
            for e in m['wrist_errors'].values())
        return bool(arms and m['foot_drift_m'] < .02 and m['root_speed_mps'] < .03
            and m['root_angular_speed_radps'] < .15 and not self.stage_collision
            and m['base_tilt_deg'] < 35. and m['root_height_m'] > .55)

    def _finish_stage(self):
        self.stage_results.append(dict(name=self.phase, passed=True, duration_s=float(self.data.time-self.phase_start),
            held_s=float(self.data.time-self.stable_since), wrist_errors=copy.deepcopy(self.current_metrics['wrist_errors']),
            foot_drift_m=self.current_metrics['foot_drift_m']))

    def _advance_path(self):
        name = self.phase
        if name == 'RETURN_HOME': self.enter('FINAL_HOLD')
        elif self.mode == 'contact' and name == 'GRASP': self.enter('CLOSE')
        elif self.mode == 'contact' and name == 'PROBE': self.enter('VERIFY_GRASP')
        elif self.mode == 'contact' and name == 'DROP_RELEASE': self.enter('OPEN')
        else: self.enter(PHASES[PHASES.index(name)+1])

    def _gate(self):
        t, m = float(self.data.time-self.phase_start), self.current_metrics
        if self.phase == 'STAND':
            if t >= .4-1e-8: self.enter('HOME')
            return
        if self.phase in PHASES:
            p = self.path.phases[self.path.indices[self.phase]]
            valid = self._pose_good() and t >= p['motion_s']
            if self.mode == 'contact' and self.phase in ('LIFT_CLEAR', 'TRANSFER_ABOVE', 'DROP_RELEASE'):
                valid &= measured_pickup(m, self.weight_N)
                if self.phase == 'LIFT_CLEAR':
                    valid &= m['bottom_above_rim_m'] >= .010 and m['hand_above_rim_m'] >= .010
                if self.phase in ('TRANSFER_ABOVE', 'DROP_RELEASE'):
                    valid &= valid_drop_location(m)
            ready = self._stable(valid, .3)
            if ready and t >= p['duration_s']-1e-8:
                self._finish_stage(); self._advance_path(); return
        elif self.phase == 'CLOSE':
            if self._stable(t >= 2.5 and m['opposed_contact'] and self._pose_good(), .3):
                self.enter('PROBE'); return
        elif self.phase == 'VERIFY_GRASP':
            physical = unsupported_load_conditions(m, self.weight_N) and not m['crate_object_contact']
            if physical and self.baseline_relation is None:
                self.baseline_relation = np.asarray(m['T_wrist_object']).copy()
                self.sync(); m = self.current_metrics
            if self._stable(measured_pickup(m, self.weight_N) and self._pose_good(), 1.):
                self.grasp_verified = True
                self.max_verified_hold_s = float(self.data.time-self.stable_since)
                self.sync(); m = self.current_metrics
                self.pickup_snapshot = copy.deepcopy(m)
                self.enter('LIFT_CLEAR'); return
        elif self.phase == 'OPEN':
            if t >= 2.5 and m['active_hand_object_contact_count'] == 0:
                self.enter('WAIT_DROP'); return
        elif self.phase == 'WAIT_DROP':
            if self._stable(deposited(m, self.weight_N), 1.):
                self.drop_verified = True
                self.sync(); m = self.current_metrics
                self.drop_snapshot = copy.deepcopy(m)
                self.enter('RETREAT_HIGH'); return
        elif self.phase == 'FINAL_HOLD':
            valid = self._pose_good() and (self.mode == 'air' or deposited(m, self.weight_N))
            if self._stable(valid, 1.) and t >= 2.:
                self._finish_stage(); self.completed = True; self.enter('COMPLETE'); return
        if t >= self.config['stage_timeout_s']-1e-8:
            self.fail(self.phase+' exceeded measured pose/contact deadline; no automatic release')

    def _stream(self):
        if self.phase == 'STAND':
            if self.config['startup_mode'] == 'hold_home':
                self._set_goals(self.path.target('HOME'))
            else:
                self.policy.follow_current(self.scratch)
                self.goal_wrist_transforms = {s: body_transform(self.scratch, b) for s, b in self.wrist_bodies.items()}
                self.active_goal = self.goal_wrist_transforms['left'].copy()
        elif self.phase in PHASES:
            # Same cubic path, reference speed conditioning and continuity as
            # training. Only the phase clock waits at actual success gates.
            self._set_goals(self.path.sample_phase(self.phase, max(0., float(self.data.time-self.phase_start))+.02))
        self.targets.append(dict(time_s=float(self.data.time), phase=self.phase,
            T_world_wrist_goal={s: v.tolist() for s, v in self.goal_wrist_transforms.items()}))

    def sync(self):
        TopGraspExperiment.sync(self)
        m, d, row = self.model, self.scratch, self.current_metrics
        errors = {}
        for side, body in self.wrist_bodies.items():
            actual, goal = body_transform(d, body), self.goal_wrist_transforms[side]
            v, w = self.policy.wrist_twist(d, side)
            errors[side] = dict(position_m=float(np.linalg.norm(actual[:3, 3]-goal[:3, 3])),
                orientation_deg=rotation_error_deg(actual[:3, :3], goal[:3, :3]),
                linear_speed_mps=float(np.linalg.norm(v)), angular_speed_radps=float(np.linalg.norm(w)),
                T_world_wrist=actual.tolist(), T_world_wrist_goal=goal.tolist())
        crate = body_transform(d, self.crate_body)
        dims = self.path.scene['crate_dimensions_depth_width_height_m']
        envelope = cylinder_envelope_in_crate(np.asarray(row['T_world_object']), crate, .02, .06,
            dims, self.path.scene['crate_wall_thickness_m'], self.layout['crate_opening_bounds_local_m'])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, self.base)
        root_v, root_w = np.linalg.norm(jp @ d.qvel), np.linalg.norm(jr @ d.qvel)
        mujoco.mj_jacBody(m, d, jp, jr, self.crate_body)
        crate_v, crate_w = np.linalg.norm(jp @ d.qvel), np.linalg.norm(jr @ d.qvel)
        crate_force, crate_contact, robot_prop, nonhand_object, self_depth = 0., False, [], 0, 0.
        crate_table = False
        for index, c in enumerate(d.contact):
            a, b = map(int, c.geom)
            if c.dist > 0.: continue
            if a in self.robot_geoms and b in self.robot_geoms:
                self_depth = max(self_depth, -float(c.dist))
            if (a in self.crate_geoms and b in self.table_geoms) or (b in self.crate_geoms and a in self.table_geoms):
                crate_table = True
            if ((a in self.robot_geoms and b in self.table_geoms | self.crate_geoms)
                    or (b in self.robot_geoms and a in self.table_geoms | self.crate_geoms)):
                robot_prop.append(dict(geoms=[m.geom(a).name, m.geom(b).name], penetration_m=-float(c.dist)))
            if self.object_geom in (a, b):
                other = b if a == self.object_geom else a
                if other in self.robot_geoms and other not in self.geom_sides:
                    nonhand_object += 1
                if other in self.crate_geoms:
                    force = np.zeros(6); mujoco.mj_contactForce(m, d, index, force)
                    if force[0] > .001:
                        crate_contact = True
                        world_force = c.frame.reshape(3, 3).T @ force[:3]
                        crate_force += float(world_force[2])*(1. if b == self.object_geom else -1.)
        if self.mode == 'air':
            diagnostic = self.diagnostic_data
            diagnostic.qpos[:] = d.qpos; diagnostic.qvel[:] = d.qvel; diagnostic.ctrl[:] = d.ctrl
            mujoco.mj_forward(self.diagnostic_model, diagnostic)
            for c in diagnostic.contact:
                a, b = map(int, c.geom)
                if c.dist < 0. and ((a in self.robot_geoms and b in self.table_geoms | self.crate_geoms)
                        or (b in self.robot_geoms and a in self.table_geoms | self.crate_geoms)):
                    robot_prop.append(dict(geoms=[m.geom(a).name, m.geom(b).name], penetration_m=-float(c.dist)))
        hand_bottom = float('inf')
        # World AABB support of each left-hand collision mesh, conservative
        # for mesh concavity. Used for over-rim clearance, never physics.
        for g, side in self.geom_sides.items():
            if side != 'left' or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
            center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            rotation = d.geom_xmat[g].reshape(3, 3)
            hand_bottom = min(hand_bottom, float((d.geom_xpos[g]+rotation @ center)[2]-np.abs(rotation[2]) @ half))
        rim = float(crate[2, 3]+crate[2, 2]*dims[2]
                    +abs(crate[2, 0])*dims[0]/2.+abs(crate[2, 1])*dims[1]/2.)
        low, high = np.asarray(envelope['local_min_m']), np.asarray(envelope['local_max_m'])
        row.update(wrist_errors=errors, **envelope, root_speed_mps=float(root_v), root_angular_speed_radps=float(root_w),
            root_height_m=float(d.xpos[self.base, 2]), base_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1., 1.)))),
            foot_drift_m=float(np.max(np.linalg.norm(self._feet(d)-self.foot_anchor, axis=1))),
            crate_object_contact=crate_contact, crate_table_contact=crate_table,
            crate_vertical_force_N=crate_force, T_world_crate=crate.tolist(),
            crate_tilt_deg=float(np.degrees(np.arccos(np.clip(crate[2, 2], -1., 1.)))),
            crate_speed_mps=float(crate_v), crate_angular_speed_radps=float(crate_w),
            object_inside_crate=bool(envelope['interior_xy_margin_m'] >= 0. and low[2] >= self.path.scene['crate_bottom_thickness_m']-.003 and high[2] <= dims[2]+.003),
            hand_above_rim_m=hand_bottom-rim, robot_prop_contacts=robot_prop, nonhand_object_contact_count=nonhand_object,
            robot_self_penetration_m=self_depth, mode=self.mode, grasp_verified=self.grasp_verified,
            drop_verified=getattr(self, 'drop_verified', False))
        if robot_prop: self.stage_collision = True
        return row

    def _safety(self):
        d, m, row = self.data, self.model, self.current_metrics
        if not row['finite_state'] or np.any(d.warning.number): self.fail('Nonfinite state or MuJoCo warning'); return
        if m.nmocap or np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD) or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            self.fail('Unexpected fixture or external force'); return
        if self.mode == 'air' and any(c.command != 0 for c in self.hands.controllers.values()):
            self.fail('AIR attempted finger closure'); return
        q, limits = d.qpos[self.body_map.qpos], m.jnt_range[self.body_map.joints]
        excess = float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1])))
        row['body_joint_violation_rad'] = excess
        for key in self.peaks:
            if row.get(key) is not None: self.peaks[key] = max(self.peaks[key], float(row[key]))
        if excess > .05: self.fail('Body joint physical limit excess > 0.05 rad'); return
        if row['base_tilt_deg'] > 35. or row['root_height_m'] < .55:
            self.fail('Body tilt/height safety limit'); return
        if row['robot_self_penetration_m'] > .005:
            self.fail('Deep robot self-collision'); return
        if row['hand_joint_violation_rad'] > .01 or row['mimic_error_rad'] > .015:
            self.fail('Finger joint or mimic limit'); return
        for c in d.contact:
            if c.dist > 0.: continue
            a, b = map(int, c.geom)
            if self.floor_geom in (a, b):
                other = b if a == self.floor_geom else a
                if other in self.robot_geoms and other not in self.foot_geoms:
                    self.fail('Non-foot ground contact'); return
        if self.mode != 'contact':
            if row['robot_prop_contacts']:
                self.fail('AIR collision diagnostic: robot intersects table/crate, including startup')
            return
        if row['parked_hand_contact_count'] or row['nonhand_object_contact_count']:
            self.fail('Non-grasping robot part contacted object'); return
        if row['hand_object_penetration_m'] > .003:
            self.fail('Hand/object penetration exceeded 3mm'); return
        if any(c['penetration_m'] > .003 for c in row['robot_prop_contacts']):
            self.fail('Robot/table or crate penetration exceeded 3mm'); return
        if row['floor_contact']:
            self.fail('Object reached floor, not crate'); return
        if self.grasp_verified and not self.release_commanded:
            retained = (row['opposed_contact'] and row['grasp_slip_m'] is not None and row['grasp_slip_m'] < .015
                and row['grasp_rotation_slip_deg'] < 5. and not row['table_contact'] and not row['crate_object_contact'])
            if not retained:
                self.fail('Verified grasp lost/slipped; fingers not automatically opened'); return

    def step(self):
        if self.done: return
        if self.steps % 20 == 0:
            self.sync(); self._safety()
            if self.done: self.record(); return
            self._gate()
            if self.done: self.record(); return
            self._stream(); self.policy.act(self.scratch)
        if self.steps % 10 == 0: self.hands.update()
        self.policy.apply(self.data); self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or np.any(self.data.warning.number):
            self.fail('Nonfinite state or MuJoCo warning')
        if self.steps % 10 == 0 or self.done:
            self.sync(); self._safety(); self.record()

    def record(self):
        TopGraspExperiment.record(self)
        if self.keep_trace:
            self.samples[-1].update(body_action=self.policy.last_action.tolist(),
                hand_commands={s: c.command for s, c in self.hands.controllers.items()})

    def report(self):
        air = self.mode == 'air' and self.completed and not self.failure
        contact = (self.mode == 'contact' and self.completed and self.grasp_verified
                   and self.release_commanded and getattr(self, 'drop_verified', False) and not self.failure)
        return dict(schema_version=1, task=TASK, task_scope=SCOPE, mode=self.mode,
            scope='Real whole-body policy and independent fingers; AIR has no grasp evidence',
            success=bool(air or contact), air_passed=bool(air), phase=self.phase,
            experiment_completed=self.completed, failure=self.failure, failure_phase=self.failure_phase,
            duration_s=float(self.data.time), grasp_verified=self.grasp_verified, release_commanded=self.release_commanded,
            drop_verified=getattr(self, 'drop_verified', False), bindings=self.bindings, config=self.config,
            initial_metrics=self.initial_metrics, final_metrics=self.current_metrics, peaks=self.peaks,
            stage_results=self.stage_results, transitions=self.transitions, safety_events=self.safety_events,
            pickup_snapshot=getattr(self, 'pickup_snapshot', None), release_snapshot=getattr(self, 'release_snapshot', None),
            drop_snapshot=getattr(self, 'drop_snapshot', None),
            grasp_relation=None if self.baseline_relation is None else self.baseline_relation.tolist(),
            physics=dict(wrist_fixture=False, object_pose_replay=False, object_weld=False, additional_external_forces=False,
                physical_props=self.mode == 'contact', dynamic_fingers=True, body_policy_hz=50, fingers_hz=100,
                physics_hz=1000, detailed_safety_metrics_hz=100, nonfinite_state_check_hz=1000),
            acceptance=dict(wrist_position_m=.005, wrist_orientation_deg=3., stable_s=.3,
                pickup_stable_s=1., pickup_clearance_m=.008, slip_m=.015, slip_deg=5.,
                drop_xy_clearance_m=.005, drop_rim_clearance_m=.010, deposited_stable_s=1.,
                deposited_upright_required=False, deadline_s=self.config['stage_timeout_s']),
            warnings=self.data.warning.number.tolist())
