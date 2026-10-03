"""Success-gated transfer of calibrated can grasps to an unassisted Reach actor.

AIR is a pose/collision diagnostic with static, noncolliding props, never grasp
evidence. CONTACT requires a matching successful AIR report. Robot physics and
the existing hand/actor channels are unchanged; no runtime pose replay or IK.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_cylinder_test import load_cylinder_profile
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import (foot_collision_ids, initialize_robot,
    load_reach_config, require_parity, resolve_asset, sha256)
from common.r2v2_tabletop_scene import build_tabletop_xml, _descendant
from common.r2v2_top_grasp import (TopGraspExperiment, pose_quaternion,
    pickup_conditions, placed_conditions, unsupported_load_conditions,
    rotation_error_deg)
from common.r2v2_top_grasp_path import (DEFAULT_CALIBRATION_PATH,
    load_top_grasp_calibration, world_path_for_scene)
from common.r2v2_front_grasp_path import (DEFAULT_FRONT_CALIBRATION_PATH,
    FRONT_RETREAT_OFFSET_M, load_front_grasp_calibration, front_world_path_for_scene)
from common.r2v2_top_grasp_scene import TopGraspCandidate
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, SOURCE, urdf_hand_joints


PATH_NAMES = ('HOVER', 'APPROACH', 'GRASP', 'PROBE', 'UPRIGHT', 'LIFT', 'TRANSLATE', 'PLACE', 'RETREAT')
AIR_STAGES = ('HOME', 'STAND', 'SAFE_OUT', 'TURN_WRIST', *PATH_NAMES)
FROZEN_POLICY = Path('/root/autodl-tmp/Postman_Deploy/wrist_payload_contact_20260913')
DEFAULT_CONFIG = dict(schema_version=1, grasp_style='upper',
    reach_config=str(FROZEN_POLICY/'policy/reach_config.yaml'),
    parity_report=str(FROZEN_POLICY/'parity/report.json'),
    calibration_path=str(DEFAULT_CALIBRATION_PATH),
    profile='baseline_40mm_100g', table_height_m=.8, can_xy_m=[.36, .18],
    yaw_deg=180., table_center_xy_m=[.48, .08], table_half_size_m=[.27, .30, .02],
    stage_timeout_s=10., air_continue_on_precision_failure=True,
    air_evidence=None)


def load_config(source=None):
    values = {} if source is None else (json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else copy.deepcopy(source))
    if not isinstance(values, dict) or set(values)-set(DEFAULT_CONFIG):
        raise ValueError('Unknown fullbody top-grasp configuration fields')
    cfg = {**copy.deepcopy(DEFAULT_CONFIG), **values}
    if cfg['grasp_style'] not in ('upper', 'front'):
        raise ValueError('grasp_style must be upper or front')
    if cfg['grasp_style'] == 'front':
        if 'calibration_path' not in values:
            cfg['calibration_path'] = str(DEFAULT_FRONT_CALIBRATION_PATH)
        if 'yaw_deg' not in values:
            cfg['yaw_deg'] = 0.
        if cfg['yaw_deg'] != 0.:
            raise ValueError('The calibrated frontal path requires yaw_deg=0')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('Expected schema_version=1')
    for key, lower, upper in (('table_height_m', .5, 1.2), ('yaw_deg', -180., 180.), ('stage_timeout_s', .1, 10.)):
        v = cfg[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or not lower <= v <= upper:
            raise ValueError(f'Invalid {key}')
    for key, size in (('can_xy_m', 2), ('table_center_xy_m', 2), ('table_half_size_m', 3)):
        a = np.asarray(cfg[key], dtype=float)
        if a.shape != (size,) or not np.isfinite(a).all() or any(isinstance(x, bool) for x in cfg[key]):
            raise ValueError(f'Invalid {key}')
        if key == 'table_half_size_m' and (np.any(a <= 0) or a[2]*2 >= cfg['table_height_m']):
            raise ValueError('Invalid table extents')
        cfg[key] = a.tolist()
    if type(cfg['air_continue_on_precision_failure']) is not bool:
        raise ValueError('air_continue_on_precision_failure must be bool')
    if cfg['profile'] != 'baseline_40mm_100g':
        raise ValueError('Fullbody migration currently uses only the independently verified 100 g baseline')
    for key in ('reach_config', 'parity_report', 'calibration_path'):
        cfg[key] = str(resolve_asset(cfg[key]))
    return cfg


def fingerprint(config):
    cfg = load_config(config)
    cfg.pop('air_evidence')
    cfg.pop('air_continue_on_precision_failure')
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


def at_wrist_goal(errors, position=.005, orientation=3., speed=.02):
    try:
        a = np.array([errors[k] for k in ('position_m', 'orientation_deg', 'linear_speed_mps')], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(a.shape == (3,) and np.isfinite(a).all() and np.all(a >= 0)
                and a[0] < position and a[1] < orientation and a[2] < speed)


def evidence_input_bindings(config):
    """Bind actual asset/source bytes, not just their configured path strings.

    Follow the source manifest convention of r2v2_grasp_recording: include all
    supplied XML/URDF/meshes and the scene/control code which consumes them.
    Capture this at initialization, so a report describes the loaded run's
    inputs rather than silently blessing files changed after simulation.
    """
    cfg = load_config(config)
    files = [p for p in SOURCE.rglob('*') if p.is_file()]
    files += [PROJECT_ROOT/p for p in (
        'r2v2_description/model.py', 'common/r2v2_hand_control.py',
        'common/r2v2_reach_sim.py', 'common/r2v2_tabletop_scene.py',
        'common/r2v2_cylinder_test.py', 'common/r2v2_can_visual.py',
        'common/r2v2_top_grasp.py', 'common/r2v2_top_grasp_path.py',
        'common/r2v2_front_grasp_path.py',
        'common/r2v2_top_grasp_scene.py', 'common/r2v2_grasp_recording.py',
        'deploy_mujoco/config/r2v2_hands.yaml',
        'deploy_mujoco/config/r2v2_top_grasp_upper_40mm.json',
        'deploy_mujoco/config/cylinder_profiles/baseline_40mm_100g.yaml',
        'requirements-r2v2-sim.txt')]
    manifest = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sorted(set(files))}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(model_source_sha256=digest, source_asset_manifest=manifest,
                reach_config_sha256=sha256(cfg['reach_config']), mujoco_version=mujoco.__version__)


def _air_stage_evidence_valid(stage):
    if (not isinstance(stage, dict) or stage.get('passed') is not True
            or stage.get('possible_table_collision') is not False):
        return False
    try:
        duration = float(stage['duration_s'])
        if not np.isfinite(duration) or duration < 0.:
            return False
        if stage['name'] == 'STAND':
            values = np.asarray([stage['standing_final_base_tilt_deg'],
                                 stage['standing_final_root_speed']], dtype=float)
            return bool(duration >= 20.-1e-8 and values.shape == (2,)
                        and np.isfinite(values).all() and np.all(values >= 0.)
                        and values[0] < 8. and values[1] < .2)
        held = float(stage['settled_duration_s'])
        if not np.isfinite(held) or min(duration, held) < .3-1e-8 or held > duration+1e-8:
            return False
        errors = stage['final_errors']
        home = stage['name'] == 'HOME'
        return (at_wrist_goal(errors['left'], .025 if home else .005,
                              7. if home else 3., .05 if home else .02)
                and at_wrist_goal(errors['right'], .02, 5., .05))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def require_air_evidence(path, config, calibration_sha, policy_sha):
    if path is None:
        raise ValueError('CONTACT requires a successful matching AIR report')
    r = json.loads(Path(path).read_text())
    if not isinstance(r, dict):
        raise ValueError('AIR report must be a structured report object')
    stages = r.get('stage_results', [])
    if (type(r.get('schema_version')) is not int or r.get('schema_version') != 1 or r.get('mode') != 'air'
            or r.get('air_passed') is not True or r.get('success') is not True
            or r.get('experiment_completed') is not True or r.get('phase') != 'COMPLETE'
            or r.get('failure') is not None or r.get('safety_violations_seen') != []
            or any(r.get(key) is not False for key in ('wrist_fixture', 'object_pose_replay',
                'object_weld', 'additional_external_forces', 'grasp_verified', 'release_commanded'))
            or not isinstance(stages, list) or not all(isinstance(s, dict) for s in stages)
            or [s.get('name') for s in stages] != list(AIR_STAGES)
            or not all(_air_stage_evidence_valid(s) for s in stages)):
        raise ValueError('AIR report did not verify the entire safe, precise sequence')
    for key, expected in (('config_sha256', fingerprint(config)),
                          ('calibration_sha256', calibration_sha), ('onnx_sha256', policy_sha),
                          ('controller_sha256', sha256(__file__)),
                          ('adapter_sha256', sha256(PROJECT_ROOT/'common/r2v2_reach_policy.py')),
                          ('endpoint_contract', 'wrist_world_v2')):
        if r.get(key) != expected:
            raise ValueError(f'AIR evidence mismatch: {key}')
    for key, expected in evidence_input_bindings(config).items():
        if r.get(key) != expected:
            raise ValueError(f'AIR source/asset evidence mismatch: {key}')
    return r


def build_fullbody_model(config=None, mode='air'):
    cfg = load_config(config)
    if mode not in ('air', 'contact'):
        raise ValueError('Expected air or contact mode')
    reach = load_reach_config(cfg['reach_config'])
    if reach.get('endpoint_contract') != 'wrist_world_v2':
        raise ValueError('Fullbody can grasp requires physical world wrist endpoints')
    profile = load_cylinder_profile(cfg['profile'])
    h, half = cfg['table_height_m'], cfg['table_half_size_m']
    yaw = np.deg2rad(cfg['yaw_deg'])
    quaternion = [np.cos(yaw/2.), 0., 0., np.sin(yaw/2.)]
    scene = dict(object_appearance='cola_can', cylinder_profile=profile,
        table_center_xyz=[*cfg['table_center_xy_m'], h-half[2]], table_half_size=half,
        cylinder_position_xyz=[*cfg['can_xy_m'], h+profile['height_m']/2.+.001],
        cylinder_quaternion_wxyz=quaternion)
    xml, hands = build_tabletop_xml(reach, scene)
    root = ET.fromstring(xml)
    root.set('model', f"R2V2_{cfg['grasp_style']}_can_fullbody_{mode}")
    for side in SIDES:
        wrist = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        site = wrist.find(f'site[@name="{side}_tcp"]')
        site.set('name', f'{side}_wrist'); site.set('pos', '0 0 0')
    if mode == 'air':
        for name in ('test_cylinder', 'tabletop'):
            body = root.find(f'.//body[@name="{name}"]')
            for free in body.findall('freejoint'):
                body.remove(free)
            for geom in body.iter('geom'):
                geom.set('contype', '0'); geom.set('conaffinity', '0'); geom.set('group', '4')
                # Draw these static planning props as renderer-only wireframes.
                geom.set('rgba', '0 0 0 0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    if model.nmocap or np.any(model.eq_type == mujoco.mjtEq.mjEQ_WELD):
        raise ValueError('Fullbody experiment forbids support fixtures')
    layout = dict(table_top_m=h, table_center=np.array(scene['table_center_xyz']),
        table_half_size=np.array(half), cylinder_initial_position=np.array(scene['cylinder_position_xyz']),
        cylinder_place_position=None, mode=mode, object_free=mode == 'contact')
    return model, hands, layout


class TopGraspFullbodyExperiment(TopGraspExperiment):
    """Actor controls the body; actual endpoint gates control task progression."""

    def __init__(self, config=None, mode='air', keep_trace=True):
        self.config, self.mode = load_config(config), mode
        self.input_bindings = evidence_input_bindings(self.config)
        self.controller_sha = sha256(__file__)
        self.adapter_sha = sha256(PROJECT_ROOT/'common/r2v2_reach_policy.py')
        self.cfg = load_reach_config(self.config['reach_config'])
        self.parity = require_parity(self.config['parity_report'], self.cfg)
        frontal = self.config['grasp_style'] == 'front'
        self.calibration = (load_front_grasp_calibration if frontal else load_top_grasp_calibration)(self.config['calibration_path'])
        self.calibration_sha = sha256(self.config['calibration_path'])
        self.policy_sha = sha256(resolve_asset(self.cfg['policy_path']))
        if mode == 'contact':
            self.air_evidence = require_air_evidence(self.config['air_evidence'], self.config, self.calibration_sha, self.policy_sha)
        self.model, self.hand_cfg, self.layout = build_fullbody_model(self.config, mode)
        self.profile, self.side = load_cylinder_profile(self.config['profile']), 'left'
        self.candidate = TopGraspCandidate(**json.loads((PROJECT_ROOT/'deploy_mujoco/config/r2v2_top_grasp_upper_40mm.json').read_text()))
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, self.hand_cfg)
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        limits = self.model.jnt_range[self.body_map.joints]
        center, half = limits.mean(1), .45*(limits[:, 1]-limits[:, 0])
        self.data.qpos[self.body_map.qpos] = np.clip(self.data.qpos[self.body_map.qpos], center-half, center+half)
        mujoco.mj_forward(self.model, self.data)
        self.policy = ReachPolicy(self.model, self.data, resolve_asset(self.cfg['policy_path']), expected_endpoint_contract='wrist_world_v2')
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.object_body, self.object_geom = self.model.body('test_cylinder').id, self.model.geom('cylinder_geom').id
        self.table_geom, self.floor_geom = self.model.geom('tabletop_geom').id, self.model.geom('floor').id
        self.base = self.model.body('base_link').id
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.wrist_bodies = {s: self.model.body(f'{s}_hand_roll_link').id for s in SIDES}
        self.robot_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.geom_sides = {g: s for g in self.robot_geoms for s, wrist in self.wrist_bodies.items()
                           if _descendant(self.model, int(self.model.geom_bodyid[g]), wrist)}
        self.table_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.model.body('tabletop').id)}
        self.hand_joints = np.array([self.model.joint(n).id for n in urdf_hand_joints()])
        self.mimics = []
        for name, element in urdf_hand_joints().items():
            mimic = element.find('mimic')
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0], self.model.joint(mimic.get('joint')).qposadr[0],
                    float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))))
        self.diagnostic_model = copy.copy(self.model)
        for g in self.table_geoms:
            self.diagnostic_model.geom_contype[g] = self.diagnostic_model.geom_conaffinity[g] = 1
        self.diagnostic_data = mujoco.MjData(self.diagnostic_model)
        self.path = (front_world_path_for_scene if frontal else world_path_for_scene)(
            self.config['table_height_m'], self.config['can_xy_m'], self.config['yaw_deg'], self.calibration)
        self.path_targets = {item['name'].upper(): item['T_world_wrist_goal'].copy() for item in self.path}
        if tuple(self.path_targets) != PATH_NAMES:
            raise ValueError('Unexpected calibration stage list')
        delta = self.path_targets['TRANSLATE'][:3, 3]-self.path_targets['LIFT'][:3, 3]
        self.place_xy = np.asarray(self.config['can_xy_m'])+delta[:2]
        self.layout['cylinder_place_position'] = np.r_[self.place_xy, self.config['table_height_m']+.06]
        # Every planned object landing must fit on the tabletop before any run.
        for xy in (self.config['can_xy_m'], self.place_xy):
            if np.any(np.abs(np.asarray(xy)-self.config['table_center_xy_m'])+.02 > np.asarray(self.config['table_half_size_m'])[:2]):
                raise ValueError('Initial/placement cylinder footprint falls outside configured tabletop')
        self.weight_N, self.table_height = self.profile['mass_kg']*9.81, self.config['table_height_m']
        self.phase, self.phase_start, self.steps = 'RESET_SETTLE', 0., 0
        self.failure = self.failure_phase = None
        self.grasp_verified = self.release_commanded = self.completed = False
        self.baseline_relation = self.alignment_relation = None
        self.upright_attempted = False
        self.max_verified_hold_s = 0.
        self.stable_since = self.bad_since = None
        self.keep_trace = bool(keep_trace)
        self.samples, self.targets = [], []
        self.transitions = [dict(time_s=0., state=self.phase)]
        self.stage_results, self.safety_violations_seen = [], []
        self.feedback_target_checks = []
        self.stage_peak = dict(position_m=0., orientation_deg=0., right_position_m=0., right_orientation_deg=0.)
        self.stage_collision = False
        self.right_locked = False
        self.inactive_bad_since = None
        self.initial_object_position = self.data.xpos[self.object_body].copy()
        self.goal_wrist_transforms = {s: body_transform(self.data, w) for s, w in self.wrist_bodies.items()}
        self.active_goal = self.goal_wrist_transforms['left'].copy()
        self.foot_anchor = np.array([self.data.xpos[self.model.body(f'{s}_ankle_roll_link').id] for s in SIDES])
        self.peaks = dict.fromkeys(('hand_object_penetration_m', 'hand_self_penetration_m', 'hand_joint_violation_rad',
            'mimic_error_rad', 'actuator_torque_Nm', 'object_tilt_deg', 'clearance_m', 'grasp_slip_m',
            'grasp_rotation_slip_deg', 'base_tilt_deg', 'foot_drift_m', 'body_joint_violation_rad',
            'robot_self_penetration_m', 'robot_table_penetration_m'), 0.)
        self.current_metrics = {}
        self.sync()
        self.initial_metrics = copy.deepcopy(self.current_metrics)
        self._safety()
        self.record()

    def _set_goal(self, side, target):
        target = np.asarray(target).copy()
        self.goal_wrist_transforms[side] = target
        self.policy.set_wrist_target_world(side, target[:3, 3], pose_quaternion(target))
        if side == 'left':
            self.active_goal = target

    def enter(self, phase, target=None, duration=0.):
        if self.mode == 'contact' and phase in ('UPRIGHT', 'PLACE') and target is not None:
            # AIR tests nominal unloaded poses, not every feedback pose. Keep
            # this first transfer inside a small explicit guard; even passing
            # this guard is not a collision-free or reachability certificate.
            actual, nominal = np.asarray(target, dtype=float), self.path_targets[phase]
            valid = (actual.shape == (4, 4) and np.isfinite(actual).all()
                and np.allclose(actual[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
                and np.allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-7, rtol=0.)
                and np.isclose(np.linalg.det(actual[:3, :3]), 1., atol=1e-7, rtol=0.))
            distance = float(np.linalg.norm(actual[:3, 3]-nominal[:3, 3])) if valid else None
            angle = rotation_error_deg(actual[:3, :3], nominal[:3, :3]) if valid else None
            accepted = bool(valid and distance <= .02 and angle <= 5.)
            self.feedback_target_checks.append(dict(time_s=float(self.data.time), phase=phase,
                position_deviation_m=distance, orientation_deviation_deg=angle, accepted=accepted,
                maximum_position_deviation_m=.02, maximum_orientation_deviation_deg=5.,
                is_reachability_or_collision_certificate=False))
            if not accepted:
                self.fail(f'{phase} feedback target exceeds nominal AIR pose guard (2 cm / 5 deg)')
                return
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since = self.bad_since = None
        self.stage_collision = False
        self.stage_peak = dict(position_m=0., orientation_deg=0., right_position_m=0., right_orientation_deg=0.)
        self.transitions.append(dict(time_s=float(self.data.time), state=phase))
        if target is not None:
            self._set_goal('left', target)
        if phase in ('CLOSE', 'OPEN'):
            if self.mode != 'contact':
                self.fail('AIR may never command a grasp or release'); return
            self.hands.command('left', int(phase == 'CLOSE'))
            self.release_commanded |= phase == 'OPEN'
        self.targets.append(dict(time_s=float(self.data.time), phase=phase,
            T_world_wrist_goal={s: x.tolist() for s, x in self.goal_wrist_transforms.items()},
            hand_commands={s: c.command for s, c in self.hands.controllers.items()}))

    def _pose_good(self, loose=False):
        e = self.current_metrics['wrist_errors']
        return (at_wrist_goal(e['left'], .025 if loose else .005, 7. if loose else 3., .05 if loose else .02)
            and at_wrist_goal(e['right'], .02, 5., .05)
            and not self.stage_collision)

    def _finish_air_stage(self, passed):
        self.stage_results.append(dict(name=self.phase, passed=bool(passed),
            duration_s=float(self.data.time-self.phase_start),
            settled_duration_s=(0. if self.stable_since is None else float(self.data.time-self.stable_since)),
            standing_final_base_tilt_deg=(self.current_metrics['base_tilt_deg'] if self.phase == 'STAND' else None),
            standing_final_root_speed=(float(np.linalg.norm(self.data.qvel[:6])) if self.phase == 'STAND' else None),
            final_errors=copy.deepcopy(self.current_metrics['wrist_errors']), peaks=self.stage_peak.copy(),
            possible_table_collision=self.stage_collision))

    def _next_air(self):
        index = PATH_NAMES.index(self.phase)
        if index+1 == len(PATH_NAMES):
            self.completed = True; self.enter('COMPLETE')
        else:
            name = PATH_NAMES[index+1]
            self.enter(name, self.path_targets[name])

    def _gate(self):
        t, m = float(self.data.time-self.phase_start), self.current_metrics
        if self.phase == 'RESET_SETTLE':
            if t < .4-1e-8:
                self.policy.follow_current(self.scratch)
                for s in SIDES:
                    self.goal_wrist_transforms[s] = body_transform(self.scratch, self.wrist_bodies[s])
                self.active_goal = self.goal_wrist_transforms['left'].copy()
            else:
                for s, y in (('left', .19698), ('right', -.19643)):
                    target = np.eye(4); target[:3, 3] = [.246, y, 1.087902013081883]
                    self._set_goal(s, target)
                self.enter('HOME')
            return
        if self.phase in ('HOME', 'SAFE_OUT', 'TURN_WRIST') or self.mode == 'air' and self.phase in PATH_NAMES:
            passed = self._stable(self._pose_good(loose=self.phase == 'HOME'), .3)
            timed_out = t >= self.config['stage_timeout_s']-1e-8
            if not passed and not timed_out:
                return
            self._finish_air_stage(passed)
            if not passed and (self.mode == 'contact' or not self.config['air_continue_on_precision_failure']):
                self.fail(f'{self.phase} did not meet actual wrist/collision gates'); return
            if self.phase == 'HOME':
                self.enter('STAND')
            elif self.phase == 'SAFE_OUT':
                turn = self.active_goal.copy(); turn[:3, :3] = self.path_targets['HOVER'][:3, :3]
                self.enter('TURN_WRIST', turn)
            elif self.phase == 'TURN_WRIST':
                self.enter('HOVER', self.path_targets['HOVER'])
            else:
                self._next_air()
            return
        if self.phase == 'STAND':
            if t >= 20.-1e-8:
                stable = m['base_tilt_deg'] < 8. and np.linalg.norm(self.data.qvel[:6]) < .2 and not self.stage_collision
                self._finish_air_stage(stable)
                if not stable and self.mode == 'contact':
                    self.fail('Standing did not settle safely'); return
                self._set_goal('right', body_transform(self.scratch, self.wrist_bodies['right']))
                self.right_locked = True
                if self.config.get('grasp_style', 'upper') == 'front':
                    # Near-HOME exterior alignment; TURN_WRIST retains the
                    # legacy phase name but holds the same upright rotation.
                    target = self.path_targets['HOVER'].copy()
                else:
                    target = np.eye(4); target[:3, 3] = [.16, .36, max(1.10, self.path_targets['HOVER'][2, 3])]
                self.enter('SAFE_OUT', target)
            return
        if self.mode == 'air':
            self.fail(f'Unexpected AIR phase {self.phase}'); return
        # CONTACT progression is based on actual endpoints, not a replay clock.
        if self.phase in ('HOVER', 'APPROACH', 'GRASP'):
            if self._stable(self._pose_good(), .3):
                next_name = {'HOVER': 'APPROACH', 'APPROACH': 'GRASP', 'GRASP': 'CLOSE'}[self.phase]
                self.enter(next_name, self.path_targets.get(next_name))
        elif self.phase == 'CLOSE':
            if self._stable(t >= 2.5 and m['opposed_contact'] and self._pose_good(), .3):
                self.enter('PROBE', self._offset_target([0., 0., .02]))
        elif self.phase == 'PROBE':
            if self._stable(self._pose_good(), .3):
                self.enter('VERIFY')
        elif self.phase == 'VERIFY':
            valid = pickup_conditions(m, self.weight_N) and self._pose_good()
            if valid and self.baseline_relation is None:
                self.baseline_relation = np.asarray(m['T_wrist_object']).copy()
                self.sync(); m = self.current_metrics
            valid = valid and m['grasp_slip_m'] is not None and m['grasp_slip_m'] < .015 and m['grasp_rotation_slip_deg'] < 5.
            if self._stable(valid, 1.):
                self.grasp_verified = True
                self.enter('LIFT', self._offset_target([0., 0., .06]))
            elif not self.upright_attempted and t >= 1. and m['object_tilt_deg'] > 10. and unsupported_load_conditions(m, self.weight_N):
                self.upright_attempted = True
                self.alignment_relation = np.asarray(m['T_wrist_object']).copy()
                desired = np.eye(4); desired[:3, 3] = m['object_position_m']
                self.enter('UPRIGHT', desired @ np.linalg.inv(self.alignment_relation))
        elif self.phase == 'UPRIGHT':
            if self._stable(self._pose_good(), .3): self.enter('VERIFY')
        elif self.phase in ('LIFT', 'TRANSLATE'):
            if self._stable(pickup_conditions(m, self.weight_N) and self._pose_good(), 1.):
                if self.phase == 'LIFT':
                    delta = self.path_targets['TRANSLATE'][:3, 3]-self.path_targets['LIFT'][:3, 3]
                    self.enter('TRANSLATE', self._offset_target(delta))
                else:
                    desired = np.eye(4); desired[:3, 3] = [*self.place_xy, self.table_height+.06]
                    self.placement_relation = np.asarray(m['T_wrist_object']).copy()
                    self.enter('PLACE', desired @ np.linalg.inv(self.placement_relation))
        elif self.phase == 'PLACE':
            valid = (m['table_contact'] and m['table_vertical_force_N'] > .05 and m['object_tilt_deg'] <= 10.
                and m['object_linear_speed_mps'] < .02 and m['object_angular_speed_radps'] < .15
                and np.linalg.norm(np.asarray(m['object_position_m'])[:2]-self.place_xy) <= .02)
            if self._stable(valid, .3): self.enter('OPEN')
        elif self.phase == 'OPEN':
            if self._stable(t >= 2.5 and placed_conditions(m, self.weight_N, self.place_xy) and m['active_hand_object_contact_count'] == 0, .3):
                offset = FRONT_RETREAT_OFFSET_M if self.config.get('grasp_style', 'upper') == 'front' else [0., 0., .10]
                self.enter('RETREAT', self._offset_target(offset))
        elif self.phase == 'RETREAT':
            if self._stable(self._pose_good() and placed_conditions(m, self.weight_N, self.place_xy) and m['active_hand_object_contact_count'] == 0, 1.):
                self.completed = True; self.enter('COMPLETE')
        if not self.done and self.data.time-self.phase_start >= self.config['stage_timeout_s']-1e-8:
            self.fail(f'{self.phase} exceeded actual pose/contact gate deadline')

    def sync(self):
        TopGraspExperiment.sync(self)
        m, d, metrics = self.model, self.scratch, self.current_metrics
        errors = {}
        for s, body in self.wrist_bodies.items():
            actual, goal = body_transform(d, body), self.goal_wrist_transforms[s]
            v, w = self.policy.wrist_twist(d, s)
            errors[s] = dict(position_m=float(np.linalg.norm(actual[:3, 3]-goal[:3, 3])),
                orientation_deg=rotation_error_deg(actual[:3, :3], goal[:3, :3]),
                linear_speed_mps=float(np.linalg.norm(v)), angular_speed_radps=float(np.linalg.norm(w)),
                T_world_wrist=actual.tolist(), T_world_wrist_goal=goal.tolist())
        self.diagnostic_data.qpos[:] = d.qpos
        self.diagnostic_data.qvel[:] = d.qvel
        self.diagnostic_data.ctrl[:] = d.ctrl
        mujoco.mj_forward(self.diagnostic_model, self.diagnostic_data)
        table_depth, table_count, nonhand = 0., 0, 0
        for c in self.diagnostic_data.contact:
            a, b = map(int, c.geom)
            if c.dist < 0 and ((a in self.robot_geoms and b in self.table_geoms) or (b in self.robot_geoms and a in self.table_geoms)):
                table_depth = max(table_depth, -float(c.dist)); table_count += 1
        for c in d.contact:
            a, b = map(int, c.geom)
            if c.dist <= 0 and self.object_geom in (a, b):
                other = b if a == self.object_geom else a
                if other in self.robot_geoms and other not in self.geom_sides:
                    nonhand += 1
        feet = np.array([d.xpos[m.body(f'{s}_ankle_roll_link').id] for s in SIDES])
        metrics.update(wrist_errors=errors,
            base_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1., 1.)))),
            foot_drift_m=float(np.max(np.linalg.norm(feet-self.foot_anchor, axis=1))),
            robot_table_penetration_m=table_depth, robot_table_contact_count=table_count,
            nonhand_object_contact_count=nonhand, mode=self.mode,
            contact_physics_evaluated=self.mode == 'contact', grasp_evaluated=self.mode == 'contact')
        if table_count:
            self.stage_collision = True
        for key in ('position_m', 'orientation_deg'):
            self.stage_peak[key] = max(self.stage_peak[key], errors['left'][key])
            self.stage_peak['right_'+key] = max(self.stage_peak['right_'+key], errors['right'][key])
        return metrics

    def _safety(self):
        d, m = self.data, self.model
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number):
            self.fail('Nonfinite state or MuJoCo warning'); return
        if m.nmocap or np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD) or np.any(d.xfrc_applied) or np.any(d.qfrc_applied):
            self.fail('Unexpected support fixture or external force'); return
        if self.mode == 'air' and any(c.command != 0 for c in self.hands.controllers.values()):
            self.fail('AIR must keep both hand commands open'); return
        q, limits = d.qpos[self.body_map.qpos], m.jnt_range[self.body_map.joints]
        violation = float(max(0., np.max(limits[:, 0]-q), np.max(q-limits[:, 1])))
        self.peaks['body_joint_violation_rad'] = max(self.peaks['body_joint_violation_rad'], violation)
        if violation > .05:
            excess = np.maximum(limits[:, 0]-q, q-limits[:, 1])
            i = int(np.argmax(excess))
            self.fail(f'Measured body joint exceeded physical limit tolerance: {BODY_JOINTS[i]} '
                f'q={q[i]:.5f} rad, limit=[{limits[i, 0]:.5f}, {limits[i, 1]:.5f}], '
                f'excess={violation:.5f} rad'); return
        depth = 0.
        for c in d.contact:
            if c.dist > 0: continue
            a, b = map(int, c.geom)
            if self.floor_geom in (a, b):
                other = b if a == self.floor_geom else a
                if other in self.robot_geoms and other not in self.foot_geoms:
                    self.fail(f'Non-foot ground contact: {m.geom(other).name}'); return
            if a in self.robot_geoms and b in self.robot_geoms:
                depth = max(depth, -float(c.dist))
        self.peaks['robot_self_penetration_m'] = max(self.peaks['robot_self_penetration_m'], depth)
        if depth > .005:
            self.fail('Deep robot self-contact'); return
        metrics = self.current_metrics
        for key in self.peaks:
            if metrics.get(key) is not None:
                self.peaks[key] = max(self.peaks[key], float(metrics[key]))
        if metrics['base_tilt_deg'] > 35. or d.xpos[self.base, 2] < .55:
            self.fail('Base tilt/height safety limit')
        elif metrics['hand_joint_violation_rad'] > .01 or metrics['mimic_error_rad'] > .015:
            self.fail('Finger joint limit or mimic constraint limit')
        elif self.mode == 'contact':
            right = metrics['wrist_errors']['right']
            if self.right_locked and (right['position_m'] > .02 or right['orientation_deg'] > 5.):
                if self.inactive_bad_since is None:
                    self.inactive_bad_since = float(d.time)
                if d.time-self.inactive_bad_since >= .3:
                    self.fail('Inactive wrist left its locked world-pose tolerance for 0.3 s'); return
            else:
                self.inactive_bad_since = None
            if metrics['robot_table_contact_count'] or metrics['nonhand_object_contact_count'] or metrics['parked_hand_contact_count']:
                self.fail('Unexpected robot/table or non-grasping-part/object contact')
            elif self.phase in ('RESET_SETTLE', 'HOME', 'STAND', 'SAFE_OUT', 'TURN_WRIST', 'HOVER') and metrics['active_hand_object_contact_count']:
                self.fail('Premature hand/object contact before approach')
            else:
                TopGraspExperiment._safety(self)
                if self.grasp_verified and self.phase == 'PLACE' and not metrics['table_contact']:
                    if not (metrics['opposed_contact'] and metrics['object_tilt_deg'] <= 10.
                            and metrics['grasp_slip_m'] is not None and metrics['grasp_slip_m'] < .015 and metrics['grasp_rotation_slip_deg'] < 5.):
                        self.fail('Grasp lost in placement; no airborne release')

    def fail(self, reason):
        if self.done: return
        self.safety_violations_seen.append(dict(time_s=float(self.data.time), phase=self.phase, reason=str(reason)))
        TopGraspExperiment.fail(self, reason)

    def record(self):
        if not self.keep_trace: return
        TopGraspExperiment.record(self)
        self.samples[-1].update(body_action=self.policy.last_action.tolist(),
            body_target_q_rad=self.policy.q_des.tolist(),
            hand_commands={s: c.command for s, c in self.hands.controllers.items()})

    def step(self):
        if self.done: return
        if self.steps % 20 == 0:
            self.sync(); self._safety()
            if self.done: self.record(); return
            self._gate()
            if self.done: self.record(); return
            self.policy.act(self.scratch)
        if self.steps % 10 == 0: self.hands.update()
        self.policy.apply(self.data); self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self._safety()
        if self.steps % 10 == 0 or self.done:
            self.sync(); self._safety(); self.record()

    def report(self):
        air_passed = bool(self.mode == 'air' and self.completed and not self.failure
            and [s['name'] for s in self.stage_results] == list(AIR_STAGES) and all(s['passed'] for s in self.stage_results))
        report = TopGraspExperiment.report(self)
        report.update(scope='Unassisted whole-body Reach migration; AIR is not physical grasp evidence',
            mode=self.mode, success=air_passed if self.mode == 'air' else report['success'],
            air_passed=air_passed, stage_results=self.stage_results,
            safety_violations_seen=self.safety_violations_seen,
            feedback_target_checks=self.feedback_target_checks, **self.input_bindings,
            config=self.config, config_sha256=fingerprint(self.config), calibration_sha256=self.calibration_sha,
            onnx_sha256=self.policy_sha, checkpoint_sha256=self.parity['checkpoint_sha256'],
            controller_sha256=sha256(__file__), adapter_sha256=sha256(PROJECT_ROOT/'common/r2v2_reach_policy.py'),
            endpoint_contract='wrist_world_v2', training_task=self.parity['task'],
            novel_task=f"{self.config['grasp_style']}_can_grasp_transfer", training_path_claimed_as_grasp_evidence=False,
            wrist_fixture=False, object_free=self.mode == 'contact',
            timing_hz=dict(physics=1000, body_policy=50, hand_reference=100, metrics=100))
        report.update(controller_sha256=self.controller_sha, adapter_sha256=self.adapter_sha,
            grasp_style=self.config['grasp_style'],
            active_stage_diagnostics=dict(name=self.failure_phase or self.phase,
                possible_table_collision=self.stage_collision, peaks=self.stage_peak.copy()),
            acceptance=dict(active_wrist_position_m=.005, active_wrist_orientation_deg=3.,
                active_wrist_linear_speed_mps=.02, stable_s=.3,
                inactive_wrist_position_m=.02, inactive_wrist_orientation_deg=5.,
                standing_duration_s=20., motion_deadline_s=self.config['stage_timeout_s']))
        if self.config['grasp_style'] == 'front':
            report['candidate'] = dict(side='left', grasp_style='front',
                wrist_cylinder_initial=self.calibration['T_wrist_cylinder_initial'],
                source_record_sha256=self.calibration['source_record_sha256'],
                transport_path_is_recorded=False)
            report['scope'] = 'Unassisted frontal Reach migration; AIR is not physical grasp evidence'
        return report


__all__ = ['TopGraspFullbodyExperiment', 'build_fullbody_model', 'load_config',
           'at_wrist_goal', 'require_air_evidence', 'fingerprint', 'DEFAULT_CONFIG', 'AIR_STAGES']
