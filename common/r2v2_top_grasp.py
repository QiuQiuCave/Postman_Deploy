"""Single-hand overhead pick/translate/place probe with a driven wrist fixture.

The wrist alone is externally supported. The cylinder is a free rigid body;
its measured state is never written after initialization. This tests a hand
grasp, not full-body Reach or real-robot readiness.
"""
from __future__ import annotations

from dataclasses import asdict
import copy

import mujoco
import numpy as np

from common.r2v2_cylinder_test import load_cylinder_profile
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_top_grasp_scene import TopGraspCandidate, build_top_grasp_model
from r2v2_description.model import SIDES, initialize_hands, urdf_hand_joints


def blend(value):
    x = float(np.clip(value, 0., 1.))
    return x*x*x*(10.+x*(-15.+6.*x))


def rotation_error_deg(a, b):
    return float(np.rad2deg(np.arccos(np.clip((np.trace(a @ b.T)-1.)/2., -1., 1.))))


def pose_quaternion(transform):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(transform[:3, :3]).ravel())
    return result


def interpolate_pose(start, end, fraction):
    x = blend(fraction)
    result = np.eye(4)
    result[:3, 3] = (1.-x)*start[:3, 3]+x*end[:3, 3]
    q1, q2 = pose_quaternion(start), pose_quaternion(end)
    if q1 @ q2 < 0:
        q2 = -q2
    dot = np.clip(q1 @ q2, -1., 1.)
    if dot > .9995:
        q = (1.-x)*q1+x*q2
        q /= np.linalg.norm(q)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1.-x)*angle)*q1+np.sin(x*angle)*q2)/np.sin(angle)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q)
    result[:3, :3] = matrix.reshape(3, 3)
    return result


def contact_opposition_angle(thumb_normals, finger_normals):
    """Largest cross-group angle of normals acting on the object, in degrees.

    Directional opposition is a useful filter, not a force-closure proof.
    Both sets must use the same frame and the same force-on-object convention.
    """
    groups = []
    for values in (thumb_normals, finger_normals):
        normals = np.asarray(values, dtype=float)
        if normals.shape == (0,):
            normals = normals.reshape(0, 3)
        if normals.ndim != 2 or normals.shape[1] != 3 or not np.all(np.isfinite(normals)):
            raise ValueError('Contact normals must be finite arrays of shape (N, 3)')
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError('Contact normals must be nonzero')
        groups.append(normals/lengths[:, None])
    if any(len(group) == 0 for group in groups):
        return None
    return float(np.rad2deg(np.arccos(np.clip(np.min(groups[0] @ groups[1].T), -1., 1.))))


def unsupported_load_conditions(metrics, weight_N):
    """Real, stationary load support before any optional upright correction."""
    m = metrics
    return bool(m['finite_state'] and m['opposed_contact']
        and m['clearance_m'] >= .008 and not m['table_contact'] and not m['floor_contact']
        and m['hand_table_contact_count'] == 0 and m['parked_hand_contact_count'] == 0
        and m['hand_vertical_force_N'] >= .8*weight_N
        and m['object_tilt_deg'] <= 30.
        and m['object_linear_speed_mps'] < .02 and m['object_angular_speed_radps'] < .15)


def pickup_conditions(metrics, weight_N):
    """Final grasp gate stays upright; tentative support is not verified success."""
    return unsupported_load_conditions(metrics, weight_N) and metrics['object_tilt_deg'] <= 10.


def placed_conditions(metrics, weight_N, target_xy):
    m = metrics
    return bool(m['finite_state'] and m['table_contact'] and m['table_vertical_force_N'] >= .8*weight_N
        and not m['floor_contact'] and m['object_tilt_deg'] <= 10.
        and m['object_linear_speed_mps'] < .02 and m['object_angular_speed_radps'] < .15
        and np.linalg.norm(np.asarray(m['object_position_m'])[:2]-target_xy) <= .02)


class TopGraspExperiment:
    """Physical, success-gated sequence; every phase has a finite deadline."""

    def __init__(self, profile='baseline_40mm_100g', candidate=None, keep_trace=True):
        self.profile = load_cylinder_profile(profile)
        self.candidate = candidate or TopGraspCandidate()
        self.side = self.candidate.side
        self.keep_trace = bool(keep_trace)
        self.model, self.hand_cfg, self.layout = build_top_grasp_model(
            profile=self.profile, candidate=self.candidate)
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_hands(self.model, self.data, self.hand_cfg)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.object_body = self.model.body('test_cylinder').id
        self.object_geom = self.model.geom('cylinder_geom').id
        self.table_geom, self.floor_geom = self.model.geom('table_geom').id, self.model.geom('floor').id
        self.wrist_bodies = {s: self.model.body(f'{s}_hand_roll_link').id for s in SIDES}
        self.geom_sides = {}
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body:
                for side, wrist in self.wrist_bodies.items():
                    if body == wrist:
                        self.geom_sides[geom] = side
                body = int(self.model.body_parentid[body])
        self.hand_joints = np.array([self.model.joint(n).id for n in urdf_hand_joints()])
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find('mimic')
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0],
                    self.model.joint(mimic.get('joint')).qposadr[0],
                    float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))))
        self.weight_N = self.profile['mass_kg']*9.81
        self.table_height = float(self.layout['table_top_m'])
        self.phase, self.failure, self.failure_phase = 'READY', None, None
        self.phase_start, self.steps = 0., 0
        self.samples, self.transitions, self.targets = [], [dict(time_s=0., state='READY')], []
        self.baseline_relation = None
        self.alignment_relation = None
        self.upright_attempted = False
        self.grasp_verified = False
        self.release_commanded = False
        self.completed = False
        self.stable_since, self.bad_since = None, None
        self.max_verified_hold_s = 0.
        self.initial_object_position = self.data.xpos[self.object_body].copy()
        self.place_xy = self.initial_object_position[:2]+np.array([.10, 0.])
        self.active_goal = body_transform(self.data, self.wrist_bodies[self.side])
        self.move_start, self.move_end = self.active_goal.copy(), self.active_goal.copy()
        self.motion_duration = 0.
        self.grasp_pose = self.active_goal.copy()
        self.grasp_pose[:3, 3] = np.asarray(self.layout['grasp_wrist_position'])
        self.peaks = dict(hand_object_penetration_m=0., hand_self_penetration_m=0.,
            hand_joint_violation_rad=0., mimic_error_rad=0., actuator_torque_Nm=0.,
            object_tilt_deg=0., clearance_m=0., grasp_slip_m=0., grasp_rotation_slip_deg=0.)
        self.current_metrics = {}
        self.sync()
        self.initial_metrics = copy.deepcopy(self.current_metrics)
        if self.current_metrics['hand_object_penetration_m'] > .001 or self.current_metrics['hand_table_contact_count']:
            self.fail('Initial hand/object or hand/table overlap')
        if self.layout.get('geometry', {}).get('valid_pregrasp') is False:
            self.fail('Nominal pregrasp geometry overlaps object or table')
        self._safety()
        self.record()

    @property
    def done(self):
        return self.phase in ('COMPLETE', 'FAILED')

    def fail(self, reason):
        if self.done:
            return
        self.failure, self.failure_phase = str(reason), self.phase
        self.phase = 'FAILED'
        self.transitions.append(dict(time_s=float(self.data.time), state='FAILED', reason=str(reason)))

    def _stable(self, valid, duration):
        if not valid:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = float(self.data.time)
        return self.data.time-self.stable_since >= duration-1e-8

    def enter(self, phase, target=None, duration=0.):
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since, self.bad_since = None, None
        self.transitions.append(dict(time_s=float(self.data.time), state=phase))
        if target is not None:
            self.move_start, self.move_end = self.active_goal.copy(), np.asarray(target).copy()
            self.motion_duration = float(duration)
        else:
            self.move_start = self.move_end = self.active_goal.copy()
            self.motion_duration = 0.
        if phase == 'CLOSE':
            self.hands.command(self.side, 1)
        elif phase == 'OPEN':
            self.hands.command(self.side, 0)
            self.release_commanded = True

    def _offset_target(self, xyz):
        target = self.active_goal.copy()
        target[:3, 3] += xyz
        return target

    def _gate(self):
        t, m = self.data.time-self.phase_start, self.current_metrics
        if t > 10.:
            self.fail(f'{self.phase} exceeded 10 s deadline'); return
        if self.phase == 'READY' and t >= .5-1e-8:
            self.enter('APPROACH', self.grasp_pose, 3.)
        elif self.phase == 'APPROACH' and t >= 3.-1e-8:
            self.enter('SETTLE')
        elif self.phase == 'SETTLE' and t >= .5-1e-8:
            self.enter('CLOSE')
        elif self.phase == 'CLOSE':
            if self._stable(t >= 2.5 and m['opposed_contact'], .3):
                self.enter('PROBE_LIFT', self._offset_target([0., 0., .02]), 1.5)
            elif t >= 5.:
                self.fail('No sustained thumb/opposing-finger contact after closure')
        elif self.phase == 'PROBE_LIFT' and t >= 1.5-1e-8:
            self.enter('VERIFY')
        elif self.phase == 'VERIFY':
            valid = pickup_conditions(m, self.weight_N)
            if valid and self.baseline_relation is None:
                self.baseline_relation = np.asarray(m['T_wrist_object']).copy()
                self.sync(); m = self.current_metrics
            valid = valid and m['grasp_slip_m'] is not None and m['grasp_slip_m'] < .015 and m['grasp_rotation_slip_deg'] < 5.
            if self._stable(valid, 1.):
                self.grasp_verified = True
                self.max_verified_hold_s = max(self.max_verified_hold_s, self.data.time-self.stable_since)
                self.enter('LIFT', self._offset_target([0., 0., .06]), 2.)
            elif (not self.upright_attempted and t >= 1. and m['object_tilt_deg'] > 10.
                    and unsupported_load_conditions(m, self.weight_N)):
                # A physically supported but tilted pickup is not yet a pass.
                # Reorient the wrist using the measured free-object relation;
                # the object's pose is never commanded or assigned.
                self.upright_attempted = True
                self.alignment_relation = np.asarray(m['T_wrist_object']).copy()
                desired_object = np.eye(4)
                desired_object[:3, 3] = np.asarray(m['object_position_m'])
                self.enter('UPRIGHT', desired_object @ np.linalg.inv(self.alignment_relation), 2.)
            elif t >= 3.:
                self.fail('Probe did not establish a stable unsupported grasp')
        elif self.phase == 'UPRIGHT' and t >= 2.-1e-8:
            self.enter('VERIFY')
        elif self.phase == 'LIFT' and t >= 2.-1e-8:
            self.enter('HOLD_LIFT')
        elif self.phase == 'HOLD_LIFT':
            if self._stable(pickup_conditions(m, self.weight_N), 1.):
                self.enter('TRANSLATE', self._offset_target([.10, 0., 0.]), 2.5)
            elif t >= 3.:
                self.fail('Unstable raised grasp')
        elif self.phase == 'TRANSLATE' and t >= 2.5-1e-8:
            self.enter('HOLD_MOVED')
        elif self.phase == 'HOLD_MOVED':
            if self._stable(pickup_conditions(m, self.weight_N), 1.):
                # Use the actually held object relation, not the nominal open
                # hand calibration, to put the cylinder upright on the table.
                desired_object = np.eye(4)
                desired_object[:3, 3] = [*self.place_xy, self.table_height+self.profile['height_m']/2.]
                held_relation = np.asarray(m['T_wrist_object']).copy()
                self.placement_relation = held_relation
                self.enter('LOWER', desired_object @ np.linalg.inv(held_relation), 3.)
            elif t >= 3.:
                self.fail('Unstable translated grasp')
        elif self.phase == 'LOWER':
            if m['table_contact'] and t >= .5:
                self.enter('PLACE_SETTLE')
            elif t >= 3.-1e-8:
                self.enter('PLACE_SETTLE')
        elif self.phase == 'PLACE_SETTLE':
            # Confirm geometric support and low motion before opening. The
            # fingers can still share weight while they remain closed.
            supported = (m['table_contact'] and m['table_vertical_force_N'] > .05
                and not m['floor_contact'] and m['object_tilt_deg'] <= 10.
                and m['object_linear_speed_mps'] < .02 and m['object_angular_speed_radps'] < .15
                and np.linalg.norm(np.asarray(m['object_position_m'])[:2]-self.place_xy) <= .02)
            if self._stable(supported, .3):
                self.enter('OPEN')
            elif t >= 3.:
                self.fail('No stable table support at placement target; hand remains closed')
        elif self.phase == 'OPEN':
            clear = m['active_hand_object_contact_count'] == 0
            if self._stable(t >= 2.5 and clear and placed_conditions(m, self.weight_N, self.place_xy), .3):
                self.enter('RETREAT', self._offset_target([0., 0., .10]), 2.)
            elif t >= 5.:
                self.fail('Hand did not release a stable placed object')
        elif self.phase == 'RETREAT' and t >= 2.-1e-8:
            self.enter('FINAL_HOLD')
        elif self.phase == 'FINAL_HOLD':
            if self._stable(placed_conditions(m, self.weight_N, self.place_xy)
                    and m['active_hand_object_contact_count'] == 0, 1.):
                self.completed = True
                self.enter('COMPLETE')
            elif t >= 3.:
                self.fail('Placed object did not remain stable after retreat')

    def _safety(self):
        m = self.current_metrics
        for name in self.peaks:
            value = m.get(name)
            if value is not None:
                self.peaks[name] = max(self.peaks[name], float(value))
        if not m['finite_state'] or np.any(self.data.warning.number):
            self.fail('Nonfinite state or MuJoCo warning')
        elif np.any(self.data.xfrc_applied) or np.any(self.data.qfrc_applied):
            self.fail('Unexpected external force on experiment')
        elif m['hand_table_contact_count'] or m['parked_hand_contact_count']:
            self.fail('Hand hit table or parked hand contacted object')
        elif m['floor_contact']:
            self.fail('Object fell to floor')
        elif m['hand_object_penetration_m'] > .003 or m['hand_self_penetration_m'] > .003:
            self.fail('Hand/object or hand/self penetration exceeded 3 mm')
        elif m['hand_joint_violation_rad'] > .01 or m['mimic_error_rad'] > .015:
            self.fail('Hand joint-limit or mimic residual exceeded baseline tolerance')
        elif m['object_tilt_deg'] > 30.:
            self.fail('Object tilted beyond exploratory 30 deg stop')
        transporting = self.phase in ('LIFT', 'HOLD_LIFT', 'TRANSLATE', 'HOLD_MOVED')
        lowering_in_air = self.phase in ('LOWER', 'PLACE_SETTLE') and not m['table_contact']
        if self.grasp_verified and (transporting or lowering_in_air):
            retained = (m['opposed_contact'] and not m['table_contact'] and not m['floor_contact']
                and m['grasp_slip_m'] is not None and m['grasp_slip_m'] < .015
                and m['grasp_rotation_slip_deg'] < 5. and m['object_tilt_deg'] <= 10.)
            if not retained:
                self.fail('Grasp lost/slipped during transport; no automatic opening')
        if self.phase == 'UPRIGHT':
            retained = (m['opposed_contact'] and not m['table_contact'] and not m['floor_contact']
                and m['clearance_m'] >= .008
                and m.get('alignment_slip_m') is not None and m['alignment_slip_m'] < .015
                and m['alignment_rotation_slip_deg'] < 5.)
            if not retained:
                self.fail('Tentative grasp lost during upright correction; no automatic opening')

    def sync(self):
        # Deployment's MuJoCo Python binding has no mj_copyData. Only copy
        # input state into a separate diagnostic MjData; never alter physics.
        for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat', 'qacc_warmstart',
                     'xfrc_applied', 'qfrc_applied'):
            getattr(self.scratch, name)[:] = getattr(self.data, name)
        self.scratch.time = self.data.time
        mujoco.mj_forward(self.model, self.scratch)
        m, d = self.model, self.scratch
        object_T = body_transform(d, self.object_body)
        wrist_T = body_transform(d, self.wrist_bodies[self.side])
        relation = np.linalg.inv(wrist_T) @ object_T
        rotation = object_T[:3, :3]
        axis_z = float(rotation[2, 2])
        # Exact support function of an oriented solid cylinder, not just its
        # centre-minus-half-height (which is wrong when it tilts).
        half_extent = self.profile['height_m']/2.*abs(axis_z)+self.profile['radius_m']*np.sqrt(max(0., 1.-axis_z*axis_z))
        bottom = float(object_T[2, 3]-half_extent)
        jac_p, jac_r = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jac_p, jac_r, self.object_body)
        velocity, angular = jac_p @ d.qvel, jac_r @ d.qvel
        fingers, hand_force = {}, np.zeros(3)
        contact_records = []
        thumb_normals, finger_normals = [], []
        table_force = np.zeros(3)
        table = floor = False
        active_count = parked_count = table_hand = 0
        hand_penetration = self_penetration = 0.
        for index, contact in enumerate(d.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            sa, sb = self.geom_sides.get(a), self.geom_sides.get(b)
            if contact.dist <= 0 and sa and sa == sb:
                self_penetration = max(self_penetration, -float(contact.dist))
            if contact.dist <= 0 and (a == self.table_geom and sb or b == self.table_geom and sa):
                table_hand += 1
            if self.object_geom not in (a, b):
                continue
            other = b if a == self.object_geom else a
            side = self.geom_sides.get(other)
            if side:
                hand_penetration = max(hand_penetration, max(0., -float(contact.dist)))
            local = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, local)
            if contact.dist > 0 or local[0] <= .001:
                continue
            force = np.asarray(contact.frame).reshape(3, 3).T @ local[:3]
            force *= 1. if b == self.object_geom else -1.
            if other == self.table_geom:
                table = True; table_force += force
            elif other == self.floor_geom:
                floor = True
            elif side == self.side:
                active_count += 1
                hand_force += force
                name = m.geom(other).name
                part = next((p for p in ('thumb', 'index', 'middle', 'ring', 'pinky') if f'_{p}_' in name), 'palm')
                fingers[part] = fingers.get(part, 0.)+max(0., float(local[0]))
                object_local = rotation.T @ (np.asarray(contact.pos)-object_T[:3, 3])
                contact_records.append(dict(part=part, normal_force_N=float(local[0]),
                    force_on_object_world_N=force.tolist(), position_world_m=np.asarray(contact.pos).tolist(),
                    position_object_m=object_local.tolist(),
                    depth_below_object_top_m=float(self.profile['height_m']/2.-object_local[2])))
                if local[0] > .05 and part != 'palm':
                    normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1. if b == self.object_geom else -1.)
                    (thumb_normals if part == 'thumb' else finger_normals).append(normal)
            elif side:
                parked_count += 1
        q = d.qpos[m.jnt_qposadr[self.hand_joints]]
        limits = m.jnt_range[self.hand_joints]
        violation = max(0., float(np.max(limits[:, 0]-q)), float(np.max(q-limits[:, 1])))
        mimic_error = max(abs(d.qpos[a]-r*d.qpos[b]-o) for a, b, r, o in self.mimics)
        slip = angle = None
        # After intentional release / retreat, relative separation is desired
        # motion, not grasp slip. Keep raw poses but do not pollute hold peaks.
        if self.baseline_relation is not None and not self.release_commanded:
            slip = float(np.linalg.norm(relation[:3, 3]-self.baseline_relation[:3, 3]))
            angle = rotation_error_deg(relation[:3, :3], self.baseline_relation[:3, :3])
        alignment_slip = alignment_angle = None
        if self.alignment_relation is not None:
            alignment_slip = float(np.linalg.norm(relation[:3, 3]-self.alignment_relation[:3, 3]))
            alignment_angle = rotation_error_deg(relation[:3, :3], self.alignment_relation[:3, :3])
        opposition = contact_opposition_angle(thumb_normals, finger_normals)
        pair_contact = bool(fingers.get('thumb', 0.) > .05 and any(fingers.get(p, 0.) > .05 for p in ('index', 'middle', 'ring', 'pinky')))
        self.current_metrics = dict(time_s=float(d.time), object_position_m=object_T[:3, 3].tolist(),
            object_quaternion_wxyz=pose_quaternion(object_T).tolist(), T_world_object=object_T.tolist(),
            T_world_wrist=wrist_T.tolist(), T_wrist_object=relation.tolist(),
            bottom_height_m=bottom, clearance_m=bottom-self.table_height,
            object_tilt_deg=float(np.rad2deg(np.arccos(np.clip(axis_z, -1., 1.)))),
            object_linear_speed_mps=float(np.linalg.norm(velocity)),
            object_angular_speed_radps=float(np.linalg.norm(angular)),
            table_contact=table, floor_contact=floor, table_vertical_force_N=float(table_force[2]),
            hand_vertical_force_N=float(hand_force[2]), fingers_normal_force_N=fingers,
            hand_object_contacts=contact_records,
            finger_pair_contact=pair_contact, directional_opposition_deg=opposition,
            opposed_contact=bool(pair_contact and opposition is not None and opposition >= 120.),
            active_hand_object_contact_count=active_count, parked_hand_contact_count=parked_count,
            hand_table_contact_count=table_hand, hand_object_penetration_m=hand_penetration,
            hand_self_penetration_m=self_penetration, hand_joint_violation_rad=violation,
            mimic_error_rad=float(mimic_error), actuator_torque_Nm=float(np.max(np.abs(d.actuator_force))),
            grasp_slip_m=slip, grasp_rotation_slip_deg=angle, grasp_verified=self.grasp_verified,
            alignment_slip_m=alignment_slip, alignment_rotation_slip_deg=alignment_angle,
            wrist_position_error_m=float(np.linalg.norm(wrist_T[:3, 3]-self.active_goal[:3, 3])),
            wrist_orientation_error_deg=rotation_error_deg(wrist_T[:3, :3], self.active_goal[:3, :3]),
            finite_state=bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))))
        return self.current_metrics

    def record(self):
        if not self.keep_trace:
            return
        mapping = self.hands.maps[self.side]
        reference = self.hands.controllers[self.side].reference
        self.samples.append(dict(time_s=float(self.data.time), phase=self.phase,
            command=int(self.hands.controllers[self.side].command), metrics=copy.deepcopy(self.current_metrics),
            qpos=self.data.qpos.tolist(), qvel=self.data.qvel.tolist(),
            reference_q_rad=reference.position.tolist(), reference_qd_rad_s=reference.velocity.tolist(),
            reference_qdd_rad_s2=reference.acceleration.tolist(),
            active_hand_q_rad=self.data.qpos[mapping.qpos].tolist(),
            active_hand_torque_Nm=self.data.actuator_force[mapping.actuators].tolist()))

    def step(self):
        if self.done:
            return
        if self.steps % 10 == 0:
            self.sync(); self._safety()
            if not self.done:
                self._gate()
            if self.done:
                self.record(); return
            self.hands.update()
        if self.motion_duration:
            self.active_goal = interpolate_pose(self.move_start, self.move_end,
                (self.data.time-self.phase_start)/self.motion_duration)
        target = int(self.layout['mocap_ids'][self.side])
        self.data.mocap_pos[target] = self.active_goal[:3, 3]
        self.data.mocap_quat[target] = pose_quaternion(self.active_goal)
        if self.steps % 10 == 0 and self.keep_trace:
            self.targets.append(dict(time_s=float(self.data.time), phase=self.phase,
                T_world_wrist_goal=self.active_goal.tolist(), command=int(self.hands.controllers[self.side].command)))
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or np.any(self.data.warning.number):
            self.fail('Nonfinite state or MuJoCo warning')
        if self.steps % 10 == 0 or self.done:
            self.sync(); self._safety(); self.record()

    def report(self):
        return dict(schema_version=1,
            scope='Pure hand top-grasp probe: externally driven dynamic wrist fixture; NOT full-body policy',
            experiment_completed=self.completed and self.phase == 'COMPLETE',
            success=self.completed and self.grasp_verified and self.release_commanded and not self.failure,
            phase=self.phase, failure=self.failure, failure_phase=self.failure_phase,
            duration_s=float(self.data.time), grasp_verified=self.grasp_verified,
            release_commanded=self.release_commanded, profile=self.profile, candidate=asdict(self.candidate),
            final_metrics=self.current_metrics, initial_metrics=self.initial_metrics, peaks=self.peaks,
            place_target_xy_m=self.place_xy.tolist(), max_verified_hold_s=self.max_verified_hold_s,
            grasp_relation=None if self.baseline_relation is None else self.baseline_relation.tolist(),
            placement_relation=getattr(self, 'placement_relation', None),
            upright_attempted=self.upright_attempted, alignment_relation=self.alignment_relation,
            layout=self.layout, hand_configuration=self.hand_cfg, transitions=self.transitions,
            criteria=dict(grasp_clearance_m=.008, grasp_stable_s=1.,
                minimum_weight_fraction=.8, maximum_slip_m=.015, maximum_rotation_slip_deg=5.,
                minimum_contact_opposition_deg=120., minimum_normal_force_per_contact_N=.05,
                object_tilt_deg=10., placement_xy_error_m=.02, final_stable_s=1.,
                no_hand_table_or_parked_hand_contact=True),
            wrist_fixture=True, object_free=True, object_pose_replay=False,
            object_weld=False, additional_external_forces=False,
            timing_hz=dict(physics=1000, hand_reference=100), warnings=self.data.warning.number.tolist())


__all__ = ['TopGraspExperiment', 'TopGraspCandidate', 'pickup_conditions', 'unsupported_load_conditions', 'placed_conditions', 'blend', 'interpolate_pose', 'contact_opposition_angle']
