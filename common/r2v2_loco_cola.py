"""Staged real-contact Loco/Reach cola pickup and carry experiment.

No physical-state handoff, welded object, wrist fixture, or external support.
The chosen empty-hand handoff is not assumed to qualify a loaded carry pose.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_front_drop import pose
from common.r2v2_grasp_recording import body_transform
from common.r2v2_hand_control import DualHandControl
from common.r2v2_loco_policy import LocoPolicy
from common.r2v2_reach_policy import ReachPolicy
from common.r2v2_reach_sim import initialize_robot, foot_collision_ids, require_parity, sha256
from common.r2v2_tabletop_scene import build_tabletop_xml, object_metrics, contact_metrics, _descendant


def interpolate_pose(a, b, fraction):
    p0, q0 = a; p1, q1 = b
    q1 = np.asarray(q1).copy()
    dot = float(q0 @ q1)
    if dot < 0: q1 = -q1; dot = -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995: q = (1-fraction)*q0+fraction*q1
    else:
        angle = np.arccos(dot)
        q = (np.sin((1-fraction)*angle)*q0+np.sin(fraction*angle)*q1)/np.sin(angle)
    return (1-fraction)*p0+fraction*p1, q/np.linalg.norm(q)


def grasp_target(can_transform, relation):
    """Both inputs are rigid transforms; relation is can in wrist frame."""
    return can_transform @ np.linalg.inv(relation)


def destination_layout(config):
    """Explicit world placements, with backward-compatible lateral layout."""
    offset=np.array([0.,config.get('destination_offset_y_m',0.),0.])
    center=np.asarray(config.get('destination_table_center_xyz',np.asarray(config['table_center_xyz'])+offset),dtype=float)
    half=np.asarray(config.get('destination_table_half_size',config['table_half_size']),dtype=float)
    target=np.asarray(config.get('destination_can_position_xyz',np.asarray(config['cylinder_position_xyz'])+offset),dtype=float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (center,half,target)) or np.any(half<=0):
        raise ValueError('Invalid destination layout')
    if np.any(np.abs(target[:2]-center[:2])+.02>=half[:2]):
        raise ValueError('Destination can must fit on tabletop')
    return center,half,target


def unload_height(height, anchor, table_force_N):
    """Bounded 50 Hz admittance reference; no forces are applied to physics."""
    speed=np.clip(.01*(table_force_N-.981),-.005,.03)
    return float(np.clip(height+.02*speed,anchor-.01,anchor+.02))


def fingers_clear(contact):
    """Palm may brush a seated object until the open hand withdraws."""
    return not any(c['active'] and c['part'] not in (None,'palm') for c in contact['contacts'])


class LocoColaExperiment:
    def __init__(self, config, stop_after='pickup'):
        if stop_after not in ('approach', 'pickup', 'handoff', 'carry', 'place'):
            raise ValueError('Unknown stage endpoint')
        self.config = copy.deepcopy(config)
        self.stop_after = stop_after
        r, l = Path(config['reach_root']), Path(config['loco_root'])
        cfg = dict(endpoint_contract='wrist_world_v2', simulation_dt=.001, policy_dt=.02, hand_dt=.01,
                   policy_path=str(r/'policy_front_manipulation_v1.onnx'), checkpoint_path=str(r/'model_9999.pt'))
        require_parity(r/'evidence/parity.json', cfg)
        xml, hand_cfg = build_tabletop_xml(cfg, {**config, 'object_appearance': 'cola_can'})
        root = ET.fromstring(xml)
        if stop_after == 'place':
            # Two genuine fixed tables, both present from the initial frame.
            center,half,_=destination_layout(config)
            # Reuse the table builder so leg geometry follows custom dimensions.
            table_xml,_=build_tabletop_xml(cfg,{**config,'table_center_xyz':center,'table_half_size':half})
            destination=copy.deepcopy(ET.fromstring(table_xml).find('.//body[@name="tabletop"]'))
            destination.set('name','destination_table')
            destination.set('pos',' '.join(map(str,center)))
            for geom in destination.iter('geom'):
                geom.set('name',geom.get('name').replace('tabletop','destination_table'))
                geom.set('rgba','0.25 0.42 0.32 1')
            root.find('worldbody').append(destination)
        for side in ('left', 'right'):
            site = root.find(f'.//site[@name="{side}_tcp"]')
            site.set('name', side+'_wrist'); site.set('pos', '0 0 0'); site.set('rgba', '0 0 0 0')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, hand_cfg)
        self.data.qpos[self.model.joint('floating_base_joint').qposadr[0]] += config['initial_robot_x_m']
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        self.reach = ReachPolicy(self.model, self.scratch, cfg['policy_path'], expected_endpoint_contract='wrist_world_v2')
        self.loco = LocoPolicy(self.model, self.scratch, l/'policy.onnx')
        self.loco.pose_command[1] = np.deg2rad(config['loco_pitch_deg'])
        self.loco.reset(self.scratch)
        self.hands = DualHandControl(self.model, self.data, hand_cfg)
        self.mapping = self.reach.body_map
        assert np.allclose(self.loco.kp, self.reach.kp[:14]) and np.allclose(self.loco.kd, self.reach.kd[:14])
        assert not set(self.mapping.actuators) & set(np.concatenate([v.actuators for v in self.hands.maps.values()]))
        self.base = self.reach.base_id
        self.floor = self.model.geom('floor').id
        self.foot = set(foot_collision_ids(self.model))
        self.robot_geoms = {g for g in range(self.model.ngeom) if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.arm_hold = self.data.qpos[self.mapping.qpos[14:]].copy()
        self.executed = self.reach.default_joint_pos.copy()
        self.blend_start = None
        self.blend_from = self.executed.copy()
        self.mode = 'loco'
        self.phase, self.phase_start, self.stable_s = 'STAND', 0., 0.
        self.steps = 0
        self.failure = None
        self.completed = self.grasp_verified = False
        self.transitions, self.samples = [], []
        self.baseline_relation = None
        self.baseline_time = None
        self.grasp_pose = None
        self.loco_start = None
        self.bad_grip_s = 0.
        self.release_commanded = self.place_verified = False
        self.place_wrist = None
        self.task_goals = {}
        self.place_bias = np.zeros(3)
        self.max_joint_violation = 0.
        self.hashes = dict(reach_checkpoint=sha256(r/'model_9999.pt'), loco_checkpoint=sha256(l/'model_9999.pt'))
        self.enter('STAND')

    @property
    def done(self):
        return self.phase in ('FAILED', 'COMPLETE')

    def sync(self):
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = self.data.qvel
        self.scratch.ctrl[:] = self.data.ctrl
        self.scratch.time = self.data.time
        mujoco.mj_forward(self.model, self.scratch)

    def switch(self, mode):
        before_q, before_v = self.data.qpos.copy(), self.data.qvel.copy()
        self.blend_from = self.executed.copy()
        self.blend_start = float(self.data.time)
        if mode == 'reach':
            self.reach.reset(self.scratch)
            self.reach.last_action = ((self.executed-self.reach.default_joint_pos)/self.reach.action_scale).astype(np.float32)
            self.reach.histories = {k: np.repeat(v[None], 10, axis=0) for k, v in self.reach.observation_terms(self.scratch).items()}
        else:
            self.arm_hold = self.scratch.qpos[self.mapping.qpos[14:]].copy()
            self.loco.reset(self.scratch, self.executed[:14])
        self.mode = mode
        self.task_goals={s:(r.goal_position.copy(),r.goal_quaternion.copy()) for s,r in self.reach.references.items()}
        assert np.array_equal(before_q, self.data.qpos) and np.array_equal(before_v, self.data.qvel)

    def entry_goals(self):
        virtual = mujoco.MjData(self.model)
        virtual.qpos[:] = self.scratch.qpos
        rot = self.scratch.xmat[self.base].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        adr = self.model.joint('floating_base_joint').qposadr[0]
        virtual.qpos[adr+3:adr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        q = virtual.qpos[self.mapping.qpos].copy(); q[12:14] = 0
        for i, sign in ((14, 1), (21, -1)):
            q[i:i+7] = [0, sign*np.deg2rad(15), 0, np.deg2rad(75), 0, np.deg2rad(15), 0]
        virtual.qpos[self.mapping.qpos] = q
        mujoco.mj_forward(self.model, virtual)
        c = self.config
        delta = [-c['entry_back_m']*np.cos(yaw), -c['entry_back_m']*np.sin(yaw), -c['entry_down_m']]
        result = {}
        for side in ('left', 'right'):
            p, quat = interpolate_pose(self.reach.endpoint_pose(self.scratch, side), self.reach.endpoint_pose(virtual, side), c['entry_side_fraction'])
            result[side] = (p+delta, quat)
        return result

    def set_goals(self, goals):
        for side, (p, q) in goals.items():
            self.task_goals[side]=(np.asarray(p).copy(),np.asarray(q).copy())
            self.reach.set_target_world(side, p, q)

    def enter(self, phase):
        self.phase, self.phase_start, self.stable_s = phase, float(self.data.time), 0.
        self.loco.command[:] = 0
        if phase == 'WALK_TO_TABLE':
            self.loco_start = self.scratch.xpos[self.base].copy()
        elif phase == 'ENTRY':
            self.switch('reach'); self.set_goals(self.entry_goals())
        elif phase == 'RAISE_OUTSIDE':
            goals = {}
            for side in ('left', 'right'):
                p, _ = self.reach.endpoint_pose(self.scratch, side)
                p[2] = 1.12
                goals[side] = (p, np.array([1., 0, 0, 0]))
            self.set_goals(goals)
        elif phase == 'PREGRASP':
            can = body_transform(self.scratch, self.model.body('test_cylinder').id)
            self.grasp_pose = grasp_target(can, pose([.145, -.035, 0], [1., 0, 0, 0]))
            q = np.empty(4); mujoco.mju_mat2Quat(q, self.grasp_pose[:3, :3].copy().ravel())
            self.grasp_quat = q
            self.set_goals({'left': (self.grasp_pose[:3, 3]+[0, .05, 0], q)})
        elif phase == 'GRASP_ALIGN':
            self.set_goals({'left': (self.grasp_pose[:3, 3], self.grasp_quat)})
        elif phase == 'CLOSE':
            self.hands.command('left', 1)
        elif phase == 'PROBE':
            self.set_goals({'left': (self.grasp_pose[:3, 3]+[0, 0, .025], self.grasp_quat)})
        elif phase == 'LIFT':
            self.set_goals({'left': (self.grasp_pose[:3, 3]+[0, 0, self.config['lift_m']], self.grasp_quat)})
        elif phase == 'LOCO_LOADED_HOLD':
            self.switch('loco')
        elif phase == 'CARRY_BACK':
            self.loco_start = self.scratch.xpos[self.base].copy()
        elif phase == 'CARRY_SIDE':
            self.side_start = self.scratch.xpos[self.base].copy()
        elif phase == 'REACH_LOADED_ENTRY':
            # Keep the held pose, never reuse the low empty-hand entry goal.
            self.switch('reach')
        elif phase == 'PLACE_ABOVE':
            self.place_bias[:]=0
            center,half,position=destination_layout(self.config)
            position=position.copy()
            # Calibrated release approach compensation; destination_layout
            # remains the unchanged acceptance target, not this command.
            position+=np.asarray(self.config.get('release_approach_offset_m',[0,0,0]))
            position[2]=center[2]+half[2]+.06+self.config.get('place_hover_height_m',.10)
            self.place_wrist=grasp_target(pose(position,[1.,0,0,0]),self.baseline_relation)
            q=np.empty(4);mujoco.mju_mat2Quat(q,self.place_wrist[:3,:3].copy().ravel())
            self.place_quat=q
            self.set_goals({'left':(self.place_wrist[:3,3],q)})
        elif phase == 'PLACE_DOWN':
            self.place_bias[:]=0
            # 2 mm target compression permits genuine support transfer to table.
            drop=self.config.get('place_hover_height_m',.10)+.002
            self.place_down_end=self.place_wrist[:3,3]+[0,0,-drop]
            if self.config.get('segmented_place',False):
                p=self.task_goals['left'][0].copy()
                p[2]=max(self.reach.endpoint_pose(self.scratch,'left')[0][2]-.01,self.place_down_end[2])
                self.lower_step_time=float(self.data.time)
                self.set_goals({'left':(p,self.place_quat)})
            else: self.set_goals({'left':(self.place_down_end,self.place_quat)})
        elif phase == 'PLACE_SETTLE':
            self.place_bias[:]=0
            p,q=self.task_goals['left'];p=p.copy()
            p[2]=self.reach.endpoint_pose(self.scratch,'left')[0][2]
            self.settle_anchor=p.copy()
            self.set_goals({'left':(p,q)})
        elif phase == 'OPEN':
            self.release_commanded=True
            self.hands.command('left',0)
        elif phase == 'RETREAT':
            self.place_bias[:]=0
            offset=np.asarray(self.config.get('retreat_offset_world_m',[0,.06,0]))
            p=self.reach.references['left'].goal_position.copy()+offset
            self.set_goals({'left':(p,self.place_quat)})
        self.transitions.append(dict(time_s=float(self.data.time), phase=phase, body_owner=self.mode,
            arm_hold_rad=self.arm_hold.tolist(), goals={s:dict(position=r.goal_position.tolist(), quaternion=r.goal_quaternion.tolist()) for s,r in self.reach.references.items()}))
        print(f'{self.data.time:.3f} {phase} {self.failure or ""}', flush=True)

    def fail(self, reason):
        if not self.done:
            self.failure = self.phase+': '+reason
            self.enter('FAILED')

    def metrics(self):
        d = self.scratch
        terms = self.loco.terms(d)
        obj = object_metrics(self.model, d)
        contact = contact_metrics(self.model, d)
        destination_contacts=[c for c in contact['contacts'] if c['active'] and c['other_geom'].startswith('destination_table')]
        contact['destination_table_contact']=bool(destination_contacts)
        contact['table_contact'] |= bool(destination_contacts)
        if self.stop_after=='place':
            table_body=self.model.body('destination_table').id
            for index,c in enumerate(d.contact):
                a,b=map(int,c.geom)
                on_table_a=_descendant(self.model,int(self.model.geom_bodyid[a]),table_body)
                on_table_b=_descendant(self.model,int(self.model.geom_bodyid[b]),table_body)
                if c.dist<=0 and ((on_table_a and b in self.robot_geoms) or (on_table_b and a in self.robot_geoms)):
                    force=np.zeros(6);mujoco.mj_contactForce(self.model,d,index,force)
                    contact['robot_table_contacts'].append(dict(geoms=[self.model.geom(a).name,self.model.geom(b).name],
                        penetration_m=max(0.,-float(c.dist)),normal_force_N=float(force[0])))
        table_vertical_force=sum(c['force_world_N'][2] for c in contact['contacts']
            if c['other_geom'].startswith(('tabletop','destination_table')))
        hand_vertical_force = sum(c['force_world_N'][2] for c in contact['contacts'] if c['part'] is not None)
        wrists = {}
        for side in ('left', 'right'):
            p,q = self.reach.endpoint_pose(d, side); ref = self.reach.references[side]
            goal_p,goal_q=self.task_goals.get(side,(ref.goal_position,ref.goal_quaternion))
            wrists[side] = dict(position_world=p.tolist(), quaternion_wxyz=q.tolist(),
                task_goal_world=goal_p.tolist(),policy_goal_world=ref.goal_position.tolist(),
                position_error_m=float(np.linalg.norm(p-goal_p)),
                orientation_error_deg=float(np.degrees(2*np.arccos(np.clip(abs(q@goal_q),0,1)))),
                speed_mps=float(np.linalg.norm(self.reach.wrist_twist(d,side)[0])))
        relative = np.linalg.inv(body_transform(d, self.reach.wrist_ids['left'])) @ obj['T_world_cylinder']
        slip = slip_deg = None
        if self.baseline_relation is not None:
            slip = float(np.linalg.norm(relative[:3,3]-self.baseline_relation[:3,3]))
            rot = relative[:3,:3].T @ self.baseline_relation[:3,:3]
            slip_deg = float(np.degrees(np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))))
        rotation = d.xmat[self.base].reshape(3,3)
        height = self.config['table_center_xyz'][2]+self.config['table_half_size'][2]
        return dict(time_s=float(self.data.time), phase=self.phase, body_owner=self.mode,
            velocity_command_body=self.loco.command.tolist(),
            place_target_bias_m=self.place_bias.tolist(),
            base_position_world=d.xpos[self.base].tolist(), velocity_body=terms['base_lin_vel'].tolist(),
            angular_velocity_body=terms['base_ang_vel'].tolist(), wrists=wrists,
            base_tilt_deg=float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1)))),
            object_position_world=obj['position_m'].tolist(), object_quaternion_wxyz=obj['quaternion_wxyz'].tolist(),
            object_speed_mps=float(np.linalg.norm(obj['linear_velocity_mps'])), object_tilt_deg=obj['tilt_deg'],
            object_angular_speed_radps=float(np.linalg.norm(obj['angular_velocity_radps'])),
            hand_vertical_force_N=float(hand_vertical_force),
            table_vertical_force_N=float(table_vertical_force),
            bottom_clearance_m=obj['bottom_height_m']-height, relative=relative.tolist(),
            slip_m=slip, slip_deg=slip_deg, contacts=contact,
            left_hand_command=int(self.hands.controllers['left'].command), grasp_verified=self.grasp_verified,
            arm_hold_error_rad=float(np.max(np.abs(d.qpos[self.mapping.qpos[14:]]-self.arm_hold))))

    def stable(self, valid, duration):
        self.stable_s = self.stable_s+.02 if valid else 0.
        return self.stable_s >= duration-1e-9

    def pose_good(self, row, position=.01, angle=5., speed=.02, both=False):
        return all(w['position_error_m']<position and w['orientation_error_deg']<angle and w['speed_mps']<speed
                   for s,w in row['wrists'].items() if both or s=='left')

    def gate(self, row):
        t = self.data.time-self.phase_start
        velocity = np.asarray(row['velocity_body'])
        stopped = np.linalg.norm(velocity[:2])<.03 and abs(row['angular_velocity_body'][2])<.08
        contact = row['contacts']
        supported_by_hand = (contact['opposed'] and not contact['table_contact'] and not contact['floor_contact']
            and row['bottom_clearance_m']>.008 and row['hand_vertical_force_N']>=.8*.1*9.81
            and not contact['robot_table_contacts'])
        if self.phase == 'STAND':
            if self.stable(stopped and t>=4., .3): self.enter('WALK_TO_TABLE'); return
        elif self.phase == 'WALK_TO_TABLE':
            distance = self.scratch.xpos[self.base,0]-self.loco_start[0]
            self.loco.command[0] = np.clip(1.2*(self.config['approach_distance_m']-distance),0,.15)
            if distance>=self.config['approach_distance_m']-.01: self.enter('STOP_AT_TABLE'); return
        elif self.phase == 'STOP_AT_TABLE':
            if self.stable(stopped and t>=2.,.3): self.enter('ENTRY'); return
        elif self.phase == 'ENTRY':
            # Exploratory empty-hand settling tolerance, NOT a grasp gate.
            if self.stable(self.pose_good(row,.025,6.,.02,both=True) and t>=4.,.3): self.enter('RAISE_OUTSIDE'); return
        elif self.phase == 'RAISE_OUTSIDE':
            if self.stable(self.pose_good(row,both=True),.3): self.enter('PREGRASP'); return
        elif self.phase in ('PREGRASP','GRASP_ALIGN'):
            if self.stable(self.pose_good(row,.005,3.),.3):
                if self.phase=='PREGRASP': self.enter('GRASP_ALIGN')
                elif self.stop_after=='approach': self.completed=True; self.enter('COMPLETE')
                else: self.enter('CLOSE')
                return
        elif self.phase == 'CLOSE':
            if self.stable(contact['opposed'] and t>=2.5 and self.pose_good(row,.005,3.),.3): self.enter('PROBE'); return
        elif self.phase == 'PROBE':
            # Latch only after the trial lift has stopped. Capturing during
            # first separation incorrectly mixes closure seating with later
            # transport slip. Never re-latch a verified grasp to hide drift.
            settled = self.pose_good(row) and row['object_speed_mps']<.02 and row['object_angular_speed_radps']<.15
            if supported_by_hand and settled and self.baseline_relation is None:
                self.baseline_relation=np.asarray(row['relative'])
                self.baseline_time=float(self.data.time)
            retained = supported_by_hand and row['slip_m'] is not None and row['slip_m']<.015 and row['slip_deg']<5
            if self.stable(retained and settled,1.): self.grasp_verified=True; self.enter('LIFT'); return
        elif self.phase == 'LIFT':
            if self.stable(self.pose_good(row) and row['bottom_clearance_m']>.06,1.):
                if self.stop_after=='pickup': self.completed=True; self.enter('COMPLETE')
                else: self.enter('LOCO_LOADED_HOLD')
                return
        elif self.phase == 'LOCO_LOADED_HOLD':
            if self.stable(stopped and t>=4.,.5):
                if self.stop_after=='handoff': self.completed=True; self.enter('COMPLETE')
                elif self.stop_after=='place': self.enter('CARRY_BACK' if self.config.get('carry_route')=='backward' else 'CARRY_SIDE')
                else: self.enter('CARRY_BACK')
                return
        elif self.phase == 'CARRY_SIDE':
            delta=self.side_start+np.array([0,self.config['destination_offset_y_m'],0])-self.scratch.xpos[self.base]
            rot=self.scratch.xmat[self.base].reshape(3,3)
            yaw=np.arctan2(rot[1,0],rot[0,0])
            side_speed=self.config.get('side_speed_mps',.12)
            if not 0<side_speed<=.15: raise ValueError('Lateral speed must stay in trained range')
            world_velocity=np.clip(1.2*delta[:2],[-.12,-side_speed],[.12,side_speed])
            body_velocity=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@world_velocity
            self.loco.command[:2]=np.clip(body_velocity,[-.15,-.15],[.15,.15])
            # Translate in the world frame without also requesting an
            # unqualified turn back to global yaw zero.
            self.loco.command[2]=0.
            if np.linalg.norm(delta[:2])<.015: self.enter('STOP_AT_B'); return
        elif self.phase == 'STOP_AT_B':
            if self.stable(stopped and t>=2.,.5): self.enter('REACH_LOADED_ENTRY'); return
        elif self.phase == 'REACH_LOADED_ENTRY':
            if self.stable(self.pose_good(row,both=True) and t>=3.,.3): self.enter('PLACE_ABOVE'); return
        elif self.phase == 'PLACE_ABOVE':
            if self.stable(self.pose_good(row) and row['object_tilt_deg']<10.,.5): self.enter('PLACE_DOWN'); return
        elif self.phase == 'PLACE_DOWN':
            if self.config.get('segmented_place',False):
                if contact['destination_table_contact'] and row['table_vertical_force_N']>=.8*.1*9.81:
                    self.enter('PLACE_SETTLE'); return
                if (self.data.time-self.lower_step_time>=.45 and row['wrists']['left']['speed_mps']<.02):
                    p,q=self.task_goals['left'];p=p.copy()
                    p[2]=max(p[2]-.01,self.place_down_end[2])
                    self.set_goals({'left':(p,q)});self.lower_step_time=float(self.data.time)
            if self.stable(self.placed(row) and row['table_vertical_force_N']>=.8*.1*9.81,.3): self.enter('OPEN'); return
        elif self.phase == 'PLACE_SETTLE':
            # Unload the hand onto the table instead of freezing an unloaded
            # position servo that may continue compressing the seated can.
            if self.config.get('place_force_feedback',False):
                p,q=self.task_goals['left'];p=p.copy()
                p[2]=unload_height(p[2],self.settle_anchor[2],row['table_vertical_force_N'])
                self.set_goals({'left':(p,q)})
            if self.stable(self.placed(row) and row['table_vertical_force_N']>=.8*.1*9.81,.3): self.enter('OPEN'); return
        elif self.phase == 'OPEN':
            released=(fingers_clear(contact) and not contact['opposed'] and self.hands.controllers['left'].finished
                and row['table_vertical_force_N']>=.8*.1*9.81)
            if self.stable(self.placed(row) and released and t>=2.5,1.): self.enter('RETREAT'); return
        elif self.phase == 'RETREAT':
            no_hand=not any(c['active'] and c['part'] is not None for c in contact['contacts'])
            if self.stable(self.placed(row) and no_hand and self.pose_good(row),1.):
                self.place_verified=True;self.completed=True;self.enter('COMPLETE');return
        elif self.phase == 'CARRY_BACK':
            distance = self.loco_start[0]-self.scratch.xpos[self.base,0]
            self.loco.command[0] = -.20
            if distance>=self.config['carry_distance_m']:
                self.enter('STOP_AT_B' if self.stop_after=='place' else 'CARRY_STOP'); return
        elif self.phase == 'CARRY_STOP':
            if self.stable(stopped and t>=2.,1.): self.completed=True; self.enter('COMPLETE'); return
        if t>=self.config['stage_timeout_s']:
            self.fail('measured arrival/contact deadline exceeded; no forced phase advance')

    def placed(self,row):
        desired=destination_layout(self.config)[2][:2]
        return (row['contacts']['destination_table_contact'] and not row['contacts']['floor_contact']
            and np.linalg.norm(np.asarray(row['object_position_world'])[:2]-desired)<.02
            and row['object_tilt_deg']<10. and row['object_speed_mps']<.02 and row['object_angular_speed_radps']<.15)

    def safety(self, row):
        m,d = self.model,self.data
        q = d.qpos[self.mapping.qpos]; bounds=m.jnt_range[self.mapping.joints]
        violation=float(max(0.,np.max(bounds[:,0]-q),np.max(q-bounds[:,1])))
        self.max_joint_violation=max(self.max_joint_violation,violation)
        if violation>.03: self.fail('body joint physical limit residual > 0.03 rad'); return
        if row['base_tilt_deg']>35 or row['base_position_world'][2]<.55: self.fail('body stability limit'); return
        for c in d.contact:
            if c.dist>=0: continue
            a,b=map(int,c.geom)
            if self.floor in (a,b):
                other=b if a==self.floor else a
                if other in self.robot_geoms and other not in self.foot: self.fail('non-foot ground contact'); return
            if a in self.robot_geoms and b in self.robot_geoms and c.dist<-.02: self.fail('self penetration >20mm'); return
        if row['contacts']['floor_contact']: self.fail('can reached floor'); return
        if self.grasp_verified and not self.release_commanded:
            retained = row['contacts']['opposed'] and (not row['contacts']['table_contact'] or self.phase in ('PLACE_DOWN','PLACE_SETTLE')) and row['slip_m']<.015 and row['slip_deg']<5
            self.bad_grip_s = 0 if retained else self.bad_grip_s+.02
            if self.bad_grip_s>=.1: self.fail('verified grasp lost/slipped; fingers remain closed')

    def step(self):
        if self.done: return
        if self.steps%20==0:
            self.sync(); row=self.metrics(); self.samples.append(row); self.safety(row)
            if self.done: return
            self.gate(row)
            if self.done: return
            if (self.config.get('place_feedback',False) and (self.phase in ('PLACE_ABOVE','RETREAT') or (self.phase=='PLACE_DOWN' and not self.config.get('segmented_place',False)))
                    and self.data.time-self.phase_start>2. and row['wrists']['left']['speed_mps']<.02):
                p,q=self.task_goals['left']
                error=p-np.asarray(row['wrists']['left']['position_world'])
                self.place_bias+=.4*.02*error
                length=np.linalg.norm(self.place_bias)
                if length>.02: self.place_bias*=.02/length
                self.reach.set_target_world('left',p+self.place_bias,q)
            desired=np.r_[self.loco.act(self.scratch),self.arm_hold] if self.mode=='loco' else self.reach.act(self.scratch)
            duration=self.config['blend_in_s' if self.mode=='reach' else 'blend_out_s']
            alpha=1. if self.blend_start is None else np.clip((self.data.time-self.blend_start)/duration,0,1)
            alpha=alpha*alpha*(3-2*alpha)
            self.executed=(1-alpha)*self.blend_from+alpha*desired
            self.reach.q_des=self.executed.copy()
        if self.steps%10==0: self.hands.update()
        self.reach.apply(self.data); self.hands.apply(self.data)
        mujoco.mj_step(self.model,self.data); self.steps+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or np.any(self.data.warning.number):
            self.fail('nonfinite state or MuJoCo warning')

    def report(self):
        self.sync()
        return dict(success=self.completed and not self.failure, phase=self.phase, failure=self.failure,
            scope=self.stop_after, config=self.config, checkpoint_sha256=self.hashes,
            duration_s=float(self.data.time), grasp_verified=self.grasp_verified,
            release_commanded=self.release_commanded,place_verified=self.place_verified,
            slip_baseline_time_s=self.baseline_time,
            final_metrics=self.metrics(), max_joint_violation_rad=self.max_joint_violation,
            physics=dict(body_hz=50,hand_hz=100,simulation_hz=1000,object_weld=False,external_forces=False,
                         physical_state_reset_at_handoff=False),
            acceptance=dict(grasp_position_m=.005,grasp_orientation_deg=3.,wrist_speed_mps=.02,
                            pickup_hold_s=1.,slip_m=.015,slip_deg=5.),
            note='Stage completion is not qualification of the entire two-table task.')
