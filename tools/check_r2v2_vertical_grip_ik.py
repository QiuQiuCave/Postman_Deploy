"""Offline whole-body IK of the measured vertical-grip fixture trajectory.

All displayed robot/object motion is assigned FK state, NOT a policy rollout
or a physical full-body grasp. Feet are constrained to their initial frames;
real robot joint limits, masses, collision shapes and hand mimics stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from common.r2v2_crate_lift_metrics import _descendant
from common.r2v2_grasp_recording import body_transform
from common.r2v2_reach_sim import initialize_robot, foot_collision_ids
from common.r2v2_tabletop_scene import build_tabletop_xml
from common.r2v2_vertical_grip_crate import VerticalGripCrateParameters, add_vertical_grip_crate
from common.r2v2_vertical_grip_test import GripTestParameters, build_scene, VerticalGripExperiment
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, urdf_hand_joints
from tools.check_r2v2_crate_height_ik import GeometryScreen, _jsonable


DEFAULT_SOURCE = Path('/root/autodl-tmp/Postman_Deploy/vertical_grip_grasp_20261025/pitch_m15_video')
DEFAULT_SCENE = Path('/root/autodl-tmp/HALO-mj/beverage_crate_20261025/path_distance33/manifest.json')


def save(path, value):
    path.write_text(json.dumps(value, default=_jsonable, indent=2, allow_nan=False)+"\n")


def mesh_box_separation_bound(model, data, mesh_geom, box_geom):
    """Certified lower distance bound from the three box-face separating axes.

    A positive projected gap separates EVERY mesh vertex (and its convex
    hull) from the box. Nonpositive results prove nothing. This is used only
    to cross-check native CCD's spurious exact-zero distances; it never
    substitutes a witness-segment length or suppresses collision geometry.
    Related upstream report: google-deepmind/mujoco#3383.
    """
    if model.geom_type[mesh_geom] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_type[box_geom] != mujoco.mjtGeom.mjGEOM_BOX:
        return 0.
    mesh = model.geom_dataid[mesh_geom]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start+count]
    box_r = data.geom_xmat[box_geom].reshape(3,3)
    transform = data.geom_xmat[mesh_geom].reshape(3,3).T@box_r
    local = vertices@transform+(data.geom_xpos[mesh_geom]-data.geom_xpos[box_geom])@box_r
    half = model.geom_size[box_geom]
    return float(max(0.,np.max(local.min(0)-half),np.max(-half-local.max(0))))


def checked_distance(model, data, a, b, maximum=.03):
    distance = float(mujoco.mj_geomDistance(model,data,int(a),int(b),maximum,None))
    if distance == 0.:
        # Only a proven separating plane can override the anomalous zero.
        lower = max(mesh_box_separation_bound(model,data,a,b),
                    mesh_box_separation_bound(model,data,b,a))
        if lower > 1e-8:
            return min(maximum,lower)
    return distance


class RecordedGrip:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.report = json.loads((self.directory/'report.json').read_text())
        if not self.report.get('passed') or not self.report.get('completed'):
            raise ValueError('Expected completed, physically successful fixture evidence')
        self.params = GripTestParameters(**self.report['parameters'])
        self.crate = VerticalGripCrateParameters(**self.report['crate_parameters'])
        self.model, cfg, _, _ = build_scene(self.params, self.crate)
        if cfg != self.report['hand_configuration']:
            raise ValueError('Hand configuration changed since recording')
        self.data = mujoco.MjData(self.model)
        with np.load(self.directory/'trajectory.npz', allow_pickle=False) as a:
            self.times, self.qpos = a['time_s'].copy(), a['qpos'].copy()
        if (self.qpos.shape != (len(self.times), self.model.nq) or not np.all(np.diff(self.times)>0)
                or not np.isfinite(self.qpos).all() or not np.isfinite(self.times).all()):
            raise ValueError('Invalid trajectory layout')
        self.hand_names = tuple(urdf_hand_joints())
        self.hand_qadr = [self.model.joint(n).qposadr[0] for n in self.hand_names]
        self.sha256 = hashlib.sha256((self.directory/'trajectory.npz').read_bytes()).hexdigest()

    def sample(self, t, anchor):
        if not self.times[0]-1e-8 <= t <= self.times[-1]+1e-8:
            raise ValueError('Requested time outside measured trajectory')
        i = int(np.clip(np.searchsorted(self.times, t, side='right'), 1, len(self.times)-1))
        w = float(np.clip((t-self.times[i-1])/(self.times[i]-self.times[i-1]), 0., 1.))
        self.data.qpos[:] = (1-w)*self.qpos[i-1]+w*self.qpos[i]
        mujoco.mj_normalizeQuat(self.model, self.data.qpos)
        mujoco.mj_kinematics(self.model, self.data)
        crate = anchor@body_transform(self.data, self.model.body('cargo_crate').id)
        wrists = np.array([anchor@body_transform(self.data, self.model.body(f'{s}_hand_roll_link').id) for s in SIDES])
        fingers = self.data.qpos[self.hand_qadr].copy()
        phase = next((name for end, name in VerticalGripExperiment.timeline if t < end-1e-8), 'COMPLETE')
        return crate, wrists, fingers, phase


class VerticalGripScreen(GeometryScreen):
    def __init__(self, recording, manifest, margin=.05):
        if not 0 <= margin < .3:
            raise ValueError('Invalid joint margin')
        self.recording, self.scene = recording, manifest['scene']
        if not np.allclose(self.scene['crate_dimensions_depth_width_height_m'],
                           [recording.crate.depth, recording.crate.width, recording.crate.height],rtol=0,atol=1e-12):
            raise ValueError('Scene shell dimensions differ from grasp recording')
        if not np.isclose(self.scene['crate_mass_kg'],recording.crate.mass,rtol=0,atol=1e-12):
            raise ValueError('Scene crate mass differs from grasp recording')
        self.origin = np.array(manifest['task_origin_in_beverage_world_m'], float)
        cfg = {'simulation_dt': .001, 'hand_dt': .01}
        xml, self.hand_cfg = build_tabletop_xml(cfg, {
            'table_center_xyz': self.scene['table_center_world_m'],
            'table_half_size': self.scene['table_half_size_m'],
            'cylinder_position_xyz': self.scene['can_initial_world_m'],
        })
        root = ET.fromstring(xml)
        root.set('model', 'R2V2_vertical_grip_offline_IK_only')
        world = root.find('worldbody')
        # Empty-crate fixture evidence is reproduced; no beverage/load added.
        world.remove(world.find('body[@name="test_cylinder"]'))
        world.find('body[@name="tabletop"]/geom[@name="tabletop_geom"]').set('name', 'lift_table_geom')
        for side in SIDES:
            site = root.find(f'.//site[@name="{side}_tcp"]')
            site.set('name', f'{side}_wrist')
            site.set('pos', '0 0 0')
        add_vertical_grip_crate(root, recording.crate, position=self.scene['crate_base_world_m'])
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        initialize_robot(self.model, self.data, self.hand_cfg)
        self.root_qadr = int(self.model.joint('floating_base_joint').qposadr[0])
        self.data.qpos[self.root_qadr:self.root_qadr+2] += self.origin[:2]
        mujoco.mj_forward(self.model, self.data)
        self.mapping = JointMap.create(self.model, BODY_JOINTS)
        self.original_ranges, self.original_inertia = self.model.jnt_range.copy(), self.model.body_inertia.copy()
        self.crate_qadr = int(self.model.joint('crate_free').qposadr[0])
        self.base = self.model.body('base_link').id
        self.wrists = [self.model.body(f'{s}_hand_roll_link').id for s in SIDES]
        self.feet = [self.model.body(f'{s}_ankle_roll_link').id for s in SIDES]
        self.foot_frames = [body_transform(self.data, i) for i in self.feet]
        self.foot_ids = set(foot_collision_ids(self.model))
        colliding = {g for g in range(self.model.ngeom) if self.model.geom_contype[g] or self.model.geom_conaffinity[g]}
        self.robot_geoms = {g for g in colliding if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)}
        self.hand_geoms = {g for g in self.robot_geoms if any(_descendant(self.model, int(self.model.geom_bodyid[g]), w) for w in self.wrists)}
        self.table_geoms = {g for g in colliding if self.model.geom_bodyid[g] == self.model.body('tabletop').id}
        self.crate_geoms = {g for g in colliding if self.model.geom_bodyid[g] == self.model.body('cargo_crate').id}
        self.floor = self.model.geom('floor').id
        self.com_target = self.data.subtree_com[self.base, :2].copy()
        centers = self.data.geom_xpos[list(self.foot_ids), :2]
        radii = self.model.geom_size[list(self.foot_ids), 0]
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        cloud = centers[:, None, :]+radii[:, None, None]*np.array([np.cos(angles), np.sin(angles)]).T
        hull = ConvexHull(cloud.reshape(-1, 2))
        self.support_equations = hull.equations.copy()
        self.support_min, self.support_max = (centers-radii[:,None]).min(0), (centers+radii[:,None]).max(0)
        self.initial_qpos = self.data.qpos.copy()
        limits = self.model.jnt_range[self.mapping.joints]
        padding = np.maximum(1e-6, margin*np.diff(limits, axis=1).ravel())
        self.lower = np.r_[self.origin[:2]-.20, .55, -.60, -.60, -.6, limits[:,0]+padding]
        self.upper = np.r_[self.origin[:2]+.20, 1.05, .60, .60, .6, limits[:,1]-padding]
        self.home = np.clip(self.pack(), self.lower+1e-7, self.upper-1e-7)
        self.hand_qadr = [self.model.joint(n).qposadr[0] for n in recording.hand_names]
        self.anchor = np.eye(4)
        self.anchor[:3,:3] = Rotation.from_quat(np.asarray(self.scene['crate_quaternion_wxyz'])[[1,2,3,0]]).as_matrix()
        self.anchor[:3,3] = np.asarray(self.scene['crate_base_world_m'])-self.anchor[:3,:3]@np.array([0,0,recording.params.table_height])
        self.pairs = {}
        for key, a, b in [('robot_table', self.robot_geoms, self.table_geoms),
                          ('nonhand_crate', self.robot_geoms-self.hand_geoms, self.crate_geoms)]:
            self.pairs[key] = np.array([(x,y) for x in sorted(a) for y in sorted(b)], dtype=int)
        self.previous = self.home.copy()

    def targets(self, t):
        crate, targets, fingers, phase = self.recording.sample(t, self.anchor)
        self.data.qpos[self.hand_qadr] = fingers
        self.data.qpos[self.crate_qadr:self.crate_qadr+7] = np.r_[crate[:3,3], Rotation.from_matrix(crate[:3,:3]).as_quat()[[3,0,1,2]]]
        self.current_phase = phase
        return crate, targets

    def clearances(self):
        values = {}
        for key, pairs in self.pairs.items():
            bound = (np.linalg.norm(self.data.geom_xpos[pairs[:,0]]-self.data.geom_xpos[pairs[:,1]], axis=1)
                     - self.model.geom_rbound[pairs[:,0]]-self.model.geom_rbound[pairs[:,1]])
            values[key] = min((checked_distance(self.model, self.data, int(a), int(b))
                               for a,b in pairs[bound<.03]), default=.03)
        return values

    def residual(self, x, targets):
        residual = super().residual(x, targets)
        clear = self.clearances()
        com = self.data.subtree_com[self.base,:2]
        support = self.support_equations[:,:2]@com+self.support_equations[:,2]
        mujoco.mj_collision(self.model, self.data)
        contacts = self.contacts()
        self_depth = max((c['penetration_m'] for c in contacts['robot_self']), default=0.)
        return np.r_[residual, [max(.003-v,0)/.0003 for v in clear.values()],
                     np.maximum(support+.003,0)/.001, max(self_depth-.001,0)/.0003,
                     .05*(x-self.previous)]

    def metrics(self, x, targets):
        m = super().metrics(x, targets)
        clear = self.clearances()
        support = self.support_equations[:,:2]@self.data.subtree_com[self.base,:2]+self.support_equations[:,2]
        contacts = m['contacts']
        hand_depth = max((c['penetration_m'] for c in contacts['hand_crate_open_pose']), default=0.)
        m.update(nonhand_prop_clearance_m=clear, com_support_margin_m=float(-support.max()),
                 hand_crate_penetration_m=hand_depth, hand_pose='measured fixture finger joint angles',
                 state_kind='OFFLINE_IK_NOT_PHYSICAL_FULL_BODY_GRASP')
        m['strict_static_candidate'] = bool(m['strict_static_candidate'] and min(clear.values())>=.002
            and support.max()<=0 and hand_depth<=.002 and m['base_tilt_deg']<35.)
        return m


def compact(row):
    return {k:v for k,v in row.items() if k not in ('body_joint_names','body_q_rad','joint_limit_margin_rad','joint_use_deg')}


def render_video(screen, times, states, rows, output):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    cameras = []
    for az, el, distance, target in [(140,-15,3.2,[.28,.44,1.05]), (90,-8,3.25,[.3,.44,1.05])]:
        c = mujoco.MjvCamera()
        c.azimuth, c.elevation, c.distance = az, el, distance
        c.lookat[:] = target
        cameras.append(c)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sample_times = np.arange(0,times[-1]+1e-8,1/30)
    with mujoco.Renderer(screen.model,width=640,height=720) as renderer, imageio.get_writer(
        str(output/'wholebody_ik.mp4'),fps=30,codec='libx264',quality=8) as writer:
        saved = set()
        for t in sample_times:
            x = np.array([np.interp(t,times,states[:,j]) for j in range(states.shape[1])])
            _, goal = screen.targets(t)
            m = screen.metrics(x,goal)
            panels=[]
            for c in cameras:
                renderer.update_scene(screen.data,camera=c,scene_option=options)
                panels.append(renderer.render())
            picture=Image.fromarray(np.concatenate(panels,axis=1));draw=ImageDraw.Draw(picture)
            draw.rectangle((0,0,1280,101),fill=(20,28,36))
            draw.text((12,7),'OFFLINE WHOLE-BODY IK | KINEMATIC REPLAY ONLY / NOT POLICY OR CONTACT ROLLOUT',font=font,fill=(255,205,90))
            draw.text((12,31),f't={t:.2f}s {screen.current_phase} | sampled pose gate={m["strict_static_candidate"]}',font=font,fill='white')
            pe=max(v['position_m'] for v in m['wrist_errors'].values())*1000
            re=max(v['orientation_deg'] for v in m['wrist_errors'].values())
            draw.text((12,55),f'Wrist error {pe:.2f} mm / {re:.2f} deg | min joint margin {np.rad2deg(m["min_joint_margin_rad"]):.2f} deg',font=font,fill='white')
            draw.text((12,79),'Fixed foot targets | 98 cm table | 36 x 24 x 16 cm crate | measured fingers + object trajectory',font=font,fill='white')
            writer.append_data(np.asarray(picture))
            if screen.current_phase not in saved:
                picture.save(output/f'{screen.current_phase.lower()}.png');saved.add(screen.current_phase)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--scene',type=Path,default=DEFAULT_SCENE)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['keyframes','path'],default='keyframes')
    p.add_argument('--dt',type=float,default=.25)
    p.add_argument('--max-nfev',type=int,default=100)
    p.add_argument('--seed-report',type=Path)
    p.add_argument('--seed-time',type=float,default=0.,help='Source-time row to use as the initial IK seed')
    p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if not .05<=args.dt<=1. or args.max_nfev<1:
        p.error('Invalid sampling or solver settings')
    args.output.mkdir(parents=True,exist_ok=False)
    recording=RecordedGrip(args.source)
    manifest=json.loads(args.scene.read_text())
    screen=VerticalGripScreen(recording,manifest)
    (args.output/'scene.xml').write_text(screen.xml)
    times=np.array([0.,3.49,6.49,8.49,12.5,15.49,19.49,22.49,24.5]) if args.mode=='keyframes' else np.unique(np.r_[np.arange(0,24.5,args.dt),24.5])
    x=screen.home.copy()
    if args.seed_report:
        seed=json.loads(args.seed_report.read_text())
        if seed['source_trajectory_sha256']!=recording.sha256:
            raise ValueError('Seed belongs to different fixture motion')
        seed_row=min(seed['points'],key=lambda r: abs(r['time_s']-args.seed_time))
        x=np.array(seed_row['packed_ik'])
    report=dict(scope='offline_sampled_IK_only_not_policy_not_dynamic_grasp',mode=args.mode,
        source=str(args.source.resolve()),source_trajectory_sha256=recording.sha256,
        seed_report=None if args.seed_report is None else str(args.seed_report.resolve()),seed_time_s=args.seed_time,
        distance_method='native distance, with independent mesh/box separating-plane lower bound for contradictory exact-zero results',
        scene=manifest['scene'],task_origin_in_beverage_world_m=screen.origin,
        scene_sha256=hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        joint_margin_fraction=.05, wrist_position_threshold_m=.005, wrist_orientation_threshold_deg=3.,
        foot_position_threshold_m=.003, foot_orientation_threshold_deg=1., prop_clearance_threshold_m=.002,
        only_empty_crate=True, limits_inertias_collisions_unchanged=True,
        support_test='robot COM in convex hull of initialized foot contact spheres; geometric proxy, not dynamics',
        entry_scope='starts at outside-hand waiting pose; locomotion-to-waiting transition not included',points=[])
    rows,states=[],[];started=time.monotonic()
    bounds=(screen.lower.copy(),screen.upper.copy())
    for i,t in enumerate(times):
        crate,goal=screen.targets(t);screen.previous=x.copy()
        if args.mode=='keyframes' or i==0:
            x,row=screen.solve(t,x,args.max_nfev,3,target_override=(crate,goal))
        else:
            step=np.r_[np.full(3,.1),np.full(3,.5),np.full(28,.7)]*(t-times[i-1])
            lo,hi=np.maximum(bounds[0],x-step),np.minimum(bounds[1],x+step)
            fit=least_squares(screen.residual,np.clip(x,lo+1e-9,hi-1e-9),args=(goal,),bounds=(lo,hi),
                x_scale='jac',max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            x=fit.x.copy();row=screen.metrics(x,goal)
            row.update(solver={'nfev':fit.nfev,'success':bool(fit.success)})
        row.update(time_s=float(t),phase=screen.current_phase,packed_ik=x.copy(),desired_wrist_world=goal,desired_crate_world=crate)
        rows.append(row);states.append(x.copy())
        report['points']=[compact(r) for r in rows];report['elapsed_s']=time.monotonic()-started
        save(args.output/'report.json',report)
        print(f'{t:.2f}s {screen.current_phase}: pass={row["strict_static_candidate"]} '
            f'p={max(v["position_m"] for v in row["wrist_errors"].values())*1000:.2f}mm '
            f'r={max(v["orientation_deg"] for v in row["wrist_errors"].values()):.2f}deg '
            f'margin={np.rad2deg(row["min_joint_margin_rad"]):.2f}deg '
            f'prop={min(row["nonhand_prop_clearance_m"].values())*1000:.2f}mm',flush=True)
    states=np.asarray(states)
    np.savez_compressed(args.output/'knots.npz',time_s=times,packed_ik=states)
    if args.mode=='path':
        dense_times=np.unique(np.r_[np.arange(0,24.5,.02),24.5]);dense=[];qpos=[]
        for t in dense_times:
            state=np.array([np.interp(t,times,states[:,j]) for j in range(states.shape[1])])
            _,goal=screen.targets(t);r=screen.metrics(state,goal)
            r.update(time_s=float(t),phase=screen.current_phase);dense.append(compact(r));qpos.append(screen.data.qpos.copy())
        save(args.output/'dense_metrics.json',dense)
        np.savez_compressed(args.output/'demonstration.npz',time_s=dense_times,qpos=np.array(qpos))
        evaluated=dense
    else:
        evaluated=rows
    report['summary']=dict(samples=len(evaluated),passed=sum(r['strict_static_candidate'] for r in evaluated),
        all_passed=all(r['strict_static_candidate'] for r in evaluated),
        max_wrist_position_mm=1000*max(v['position_m'] for r in evaluated for v in r['wrist_errors'].values()),
        max_wrist_orientation_deg=max(v['orientation_deg'] for r in evaluated for v in r['wrist_errors'].values()),
        min_joint_margin_deg=float(np.rad2deg(min(r['min_joint_margin_rad'] for r in evaluated))),
        max_foot_position_mm=1000*max(v['position_m'] for r in evaluated for v in r['foot_errors'].values()),
        max_foot_orientation_deg=max(v['orientation_deg'] for r in evaluated for v in r['foot_errors'].values()),
        min_nonhand_prop_clearance_mm=1000*min(v for r in evaluated for v in r['nonhand_prop_clearance_m'].values()),
        max_forbidden_penetration_mm=1000*max(r['max_forbidden_penetration_m'] for r in evaluated),
        min_com_support_margin_mm=1000*min(r['com_support_margin_m'] for r in evaluated),
        max_joint_speed_rad_s=float(np.max(abs(np.diff(states[:,6:],axis=0)/np.diff(times)[:,None]))),
        continuous_time_certified=False,dynamics_validated=False)
    report['elapsed_s']=time.monotonic()-started
    save(args.output/'report.json',report)
    print(json.dumps(report['summary'],indent=2),flush=True)
    if args.video:
        render_video(screen,times,states,rows,args.output)


if __name__=='__main__':
    main()
