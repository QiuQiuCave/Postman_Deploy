"""14-DoF locomotion adapter: never observes or writes arm/finger channels.

The exported normalizer is inside ONNX. History is term-major, oldest first.
This adapter alone is not evidence of transfer to the articulated-hand robot.
"""
import mujoco
import numpy as np
import onnxruntime as ort

from r2v2_description.model import BODY_JOINTS, JointMap


LOCO_JOINTS = BODY_JOINTS[:14]
TERM_DIMS = dict(base_lin_vel=3, base_ang_vel=3, projected_gravity=3,
                 joint_pos=14, joint_vel=14, actions=14, command=3,
                 base_pose_command=2)


class LocoPolicy:
    def __init__(self, model, data, path):
        self.model = model
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        meta = self.session.get_modelmeta().custom_metadata_map
        if tuple(meta['joint_names'].split(',')) != LOCO_JOINTS:
            raise ValueError('Not the 14-DoF R2V2 loco joint contract')
        if tuple(meta['observation_names'].split(',')) != tuple(TERM_DIMS):
            raise ValueError('Incorrect loco observation order')
        assert self.session.get_inputs()[0].shape == [1, 560]
        assert self.session.get_outputs()[0].shape == [1, 14]
        self.default = np.fromstring(meta['default_joint_pos'], sep=',')
        self.kp = np.fromstring(meta['joint_stiffness'], sep=',')
        self.kd = np.fromstring(meta['joint_damping'], sep=',')
        self.scale = float(meta['action_scale'])
        prefix = '' if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link') >= 0 else 'robot/'
        self.mapping = JointMap.create(model, tuple(prefix+n for n in LOCO_JOINTS))
        self.base = model.body(prefix+'base_link').id
        self.command = np.zeros(3)
        self.pose_command = np.array([.82, 0.])
        self.jp = np.zeros((3,model.nv))
        self.jr = np.zeros((3,model.nv))
        self.reset(data)

    def terms(self, data):
        rotation = data.xmat[self.base].reshape(3,3)
        mujoco.mj_jacBody(self.model,data,self.jp,self.jr,self.base)
        return {k:np.asarray(v,dtype=np.float32) for k,v in dict(
            base_lin_vel=rotation.T@(self.jp@data.qvel),
            base_ang_vel=rotation.T@(self.jr@data.qvel),
            projected_gravity=rotation.T@np.array([0.,0.,-1.]),
            joint_pos=data.qpos[self.mapping.qpos]-self.default,
            joint_vel=data.qvel[self.mapping.dofs], actions=self.last_action.copy(),
            command=self.command.copy(),base_pose_command=self.pose_command.copy()).items()}

    def reset(self, data, previous_target=None):
        self.last_action = (np.zeros(14) if previous_target is None else
                            (np.asarray(previous_target)-self.default)/self.scale).astype(np.float32)
        self.histories = {k:np.repeat(v[None],10,axis=0) for k,v in self.terms(data).items()}
        self.q_des = self.default+self.scale*self.last_action

    def observe(self, data):
        for k,v in self.terms(data).items():
            self.histories[k][:-1] = self.histories[k][1:]
            self.histories[k][-1] = v
        result = np.concatenate([self.histories[k].ravel() for k in TERM_DIMS])
        if result.shape != (560,) or not np.isfinite(result).all():
            raise ValueError('Invalid loco observation')
        return result

    def act(self, data):
        self.last_action = self.session.run(['actions'],{'obs':self.observe(data)[None]})[0][0]
        if self.last_action.shape != (14,) or not np.isfinite(self.last_action).all():
            raise ValueError('Invalid loco action')
        self.q_des = self.default+self.scale*self.last_action
        return self.q_des.copy()
