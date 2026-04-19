from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import os
import onnxruntime as ort


class BoxTransportVelocity(FSMState):
    """
    ONNX sim2sim runtime for the g1_box_transport_velocity policy.

    Obs contract (matches IsaacLab ObservationManager with history_length=5,
    flatten_history_dim=true, concatenate_terms=true, per-term concatenation
    in yaml order, oldest -> newest within each term's window):

      [base_ang_vel_hist (3*5=15) |
       projected_gravity_hist (3*5=15) |
       velocity_commands_hist (3*5=15) |
       joint_pos_rel_hist (29*5=145) |
       joint_vel_rel_hist (29*5=145) |
       last_action_hist (29*5=145)]
      = 480

    Per-term scaling is applied BEFORE the value is pushed into its history
    window (matches IsaacLab term-level scale).
    """

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_BOX_TRANSPORT_V
        self.name_str = "box_transport_velocity_mode"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "BoxTransportVelocity.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path     = os.path.join(current_dir, "model", config["policy_path"])
        self.kps             = np.array(config["kps"], dtype=np.float32)
        self.kds             = np.array(config["kds"], dtype=np.float32)
        self.default_angles  = np.array(config["default_angles"], dtype=np.float32)
        self.action_scale    = np.array(config["action_scale"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)

        self.ang_vel_scale    = config["ang_vel_scale"]
        self.gravity_scale    = config["gravity_scale"]
        self.cmd_scale        = np.array(config["cmd_scale"], dtype=np.float32)
        self.dof_pos_scale    = config["dof_pos_scale"]
        self.dof_vel_scale    = config["dof_vel_scale"]
        self.obs_clip_default = config["obs_clip_default"]
        self.last_action_clip = config["last_action_clip"]
        self.history_length   = config["history_length"]
        self.num_actions      = config["num_actions"]
        self.num_obs          = config["num_obs"]
        self.control_dt       = config["control_dt"]
        self.ramp_time        = config["ramp_time"]
        self.ramp_num_step    = max(1, int(self.ramp_time / self.control_dt))

        cmd_range = config["cmd_range"]
        self.range_velx = np.array(cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.array(cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.array(cmd_range["ang_vel_z"], dtype=np.float32)

        self.qj_obs  = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.action  = np.zeros(self.num_actions, dtype=np.float32)  # raw, pre-scale

        # Per-term history windows: shape (H, dim), oldest at index 0.
        H = self.history_length
        self.hist_ang_vel = np.zeros((H, 3),                 dtype=np.float32)
        self.hist_gravity = np.zeros((H, 3),                 dtype=np.float32)
        self.hist_cmd     = np.zeros((H, 3),                 dtype=np.float32)
        self.hist_qpos    = np.zeros((H, self.num_actions),  dtype=np.float32)
        self.hist_qvel    = np.zeros((H, self.num_actions),  dtype=np.float32)
        self.hist_action  = np.zeros((H, self.num_actions),  dtype=np.float32)
        self._history_primed = False

        self.sess = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"])
        self.onnx_in  = self.sess.get_inputs()[0].name
        self.onnx_out = self.sess.get_outputs()[0].name

        # sanity: input/output dims.
        in_shape  = self.sess.get_inputs()[0].shape
        out_shape = self.sess.get_outputs()[0].shape
        assert in_shape[-1]  == self.num_obs,     f"ONNX obs dim {in_shape[-1]} != {self.num_obs}"
        assert out_shape[-1] == self.num_actions, f"ONNX act dim {out_shape[-1]} != {self.num_actions}"

        # warm-up
        warm = np.zeros((1, self.num_obs), dtype=np.float32)
        for _ in range(5):
            self.sess.run([self.onnx_out], {self.onnx_in: warm})

        print("BoxTransportVelocity policy initializing (backend=onnx) ...")

    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.kps_reorder[motor_idx]            = self.kps[i]
            self.kds_reorder[motor_idx]            = self.kds[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]

        # Fresh history on re-entry; _history_primed=False means next run() fills
        # all H slots with the current frame.
        self.hist_ang_vel.fill(0.0)
        self.hist_gravity.fill(0.0)
        self.hist_cmd.fill(0.0)
        self.hist_qpos.fill(0.0)
        self.hist_qvel.fill(0.0)
        self.hist_action.fill(0.0)
        self.action.fill(0.0)
        self._history_primed = False

        # Snapshot the previous policy's last motor pose; ramp linearly toward
        # default_angles (motor order) over ramp_time seconds before inference.
        self.ramp_init_motor_pos = self.state_cmd.q.copy().astype(np.float32)
        self.ramp_cur_step = 0
        self.ramping = True
        print(f"BoxTransportVelocity: ramping to default pose over {self.ramp_time:.2f}s "
              f"({self.ramp_num_step} ticks) before policy inference starts.")

    def _push(self, buf, new_row):
        """Roll oldest out, append newest at the end (buf[-1])."""
        buf[:-1] = buf[1:]
        buf[-1]  = new_row

    def _prime(self, buf, new_row):
        """Fill every slot with new_row (first-frame init, mirrors CircularBuffer)."""
        buf[:] = new_row

    def run(self):
        # 0. ramp-in phase: hold the box-transport PD and interpolate from the
        # entry pose to default_angles. Skip inference + history updates so the
        # policy's first real obs sees a near-default joint configuration.
        if self.ramping:
            self.ramp_cur_step += 1
            alpha = min(self.ramp_cur_step / self.ramp_num_step, 1.0)
            target = (self.ramp_init_motor_pos * (1.0 - alpha)
                      + self.default_angles_reorder * alpha).astype(np.float32)
            self.policy_output.actions = target
            self.policy_output.kps     = self.kps_reorder.copy()
            self.policy_output.kds     = self.kds_reorder.copy()
            if alpha >= 1.0:
                self.ramping = False
                print("BoxTransportVelocity: ramp complete, starting policy inference.")
            return

        # 1. read robot state
        gravity = self.state_cmd.gravity_ori.copy()
        ang_vel = self.state_cmd.ang_vel.copy()
        qj      = self.state_cmd.q.copy()
        dqj     = self.state_cmd.dq.copy()
        joycmd  = self.state_cmd.vel_cmd.copy()
        cmd     = scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])

        # 2. motor-order -> policy-order for joint state
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i]  = qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj[self.joint2motor_idx[i]]

        # 3. per-term scale + clip (matches IsaacLab obs term order/ops)
        C = self.obs_clip_default
        ang_vel_s      = np.clip(ang_vel * self.ang_vel_scale, -C, C).astype(np.float32)
        gravity_s      = (gravity * self.gravity_scale).astype(np.float32)
        cmd_s          = (cmd * self.cmd_scale).astype(np.float32)
        joint_pos_rel  = np.clip((self.qj_obs - self.default_angles) * self.dof_pos_scale,
                                 -C, C).astype(np.float32)
        joint_vel_s    = np.clip(self.dqj_obs * self.dof_vel_scale, -C, C).astype(np.float32)
        last_action_s  = np.clip(self.action, -self.last_action_clip,
                                 self.last_action_clip).astype(np.float32)

        # 4. update per-term history windows
        if not self._history_primed:
            self._prime(self.hist_ang_vel, ang_vel_s)
            self._prime(self.hist_gravity, gravity_s)
            self._prime(self.hist_cmd,     cmd_s)
            self._prime(self.hist_qpos,    joint_pos_rel)
            self._prime(self.hist_qvel,    joint_vel_s)
            self._prime(self.hist_action,  last_action_s)
            self._history_primed = True
        else:
            self._push(self.hist_ang_vel, ang_vel_s)
            self._push(self.hist_gravity, gravity_s)
            self._push(self.hist_cmd,     cmd_s)
            self._push(self.hist_qpos,    joint_pos_rel)
            self._push(self.hist_qvel,    joint_vel_s)
            self._push(self.hist_action,  last_action_s)

        # 5. assemble flattened obs: per-term history (oldest->newest) concatenated
        obs = np.concatenate([
            self.hist_ang_vel.reshape(-1),
            self.hist_gravity.reshape(-1),
            self.hist_cmd.reshape(-1),
            self.hist_qpos.reshape(-1),
            self.hist_qvel.reshape(-1),
            self.hist_action.reshape(-1),
        ]).astype(np.float32)
        assert obs.shape[0] == self.num_obs

        # 6. inference (outer clip = safety belt)
        x = np.clip(obs.reshape(1, -1), -100.0, 100.0)
        raw = self.sess.run([self.onnx_out], {self.onnx_in: x})[0]
        self.action = np.clip(raw, -100.0, 100.0).squeeze().astype(np.float32)

        # 7. action -> q_cmd (policy order, per-joint scale + default offset) -> motor order
        q_cmd_policy = self.action * self.action_scale + self.default_angles
        action_motor = q_cmd_policy.copy()
        for i in range(len(self.joint2motor_idx)):
            action_motor[self.joint2motor_idx[i]] = q_cmd_policy[i]

        self.policy_output.actions = action_motor.copy()
        self.policy_output.kps     = self.kps_reorder.copy()
        self.policy_output.kds     = self.kds_reorder.copy()

    def exit(self):
        pass

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif self.state_cmd.skill_cmd == FSMCommand.LOCO_NEW:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCO_NEW
        elif self.state_cmd.skill_cmd == FSMCommand.LOCO_NEW_ONNX:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCO_NEW_ONNX
        else:
            return FSMStateName.SKILL_BOX_TRANSPORT_V
