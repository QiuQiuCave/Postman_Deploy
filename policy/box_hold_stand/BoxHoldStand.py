from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
import numpy as np
import yaml
import os
import onnxruntime as ort


class BoxHoldStand(FSMState):
    """
    Second no-tracking box policy.

    The actor uses the same 96-dim single-frame proprioceptive observation as
    BoxTransportVelocity, but with a zero velocity command. In MuJoCo this state
    asks the deploy loop to place the shared transport_box into the grasp area.
    On real hardware the box is supplied manually before switching into this
    policy.
    """

    needs_transport_box = True
    transport_box_offset_base = (0.32, 0.0, 0.22)

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.BOX_HOLD_STAND
        self.name_str = "box_hold_stand_mode"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "BoxHoldStand.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path = os.path.join(current_dir, "model", config["policy_path"])
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)

        self.ang_vel_scale = config["ang_vel_scale"]
        self.gravity_scale = config["gravity_scale"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.obs_clip_default = config["obs_clip_default"]
        self.last_action_clip = config["last_action_clip"]
        self.history_length = config["history_length"]
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.control_dt = config["control_dt"]
        self.ramp_time = config["ramp_time"]
        self.ramp_num_step = max(1, int(self.ramp_time / self.control_dt))
        self.ramp_kp_scale = config.get("ramp_kp_scale", 1.0)
        self.ramp_kd_scale = config.get("ramp_kd_scale", 1.0)

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.zero_cmd = np.zeros(3, dtype=np.float32)

        self.sess = None
        self.onnx_in = None
        self.onnx_out = None
        self.use_policy = False

        print("BoxHoldStand policy initializing (lazy ONNX load) ...")

    def _load_policy_if_available(self):
        if self.sess is not None:
            self.use_policy = True
            return
        if not os.path.exists(self.policy_path):
            self.use_policy = False
            print(f"BoxHoldStand: missing ONNX at {self.policy_path}; holding default pose.")
            return

        self.sess = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
        self.onnx_in = self.sess.get_inputs()[0].name
        self.onnx_out = self.sess.get_outputs()[0].name

        in_shape = self.sess.get_inputs()[0].shape
        out_shape = self.sess.get_outputs()[0].shape
        assert in_shape[-1] == self.num_obs, f"ONNX obs dim {in_shape[-1]} != {self.num_obs}"
        assert out_shape[-1] == self.num_actions, f"ONNX act dim {out_shape[-1]} != {self.num_actions}"

        warm = np.zeros((1, self.num_obs), dtype=np.float32)
        for _ in range(5):
            self.sess.run([self.onnx_out], {self.onnx_in: warm})
        self.use_policy = True
        print("BoxHoldStand: ONNX loaded, starting learned policy after ramp.")

    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.kps_reorder[motor_idx] = self.kps[i]
            self.kds_reorder[motor_idx] = self.kds[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]

        self.ramp_kps = (self.kps_reorder * self.ramp_kp_scale).astype(np.float32)
        self.ramp_kds = (self.kds_reorder * self.ramp_kd_scale).astype(np.float32)

        self.action.fill(0.0)
        self.ramp_init_motor_pos = self.state_cmd.q.copy().astype(np.float32)
        self.ramp_cur_step = 0
        self.ramping = True
        self._load_policy_if_available()
        print(f"BoxHoldStand: ramping to hold pose over {self.ramp_time:.2f}s "
              f"({self.ramp_num_step} ticks).")

    def run(self):
        if self.ramping:
            self.ramp_cur_step += 1
            alpha = min(self.ramp_cur_step / self.ramp_num_step, 1.0)
            target = (self.ramp_init_motor_pos * (1.0 - alpha)
                      + self.default_angles_reorder * alpha).astype(np.float32)
            self.policy_output.actions = target
            self.policy_output.kps = self.ramp_kps.copy()
            self.policy_output.kds = self.ramp_kds.copy()
            if alpha >= 1.0:
                self.ramping = False
                if self.use_policy:
                    print("BoxHoldStand: ramp complete, starting policy inference.")
                else:
                    print("BoxHoldStand: ramp complete, holding default pose.")
            return

        if not self.use_policy:
            self.policy_output.actions = self.default_angles_reorder.copy()
            self.policy_output.kps = self.kps_reorder.copy()
            self.policy_output.kds = self.kds_reorder.copy()
            return

        gravity = self.state_cmd.gravity_ori.copy()
        ang_vel = self.state_cmd.ang_vel.copy()
        qj = self.state_cmd.q.copy()
        dqj = self.state_cmd.dq.copy()

        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj[self.joint2motor_idx[i]]

        c = self.obs_clip_default
        ang_vel_s = np.clip(ang_vel * self.ang_vel_scale, -c, c).astype(np.float32)
        gravity_s = (gravity * self.gravity_scale).astype(np.float32)
        cmd_s = (self.zero_cmd * self.cmd_scale).astype(np.float32)
        joint_pos_rel = np.clip((self.qj_obs - self.default_angles) * self.dof_pos_scale, -c, c).astype(np.float32)
        joint_vel_s = np.clip(self.dqj_obs * self.dof_vel_scale, -c, c).astype(np.float32)
        last_action_s = np.clip(self.action, -self.last_action_clip, self.last_action_clip).astype(np.float32)

        self.obs[0:3] = ang_vel_s
        self.obs[3:6] = gravity_s
        self.obs[6:9] = cmd_s
        self.obs[9:38] = joint_pos_rel
        self.obs[38:67] = joint_vel_s
        self.obs[67:96] = last_action_s

        x = np.clip(self.obs.reshape(1, -1), -100.0, 100.0)
        raw = self.sess.run([self.onnx_out], {self.onnx_in: x})[0]
        self.action = np.clip(raw, -100.0, 100.0).squeeze().astype(np.float32)

        q_cmd_policy = self.action * self.action_scale + self.default_angles
        action_motor = q_cmd_policy.copy()
        for i in range(len(self.joint2motor_idx)):
            action_motor[self.joint2motor_idx[i]] = q_cmd_policy[i]

        self.policy_output.actions = action_motor.copy()
        self.policy_output.kps = self.kps_reorder.copy()
        self.policy_output.kds = self.kds_reorder.copy()

    def exit(self):
        pass

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif self.state_cmd.skill_cmd == FSMCommand.BOX_HANDOFF_STAND:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.BOX_HANDOFF_STAND
        elif self.state_cmd.skill_cmd == FSMCommand.DUAL_AGENT_TRACK:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DUAL_AGENT_TRACK
        elif self.state_cmd.skill_cmd == FSMCommand.DUAL_AGENT_RUN_TRACK:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DUAL_AGENT_RUN_TRACK
        elif self.state_cmd.skill_cmd == FSMCommand.DUAL_AGENT_JUMP_TRACK:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DUAL_AGENT_JUMP_TRACK
        elif self.state_cmd.skill_cmd == FSMCommand.DUAL_AGENT_DANCE_TRACK:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DUAL_AGENT_DANCE_TRACK
        else:
            return FSMStateName.BOX_HOLD_STAND
