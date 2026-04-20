from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import os
import onnxruntime as ort


class DualAgentVelocity(FSMState):
    """
    ONNX sim2sim runtime for the dual-agent velocity policy (upper + lower).

    Model: single ONNX file (dual_agent_combined.onnx) with two inputs.
      upper_obs: float32[1, 480]   per-term 5-frame history
      lower_obs: float32[1,  99]   single frame
      → actions: float32[1,  29]   already in MuJoCo motor order

    Upper obs per-frame (96):
      base_ang_vel(3) | projected_gravity(3) | velocity_commands(3) |
      joint_pos_rel(29) | joint_vel_rel(29) | last_action(29)
    History concatenation: [ang_vel×5, gravity×5, cmd×5, qpos×5, qvel×5, act×5]
    (per-term oldest→newest, matches IsaacLab ObservationManager flatten_history_dim).

    Lower obs (99):
      base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) |
      velocity_commands(3) | joint_pos_rel(29) | joint_vel_rel(29) |
      last_action(29)
    base_lin_vel has no scale in the training cfg (pass-through).

    Action handling differs from BoxTransportVelocity: the CombinedActor in
    dual_agent_export_onnx.py reorders the combined action to MuJoCo order
    before exiting the graph, so `policy_output.actions` is written directly
    from the ONNX output — no scatter via joint2motor_idx needed. The stored
    `self.action_isaac` is used only to feed last_action back into the next
    obs (training recorded last_action in Isaac order, pre-reorder).
    """

    # Upper / lower share the full 29-dim last_action (the "combined" action
    # before the output reorder, in Isaac order).
    needs_transport_box = True

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.DUAL_AGENT_VEL
        self.name_str = "dual_agent_velocity_mode"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "DualAgentVelocity.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path     = os.path.join(current_dir, "model", config["policy_path"])
        self.kps             = np.array(config["kps"],             dtype=np.float32)
        self.kds             = np.array(config["kds"],             dtype=np.float32)
        self.default_angles  = np.array(config["default_angles"],  dtype=np.float32)
        self.action_scale    = np.array(config["action_scale"],    dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)

        self.ang_vel_scale    = config["ang_vel_scale"]
        self.lin_vel_scale    = config["lin_vel_scale"]
        self.gravity_scale    = config["gravity_scale"]
        self.cmd_scale        = np.array(config["cmd_scale"], dtype=np.float32)
        self.dof_pos_scale    = config["dof_pos_scale"]
        self.dof_vel_scale    = config["dof_vel_scale"]
        self.obs_clip_default = config["obs_clip_default"]
        self.last_action_clip = config["last_action_clip"]

        self.num_actions      = config["num_actions"]
        self.history_length_upper = config["history_length_upper"]
        self.num_obs_upper    = config["num_obs_upper"]
        self.num_obs_lower    = config["num_obs_lower"]

        self.control_dt    = config["control_dt"]
        self.ramp_time     = config["ramp_time"]
        self.ramp_num_step = max(1, int(self.ramp_time / self.control_dt))
        self.ramp_kp_scale = config.get("ramp_kp_scale", 1.0)
        self.ramp_kd_scale = config.get("ramp_kd_scale", 1.0)

        cmd_range = config["cmd_range"]
        self.range_velx = np.array(cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.array(cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.array(cmd_range["ang_vel_z"], dtype=np.float32)

        # Scratch buffers (Isaac order for obs, MuJoCo order for control).
        self.qj_obs         = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs        = np.zeros(self.num_actions, dtype=np.float32)
        self.action_isaac   = np.zeros(self.num_actions, dtype=np.float32)  # raw, Isaac order, feeds next last_action
        self.upper_obs_flat = np.zeros(self.num_obs_upper, dtype=np.float32)
        self.lower_obs_flat = np.zeros(self.num_obs_lower, dtype=np.float32)

        # Upper per-term history rings, shape (H, dim). Oldest at index 0.
        H = self.history_length_upper
        self.hist_ang_vel = np.zeros((H, 3),                dtype=np.float32)
        self.hist_gravity = np.zeros((H, 3),                dtype=np.float32)
        self.hist_cmd     = np.zeros((H, 3),                dtype=np.float32)
        self.hist_qpos    = np.zeros((H, self.num_actions), dtype=np.float32)
        self.hist_qvel    = np.zeros((H, self.num_actions), dtype=np.float32)
        self.hist_action  = np.zeros((H, self.num_actions), dtype=np.float32)
        self._history_primed = False

        # ONNX session.
        self.sess = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"])
        in_names  = [i.name for i in self.sess.get_inputs()]
        out_names = [o.name for o in self.sess.get_outputs()]
        assert "upper_obs" in in_names and "lower_obs" in in_names, \
            f"ONNX inputs must include 'upper_obs' and 'lower_obs', got {in_names}"
        assert "actions" in out_names, \
            f"ONNX output must include 'actions', got {out_names}"

        # Dim sanity.
        dims = {i.name: i.shape[-1] for i in self.sess.get_inputs()}
        assert dims["upper_obs"] == self.num_obs_upper, \
            f"upper_obs dim {dims['upper_obs']} != {self.num_obs_upper}"
        assert dims["lower_obs"] == self.num_obs_lower, \
            f"lower_obs dim {dims['lower_obs']} != {self.num_obs_lower}"

        # Warm-up.
        warm_u = np.zeros((1, self.num_obs_upper), dtype=np.float32)
        warm_l = np.zeros((1, self.num_obs_lower), dtype=np.float32)
        for _ in range(5):
            self.sess.run(["actions"], {"upper_obs": warm_u, "lower_obs": warm_l})

        print("DualAgentVelocity policy initializing (backend=onnx, dual-input) ...")

    def enter(self):
        # Reorder per-joint constants from Isaac order (yaml) to MuJoCo order.
        # ONNX output is MuJoCo order, so q_cmd is computed directly in MuJoCo.
        self.kps_reorder            = np.zeros_like(self.kps)
        self.kds_reorder            = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        self.action_scale_reorder   = np.zeros_like(self.action_scale)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.kps_reorder[motor_idx]            = self.kps[i]
            self.kds_reorder[motor_idx]            = self.kds[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]
            self.action_scale_reorder[motor_idx]   = self.action_scale[i]

        self.ramp_kps = (self.kps_reorder * self.ramp_kp_scale).astype(np.float32)
        self.ramp_kds = (self.kds_reorder * self.ramp_kd_scale).astype(np.float32)

        # Reset history + last-action memory.
        self.hist_ang_vel.fill(0.0)
        self.hist_gravity.fill(0.0)
        self.hist_cmd.fill(0.0)
        self.hist_qpos.fill(0.0)
        self.hist_qvel.fill(0.0)
        self.hist_action.fill(0.0)
        self._history_primed = False
        self.action_isaac.fill(0.0)

        # Ramp from entry pose to default (MuJoCo order). Both arm ±1 rad pose
        # and the fact that lower obs sees these joints make a hard switch
        # risky from LocoMode — keep ramp_time configurable.
        self.ramp_init_motor_pos = self.state_cmd.q.copy().astype(np.float32)
        self.ramp_cur_step = 0
        self.ramping = True
        print(f"DualAgentVelocity: ramping to default pose over {self.ramp_time:.2f}s "
              f"({self.ramp_num_step} ticks) before policy inference starts.")

    @staticmethod
    def _push(buf, new_row):
        buf[:-1] = buf[1:]
        buf[-1]  = new_row

    @staticmethod
    def _prime(buf, new_row):
        buf[:] = new_row

    def run(self):
        # Ramp-in: PD-hold while interpolating to default_angles (MuJoCo order).
        if self.ramping:
            self.ramp_cur_step += 1
            alpha  = min(self.ramp_cur_step / self.ramp_num_step, 1.0)
            target = (self.ramp_init_motor_pos * (1.0 - alpha)
                      + self.default_angles_reorder * alpha).astype(np.float32)
            self.policy_output.actions = target
            self.policy_output.kps     = self.ramp_kps.copy()
            self.policy_output.kds     = self.ramp_kds.copy()
            if alpha >= 1.0:
                self.ramping = False
                print("DualAgentVelocity: ramp complete, starting policy inference.")
            return

        # 1. robot state (MuJoCo order from bus).
        gravity    = self.state_cmd.gravity_ori.copy()
        ang_vel    = self.state_cmd.ang_vel.copy()
        lin_vel_b  = self.state_cmd.base_lin_vel.copy().astype(np.float32)
        qj_motor   = self.state_cmd.q.copy()
        dqj_motor  = self.state_cmd.dq.copy()
        joycmd     = self.state_cmd.vel_cmd.copy()
        cmd        = scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])

        # 2. gather motor-order → Isaac order for joint state.
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i]  = qj_motor[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj_motor[self.joint2motor_idx[i]]

        # 3. per-term scale + clip (matches training ObsManager).
        C = self.obs_clip_default
        ang_vel_s     = np.clip(ang_vel * self.ang_vel_scale, -C, C).astype(np.float32)
        lin_vel_s     = np.clip(lin_vel_b * self.lin_vel_scale, -C, C).astype(np.float32)
        gravity_s     = (gravity * self.gravity_scale).astype(np.float32)
        cmd_s         = (cmd * self.cmd_scale).astype(np.float32)
        joint_pos_rel = np.clip((self.qj_obs - self.default_angles) * self.dof_pos_scale,
                                -C, C).astype(np.float32)
        joint_vel_s   = np.clip(self.dqj_obs * self.dof_vel_scale, -C, C).astype(np.float32)
        last_action_s = np.clip(self.action_isaac, -self.last_action_clip,
                                self.last_action_clip).astype(np.float32)

        # 4. upper per-term history update (same 6 terms as per-frame upper obs).
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

        # 5. assemble upper (per-term history concat) and lower (single frame).
        self.upper_obs_flat = np.concatenate([
            self.hist_ang_vel.reshape(-1),
            self.hist_gravity.reshape(-1),
            self.hist_cmd.reshape(-1),
            self.hist_qpos.reshape(-1),
            self.hist_qvel.reshape(-1),
            self.hist_action.reshape(-1),
        ]).astype(np.float32)
        assert self.upper_obs_flat.shape[0] == self.num_obs_upper

        self.lower_obs_flat[0:3]    = lin_vel_s
        self.lower_obs_flat[3:6]    = ang_vel_s
        self.lower_obs_flat[6:9]    = gravity_s
        self.lower_obs_flat[9:12]   = cmd_s
        self.lower_obs_flat[12:41]  = joint_pos_rel
        self.lower_obs_flat[41:70]  = joint_vel_s
        self.lower_obs_flat[70:99]  = last_action_s

        # 6. inference.
        u_in = np.clip(self.upper_obs_flat.reshape(1, -1), -100.0, 100.0)
        l_in = np.clip(self.lower_obs_flat.reshape(1, -1), -100.0, 100.0)
        raw_mujoco = self.sess.run(
            ["actions"],
            {"upper_obs": u_in, "lower_obs": l_in},
        )[0]
        action_mujoco = np.clip(raw_mujoco, -100.0, 100.0).squeeze().astype(np.float32)

        # 7. Store Isaac-order raw action for next-tick last_action obs.
        for i in range(len(self.joint2motor_idx)):
            self.action_isaac[i] = action_mujoco[self.joint2motor_idx[i]]

        # 8. q_cmd in MuJoCo order (action is already MuJoCo-ordered).
        q_cmd_motor = action_mujoco * self.action_scale_reorder + self.default_angles_reorder
        self.policy_output.actions = q_cmd_motor.astype(np.float32)
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
        elif self.state_cmd.skill_cmd == FSMCommand.SKILL_BOX_TRANSPORT_V:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_BOX_TRANSPORT_V
        else:
            return FSMStateName.DUAL_AGENT_VEL
