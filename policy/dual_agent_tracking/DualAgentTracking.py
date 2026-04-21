from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import os
import onnxruntime as ort


# -----------------------------------------------------------------------------
# Small numpy quaternion utilities (wxyz convention, matches MuJoCo & Isaac).
# -----------------------------------------------------------------------------
def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_rotate_inverse(q, v):
    """Rotate vector v from world to body frame, given body quat q (wxyz)."""
    w, x, y, z = q
    # r = q^-1 * [0, v] * q  → equivalent to this closed form
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


def quat_to_mat(q):
    """wxyz quaternion → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def subtract_frame_transforms_b(robot_pos, robot_quat, target_pos, target_quat):
    """
    Express (target_pos, target_quat) in robot's body frame.
    Matches beyondMimic.mdp.observations.subtract_frame_transforms.

    Returns:
        pos_b: (3,) target pos in robot body frame
        quat_b: (4,) target quat in robot body frame (wxyz)
    """
    pos_b = quat_rotate_inverse(robot_quat, target_pos - robot_pos)
    quat_b = quat_mul(quat_conjugate(robot_quat), target_quat)
    return pos_b.astype(np.float32), quat_b.astype(np.float32)


# -----------------------------------------------------------------------------
# Motion buffer: advances one frame per control tick, wraps on end.
# -----------------------------------------------------------------------------
class MotionBuffer:
    def __init__(self, motion_path: str):
        data = np.load(motion_path)
        self.lower_joint_pos = data["lower_joint_pos"].astype(np.float32)   # (T, 15)
        self.lower_joint_vel = data["lower_joint_vel"].astype(np.float32)   # (T, 15)
        self.torso_pos_w     = data["torso_pos_w"].astype(np.float32)       # (T, 3)
        self.torso_quat_w    = data["torso_quat_w"].astype(np.float32)      # (T, 4) wxyz
        self.fps             = int(data["fps"].item() if hasattr(data["fps"], "item")
                                   else data["fps"][0])
        self.total_frames    = self.lower_joint_pos.shape[0]
        self.frame_idx       = 0

    def reset(self):
        self.frame_idx = 0

    def read(self):
        i = self.frame_idx
        return (
            self.lower_joint_pos[i],
            self.lower_joint_vel[i],
            self.torso_pos_w[i],
            self.torso_quat_w[i],
        )

    def advance(self):
        self.frame_idx = (self.frame_idx + 1) % self.total_frames


class DualAgentTracking(FSMState):
    """
    ONNX sim2sim runtime for the dual-agent *tracking* policy (upper + lower).

    Model: single ONNX file (dual_agent_combined.onnx) with two inputs.
      upper_obs: float32[1, 480]   per-term 5-frame history (same as velocity)
      lower_obs: float32[1, 121]   single frame, tracking-specific
      → actions: float32[1,  29]   already in MuJoCo motor order

    Upper obs per-frame (96): identical to DualAgentBoxTransVel upper branch
      base_ang_vel(3) | projected_gravity(3) | velocity_commands(3) |
      joint_pos_rel(29) | joint_vel_rel(29) | last_action(29)

    Lower obs (121) — order from dual_agent_train_env_cfg.LowerBodyPolicyCfg:
      lower_body_command(30)    # 15 joint_pos + 15 joint_vel from motion npz
      motion_anchor_pos_b(3)    # motion torso pos in robot torso frame
      motion_anchor_ori_b(6)    # first 2 columns of R_{robot_torso → motion_torso}
      projected_gravity(3)
      base_lin_vel(3)           # raw, no scale
      base_ang_vel(3)           # raw, no scale (differs from velocity variant!)
      joint_pos(29)             # raw joint_pos_rel, no scale (differs!)
      joint_vel(29)             # raw joint_vel_rel, no scale (differs!)
      actions(15)               # last lower action, Isaac slots 0..14, no clip

    The CombinedActor (dual_agent_export_onnx.CombinedActor) reorders the
    combined 29-dim action to MuJoCo order inside the graph, so q_cmd is
    computed directly in MuJoCo order. We still keep an Isaac-order copy of
    the raw action to feed last_action back into the next obs.
    """

    # Align with other box-carrying policies: teleport the shared transport_box
    # into the grasp region on entry so the upper actor has something to hold.
    needs_transport_box = True

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.DUAL_AGENT_TRACK
        self.name_str = "dual_agent_tracking_mode"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "DualAgentTracking.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.policy_path     = os.path.join(current_dir, "model",  config["policy_path"])
        self.motion_path     = os.path.join(current_dir, "motion", config["motion_file"])
        self.kps             = np.array(config["kps"],             dtype=np.float32)
        self.kds             = np.array(config["kds"],             dtype=np.float32)
        self.default_angles  = np.array(config["default_angles"],  dtype=np.float32)
        self.action_scale    = np.array(config["action_scale"],    dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)

        self.ang_vel_scale    = config["ang_vel_scale"]
        self.gravity_scale    = config["gravity_scale"]
        self.cmd_scale        = np.array(config["cmd_scale"], dtype=np.float32)
        self.dof_pos_scale    = config["dof_pos_scale"]
        self.dof_vel_scale    = config["dof_vel_scale"]
        self.obs_clip_default = config["obs_clip_default"]
        self.last_action_clip = config["last_action_clip"]

        self.num_actions          = config["num_actions"]
        self.history_length_upper = config["history_length_upper"]
        self.num_obs_upper        = config["num_obs_upper"]
        self.num_obs_lower        = config["num_obs_lower"]
        self.anchor_body_name     = config["anchor_body_name"]

        self.control_dt    = config["control_dt"]
        self.ramp_time     = config["ramp_time"]
        self.ramp_num_step = max(1, int(self.ramp_time / self.control_dt))
        self.ramp_kp_scale = config.get("ramp_kp_scale", 1.0)
        self.ramp_kd_scale = config.get("ramp_kd_scale", 1.0)

        cmd_range = config["cmd_range"]
        self.range_velx = np.array(cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.array(cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.array(cmd_range["ang_vel_z"], dtype=np.float32)

        # Scratch buffers.
        self.qj_obs         = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs        = np.zeros(self.num_actions, dtype=np.float32)
        # Upper last_action = full 29-dim Isaac-order raw action (pre-reorder).
        # Lower last_action = first 15 slots of that same Isaac-order action.
        self.action_isaac   = np.zeros(self.num_actions, dtype=np.float32)
        self.upper_obs_flat = np.zeros(self.num_obs_upper, dtype=np.float32)
        self.lower_obs_flat = np.zeros(self.num_obs_lower, dtype=np.float32)

        H = self.history_length_upper
        self.hist_ang_vel = np.zeros((H, 3),                dtype=np.float32)
        self.hist_gravity = np.zeros((H, 3),                dtype=np.float32)
        self.hist_cmd     = np.zeros((H, 3),                dtype=np.float32)
        self.hist_qpos    = np.zeros((H, self.num_actions), dtype=np.float32)
        self.hist_qvel    = np.zeros((H, self.num_actions), dtype=np.float32)
        self.hist_action  = np.zeros((H, self.num_actions), dtype=np.float32)
        self._history_primed = False

        # Motion reference buffer.
        self.motion = MotionBuffer(self.motion_path)

        # ONNX session.
        self.sess = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"])
        in_names = [i.name for i in self.sess.get_inputs()]
        out_names = [o.name for o in self.sess.get_outputs()]
        assert "upper_obs" in in_names and "lower_obs" in in_names, \
            f"ONNX inputs must include 'upper_obs' and 'lower_obs', got {in_names}"
        assert "actions" in out_names, \
            f"ONNX output must include 'actions', got {out_names}"
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

        # anchor_body_id filled lazily from state bus on enter() — see enter().
        self.anchor_body_id = None

        print(
            f"DualAgentTracking policy initializing (backend=onnx, dual-input) "
            f"| motion frames={self.motion.total_frames} @ {self.motion.fps}Hz "
            f"| duration={self.motion.total_frames/self.motion.fps:.2f}s"
        )

    def enter(self):
        # Reorder per-joint constants from Isaac order → MuJoCo order.
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

        self.hist_ang_vel.fill(0.0)
        self.hist_gravity.fill(0.0)
        self.hist_cmd.fill(0.0)
        self.hist_qpos.fill(0.0)
        self.hist_qvel.fill(0.0)
        self.hist_action.fill(0.0)
        self._history_primed = False
        self.action_isaac.fill(0.0)

        self.motion.reset()

        self.ramp_init_motor_pos = self.state_cmd.q.copy().astype(np.float32)
        self.ramp_cur_step = 0
        self.ramping = True
        print(f"DualAgentTracking: ramping to default pose over {self.ramp_time:.2f}s "
              f"({self.ramp_num_step} ticks) before policy inference starts.")

    @staticmethod
    def _push(buf, new_row):
        buf[:-1] = buf[1:]
        buf[-1] = new_row

    @staticmethod
    def _prime(buf, new_row):
        buf[:] = new_row

    def run(self):
        # Ramp-in: PD-hold to default_angles (MuJoCo order). Motion clock stays
        # at 0; starts advancing only once policy takes over.
        if self.ramping:
            self.ramp_cur_step += 1
            alpha = min(self.ramp_cur_step / self.ramp_num_step, 1.0)
            target = (self.ramp_init_motor_pos * (1.0 - alpha)
                      + self.default_angles_reorder * alpha).astype(np.float32)
            self.policy_output.actions = target
            self.policy_output.kps     = self.ramp_kps.copy()
            self.policy_output.kds     = self.ramp_kds.copy()
            if alpha >= 1.0:
                self.ramping = False
                print("DualAgentTracking: ramp complete, starting policy inference.")
            return

        # 1. robot state (MuJoCo order from bus).
        gravity    = self.state_cmd.gravity_ori.copy()
        ang_vel    = self.state_cmd.ang_vel.copy()
        lin_vel_b  = self.state_cmd.base_lin_vel.copy().astype(np.float32)
        qj_motor   = self.state_cmd.q.copy()
        dqj_motor  = self.state_cmd.dq.copy()
        joycmd     = self.state_cmd.vel_cmd.copy()
        cmd        = scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])

        # Anchor (torso_link) world pose — injected each tick via state_cmd
        # (set by deploy_mujoco before FSM.run()). wxyz quat.
        robot_anchor_pos  = self.state_cmd.anchor_pos_w.copy().astype(np.float32)
        robot_anchor_quat = self.state_cmd.anchor_quat_w.copy().astype(np.float32)

        # 2. motor-order → Isaac order for joint state.
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i]  = qj_motor[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj_motor[self.joint2motor_idx[i]]

        # 3. UPPER per-term scaling + clip (matches velocity variant).
        C = self.obs_clip_default
        ang_vel_s_u      = np.clip(ang_vel * self.ang_vel_scale, -C, C).astype(np.float32)
        gravity_s_u      = (gravity * self.gravity_scale).astype(np.float32)
        cmd_s            = (cmd * self.cmd_scale).astype(np.float32)
        joint_pos_rel_u  = np.clip((self.qj_obs - self.default_angles) * self.dof_pos_scale,
                                   -C, C).astype(np.float32)
        joint_vel_s_u    = np.clip(self.dqj_obs * self.dof_vel_scale, -C, C).astype(np.float32)
        last_action_s_u  = np.clip(self.action_isaac, -self.last_action_clip,
                                   self.last_action_clip).astype(np.float32)

        # 4. upper history ring update.
        if not self._history_primed:
            self._prime(self.hist_ang_vel, ang_vel_s_u)
            self._prime(self.hist_gravity, gravity_s_u)
            self._prime(self.hist_cmd,     cmd_s)
            self._prime(self.hist_qpos,    joint_pos_rel_u)
            self._prime(self.hist_qvel,    joint_vel_s_u)
            self._prime(self.hist_action,  last_action_s_u)
            self._history_primed = True
        else:
            self._push(self.hist_ang_vel, ang_vel_s_u)
            self._push(self.hist_gravity, gravity_s_u)
            self._push(self.hist_cmd,     cmd_s)
            self._push(self.hist_qpos,    joint_pos_rel_u)
            self._push(self.hist_qvel,    joint_vel_s_u)
            self._push(self.hist_action,  last_action_s_u)

        self.upper_obs_flat = np.concatenate([
            self.hist_ang_vel.reshape(-1),
            self.hist_gravity.reshape(-1),
            self.hist_cmd.reshape(-1),
            self.hist_qpos.reshape(-1),
            self.hist_qvel.reshape(-1),
            self.hist_action.reshape(-1),
        ]).astype(np.float32)

        # 5. LOWER obs (tracking): raw values, no scales/clips on dynamics.
        lower_cmd_pos, lower_cmd_vel, motion_pos, motion_quat = self.motion.read()

        pos_b, quat_b = subtract_frame_transforms_b(
            robot_anchor_pos, robot_anchor_quat, motion_pos, motion_quat)
        # ori_b = first 2 columns of rotation matrix, flattened row-major → 6-dim.
        mat_b = quat_to_mat(quat_b)
        ori_b = mat_b[:, :2].reshape(-1).astype(np.float32)   # (3,2) → (6,) row-major

        # Lower joint_pos / joint_vel observations are RAW in this cfg
        # (no dof_*_scale, no clip, no noise — training had noise but inference doesn't).
        joint_pos_rel_l = (self.qj_obs - self.default_angles).astype(np.float32)
        joint_vel_l     = self.dqj_obs.astype(np.float32)
        gravity_s_l     = gravity.astype(np.float32)
        ang_vel_s_l     = ang_vel.astype(np.float32)
        lin_vel_s_l     = lin_vel_b.astype(np.float32)
        last_action_lower = self.action_isaac[:15].astype(np.float32)

        o = self.lower_obs_flat
        o[0:15]    = lower_cmd_pos
        o[15:30]   = lower_cmd_vel
        o[30:33]   = pos_b
        o[33:39]   = ori_b
        o[39:42]   = gravity_s_l
        o[42:45]   = lin_vel_s_l
        o[45:48]   = ang_vel_s_l
        o[48:77]   = joint_pos_rel_l
        o[77:106]  = joint_vel_l
        o[106:121] = last_action_lower

        # 6. inference.
        u_in = np.clip(self.upper_obs_flat.reshape(1, -1), -100.0, 100.0)
        l_in = np.clip(self.lower_obs_flat.reshape(1, -1), -100.0, 100.0)
        raw_mujoco = self.sess.run(
            ["actions"],
            {"upper_obs": u_in, "lower_obs": l_in},
        )[0]
        action_mujoco = np.clip(raw_mujoco, -100.0, 100.0).squeeze().astype(np.float32)

        # 7. Isaac-order copy of the raw action (for next-tick last_action obs).
        for i in range(len(self.joint2motor_idx)):
            self.action_isaac[i] = action_mujoco[self.joint2motor_idx[i]]

        # 8. advance motion clock AFTER reading this frame, so next tick reads next.
        self.motion.advance()

        # 9. q_cmd in MuJoCo order.
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
        elif self.state_cmd.skill_cmd == FSMCommand.DUAL_AGENT_BOX_TRANS_VEL:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.DUAL_AGENT_BOX_TRANS_VEL
        else:
            return FSMStateName.DUAL_AGENT_TRACK
