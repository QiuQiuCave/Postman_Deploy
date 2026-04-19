"""
BoxTransport FSM state for sim2sim deployment of BoxTransport-G1-Velocity policy.

The ONNX model outputs raw actions in Isaac joint order (no baked scale/reorder).
This script handles:
  - Joint order conversion (MuJoCo ↔ Isaac)
  - Observation construction with correct scaling/clipping
  - 5-frame observation history
  - Action scale + default offset application
"""

from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, get_gravity_orientation, scale_values
import numpy as np
import yaml
import onnx
import onnxruntime
import os


class BoxTransport(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_BOX_TRANSPORT
        self.name_str = "box_transport"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "BoxTransport.yaml")

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)

        cmd_range = config.get("cmd_range", {})
        self.range_velx = np.array(cmd_range.get("lin_vel_x", [-0.5, 1.0]), dtype=np.float32)
        self.range_vely = np.array(cmd_range.get("lin_vel_y", [-0.3, 0.3]), dtype=np.float32)
        self.range_velz = np.array(cmd_range.get("ang_vel_z", [-0.2, 0.2]), dtype=np.float32)

        # Load ONNX model
        self.onnx_model = onnx.load(onnx_path)
        self.ort_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        # Parse metadata from ONNX
        meta = {p.key: p.value for p in self.onnx_model.metadata_props}

        # Joint order mappings
        self.isaac_to_mujoco_idx = self._parse_int_csv(
            meta.get("isaac_to_mujoco_idx", ""),
            fallback=[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                      4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
        )
        self.mujoco_to_isaac_idx = self._parse_int_csv(
            meta.get("mujoco_to_isaac_idx", ""),
            fallback=[0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
                      2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
        )

        # Action scale (Isaac order)
        self.action_scale_isaac = self._parse_float_csv(
            meta.get("action_scale_isaac", ""),
            fallback=np.full(29, 0.25),
        )

        # Default joint positions (Isaac order)
        self.default_pos_isaac = self._parse_float_csv(
            meta.get("default_pos_isaac", ""),
            fallback=np.zeros(29),
        )

        # History config
        self.history_length = int(meta.get("obs_history_length", "5"))
        self.obs_per_step = int(meta.get("obs_per_step", "96"))

        # Runtime state
        self.last_action_isaac = np.zeros(29, dtype=np.float32)
        self.obs_history = []

        print(f"BoxTransport policy initialized (action_order={meta.get('action_order', '?')}, "
              f"history={self.history_length}, obs_dim={self.history_length * self.obs_per_step})")

    @staticmethod
    def _parse_float_csv(csv_str, fallback):
        if not csv_str:
            return np.array(fallback, dtype=np.float32)
        return np.array([float(x.strip()) for x in csv_str.split(",")], dtype=np.float32)

    @staticmethod
    def _parse_int_csv(csv_str, fallback):
        if not csv_str:
            return np.array(fallback, dtype=np.int32)
        return np.array([int(float(x.strip())) for x in csv_str.split(",")], dtype=np.int32)

    def _init_history(self):
        self.obs_history = [np.zeros(self.obs_per_step, dtype=np.float32)
                           for _ in range(self.history_length)]

    def enter(self):
        self.last_action_isaac = np.zeros(29, dtype=np.float32)
        self._init_history()
        print("Entered BoxTransport state")
        print(f"  isaac_to_mujoco_idx = {self.isaac_to_mujoco_idx.tolist()}")
        print(f"  mujoco_to_isaac_idx = {self.mujoco_to_isaac_idx.tolist()}")
        print(f"  action_scale_isaac  = {np.round(self.action_scale_isaac, 4).tolist()}")
        print(f"  default_pos_isaac   = {np.round(self.default_pos_isaac, 4).tolist()}")
        print(f"  cmd_range: vx={self.range_velx}, vy={self.range_vely}, wz={self.range_velz}")

    def run(self):
        # --- 1. Read MuJoCo state ---
        qj_mj = self.state_cmd.q.copy()         # [29] MuJoCo order
        dqj_mj = self.state_cmd.dq.copy()       # [29] MuJoCo order
        ang_vel = self.state_cmd.ang_vel.copy()  # [3] body frame
        gravity = self.state_cmd.gravity_ori.copy()  # [3]

        # Velocity command: map joystick [-1, 1] → training command range
        vel_cmd = scale_values(
            self.state_cmd.vel_cmd.copy(),
            [self.range_velx, self.range_vely, self.range_velz],
        )

        # --- 2. MuJoCo → Isaac joint order ---
        qj_isaac = qj_mj[self.isaac_to_mujoco_idx]
        dqj_isaac = dqj_mj[self.isaac_to_mujoco_idx]

        # --- 3. Build single-step obs (96D, matching training ObsTerm order & scaling) ---
        joint_pos_rel = qj_isaac - self.default_pos_isaac

        obs_step = np.concatenate([
            np.clip(ang_vel * 0.2, -100, 100),                      # base_ang_vel [3]
            gravity,                                                  # projected_gravity [3]
            np.array(vel_cmd, dtype=np.float32),                     # velocity_commands [3]
            np.clip(joint_pos_rel, -100, 100),                       # joint_pos_rel [29]
            np.clip(dqj_isaac * 0.05, -100, 100),                   # joint_vel_rel [29]
            np.clip(self.last_action_isaac, -12, 12),                # last_action [29]
        ], dtype=np.float32)

        # --- 4. Maintain 5-frame history ---
        self.obs_history.append(obs_step)
        if len(self.obs_history) > self.history_length:
            self.obs_history.pop(0)
        obs_full = np.concatenate(self.obs_history, axis=0).astype(np.float32)

        # --- 5. ONNX inference → raw action (Isaac order) ---
        raw_action_isaac = self.ort_session.run(
            None, {self.input_name: obs_full.reshape(1, -1)}
        )[0].squeeze()

        # --- 6. Store raw action for next frame's last_action obs ---
        self.last_action_isaac = raw_action_isaac.copy()

        # --- 7. Apply action scale + default offset (Isaac order) ---
        target_isaac = raw_action_isaac * self.action_scale_isaac + self.default_pos_isaac

        # --- 8. Isaac → MuJoCo joint order ---
        target_mj = target_isaac[self.mujoco_to_isaac_idx]

        # --- 9. Output ---
        self.policy_output.actions = target_mj
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def exit(self):
        self.last_action_isaac = np.zeros(29, dtype=np.float32)
        self._init_history()
        print("Exited BoxTransport state")

    def checkChange(self):
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_BOX_TRANSPORT
