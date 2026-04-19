from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import torch
import os


class LocoNew(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput,
                 config_dir: str = None,
                 config_name: str = "LocoNew.yaml",
                 state_name: FSMStateName = FSMStateName.LOCO_NEW,
                 name_str: str = "loco_new_mode"):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = state_name
        self.name_str = name_str
        self._own_state = state_name

        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, "config", config_name)
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.backend = str(config.get("backend", "pt")).lower()
        self.policy_path = os.path.join(config_dir, "model", config["policy_path"])
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
        self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.base_lin_vel_scale = config["base_lin_vel_scale"]
        self.ang_vel_scale = config["ang_vel_scale"]
        self.gravity_scale = config["gravity_scale"]
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]
        self.action_scale = config["action_scale"]
        self.obs_clip = float(config["obs_clip_default"])
        self.last_action_clip = float(config["last_action_clip"])
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        self.cmd_range = config["cmd_range"]
        self.range_velx = np.array([self.cmd_range["lin_vel_x"][0], self.cmd_range["lin_vel_x"][1]], dtype=np.float32)
        self.range_vely = np.array([self.cmd_range["lin_vel_y"][0], self.cmd_range["lin_vel_y"][1]], dtype=np.float32)
        self.range_velz = np.array([self.cmd_range["ang_vel_z"][0], self.cmd_range["ang_vel_z"][1]], dtype=np.float32)

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)
        self.obs = np.zeros(self.num_obs)
        self.action = np.zeros(self.num_actions)

        if self.backend == "pt":
            self.policy = torch.jit.load(self.policy_path)
            with torch.inference_mode():
                probe = self.policy(torch.zeros(1, self.num_obs, dtype=torch.float32))
            out_dim = int(probe.shape[-1])
        elif self.backend == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
            self.onnx_in = self.sess.get_inputs()[0].name
            self.onnx_out = self.sess.get_outputs()[0].name
            probe = self.sess.run([self.onnx_out],
                                  {self.onnx_in: np.zeros((1, self.num_obs), dtype=np.float32)})[0]
            out_dim = int(probe.shape[-1])
        else:
            raise ValueError(f"unknown backend '{self.backend}' (expected 'pt' or 'onnx')")

        assert out_dim == self.num_actions, (
            f"{self.name_str} model output dim {out_dim} != num_actions={self.num_actions}"
        )

        spot = (int(self.joint2motor_idx[0]), int(self.joint2motor_idx[11]), int(self.joint2motor_idx[2]))
        assert spot == (0, 15, 12), f"joint2motor_idx spot-check failed: got {spot}, expected (0, 15, 12)"

        warmup = self.obs.reshape(1, -1).astype(np.float32)
        for _ in range(50):
            self._infer(warmup)

        print(f"{self.name_str} policy initializing (backend={self.backend}) ...")

    def _infer(self, obs_np: np.ndarray) -> np.ndarray:
        obs_clipped = np.clip(obs_np, -100.0, 100.0).astype(np.float32)
        if self.backend == "pt":
            with torch.inference_mode():
                raw = self.policy(torch.from_numpy(obs_clipped)).detach().numpy()
        else:
            raw = self.sess.run([self.onnx_out], {self.onnx_in: obs_clipped})[0]
        return np.clip(raw, -100.0, 100.0).squeeze()

    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.kps_reorder[motor_idx] = self.kps[i]
            self.kds_reorder[motor_idx] = self.kds[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]

    def run(self):
        base_lin_vel = self.state_cmd.base_lin_vel.copy()
        gravity = self.state_cmd.gravity_ori.copy()
        self.qj = self.state_cmd.q.copy()
        self.dqj = self.state_cmd.dq.copy()
        ang_vel = self.state_cmd.ang_vel.copy()
        joycmd = self.state_cmd.vel_cmd.copy()
        cmd = scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])

        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = self.qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = self.dqj[self.joint2motor_idx[i]]

        C = self.obs_clip
        base_lin_vel_s = np.clip(base_lin_vel * self.base_lin_vel_scale, -C, C)
        ang_vel_s = np.clip(ang_vel * self.ang_vel_scale, -C, C)
        gravity_s = gravity * self.gravity_scale
        cmd_s = cmd * self.cmd_scale
        joint_pos_rel = np.clip((self.qj_obs - self.default_angles) * self.dof_pos_scale, -C, C)
        joint_vel_s = np.clip(self.dqj_obs * self.dof_vel_scale, -C, C)
        last_action_s = np.clip(self.action, -self.last_action_clip, self.last_action_clip)

        n = self.num_actions
        self.obs[0:3] = base_lin_vel_s
        self.obs[3:6] = ang_vel_s
        self.obs[6:9] = gravity_s
        self.obs[9:12] = cmd_s
        self.obs[12:12 + n] = joint_pos_rel
        self.obs[12 + n:12 + 2 * n] = joint_vel_s
        self.obs[12 + 2 * n:12 + 3 * n] = last_action_s

        self.action = self._infer(self.obs.reshape(1, -1))

        loco_action = self.action * self.action_scale + self.default_angles
        action_reorder = loco_action.copy()
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            action_reorder[motor_idx] = loco_action[i]

        self.policy_output.actions = action_reorder.copy()
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
            return self._own_state
