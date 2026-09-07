"""50 Hz adapter for the trained 28-DoF dual-arm Reach ONNX policy.

The ONNX already contains the empirical observation normalizer. This module
supplies unnormalized training observations, with ten *per-term* history frames
in oldest-to-newest order. Fingers are never observed or actuated here.

Positions and reference twists use world coordinates internally; the policy's
position errors/twists use base-yaw coordinates, whereas IMU observations use
the full base orientation. Orientation errors are inv(TCP) * target, wxyz.
The reference stepping equations mirror AMO_R2/tasks/reach/trajectory.py,
including its final-step snap; reference bounds are not hard physical bounds.
"""

from common.path_config import PROJECT_ROOT  # Establish repository imports.

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from r2v2_description.model import BODY_JOINTS, JointMap, SIDES


HISTORY_LENGTH = 10
POLICY_DT = 0.02
OBSERVATION_DIM = 1460
ACTION_DIM = 28
LINEAR_SPEED_LIMIT = 0.25
ANGULAR_SPEED_LIMIT = 0.80
LINEAR_ACCELERATION_LIMIT = 0.80
ANGULAR_ACCELERATION_LIMIT = 2.00

# A compatibility *virtual* TCP, not the new hand's palm or cylinder centre.
# These transforms were obtained by FK of the existing training model. Its
# rounded fixed quaternions account for the sub-micrometre lateral offsets.
TCP_WRIST_POSITIONS = {
    "left": np.array([0.1735, -0.0000004, -0.0317504]),
    "right": np.array([0.1735, 0.0000004, -0.0317504]),
}
TCP_WRIST_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])

TERM_DIMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": 28,
    "joint_vel": 28,
    "actions": 28,
    "command": 3,
    "base_pose_command": 2,
    "reach_position_error": 3,
    "reach_orientation_error": 4,
    "active_arm_mask": 2,
    "final_reach_position_error": 3,
    "final_reach_orientation_error": 4,
    "reach_reference_linear_velocity": 3,
    "reach_reference_angular_velocity": 3,
    "reach_speed_limits": 2,
    "right_reach_position_error": 3,
    "right_reach_orientation_error": 4,
    "right_active_arm_mask": 2,
    "right_final_reach_position_error": 3,
    "right_final_reach_orientation_error": 4,
    "right_reach_reference_linear_velocity": 3,
    "right_reach_reference_angular_velocity": 3,
    "right_reach_speed_limits": 2,
}

# Effort limits are not present in the original ONNX metadata. Preserve its
# training motor contract, intersected with (never replacing) model limits.
TRAINING_EFFORT_LIMITS = np.array(
    [150, 130, 130, 150, 75, 75] * 2 + [130, 130]
    + [75, 75, 75, 36, 36, 36, 36] * 2,
    dtype=float,
)


def _finite_vector(value, length, label):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be {length} finite numbers")
    return result.copy()


def _quaternion(value):
    q = _finite_vector(value, 4, "wxyz quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Quaternion must have nonzero norm")
    q /= norm
    return -q if q[0] < 0 else q


def _quat_mul(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotation_vector(q, eps=1e-6):
    q = _quaternion(q)
    half_angle = np.arctan2(np.linalg.norm(q[1:]), q[0])
    angle = 2 * half_angle
    factor = np.sin(half_angle) / angle if abs(angle) > eps else 0.5 - angle**2 / 48
    return q[1:] / factor


def _box_plus(q, delta, eps=1e-6):
    # Match training's quat_box_plus/quat_from_angle_axis Taylor convention,
    # including the minimum angle used for extremely small nonzero steps.
    angle = max(float(np.linalg.norm(delta)), eps)
    axis = delta / angle
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    increment = _quaternion(np.r_[np.cos(angle/2), np.sin(angle/2) * axis])
    return _quaternion(_quat_mul(increment, q))


def _next_velocity(velocity, delta, speed_limit, acceleration, dt, eps=1e-6):
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, eps)
    speed = min(speed_limit, np.sqrt(2.0 * acceleration * distance))
    desired = direction * speed if distance > eps else np.zeros(3)
    change = desired - velocity
    result = velocity + change * min(1.0, acceleration*dt / max(np.linalg.norm(change), eps))
    result *= min(1.0, speed_limit / max(np.linalg.norm(result), eps))
    reached = distance <= eps or np.dot(result*dt, direction) >= distance
    return result, reached


@dataclass
class ReachReference:
    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    goal_position: np.ndarray
    goal_quaternion: np.ndarray

    def step(self, dt=POLICY_DT):
        velocity, reached = _next_velocity(
            self.linear_velocity, self.goal_position-self.position,
            LINEAR_SPEED_LIMIT, LINEAR_ACCELERATION_LIMIT, dt,
        )
        self.position = self.goal_position.copy() if reached else self.position + dt*velocity
        self.linear_velocity = np.zeros(3) if reached else velocity
        rotation_delta = _rotation_vector(_quat_mul(self.goal_quaternion, _quat_inv(self.quaternion)))
        omega, reached = _next_velocity(
            self.angular_velocity, rotation_delta,
            ANGULAR_SPEED_LIMIT, ANGULAR_ACCELERATION_LIMIT, dt,
        )
        self.quaternion = (self.goal_quaternion.copy() if reached
                           else _box_plus(self.quaternion, dt*omega))
        self.quaternion = _quaternion(self.quaternion)
        self.angular_velocity = np.zeros(3) if reached else omega


class ReachPolicy:
    """Independent whole-body inference and torque bus; hand bus stays intact.

    Call ``act`` exactly once per 20 ms; call ``apply`` every physics step.
    ``reset`` and ``follow_current`` never write physical state. The caller
    must have run FK for the supplied data before observing or resetting it.
    ``observe(..., False)`` is a diagnostic snapshot: it does not advance the
    reference or push history, but substitutes current features in the newest
    slot of the returned copy. ``observe(..., True)`` pushes one policy frame.
    Explicit ``update_history=True`` permits a numerical audit to push copied
    training observations without independently advancing their references.
    """

    dt = POLICY_DT

    def __init__(self, model, data, onnx_path):
        import onnxruntime as ort

        self.model = model
        self.onnx_path = Path(onnx_path).resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(self.onnx_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs"
                or list(inputs[0].shape) != [1, OBSERVATION_DIM]
                or inputs[0].type != "tensor(float)"):
            raise ValueError("Expected dual-arm ONNX input obs float32[1,1460]")
        if (len(outputs) != 1 or outputs[0].name != "actions"
                or list(outputs[0].shape) != [1, ACTION_DIM]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("Expected dual-arm ONNX output actions float32[1,28]")
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        for key, expected in (
            ("joint_names", BODY_JOINTS),
            ("observation_names", tuple(TERM_DIMS)),
            ("command_names", ("twist", "base_pose", "reach", "reach_right")),
        ):
            actual = tuple(self.metadata.get(key, "").split(","))
            if actual != tuple(expected):
                raise ValueError(f"ONNX {key} differs from the trained dual-arm contract")
        self.action_scale = float(self.metadata.get("action_scale", "nan"))
        if not np.isfinite(self.action_scale) or self.action_scale != 0.25:
            raise ValueError("Expected ONNX action_scale=0.25")
        self.kp = self._metadata_vector("joint_stiffness")
        self.kd = self._metadata_vector("joint_damping")
        self.default_joint_pos = self._metadata_vector("default_joint_pos")
        if np.any(self.kp <= 0) or np.any(self.kd < 0):
            raise ValueError("ONNX stiffness must be positive and damping non-negative")

        # The optional training prefix permits same-state numerical audits on
        # mjlab's original model; deployment continues to use plain names.
        self.prefix = "" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link") >= 0 else "robot/"
        self.base_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.site_ids = {side: self._named_id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp") for side in SIDES}
        self.body_map = JointMap.create(model, tuple(self.prefix+n for n in BODY_JOINTS))
        acts = self.body_map.actuators
        gear = model.actuator_gear[acts]
        self._direct_torque = bool(
            np.all(model.actuator_dyntype[acts] == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(model.actuator_gaintype[acts] == mujoco.mjtGain.mjGAIN_FIXED)
            and np.all(model.actuator_biastype[acts] == mujoco.mjtBias.mjBIAS_NONE)
            and np.allclose(model.actuator_gainprm[acts, 0], 1.0)
            and np.allclose(gear, np.tile([1, 0, 0, 0, 0, 0], (ACTION_DIM, 1)))
        )
        self.torque_lower = -TRAINING_EFFORT_LIMITS.copy()
        self.torque_upper = TRAINING_EFFORT_LIMITS.copy()
        for enabled, bounds in (
            (model.actuator_ctrllimited[acts], model.actuator_ctrlrange[acts]),
            (model.actuator_forcelimited[acts], model.actuator_forcerange[acts]),
            (model.jnt_actfrclimited[self.body_map.joints],
             model.jnt_actfrcrange[self.body_map.joints]),
        ):
            limited = np.asarray(enabled, dtype=bool)
            self.torque_lower[limited] = np.maximum(self.torque_lower[limited], bounds[limited, 0])
            self.torque_upper[limited] = np.minimum(self.torque_upper[limited], bounds[limited, 1])
        if (np.any(self.torque_lower >= 0) or np.any(self.torque_upper <= 0)
                or not np.all(np.isfinite(np.r_[self.torque_lower, self.torque_upper]))):
            raise ValueError("Body torque limits must be finite and contain zero")
        self._jacp = np.empty((3, model.nv))
        self._jacr = np.empty((3, model.nv))
        self.reset(data)

    def _named_id(self, object_type, name):
        value = mujoco.mj_name2id(self.model, object_type, self.prefix+name)
        if value < 0:
            raise ValueError(f"Model missing {self.prefix+name}")
        return value

    def _metadata_vector(self, key):
        try:
            values = [float(value) for value in self.metadata[key].split(",")]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Missing/invalid ONNX metadata {key}") from exc
        return _finite_vector(values, ACTION_DIM, f"ONNX {key}")

    def tcp_pose(self, data, side):
        if side not in SIDES:
            raise ValueError(f"Unknown side: {side}")
        site = self.site_ids[side]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        return data.site_xpos[site].copy(), _quaternion(quat)

    def reset(self, data):
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.q_des = self.default_joint_pos.copy()
        self.last_torque = np.zeros(ACTION_DIM)
        self.references = {}
        self.follow_current(data)
        current = self.observation_terms(data)
        self.histories = {name: np.repeat(value[None, :], HISTORY_LENGTH, axis=0)
                          for name, value in current.items()}
        self.last_observation = self._flatten(self.histories)

    def follow_current(self, data):
        """Training's reset-settle phase, without clearing actions or history."""
        for side in SIDES:
            position, quaternion = self.tcp_pose(data, side)
            if side not in self.references:
                self.references[side] = ReachReference(
                    position, quaternion, np.zeros(3), np.zeros(3),
                    position.copy(), quaternion.copy(),
                )
            else:
                ref = self.references[side]
                ref.position = position
                ref.quaternion = quaternion
                ref.goal_position = position.copy()
                ref.goal_quaternion = quaternion.copy()
                ref.linear_velocity = np.zeros(3)
                ref.angular_velocity = np.zeros(3)

    def set_target_world(self, side, position, quaternion):
        if side not in SIDES:
            raise ValueError(f"Unknown side: {side}")
        position = _finite_vector(position, 3, "World goal position")
        quaternion = _quaternion(quaternion)
        ref = self.references[side]
        # No reference reset, including when goals reverse or are repeated.
        ref.goal_position = position
        ref.goal_quaternion = quaternion

    def observation_terms(self, data):
        """Unnormalized current features; useful for deployment parity tests."""
        rotation = data.xmat[self.base_id].reshape(3, 3)
        w, x, y, z = data.xquat[self.base_id]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = np.cos(yaw), np.sin(yaw)
        yaw_inverse = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        # mj_jacBody uses the body-frame origin (not its centre of mass).
        mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, self.base_id)
        terms = {
            "base_lin_vel": rotation.T @ (self._jacp @ data.qvel),
            "base_ang_vel": rotation.T @ (self._jacr @ data.qvel),
            "projected_gravity": rotation.T @ np.array([0.0, 0.0, -1.0]),
            "joint_pos": data.qpos[self.body_map.qpos] - self.default_joint_pos,
            "joint_vel": data.qvel[self.body_map.dofs].copy(),
            "actions": self.last_action.copy(),
            "command": np.zeros(3),
            "base_pose_command": np.array([0.82, 0.0]),
        }
        for side in SIDES:
            ref = self.references[side]
            tcp_position, tcp_quaternion = self.tcp_pose(data, side)
            pre = "" if side == "left" else "right_"
            terms.update({
                pre+"reach_position_error": yaw_inverse @ (ref.position-tcp_position),
                pre+"reach_orientation_error": _quaternion(_quat_mul(_quat_inv(tcp_quaternion), ref.quaternion)),
                pre+"active_arm_mask": np.array([1.0, 0.0] if side == "left" else [0.0, 1.0]),
                pre+"final_reach_position_error": yaw_inverse @ (ref.goal_position-tcp_position),
                pre+"final_reach_orientation_error": _quaternion(_quat_mul(_quat_inv(tcp_quaternion), ref.goal_quaternion)),
                pre+"reach_reference_linear_velocity": yaw_inverse @ ref.linear_velocity,
                pre+"reach_reference_angular_velocity": yaw_inverse @ ref.angular_velocity,
                pre+"reach_speed_limits": np.array([LINEAR_SPEED_LIMIT, ANGULAR_SPEED_LIMIT]),
            })
        for name, length in TERM_DIMS.items():
            value = np.asarray(terms[name], dtype=np.float32)
            if value.shape != (length,) or not np.all(np.isfinite(value)):
                raise RuntimeError(f"Non-finite or incorrect-size policy observation: {name}")
            terms[name] = value
        return terms

    @staticmethod
    def _flatten(histories):
        return np.concatenate([histories[name].reshape(-1) for name in TERM_DIMS]).astype(np.float32)

    def observe(self, data, advance_reference=True, *, update_history=None):
        if update_history is None:
            update_history = advance_reference
        if advance_reference:
            for ref in self.references.values():
                ref.step(self.dt)
        current = self.observation_terms(data)
        if update_history:
            for name, value in current.items():
                self.histories[name][:-1] = self.histories[name][1:]
                self.histories[name][-1] = value
            self.last_observation = self._flatten(self.histories)
            return self.last_observation.copy()
        snapshot = {name: history.copy() for name, history in self.histories.items()}
        for name, value in current.items():
            snapshot[name][-1] = value
        return self._flatten(snapshot)

    def act(self, data):
        observation = self.observe(data)
        output = np.asarray(self.session.run(["actions"], {"obs": observation[None, :]})[0])
        if output.shape != (1, ACTION_DIM) or not np.all(np.isfinite(output)):
            raise RuntimeError("ONNX returned non-finite/incorrect-size actions")
        self.last_action = output[0].astype(np.float32).copy()
        self.q_des = self.default_joint_pos + self.action_scale*self.last_action
        return self.q_des.copy()

    def apply(self, data):
        if not self._direct_torque:
            raise RuntimeError("ReachPolicy.apply requires unit-gear direct torque motors")
        torque = self.kp*(self.q_des-data.qpos[self.body_map.qpos]) - self.kd*data.qvel[self.body_map.dofs]
        if not np.all(np.isfinite(torque)):
            raise RuntimeError("Non-finite whole-body torque request")
        self.last_torque = np.clip(torque, self.torque_lower, self.torque_upper)
        data.ctrl[self.body_map.actuators] = self.last_torque
        return self.last_torque.copy()
