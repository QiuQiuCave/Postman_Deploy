"""Binary commands -> continuous six-channel references, outside the RL bus.

This is an EMPTY-HAND motion controller, not a grasp-success detector. CLOSED
means the configured pose is reached, never that an object has been grasped.
"""

from common.path_config import PROJECT_ROOT

from dataclasses import dataclass

import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from r2v2_description.model import JointMap, SIDES, hand_names, urdf_hand_joints


@dataclass(frozen=True)
class HandReference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class BinaryHandController:
    def __init__(self, cfg, side, initial_position):
        self.side = side
        self.dt = float(cfg["control_dt"])
        self.poses = {int(key == "closed"): np.array(cfg["hands"][side][key], dtype=float)
                      for key in ("open", "closed")}
        self.generator = Ruckig(6, self.dt)
        self.input = InputParameter(6)
        self.output = OutputParameter(6)
        initial = np.asarray(initial_position, dtype=float)
        if initial.shape != (6,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_position must be six finite angles")
        self.input.current_position = initial.tolist()
        self.input.current_velocity = [0.0] * 6
        self.input.current_acceleration = [0.0] * 6
        self.input.target_position = self.poses[0].tolist()
        self.input.target_velocity = [0.0] * 6
        self.input.target_acceleration = [0.0] * 6
        for key in ("max_velocity", "max_acceleration", "max_jerk"):
            setattr(self.input, key, cfg["trajectory"][key])
        self.command = 0
        self.command_changes = 0
        self.finished = bool(np.allclose(initial, self.poses[0], atol=1e-8))
        self.reference = HandReference(initial.copy(), np.zeros(6), np.zeros(6))
        self.settle_tolerance = cfg["validation"]["settle_tolerance"]
        joints = urdf_hand_joints()
        self.lower = np.array([float(joints[n].find("limit").get("lower"))
                               for n in hand_names(side)])
        self.upper = np.array([float(joints[n].find("limit").get("upper"))
                               for n in hand_names(side)])
        if np.any(initial < self.lower) or np.any(initial > self.upper):
            raise ValueError("Initial hand position outside source joint limits")

    def set_command(self, value):
        if not isinstance(value, (bool, int, np.integer)) or value not in (0, 1):
            raise ValueError("Hand command must be integer 0 (open) or 1 (close)")
        if int(value) == self.command:
            return False  # Repeated binary commands must not restart motion.
        self.command = int(value)
        self.command_changes += 1
        self.input.target_position = self.poses[self.command].tolist()
        # Retain the current reference position/velocity/acceleration. Ruckig
        # computes a feasible reversal; do not zero the velocity at a toggle.
        self.finished = False
        return True

    def step(self):
        result = self.generator.update(self.input, self.output)
        if result < 0:
            raise RuntimeError(f"{self.side} hand trajectory generation failed: {result}")
        q = np.array(self.output.new_position)
        if np.any(q < self.lower - 1e-8) or np.any(q > self.upper + 1e-8):
            raise RuntimeError(f"{self.side} hand reference crossed joint limits")
        self.reference = HandReference(q, np.array(self.output.new_velocity),
                                       np.array(self.output.new_acceleration))
        self.output.pass_to_input(self.input)
        self.finished = result == Result.Finished
        return self.reference

    def status(self, measured_q):
        if self.finished and np.max(np.abs(measured_q - self.poses[self.command])) < self.settle_tolerance:
            return "CLOSED" if self.command else "OPEN"
        return "CLOSING" if self.command else "OPENING"


class DualHandControl:
    """Independent hand state/control. Does not resize StateAndCmd/PolicyOutput."""

    def __init__(self, model, data, cfg):
        self.maps = {side: JointMap.create(model, hand_names(side)) for side in SIDES}
        self.controllers = {
            side: BinaryHandController(cfg, side, data.qpos[self.maps[side].qpos])
            for side in SIDES
        }
        self.kp = np.array(cfg["servo"]["kp"])
        self.kd = np.array(cfg["servo"]["kd"])
        self.torque_limit = np.array(cfg["servo"]["torque_limit"])

    def command(self, side, value):
        if side not in SIDES:
            raise ValueError(f"Unknown hand: {side}")
        return self.controllers[side].set_command(value)

    def update(self):
        for controller in self.controllers.values():
            controller.step()

    def apply(self, data):
        for side, mapping in self.maps.items():
            ref = self.controllers[side].reference
            torque = self.kp * (ref.position - data.qpos[mapping.qpos])
            torque += self.kd * (ref.velocity - data.qvel[mapping.dofs])
            data.ctrl[mapping.actuators] = np.clip(torque, -self.torque_limit, self.torque_limit)
