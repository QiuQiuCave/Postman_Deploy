"""Deterministic fixed-wrist experiment and physics/reference diagnostics."""

from common.path_config import PROJECT_ROOT

import mujoco
import numpy as np

from common.r2v2_hand_control import DualHandControl
from r2v2_description.model import SIDES, build_model, initialize_hands, load_config, urdf_hand_joints

# (simulation seconds, left command, right command). Includes independent,
# symmetric, and intentionally interrupted movements. No object is present.
DEMO_EVENTS = (
    (0.0, 0, 0), (1.0, 1, 0), (5.0, 0, 0), (6.0, 0, 1),
    (10.0, 0, 0), (12.0, 1, 1), (16.0, 0, 0),
    (19.0, 1, 0), (19.55, 0, 1), (20.10, 1, 0), (20.65, 0, 1),
    (22.0, 1, 1), (27.0, 0, 0),
)
DEMO_DURATION = 32.0
CHECKPOINTS = (
    (4.5, "CLOSED", "OPEN"), (9.5, "OPEN", "CLOSED"),
    (15.5, "CLOSED", "CLOSED"), (18.5, "OPEN", "OPEN"),
    (26.5, "CLOSED", "CLOSED"), (31.5, "OPEN", "OPEN"),
)


class HandExperiment:
    def __init__(self, cfg=None, automatic=True):
        self.cfg = cfg or load_config()
        self.model = build_model(self.cfg, fixture=True)
        self.data = mujoco.MjData(self.model)
        initialize_hands(self.model, self.data, self.cfg)
        self.control = DualHandControl(self.model, self.data, self.cfg)
        self.decimation = round(self.cfg["control_dt"] / self.cfg["simulation_dt"])
        self.steps = 0
        self.automatic = automatic
        self.event_index = 0
        self.check_index = 0
        self.events = []
        self.checkpoints = []
        self.samples = []
        self.peaks = {
            side: {key: np.zeros(6) for key in (
                "tracking_error_rad", "measured_velocity_rad_s", "reference_velocity_rad_s",
                "reference_acceleration_rad_s2", "reference_jerk_rad_s3",
            )} for side in SIDES
        }
        self.previous_acceleration = {side: np.zeros(6) for side in SIDES}
        self.command_jump = 0.0
        self.limit_violation = 0.0
        self.mimic_error = 0.0
        self.penetration = 0.0
        self.contacts = {}
        self.finite = True
        self.mimics = []
        self.all_hand_qpos = []
        self.all_hand_dofs = []
        self.all_hand_joint_ids = []
        self.source_velocity_limits = []
        for name, joint in urdf_hand_joints().items():
            j = self.model.joint(name)
            self.all_hand_qpos.append(j.qposadr[0])
            self.all_hand_dofs.append(j.dofadr[0])
            self.all_hand_joint_ids.append(j.id)
            self.source_velocity_limits.append(float(joint.find("limit").get("velocity")))
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append((j.qposadr[0], self.model.joint(mimic.get("joint")).qposadr[0],
                                    float(mimic.get("multiplier", "1")),
                                    float(mimic.get("offset", "0"))))
        self.source_velocity_excess = 0.0

    def command(self, left, right):
        # Check continuity at the exact command boundary, before advancing time.
        for side, value in zip(SIDES, (left, right)):
            c = self.control.controllers[side]
            before = np.r_[c.input.current_position, c.input.current_velocity, c.input.current_acceleration]
            self.control.command(side, value)
            after = np.r_[c.input.current_position, c.input.current_velocity, c.input.current_acceleration]
            self.command_jump = max(self.command_jump, float(np.max(np.abs(after - before))))

    def statuses(self):
        return {side: self.control.controllers[side].status(self.data.qpos[self.control.maps[side].qpos])
                for side in SIDES}

    def step(self):
        t = self.data.time
        if self.steps % self.decimation == 0:
            if self.automatic:
                while self.event_index < len(DEMO_EVENTS) and t + 1e-8 >= DEMO_EVENTS[self.event_index][0]:
                    when, left, right = DEMO_EVENTS[self.event_index]
                    self.command(left, right)
                    self.events.append({"time": when, "left": left, "right": right})
                    self.event_index += 1
            # Exercise the idempotent command path every control tick.
            for side, c in self.control.controllers.items():
                self.control.command(side, c.command)
            self.control.update()
            for side, c in self.control.controllers.items():
                ref = c.reference
                jerk = (ref.acceleration - self.previous_acceleration[side]) / self.cfg["control_dt"]
                self.previous_acceleration[side] = ref.acceleration.copy()
                for key, values in (("reference_velocity_rad_s", ref.velocity),
                                    ("reference_acceleration_rad_s2", ref.acceleration),
                                    ("reference_jerk_rad_s3", jerk)):
                    self.peaks[side][key] = np.maximum(self.peaks[side][key], np.abs(values))
            self.samples.append(np.r_[t, *[
                np.r_[self.data.qpos[self.control.maps[s].qpos],
                      self.control.controllers[s].reference.position,
                      self.data.qvel[self.control.maps[s].dofs],
                      self.control.controllers[s].reference.velocity]
                for s in SIDES
            ]])
        self.control.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.finite &= bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        for side, mapping in self.control.maps.items():
            c = self.control.controllers[side]
            for key, values in (("tracking_error_rad", self.data.qpos[mapping.qpos] - c.reference.position),
                                ("measured_velocity_rad_s", self.data.qvel[mapping.dofs])):
                self.peaks[side][key] = np.maximum(self.peaks[side][key], np.abs(values))
        for dst, src, ratio, offset in self.mimics:
            self.mimic_error = max(self.mimic_error, abs(self.data.qpos[dst] - ratio * self.data.qpos[src] - offset))
        q = self.data.qpos[self.all_hand_qpos]
        limits = self.model.jnt_range[self.all_hand_joint_ids]
        self.limit_violation = max(self.limit_violation, float(np.max(limits[:, 0] - q)),
                                   float(np.max(q - limits[:, 1])))
        self.source_velocity_excess = max(self.source_velocity_excess, float(np.max(
            np.abs(self.data.qvel[self.all_hand_dofs]) - self.source_velocity_limits)))
        for contact in self.data.contact:
            depth = max(0.0, -float(contact.dist))
            self.penetration = max(self.penetration, depth)
            pair = f"{self.model.geom(contact.geom1).name} | {self.model.geom(contact.geom2).name}"
            self.contacts[pair] = max(self.contacts.get(pair, 0.0), depth)
        if self.automatic:
            while self.check_index < len(CHECKPOINTS) and self.data.time + 1e-8 >= CHECKPOINTS[self.check_index][0]:
                when, left, right = CHECKPOINTS[self.check_index]
                states = self.statuses()
                self.checkpoints.append({"time": when, "states": states,
                                         "passed": states == dict(left=left, right=right)})
                self.check_index += 1

    def report(self):
        cfg, checks = self.cfg, {}
        checks["finite_state"] = self.finite
        checks["no_mujoco_warnings"] = bool(not np.any(self.data.warning.number))
        checks["no_command_jump"] = self.command_jump < 1e-12
        checks["joint_limits"] = self.limit_violation <= cfg["validation"]["max_limit_violation"]
        checks["mimic_constraints"] = self.mimic_error <= cfg["validation"]["max_mimic_error"]
        checks["no_deep_penetration"] = self.penetration <= cfg["validation"]["max_penetration"]
        checks["all_joints_source_velocity_limits"] = self.source_velocity_excess <= 1e-6
        if self.automatic:
            checks["all_demo_checkpoints"] = (len(self.checkpoints) == len(CHECKPOINTS)
                                              and all(c["passed"] for c in self.checkpoints))
        for side in SIDES:
            p = self.peaks[side]
            checks[f"{side}_tracking"] = bool(np.max(p["tracking_error_rad"]) <= cfg["validation"]["max_tracking_error"])
            for field, limit in (("reference_velocity_rad_s", "max_velocity"),
                                 ("reference_acceleration_rad_s2", "max_acceleration"),
                                 ("reference_jerk_rad_s3", "max_jerk")):
                checks[f"{side}_{limit}"] = bool(np.all(p[field] <= np.array(cfg["trajectory"][limit]) + 1e-6))
            checks[f"{side}_measured_speed"] = bool(np.all(
                p["measured_velocity_rad_s"] <= np.array(cfg["trajectory"]["max_velocity"])
                + cfg["validation"]["physical_velocity_tolerance"]))
        checks = {key: bool(value) for key, value in checks.items()}
        return {
            "passed": all(checks.values()), "checks": checks, "scope": "empty-hand fixed-wrist motion only",
            "mujoco_version": mujoco.__version__, "duration_s": self.data.time,
            "model": {"nq": self.model.nq, "nv": self.model.nv, "nu": self.model.nu, "neq": self.model.neq},
            "limits": cfg["trajectory"], "thresholds": cfg["validation"],
            "peaks": {s: {k: a.tolist() for k, a in p.items()} for s, p in self.peaks.items()},
            "max_mimic_error_rad": self.mimic_error, "max_limit_violation_rad": self.limit_violation,
            "max_penetration_m": self.penetration, "max_command_state_jump": self.command_jump,
            "max_source_velocity_excess_rad_s": self.source_velocity_excess,
            "contacts": self.contacts, "warnings": self.data.warning.number.tolist(),
            "events": self.events, "checkpoints": self.checkpoints, "final_states": self.statuses(),
            "command_changes": {s: c.command_changes for s, c in self.control.controllers.items()},
        }
