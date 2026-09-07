"""Free-cylinder grasp probe. Support withdraws physically; object is never pinned."""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_hand_control import DualHandControl
from r2v2_description.model import build_model_xml, initialize_hands, load_config, urdf_hand_joints


@dataclass
class CylinderParameters:
    side: str = "left"
    x: float = 0.015
    palm_offset: float = 0.035  # left: y=0.15-offset, right: y=-0.15+offset
    z: float = 0.45
    radius: float = 0.020
    height: float = 0.120
    mass: float = 0.100
    close_time: float = 0.5
    withdraw_start: float = 5.0
    withdraw_end: float = 7.0
    release_time: float = 12.0
    duration: float = 15.0
    grasp: bool = True

    @property
    def center(self):
        return np.array([self.x, (0.15 - self.palm_offset) * (1 if self.side == "left" else -1), self.z])


class CylinderExperiment:
    def __init__(self, params=None, hand_cfg=None):
        self.params = params or CylinderParameters()
        p = self.params
        scalars = [p.x, p.palm_offset, p.z, p.radius, p.height, p.mass,
                   p.close_time, p.withdraw_start, p.withdraw_end, p.release_time, p.duration]
        if (p.side not in ("left", "right") or not np.all(np.isfinite(scalars))
                or min(p.radius, p.height, p.mass) <= 0):
            raise ValueError("Invalid cylinder side or dimensions")
        if not 0 <= p.close_time < p.withdraw_start < p.withdraw_end < p.release_time < p.duration:
            raise ValueError("Invalid grasp experiment timeline")
        self.cfg = copy.deepcopy(hand_cfg or load_config())
        # Resolve object impacts at 1 kHz; keep the hand's 100 Hz command rate.
        self.cfg["simulation_dt"] = 0.001
        root = ET.fromstring(build_model_xml(self.cfg, fixture=True))
        root.set("model", "R2V2_free_cylinder_grasp_probe")
        world = root.find("worldbody")
        body = ET.SubElement(world, "body", name="test_cylinder", pos=" ".join(map(str, p.center)))
        ET.SubElement(body, "freejoint", name="cylinder_free")
        ET.SubElement(body, "geom", name="cylinder_geom", type="cylinder",
                      size=f"{p.radius} {p.height / 2}", mass=str(p.mass),
                      rgba="0.95 0.45 0.08 1", friction="1 0.005 0.0001", condim="3",
                      priority="1", solref="0.008 1", solimp="0.95 0.99 0.001")
        # Real support contact during closure, not a weld or object teleport.
        support_z = p.z - p.height / 2 - 0.012
        stand = ET.SubElement(world, "body", name="cylinder_support", mocap="true",
                              pos=f"{p.center[0]} {p.center[1]} {support_z}")
        ET.SubElement(stand, "geom", name="cylinder_support_geom", type="box",
                      size="0.035 0.035 0.012", rgba="0.15 0.55 0.7 1", friction="1 0.005 0.0001")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        initialize_hands(self.model, self.data, self.cfg)
        self.control = DualHandControl(self.model, self.data, self.cfg)
        self.cylinder_body = self.model.body("test_cylinder").id
        self.cylinder_geom = self.model.geom("cylinder_geom").id
        self.support_geom = self.model.geom("cylinder_support_geom").id
        self.support_mocap = self.model.body("cylinder_support").mocapid[0]
        self.floor_geom = self.model.geom("floor").id
        self.support_z = support_z
        self.steps = 0
        self.decimation = round(self.cfg["control_dt"] / self.cfg["simulation_dt"])
        self.max_penetration = 0.0
        self.initial_penetration = max([max(0.0, -c.dist) for c in self.data.contact
                                        if self.cylinder_geom in c.geom] or [0.0])
        # Track non-adjacent hand self-contact separately from hand/object contact.
        # The 75-degree open pose has a small palm/thumb collision-mesh overlap;
        # retain collision response and measure it rather than filtering it out.
        self.hand_geom_sides = [next((i for i, side in enumerate(("left", "right"), 1)
                                     if self.model.geom(g).name.startswith(side + "_")), 0)
                                for g in range(self.model.ngeom)]
        self.initial_hand_self_penetration = max(
            [max(0.0, -float(c.dist)) for c in self.data.contact
             if self.hand_geom_sides[c.geom1]
             and self.hand_geom_sides[c.geom1] == self.hand_geom_sides[c.geom2]] or [0.0])
        self.max_hand_self_penetration = self.initial_hand_self_penetration
        self.max_contact_force = 0.0
        self.max_hand_penetration = 0.0
        self.max_pre_release_penetration = 0.0
        self.max_actuator_torque = 0.0
        self.max_mimic_error = 0.0
        self.max_joint_limit_violation = 0.0
        self.hand_joints = [self.model.joint(name).id for name in urdf_hand_joints()]
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append((self.model.joint(name).qposadr[0],
                                    self.model.joint(mimic.get("joint")).qposadr[0],
                                    float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
        self.samples = []
        self.contacts = {}
        self.current_contacts = {}
        self.hold_samples = []
        self.floor_during_hold = False
        self.support_during_hold = False
        self.finite = True
        self.release_position = None
        self.start_position = self.data.xpos[self.cylinder_body].copy()

    def phase(self):
        t, p = self.data.time, self.params
        if t < p.close_time: return "READY / SUPPORTED"
        if t < p.withdraw_start: return "CLOSING / SUPPORTED"
        if t < p.withdraw_end: return "WITHDRAWING SUPPORT"
        if t < p.release_time: return "UNSUPPORTED HOLD TEST"
        return "OPEN / RELEASE TEST"

    def step(self):
        d, m, p = self.data, self.model, self.params
        t = d.time
        if self.steps % self.decimation == 0:
            close = p.grasp and p.close_time <= t + 1e-8 < p.release_time
            self.control.command(p.side, int(close))
            self.control.update()
        alpha = np.clip((t - p.withdraw_start) / (p.withdraw_end - p.withdraw_start), 0, 1)
        # Quintic smooth withdrawal to 18 cm below the starting platform.
        blend = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
        d.mocap_pos[self.support_mocap, 2] = self.support_z - 0.18 * blend
        self.control.apply(d)
        mujoco.mj_step(m, d)
        self.steps += 1
        self.finite &= bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
        self.max_actuator_torque = max(self.max_actuator_torque, float(np.max(np.abs(d.actuator_force))))
        q = d.qpos[m.jnt_qposadr[self.hand_joints]]
        limits = m.jnt_range[self.hand_joints]
        self.max_joint_limit_violation = max(self.max_joint_limit_violation,
                                             float(np.max(limits[:, 0] - q)), float(np.max(q - limits[:, 1])))
        for dst, src, ratio, offset in self.mimics:
            self.max_mimic_error = max(self.max_mimic_error, abs(d.qpos[dst] - ratio * d.qpos[src] - offset))
        fingers = {}
        support, floor = False, False
        for index, c in enumerate(d.contact):
            if (self.hand_geom_sides[c.geom1]
                    and self.hand_geom_sides[c.geom1] == self.hand_geom_sides[c.geom2]):
                self.max_hand_self_penetration = max(self.max_hand_self_penetration,
                                                      max(0.0, -float(c.dist)))
            if self.cylinder_geom not in c.geom:
                continue
            other = c.geom2 if c.geom1 == self.cylinder_geom else c.geom1
            name = m.geom(other).name
            depth = max(0.0, -float(c.dist))
            self.max_penetration = max(self.max_penetration, depth)
            if d.time < p.release_time:
                self.max_pre_release_penetration = max(self.max_pre_release_penetration, depth)
            if name.startswith(p.side + "_"):
                self.max_hand_penetration = max(self.max_hand_penetration, depth)
            self.contacts[name] = max(self.contacts.get(name, 0.0), depth)
            support |= other == self.support_geom
            floor |= other == self.floor_geom
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, force)
            normal = max(0.0, float(force[0]))
            self.max_contact_force = max(self.max_contact_force, normal)
            if name.startswith(p.side + "_"):
                part = next((f for f in ("thumb", "index", "middle", "ring", "pinky") if f"_{f}_" in name), "palm")
                fingers[part] = fingers.get(part, 0.0) + normal
        self.current_contacts = fingers
        pos = d.xpos[self.cylinder_body].copy()
        if p.withdraw_end <= d.time < p.release_time:
            opposed = fingers.get("thumb", 0.0) > 0.05 and any(fingers.get(f, 0.0) > 0.05 for f in ("index", "middle", "ring", "pinky"))
            self.hold_samples.append((d.time, *pos, opposed))
            self.floor_during_hold |= floor
            self.support_during_hold |= support
        if d.time >= p.release_time and self.release_position is None:
            self.release_position = pos.copy()
        if self.steps % self.decimation == 0:
            self.samples.append({"time": d.time, "position": pos.tolist(), "fingers_normal_force_N": fingers,
                                 "support_contact": bool(support), "floor_contact": bool(floor)})

    def report(self):
        p = self.params
        held = np.array(self.hold_samples, dtype=float)
        final_pos = self.data.xpos[self.cylinder_body].copy()
        if len(held):
            positions = held[:, 1:4]
            max_displacement = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
            min_height = float(np.min(positions[:, 2]))
            contact_fraction = float(held[:, 4].mean())
        else:
            max_displacement, min_height, contact_fraction = float("inf"), 0.0, 0.0
        release_drop = float(self.release_position[2] - final_pos[2]) if self.release_position is not None else 0.0
        checks = {
            "completed": self.data.time >= p.duration - 0.005,
            "finite_no_warnings": self.finite and not np.any(self.data.warning.number),
            "hand_joint_limits": self.max_joint_limit_violation < 0.01,
            "hand_mimic_constraints": self.max_mimic_error < 0.015,
            "no_initial_overlap": self.initial_penetration < 0.001,
            # Intentional post-release floor impact is reported separately;
            # it is not evidence for/against a physically supported grasp.
            "limited_pre_release_penetration": self.max_pre_release_penetration < 0.003,
            "limited_hand_penetration": self.max_hand_penetration < 0.003,
            "limited_hand_self_penetration": self.max_hand_self_penetration < 0.003,
            "support_clear_during_hold": not self.support_during_hold,
            "floor_clear_during_hold": not self.floor_during_hold,
            "retained_height": min_height > p.z - 0.025,
            "stable_unsupported_hold": max_displacement < 0.015,
            "opposing_finger_contacts": contact_fraction > 0.95,
            "drops_after_open": release_drop > 0.08,
        }
        checks = {k: bool(v) for k, v in checks.items()}
        return {"grasp_passed": all(checks.values()), "checks": checks, "parameters": asdict(p),
                "scope": "fixed wrist; free object; no object weld, teleport, adhesion or external force",
                "duration_s": self.data.time, "start_position": self.start_position.tolist(),
                "final_position": final_pos.tolist(), "initial_penetration_m": self.initial_penetration,
                "max_penetration_m": self.max_penetration, "max_contact_normal_force_N": self.max_contact_force,
                "max_hand_penetration_m": self.max_hand_penetration,
                "initial_hand_self_penetration_m": self.initial_hand_self_penetration,
                "max_hand_self_penetration_m": self.max_hand_self_penetration,
                "max_pre_release_penetration_m": self.max_pre_release_penetration,
                "max_actuator_torque_Nm": self.max_actuator_torque,
                "max_mimic_error_rad": self.max_mimic_error,
                "max_joint_limit_violation_rad": self.max_joint_limit_violation,
                "hold_displacement_m": max_displacement, "hold_min_height_m": min_height,
                "opposing_contact_fraction": contact_fraction, "release_drop_m": release_drop,
                "warnings": self.data.warning.number.tolist(), "contact_geoms": self.contacts}
