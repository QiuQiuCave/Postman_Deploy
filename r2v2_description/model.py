"""Build deployment/fixture models without modifying the supplied XML/URDF.

Body state is always explicitly ordered: never slice qpos by actuator count.
The fixed-wrist fixture contains the exact two wrist/hand subtrees, not a
different hand model. The full robot retains its floating base and 28 motors.
"""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

SOURCE = PROJECT_ROOT / "r2v2_description/source/r2v2_with_hand"
CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_hands.yaml"
SIDES = ("left", "right")
HAND_CHANNELS = (
    "thumb_metacarpal_joint", "thumb_proximal_joint", "index_proximal_joint",
    "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint",
)
BODY_JOINTS = tuple(
    f"{side}_{part}_joint"
    for side in SIDES
    for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint", "waist_pitch_joint") + tuple(
    f"{side}_{part}_joint"
    for side in SIDES
    for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "arm_pitch",
                 "arm_yaw", "hand_pitch", "hand_roll")
)


def hand_names(side):
    if side not in SIDES:
        raise ValueError(f"Unknown hand: {side}")
    return tuple(f"{side}_{name}" for name in HAND_CHANNELS)


def urdf_hand_joints():
    root = ET.parse(SOURCE / "r2v2_with_hand.urdf").getroot()
    return {
        j.get("name"): j for j in root.findall("joint")
        if j.get("type") == "revolute"
        and any(f"_{finger}_" in j.get("name", "")
                for finger in ("thumb", "index", "middle", "ring", "pinky"))
    }


def load_config(path=CONFIG):
    with Path(path).open() as stream:
        cfg = yaml.safe_load(stream)
    if tuple(cfg["channels"]) != HAND_CHANNELS:
        raise ValueError("Hand channels must match the documented named order")
    dt, control_dt = float(cfg["simulation_dt"]), float(cfg["control_dt"])
    if not (0 < dt <= control_dt and np.isclose(control_dt / dt, round(control_dt / dt))):
        raise ValueError("control_dt must be a positive integer multiple of simulation_dt")
    for group in ("trajectory", "servo"):
        for key, value in cfg[group].items():
            a = np.asarray(value, dtype=float)
            if a.shape != (6,) or not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError(f"{group}.{key}: expected six finite positive values")
    joints = urdf_hand_joints()
    for side in SIDES:
        for pose in ("open", "closed"):
            q = np.asarray(cfg["hands"][side][pose], dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)):
                raise ValueError(f"{side}.{pose}: expected six finite values")
            for i, name in enumerate(hand_names(side)):
                lim = joints[name].find("limit")
                if not float(lim.get("lower")) <= q[i] <= float(lim.get("upper")):
                    raise ValueError(f"{side}.{pose}: {name} outside joint limits")
                if cfg["servo"]["torque_limit"][i] > float(lim.get("effort")):
                    raise ValueError(f"{name}: configured torque exceeds source effort limit")
        # Coupled distal joints constrain the proximal velocity/pose as well.
        for name, joint in joints.items():
            mimic = joint.find("mimic")
            if mimic is None or not name.startswith(side + "_"):
                continue
            index = hand_names(side).index(mimic.get("joint"))
            ratio = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            lim = joint.find("limit")
            for pose in ("open", "closed"):
                q = cfg["hands"][side][pose][index] * ratio + offset
                if not float(lim.get("lower")) <= q <= float(lim.get("upper")):
                    raise ValueError(f"{name}: coupled {pose} pose outside limits")
            if cfg["trajectory"]["max_velocity"][index] * abs(ratio) > float(lim.get("velocity")):
                raise ValueError(f"{name}: coupled velocity exceeds source limit")
        for i, name in enumerate(hand_names(side)):
            if cfg["trajectory"]["max_velocity"][i] > float(joints[name].find("limit").get("velocity")):
                raise ValueError(f"{name}: velocity exceeds source limit")
    return cfg


@dataclass(frozen=True)
class JointMap:
    names: tuple
    joints: np.ndarray
    qpos: np.ndarray
    dofs: np.ndarray
    actuators: np.ndarray

    @classmethod
    def create(cls, model, names):
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate joints in mapping")
        joints, acts = [], []
        for name in names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise ValueError(f"Missing joint: {name}")
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected hinge joint: {name}")
            candidates = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == j)
            )
            if len(candidates) != 1:
                raise ValueError(f"Expected one actuator for {name}, found {len(candidates)}")
            joints.append(j)
            acts.append(candidates[0])
        j = np.asarray(joints, dtype=int)
        return cls(names, j, model.jnt_qposadr[j].copy(), model.jnt_dofadr[j].copy(),
                   np.asarray(acts, dtype=int))


def build_model_xml(cfg, fixture=False):
    root = ET.parse(SOURCE / "r2v2_with_hand.xml").getroot()
    root.set("model", "R2V2_fixed_wrist_test" if fixture else "R2V2_with_controlled_hands")
    root.find("compiler").set("meshdir", str(SOURCE / "meshes"))
    root.find("compiler").set("autolimits", "true")
    ET.SubElement(root, "option", timestep=str(cfg["simulation_dt"]),
                  integrator="implicitfast", iterations="100", tolerance="1e-10")
    ET.SubElement(root, "visual")
    ET.SubElement(root.find("visual"), "global", offwidth="1280", offheight="720")
    ET.SubElement(root.find("visual"), "headlight", ambient="0.35 0.35 0.35",
                  diffuse="0.65 0.65 0.65", specular="0.2 0.2 0.2")
    world = root.find("worldbody")
    if fixture:
        hands = [copy.deepcopy(root.find(f'.//body[@name="{side}_hand_roll_link"]'))
                 for side in SIDES]
        for body in list(world.findall("body")):
            world.remove(body)
        for side, hand in zip(SIDES, hands):
            # Wrist frames clamped to fixtures. Fingers remain fully dynamic.
            hand.set("pos", f"-0.13 {0.15 if side == 'left' else -0.15} 0.45")
            for joint in list(hand.findall("joint")):
                hand.remove(joint)
            world.append(hand)
            support = ET.SubElement(world, "body", name=f"{side}_fixture",
                                    pos=f"-0.15 {0.15 if side == 'left' else -0.15} 0.22")
            ET.SubElement(support, "geom", name=f"{side}_fixture_visual", type="box",
                          size="0.025 0.035 0.22", rgba="0.2 0.3 0.4 1",
                          contype="0", conaffinity="0")
        root.remove(root.find("sensor"))
        root.find("actuator").clear()
    else:
        existing = tuple(a.get("joint") for a in root.find("actuator"))
        if existing != BODY_JOINTS:
            raise ValueError("Source body actuator layout differs from the 28-DoF contract")

    # Hide collision duplicates in rendering, but keep their physics enabled.
    for body in world.iter("body"):
        for index, geom in enumerate(body.findall("geom")):
            visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            if not geom.get("name"):
                geom.set("name", f'{body.get("name")}_{"vis" if visual else "col"}_{index}')
            # Separate left/right visuals lets close-up recording hide the
            # opposite hand without disabling any contacts in the simulation.
            geom.set("group", ("2" if body.get("name", "").startswith("right_") else "1")
                     if visual else "3")
    equality = ET.SubElement(root, "equality")
    contact = ET.SubElement(root, "contact")
    # MuJoCo does not apply its moving-parent collision filter when the palm
    # is welded to the world. Preserve the same adjacent-link exclusions in
    # the fixture and the floating robot; all non-adjacent collisions remain.
    for side in SIDES:
        for part in ("thumb_metacarpal", "index_proximal", "middle_proximal",
                     "ring_proximal", "pinky_proximal"):
            ET.SubElement(contact, "exclude", body1=f"{side}_hand_roll_link",
                          body2=f"{side}_{part}_link")
    source_joints = urdf_hand_joints()
    for name, joint in source_joints.items():
        mimic = joint.find("mimic")
        if mimic is not None:
            ET.SubElement(equality, "joint", name=f"couple_{name}", joint1=name,
                          joint2=mimic.get("joint"),
                          polycoef=f'{mimic.get("offset", "0")} {mimic.get("multiplier", "1")} 0 0 0',
                          solref="0.008 1", solimp="0.99 0.999 0.001")
    for side in SIDES:
        for index, name in enumerate(hand_names(side)):
            limit = cfg["servo"]["torque_limit"][index]
            ET.SubElement(root.find("actuator"), "motor", name=f"drive_{name}",
                          joint=name, gear="1", ctrllimited="true",
                          ctrlrange=f"{-limit} {limit}", forcelimited="true",
                          forcerange=f"{-limit} {limit}")
    # Fixture doesn't need unused whole-body meshes in GPU memory.
    used = {g.get("mesh") for g in world.iter("geom") if g.get("mesh")}
    for mesh in list(root.find("asset").findall("mesh")):
        if mesh.get("name") not in used:
            root.find("asset").remove(mesh)
    return ET.tostring(root, encoding="unicode")


def build_model(cfg=None, fixture=False):
    return mujoco.MjModel.from_xml_string(build_model_xml(cfg or load_config(), fixture))


def initialize_hands(model, data, cfg):
    for side in SIDES:
        mapping = JointMap.create(model, hand_names(side))
        data.qpos[mapping.qpos] = cfg["hands"][side]["open"]
    for name, joint in urdf_hand_joints().items():
        mimic = joint.find("mimic")
        if mimic is not None:
            dst = model.joint(name).qposadr[0]
            src = model.joint(mimic.get("joint")).qposadr[0]
            data.qpos[dst] = (data.qpos[src] * float(mimic.get("multiplier", "1"))
                             + float(mimic.get("offset", "0")))
    mujoco.mj_forward(model, data)
