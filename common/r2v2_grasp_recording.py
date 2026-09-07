"""Lossless, named hand/reference and object-in-wrist logs for FSM handoff.

Kinematics are evaluated on private scratch data: recording never forwards,
steps or resets the live physics state. These traces are observations of a
fixed-wrist experiment, not commands to replay blindly on the full robot.
"""

from common.path_config import PROJECT_ROOT

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from r2v2_description.model import SOURCE, urdf_hand_joints


CONTACT_PARTS = ("palm", "thumb", "index", "middle", "ring", "pinky")


def body_transform(data, body_id):
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def relative_transform(parent, child):
    """T_parent_child: child coordinates -> parent coordinates."""
    result = np.eye(4)
    result[:3, :3] = parent[:3, :3].T @ child[:3, :3]
    result[:3, 3] = parent[:3, :3].T @ (child[:3, 3] - parent[:3, 3])
    return result


def transform_pose(transform):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).ravel())
    if quat[0] < 0:
        quat *= -1
    return {"position_m": transform[:3, 3].tolist(), "quaternion_wxyz": quat.tolist()}


class GraspRecorder:
    def __init__(self, experiment):
        self.exp = experiment
        self.side = experiment.params.side
        self.wrist_name = f"{self.side}_hand_roll_link"
        self.wrist_id = experiment.model.body(self.wrist_name).id
        self.joint_names = tuple(n for n in urdf_hand_joints() if n.startswith(self.side + "_"))
        joint_ids = [experiment.model.joint(n).id for n in self.joint_names]
        self.qpos = experiment.model.jnt_qposadr[joint_ids]
        self.dofs = experiment.model.jnt_dofadr[joint_ids]
        self.scratch = mujoco.MjData(experiment.model)
        self.rows = []
        self.capture()

    def capture(self):
        exp, scratch = self.exp, self.scratch
        d, m = exp.data, exp.model
        if self.rows and d.time <= self.rows[-1]["time_s"]:
            raise ValueError("Recording timestamps must strictly increase")
        # mj_step's derived poses may precede its integration. Recompute only
        # kinematics at the recorded qpos on scratch, never on the live data.
        scratch.qpos[:] = d.qpos
        scratch.mocap_pos[:] = d.mocap_pos
        scratch.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(m, scratch)
        wrist = body_transform(scratch, self.wrist_id)
        cylinder = body_transform(scratch, exp.cylinder_body)
        mapping = exp.control.maps[self.side]
        controller = exp.control.controllers[self.side]
        ref = controller.reference
        support = floor = False
        for contact in d.contact:
            if exp.cylinder_geom in contact.geom:
                support |= exp.support_geom in contact.geom
                floor |= exp.floor_geom in contact.geom
        self.rows.append({
            "time_s": float(d.time), "command": controller.command, "phase": exp.phase(),
            "reference_q_rad": ref.position.copy(),
            "reference_qd_rad_s": ref.velocity.copy(),
            "reference_qdd_rad_s2": ref.acceleration.copy(),
            "measured_q_rad": d.qpos[self.qpos].copy(),
            "measured_qd_rad_s": d.qvel[self.dofs].copy(),
            "commanded_torque_Nm": d.ctrl[mapping.actuators].copy(),
            "actuator_torque_Nm": d.actuator_force[mapping.actuators].copy(),
            "T_world_wrist": wrist, "T_world_cylinder": cylinder,
            "T_wrist_cylinder": relative_transform(wrist, cylinder),
            "contact_normal_force_N": np.array([exp.current_contacts.get(k, 0.0) for k in CONTACT_PARTS]),
            "support_contact": bool(support), "floor_contact": bool(floor),
        })

    def arrays(self):
        return {
            "schema_version": np.array(1),
            "side": np.array(self.side), "wrist_body": np.array(self.wrist_name),
            "reference_joint_names": np.array(self.exp.control.maps[self.side].names),
            "measured_joint_names": np.array(self.joint_names),
            "contact_part_names": np.array(CONTACT_PARTS),
            **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]},
        }

    def save(self, output, report):
        output = Path(output)
        trajectory = output / "grasp_trajectory.npz"
        manifest = output / "grasp_record.json"
        if trajectory.exists() or manifest.exists():
            raise FileExistsError("Refusing to overwrite a grasp recording")
        output.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        np.savez_compressed(trajectory, **arrays)
        keyframes = {}
        p = self.exp.params
        for label, when in (("initial", 0), ("pre_close", p.close_time),
                            ("supported_grasp", p.withdraw_start - 0.5),
                            ("hold_start", p.withdraw_end),
                            ("hold_mid", (p.withdraw_end + p.release_time) / 2),
                            ("pre_release", p.release_time),
                            ("released", p.duration - 0.5), ("final", p.duration)):
            index = int(np.argmin(np.abs(arrays["time_s"] - when)))
            if abs(float(arrays["time_s"][index]) - when) > self.exp.cfg["control_dt"] / 2:
                continue  # Do not label a partial recording with future events.
            relation = arrays["T_wrist_cylinder"][index]
            keyframes[label] = {
                "index": index, "time_s": float(arrays["time_s"][index]),
                "command_during_preceding_step": int(arrays["command"][index]),
                "cylinder_in_wrist": transform_pose(relation),
                "wrist_in_cylinder": transform_pose(np.linalg.inv(relation)),
                "reference_q_rad": arrays["reference_q_rad"][index].tolist(),
                "measured_q_rad": arrays["measured_q_rad"][index].tolist(),
            }
        source_files = [p for p in SOURCE.rglob("*") if p.is_file()]
        source_files += [PROJECT_ROOT / path for path in (
            "common/r2v2_hand_control.py", "common/r2v2_cylinder_test.py",
            "common/r2v2_grasp_recording.py", "r2v2_description/model.py",
            "deploy_mujoco/r2v2_cylinder_test.py", "deploy_mujoco/config/r2v2_hands.yaml",
            "requirements-r2v2-sim.txt")]
        metadata = {
            "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "trajectory_file": trajectory.name,
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "samples": len(self.rows), "sample_dt_s": self.exp.cfg["control_dt"],
            "physics_dt_s": self.exp.cfg["simulation_dt"], "side": self.side,
            "wrist_body": self.wrist_name, "cylinder_body": "test_cylinder",
            "units": {"distance": "m", "angle": "rad", "time": "s", "torque": "Nm", "force": "N"},
            "transform_convention": "T_A_B maps B-frame coordinates into A; column vectors; translation in A",
            "quaternion_convention": "wxyz, scalar first; rotation maps local coordinates into parent",
            "sampling_convention": (
                "t=0 initialization then post-step qpos/qvel every control_dt. Pose transforms are FK at that qpos. "
                "command/reference/torque/contact describe the preceding physics step; contact forces are instantaneous, "
                "not 10 ms averages. The boundary command change appears in the next sampled row. "
                "phase describes the timeline at the row time, not measured grasp success."),
            "scope": "fixed wrist, selected hand only; no full-body trajectory or verified real-robot replay",
            "reference_joint_names": arrays["reference_joint_names"].tolist(),
            "measured_joint_names": list(self.joint_names), "contact_part_names": list(CONTACT_PARTS),
            "experiment_parameters": asdict(p), "hand_configuration": self.exp.cfg,
            "mujoco_version": mujoco.__version__, "keyframes": keyframes,
            "validation": report,
            "source_sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(source_files)},
        }
        manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        return metadata
