"""Policy-generated common initial state for independent crate-height trials.

The empty-scene preparation is not an obstacle-avoidance demonstration. Saved
states may initialize a NEW trial at t=0 only. No rollout teleport or IK occurs.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from common.r2v2_crate_motion_recording import transform_from_pose
from common.r2v2_hand_control import HandReference
from common.r2v2_reach_sim import build_reach_model, require_parity, sha256
from common.r2v2_simple_dual_reach import SimpleDualReachExperiment
from common.r2v2_tabletop_demo import copy_robot_initial_state, serializable
from r2v2_description.model import SIDES


COMMON_START = {s: transform_from_pose([.18, y, 1.13], [1, 0, 0, 0])
                for s, y in (("left", .27), ("right", -.27))}


def prepare_common_start(reach_config, parity_report):
    exp = SimpleDualReachExperiment(reach_config, parity_report, case_names=["HOME_HOLD"])
    exp.cases.append(dict(name="COMMON_OUTSIDE", duration_s=8., targets=copy.deepcopy(COMMON_START),
                         note="Empty-scene policy-generated start, not a collision-free approach claim"))
    exp.case_names = [c["name"] for c in exp.cases]
    return exp


def save_common_start(exp, directory):
    if exp.phase != "COMPLETE" or exp.failure:
        raise ValueError("Unsafe/incomplete preparation cannot initialize trials")
    exp.sync()
    if any(exp.errors(s)["wrist_linear_speed_mps"] >= .02 for s in SIDES):
        raise ValueError("Actual wrists have not stopped")
    if np.linalg.norm(exp.data.qvel[:6]) >= .05:
        raise ValueError("Base has not stopped")
    if any(h.command != 0 or not h.finished for h in exp.hands.controllers.values()):
        raise ValueError("Hands must be settled open")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    archive, manifest = out/"prepared_state.npz", out/"prepared_state.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("Refusing to overwrite common initial state")
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(exp.model, spec))
    mujoco.mj_getState(exp.model, exp.data, integration, spec)
    arrays = {"integration_state": integration}
    for key in ("last_action", "q_des", "last_torque", "last_observation"):
        arrays[f"policy_{key}"] = getattr(exp.policy, key).copy()
    for key, value in exp.policy.histories.items():
        arrays[f"history_{key}"] = value.copy()
    for side, ref in exp.policy.references.items():
        for key, value in asdict(ref).items():
            arrays[f"reference_{side}_{key}"] = np.array(value, copy=True)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    info = dict(schema_version=1, scope="empty-scene real policy preparation; only NEW trial initialization",
        sha256=sha256(archive), state_file=archive.name, integration_spec=int(spec),
        endpoint_contract="wrist_world_v2", onnx_sha256=exp.parity["onnx_sha256"],
        checkpoint_sha256=exp.parity["checkpoint_sha256"], adapter_sha256=exp.parity["adapter_sha256"],
        source_model_dimensions=[exp.model.nq, exp.model.nv, exp.model.nu],
        joint_names=[exp.model.joint(i).name for i in range(exp.model.njnt)],
        actuator_names=[exp.model.actuator(i).name for i in range(exp.model.nu)],
        hand_configuration=exp.hand_cfg, source_duration_s=float(exp.data.time), source_steps=exp.steps,
        actual_wrist_world={s: transform_from_pose(*exp.policy.wrist_pose(exp.scratch, s)) for s in SIDES},
        target_wrist_world=COMMON_START, final_errors={s: exp.errors(s) for s in SIDES},
        hand_states={s:dict(command=h.command, finished=h.finished, reference=asdict(h.reference))
                     for s,h in exp.hands.controllers.items()},
        preparation_report=exp.report())
    with manifest.open("x") as stream:
        json.dump(serializable(info), stream, indent=2, allow_nan=False)
    return manifest


def load_common_start(path, cfg, parity_report):
    path = Path(path)
    if path.suffix == ".npz":
        path = path.with_suffix(".json")
    info = json.loads(path.read_text())
    if info.get("schema_version") != 1 or info.get("endpoint_contract") != "wrist_world_v2":
        raise ValueError("Unsupported prepared state")
    if Path(info["state_file"]).name != info["state_file"]:
        raise ValueError("Prepared state archive must be alongside manifest")
    archive = path.parent/info["state_file"]
    if sha256(archive) != info["sha256"]:
        raise ValueError("Prepared state hash mismatch")
    parity = require_parity(parity_report, cfg)
    for key in ("onnx_sha256", "checkpoint_sha256", "adapter_sha256"):
        if info[key] != parity[key]:
            raise ValueError(f"Prepared state uses a different {key}")
    with np.load(archive, allow_pickle=False) as payload:
        arrays = {k: payload[k].copy() for k in payload.files}
    model, _ = build_reach_model(cfg)
    if ([model.nq, model.nv, model.nu] != info["source_model_dimensions"]
            or [model.joint(i).name for i in range(model.njnt)] != info["joint_names"]
            or [model.actuator(i).name for i in range(model.nu)] != info["actuator_names"]):
        raise ValueError("Prepared-state robot layout changed")
    data = mujoco.MjData(model)
    spec = mujoco.mjtState(info["integration_spec"])
    if len(arrays["integration_state"]) != mujoco.mj_stateSize(model, spec):
        raise ValueError("Prepared integration state dimensions changed")
    mujoco.mj_setState(model, data, arrays["integration_state"], spec)
    mujoco.mj_forward(model, data)
    if np.any(data.xfrc_applied) or np.any(data.qfrc_applied) or np.any(data.warning.number):
        raise ValueError("Assisted or numerically invalid preparation")
    return info, arrays, model, data


def initialize_from_common(exp, state):
    """Restore robot/controller continuity at new-scene t=0, preserving props."""
    if exp.data.time != 0. or exp.steps != 0:
        raise ValueError("Prepared robot state cannot be injected into an ongoing trial")
    info, arrays, source_model, source_data = state
    copy_robot_initial_state(source_model, source_data, exp.model, exp.data)
    for key in ("last_action", "q_des", "last_torque", "last_observation"):
        setattr(exp.policy, key, arrays[f"policy_{key}"].copy())
    for key in exp.policy.histories:
        exp.policy.histories[key] = arrays[f"history_{key}"].copy()
    for side, ref in exp.policy.references.items():
        for key in asdict(ref):
            setattr(ref, key, arrays[f"reference_{side}_{key}"].copy())
    for side, h in exp.hands.controllers.items():
        saved = info["hand_states"][side]
        if saved["command"] != 0 or not saved["finished"]:
            raise ValueError("Only idle open fingers can seed a height sweep")
        values = {key: np.array(value, dtype=float) for key, value in saved["reference"].items()}
        h.reference = HandReference(**values)
        for name, value in values.items():
            setattr(h.input, f"current_{name}", value.tolist())
        h.input.target_position = h.poses[0].tolist()
        h.finished = True
    # Control phase is preserved if preparation ended between controller ticks.
    exp.steps = int(info["source_steps"]) % 20
    return info
