"""Native Reach configuration, parity and free-standing initialization."""

from pathlib import Path
import json
import mujoco
import numpy as np
import yaml
from r2v2_deploy.resources import PROJECT_ROOT
from r2v2_deploy.control.reach_policy import (
    ENDPOINT_CONTRACTS,
    LEGACY_ENDPOINT_CONTRACT,
)
from r2v2_deploy.simulation.robot import BODY_JOINTS, JointMap, SIDES, initialize_hands

TCP_OFFSETS = {
    "left": np.array([0.1735, -4e-7, -0.0317504]),
    "right": np.array([0.1735, 4e-7, -0.0317504]),
}


def load_reach_config(path=None):
    if path is None:
        raise ValueError("An explicit native Reach configuration is required")
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    from r2v2_deploy.configuration import resolve_archived_path

    for key in ("policy_path", "checkpoint_path"):
        cfg[key] = str(resolve_archived_path(cfg[key]))
    if (cfg["simulation_dt"], cfg["policy_dt"], cfg["hand_dt"]) != (0.001, 0.02, 0.01):
        raise ValueError(
            "Validated schedule requires 1 kHz physics, 50 Hz policy, 100 Hz hands"
        )
    if cfg.get("endpoint_contract", LEGACY_ENDPOINT_CONTRACT) not in ENDPOINT_CONTRACTS:
        raise ValueError("Unknown Reach endpoint_contract in simulation config")
    return cfg


def resolve_asset(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256(path):
    from r2v2_deploy.configuration import digest

    return digest(path)


def require_parity(path, cfg):
    report = json.loads(Path(path).read_text())
    if not report.get("passed"):
        raise ValueError("Training/deployment numerical parity has not passed")
    if report.get("endpoint_contract", LEGACY_ENDPOINT_CONTRACT) != cfg.get(
        "endpoint_contract", LEGACY_ENDPOINT_CONTRACT
    ):
        raise ValueError("Parity evidence belongs to a different endpoint contract")
    for key, file in (
        ("onnx_sha256", resolve_asset(cfg["policy_path"])),
        ("checkpoint_sha256", resolve_asset(cfg["checkpoint_path"])),
        (
            "adapter_sha256",
            Path(__file__).resolve().parents[1] / "control/reach_policy.py",
        ),
    ):
        if report.get(key) != sha256(file):
            raise ValueError(f"Stale or mismatched parity evidence: {key}")
    return report


def foot_collision_ids(model):
    ids = [
        g
        for g in range(model.ngeom)
        if model.body(model.geom_bodyid[g]).name
        in ("left_ankle_roll_link", "right_ankle_roll_link")
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if len(ids) != 28 or any(
        (model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in ids)
    ):
        raise ValueError(
            "New-asset foot geometry changed; revalidate ground initialization"
        )
    return ids


def initialize_robot(model, data, hands, clearance=0.001):
    mapping = JointMap.create(model, BODY_JOINTS)
    q = np.zeros(28)
    for side in SIDES:
        for suffix, angle in (
            ("hip_pitch", -0.1),
            ("knee", 0.3),
            ("ankle_pitch", -0.2),
        ):
            q[BODY_JOINTS.index(f"{side}_{suffix}_joint")] = angle
    data.qpos[mapping.qpos] = q
    initialize_hands(model, data, hands)
    ids = foot_collision_ids(model)
    bottom = min((data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in ids))
    base_qpos = model.joint("floating_base_joint").qposadr[0]
    data.qpos[base_qpos + 2] += clearance - bottom
    mujoco.mj_forward(model, data)
    return float(data.qpos[base_qpos + 2])
