#!/usr/bin/env python3
"""Validate the deploy package on a fresh machine.

This script is intentionally read-only. It checks dependency imports, required
policy artifacts, selected reference-motion hashes, dual-agent ONNX contracts,
and a full FSM initialization pass.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IMPORTS = [
    "mujoco",
    "onnxruntime",
    "torch",
    "yaml",
    "pygame",
    "unitree_sdk2py",
]

REQUIRED_FILES = [
    "policy/loco_mode/model/policy_29dof.pt",
    "policy/box_handoff_stand/model/policy.onnx",
    "policy/box_hold_stand/model/policy.onnx",
    "policy/dual_agent_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_tracking/motion/walk_tracking_ref.npz",
    "policy/dual_agent_run_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_run_tracking/motion/run_tracking_ref.npz",
    "policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz",
    "policy/dual_agent_dance_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz",
]

MOTION_EXPECTATIONS = {
    "walk": {
        "path": "policy/dual_agent_tracking/motion/walk_tracking_ref.npz",
        "frames": 1150,
        "fps": 50,
        "sha256": "8c970e646fa59af71dd9e95959d0098496b4af81b08a6bd4ee82028b08dcfd55",
    },
    "run": {
        "path": "policy/dual_agent_run_tracking/motion/run_tracking_ref.npz",
        "frames": 350,
        "fps": 50,
        "sha256": "d1a6c74a8c1ea640882dd6b71fde8b79e2c4ebb8556620aa65956ef93955720d",
    },
    "jump": {
        "path": "policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz",
        "frames": 350,
        "fps": 50,
        "sha256": "9df2a6e567626e4939a95a2609f6d22e61bab868fe2d300ed45f5d0209199488",
    },
    "sprint_y_l1": {
        "path": "policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz",
        "frames": 350,
        "fps": 50,
        "sha256": "1f2f10098c2fe0aea93b09afcf2211815e4378f84993ab0feb6a8ae190a6e2db",
    },
}

DUAL_AGENT_MODELS = {
    "walk": "policy/dual_agent_tracking/model/dual_agent_combined.onnx",
    "run": "policy/dual_agent_run_tracking/model/dual_agent_combined.onnx",
    "jump": "policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx",
    "sprint_y_l1": "policy/dual_agent_dance_tracking/model/dual_agent_combined.onnx",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_imports() -> None:
    print("[1/5] Checking Python imports")
    for name in REQUIRED_IMPORTS:
        importlib.import_module(name)
        print(f"  ok import {name}")


def check_required_files() -> None:
    print("[2/5] Checking required policy artifacts")
    missing = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.exists():
            print(f"  ok {rel} ({path.stat().st_size} bytes)")
        else:
            missing.append(rel)
            print(f"  MISSING {rel}")
    if missing:
        raise RuntimeError("Missing required files: " + ", ".join(missing))


def check_motions() -> None:
    print("[3/5] Checking selected reference motions")
    for name, spec in MOTION_EXPECTATIONS.items():
        path = ROOT / spec["path"]
        data = np.load(path)
        frames = int(data["lower_joint_pos"].shape[0])
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        digest = sha256(path)
        if frames != spec["frames"] or fps != spec["fps"] or digest != spec["sha256"]:
            raise RuntimeError(
                f"{name} motion mismatch: frames={frames}, fps={fps}, sha256={digest}"
            )
        print(f"  ok {name}: {frames} frames @ {fps}Hz sha256={digest}")


def check_onnx_contracts() -> None:
    print("[4/5] Checking dual-agent ONNX contracts")
    for name, rel in DUAL_AGENT_MODELS.items():
        path = ROOT / rel
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = {item.name: item.shape[-1] for item in sess.get_inputs()}
        if inputs.get("upper_obs") != 96 or inputs.get("lower_obs") != 109:
            raise RuntimeError(f"{name} ONNX input mismatch: {inputs}")
        out = sess.run(
            ["actions"],
            {
                "upper_obs": np.zeros((1, 96), dtype=np.float32),
                "lower_obs": np.zeros((1, 109), dtype=np.float32),
            },
        )[0]
        if out.shape != (1, 29) or not np.isfinite(out).all():
            raise RuntimeError(f"{name} ONNX output mismatch: shape={out.shape}")
        print(f"  ok {name}: upper_obs=96 lower_obs=109 actions={out.shape}")


def check_fsm_init() -> None:
    print("[5/5] Checking FSM initialization")
    sys.path.insert(0, str(ROOT))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from FSM.FSM import FSM

    FSM(StateAndCmd(29), PolicyOutput(29))
    print("  ok FSM initialized")


def main() -> int:
    try:
        check_imports()
        check_required_files()
        check_motions()
        check_onnx_contracts()
        check_fsm_init()
    except Exception as exc:
        print(f"\nVALIDATION FAILED: {exc}", file=sys.stderr)
        return 1
    print("\nVALIDATION PASSED: deploy package is ready for MuJoCo smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
