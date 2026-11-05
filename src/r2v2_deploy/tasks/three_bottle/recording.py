"""Compact output for one continuous execution."""

from __future__ import annotations
import json
from pathlib import Path
import mujoco
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported event value: {type(value).__name__}")


def save_run(controller, output: Path, runtime: dict, interrupted=False) -> dict:
    """Save the actual 50 Hz state/action stream and terminal integration state."""
    s = controller
    completed = s.completed_bottles == [1, 2, 3]
    success = bool(s.done and completed and (not s.failure) and (not interrupted))
    report = {
        "controller": "Loco + Cartesian IK + independent finger control",
        "success": success,
        "duration_s": float(s.data.time),
        "stage": s.stage,
        "completed_bottles": s.completed_bottles,
        "failure": "Interrupted" if interrupted else s.failure,
        "runtime": runtime,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "events.json").write_text(
        json.dumps(
            {"transitions": s.transitions, "bottles": s.multi_events},
            indent=2,
            default=_json_value,
        )
        + "\n"
    )
    if s.states:
        np.savez_compressed(
            output / "trajectory_50hz.npz",
            time_s=np.array([row[0] for row in s.states]),
            physical_qpos=np.array([row[1] for row in s.states]),
            physical_qvel=np.array([row[2] for row in s.states]),
            actions28=np.array([row["actions"] for row in s.trace]),
            hand_commands2=np.array([row["hand_commands"] for row in s.trace]),
            phase=np.array([row[4] for row in s.states]),
        )
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(s.model, signature))
    mujoco.mj_getState(s.model, s.data, state, signature)
    np.savez_compressed(
        output / "terminal_state.npz",
        integration_state=state,
        state_signature=int(signature),
    )
    return report
