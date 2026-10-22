"""Exact training-reset state I/O and dynamic hand continuation."""

import copy
import mujoco
import numpy as np
from r2v2_deploy.control.hand_control import BinaryHandController

STATE = mujoco.mjtState.mjSTATE_INTEGRATION

RUNTIME = (
    "phase",
    "phase_start",
    "steps",
    "failure",
    "failure_phase",
    "grasp_verified",
    "release_commanded",
    "completed",
    "baseline_relation",
    "alignment_relation",
    "upright_attempted",
    "max_verified_hold_s",
    "stable_since",
    "bad_since",
    "transitions",
    "stage_results",
    "safety_events",
    "contact_events",
    "current_metrics",
    "goal_wrist_transforms",
    "active_goal",
    "initial_object_position",
    "foot_anchor",
    "stage_collision",
    "peaks",
    "initial_metrics",
    "top_active",
    "tracking_verified",
    "top_index",
    "top_start",
    "top_end",
    "top_motion_s",
    "can_exit_verified",
    "contact_air_verified",
    "pickup_snapshot",
    "release_snapshot",
    "drop_snapshot",
    "drop_verified",
)

POLICY = (
    "last_action",
    "q_des",
    "last_torque",
    "references",
    "histories",
    "last_observation",
)

HAND_INPUT = (
    "current_position",
    "current_velocity",
    "current_acceleration",
    "target_position",
    "target_velocity",
    "target_acceleration",
    "max_velocity",
    "max_acceleration",
    "max_jerk",
)


def record_hand_generator(controller):
    """Training-only deterministic input-event log, not a runtime intervention."""
    if hasattr(controller, "snapshot_generator_log"):
        raise ValueError("Hand generator recorder already installed")
    log = dict(ticks=0, events=[])
    controller.snapshot_generator_log = log
    original = controller.step

    def step():
        target = list(controller.input.target_position)
        if (
            not log["events"]
            or target != log["events"][-1]["inputs"]["target_position"]
        ):
            log["events"].append(
                dict(
                    tick=log["ticks"],
                    inputs={
                        k: copy.deepcopy(getattr(controller.input, k))
                        for k in HAND_INPUT
                    },
                )
            )
        result = original()
        log["ticks"] += 1
        return result

    controller.step = step


def restore_hand_generator(controller, log):
    """Rebuild Ruckig's internal trajectory/time by the original input events."""
    controller.generator = type(controller.generator)(6, controller.dt)
    controller.input = type(controller.input)(6)
    controller.output = type(controller.output)(6)
    events = {e["tick"]: e["inputs"] for e in log["events"]}
    if not log["ticks"] or 0 not in events:
        raise ValueError("Missing initial hand-generator event")
    for tick in range(log["ticks"]):
        if tick in events:
            for k, v in events[tick].items():
                setattr(controller.input, k, copy.deepcopy(v))
        BinaryHandController.step(controller)


def integration_state(exp):
    state = np.empty(mujoco.mj_stateSize(exp.model, STATE))
    mujoco.mj_getState(exp.model, exp.data, state, STATE)
    return state


def restore(exp, snapshot):
    if exp.mode != snapshot.get("mode", "air") or tuple(
        snapshot["model_dimensions"]
    ) != (
        exp.model.nq,
        exp.model.nv,
        exp.model.nu,
    ):
        raise ValueError("Matching dynamic physics mode and model required")
    if exp.plan != snapshot["plan"]:
        raise ValueError("Cannot change planned goals or durations")
    for k, v in snapshot["runtime"].items():
        setattr(exp, k, copy.deepcopy(v))
    for k, v in snapshot["policy"].items():
        setattr(exp.policy, k, copy.deepcopy(v))
    for side, saved in snapshot["hands"].items():
        c = exp.hands.controllers[side]
        if "generator_replay" in saved:
            restore_hand_generator(c, saved["generator_replay"])
            if any(
                (
                    not np.array_equal(
                        getattr(c.reference, k), getattr(saved["reference"], k)
                    )
                    for k in ("position", "velocity", "acceleration")
                )
            ):
                raise RuntimeError("Reconstructed hand-generator reference differs")
        else:
            c.generator.reset()
        for k, v in saved["inputs"].items():
            setattr(c.input, k, copy.deepcopy(v))
        for k in ("command", "command_changes", "finished", "reference"):
            setattr(c, k, copy.deepcopy(saved[k]))
    mujoco.mj_setState(exp.model, exp.data, snapshot["physics"], STATE)
    mujoco.mj_forward(exp.model, exp.data)
    mujoco.mj_setState(exp.model, exp.data, snapshot["physics"], STATE)
    exp.samples = []
    exp.targets = []
    exp.sync()
    if not np.array_equal(integration_state(exp), snapshot["physics"]):
        raise RuntimeError("Integration state was not restored exactly")
