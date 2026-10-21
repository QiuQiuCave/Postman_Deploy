"""Training-only exact CONTACT state, history and hand-reference snapshots."""

import copy
import hashlib
import numpy as np
from .state_io import RUNTIME, POLICY, HAND_INPUT, integration_state, restore


def physics_fingerprint(model):
    checksum = hashlib.sha256()
    for key in (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_type",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "jnt_range",
        "actuator_ctrlrange",
        "actuator_gear",
    ):
        checksum.update(key.encode())
        checksum.update(np.ascontiguousarray(getattr(model, key)).tobytes())
    checksum.update(
        np.asarray([model.opt.timestep, *model.opt.gravity], dtype="<f8").tobytes()
    )
    return checksum.hexdigest()


def capture_training(exp, *, policy_version):
    if (
        not getattr(exp, "training_only", False)
        or exp.mode != "contact"
        or exp.steps % 20
        or exp.done
    ):
        raise ValueError("Only actual50Hz training CONTACT states can enter this cache")
    return _capture_state(exp, policy_version=policy_version)


def _capture_state(exp, *, policy_version, native_policy=None):
    """Serialize physical/FSM state; caller establishes control/source scope."""
    hands = {}
    for side, c in exp.hands.controllers.items():
        if not hasattr(c, "snapshot_generator_log"):
            raise ValueError("Actual Ruckig input-event history required")
        hands[side] = dict(
            command=c.command,
            command_changes=c.command_changes,
            finished=c.finished,
            reference=copy.deepcopy(c.reference),
            inputs={k: copy.deepcopy(getattr(c.input, k)) for k in HAND_INPUT},
            generator_replay=copy.deepcopy(c.snapshot_generator_log),
        )
    extra = (
        "delivery_started",
        "placement_completed",
        "box_exit_snapshot",
        "drop_verified",
        "delivery_index",
    )
    snapshot = dict(
        schema="delivery_training_snapshot_v1",
        training_only=True,
        physics=integration_state(exp),
        runtime={
            k: copy.deepcopy(getattr(exp, k))
            for k in (*RUNTIME, *extra)
            if hasattr(exp, k)
        },
        policy=(
            native_policy
            if native_policy is not None
            else {k: copy.deepcopy(getattr(exp.policy, k)) for k in POLICY}
        ),
        hands=hands,
        model_dimensions=(exp.model.nq, exp.model.nv, exp.model.nu),
        plan=copy.deepcopy(exp.plan),
        source_policy_sha256=exp.policy_sha,
        mode="contact",
        policy_version=int(policy_version),
        physics_sha256=physics_fingerprint(exp.model),
        program_sha256=exp.program.sha256,
        source_initial_seed=int(exp.initial_seed),
        initialization_origin=getattr(
            exp, "training_initialization_origin", "normal_standing"
        ),
        official_air_contact_evidence=False,
        continuation_contract="delivery_fsm_index_native_history_v2",
    )
    if hasattr(exp, "demonstration_reach_program"):
        program = exp.demonstration_reach_program
        snapshot["reach_program_state"] = {
            k: copy.deepcopy(getattr(program, k))
            for k in (
                "start",
                "duration",
                "loss_since",
                "settle_since",
                "release_position",
                "table_touch",
                "stable_touch",
                "actual_release",
                "withdrawn",
                "rows",
            )
        }
        snapshot["reach_source_index"] = min(
            len(program.tube.times) - 1,
            int(round((float(exp.data.time) - program.start) / 0.02)),
        )
        snapshot["reach_source_data_sha256"] = program.tube.data_sha256
    return snapshot


def restore_training(exp, snapshot):
    if (
        not getattr(exp, "training_only", False)
        or not snapshot.get("training_only")
        or snapshot.get("schema") != "delivery_training_snapshot_v1"
        or snapshot.get("program_sha256") != exp.program.sha256
        or snapshot.get("physics_sha256") != physics_fingerprint(exp.model)
    ):
        raise ValueError("Unknown/mismatched real-contact training snapshot")
    restore(exp, snapshot)
    if "reach_program_state" in snapshot:
        if not hasattr(exp, "demonstration_reach_program"):
            raise ValueError(
                "Student Reach snapshot requires its matching reference program"
            )
        for key, value in snapshot["reach_program_state"].items():
            setattr(exp.demonstration_reach_program, key, copy.deepcopy(value))
    exp.initial_seed = int(snapshot.get("source_initial_seed", exp.initial_seed))
    exp.training_initialization_origin = snapshot.get(
        "initialization_origin", "normal_standing"
    )
    for side, c in exp.hands.controllers.items():
        # Keep the original recorder closure's dict identity. Replacing the
        # attribute alone would leave its callback logging an empty new clock.
        log = c.snapshot_generator_log
        log.clear()
        log.update(copy.deepcopy(snapshot["hands"][side]["generator_replay"]))
