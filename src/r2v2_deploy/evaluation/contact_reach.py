"""CONTACT-only v1: same deployment program, independent Reach local tests."""

import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from r2v2_deploy.tasks.contact_reach import (
    PROTOCOL,
    GoalTube,
    source_clock,
    safety_conditions,
)
from r2v2_deploy.data.artifacts import write_json


def create_student(cfg, candidate, output):
    cfg = {
        k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()
    }
    from r2v2_deploy.compatibility.release import validated_candidate
    from r2v2_deploy.simulation.factory import create_scene
    from r2v2_deploy.tasks.exit_placement import ExitPlacementProgram
    import onnx

    Path(output).mkdir(parents=True, exist_ok=True)
    program = ExitPlacementProgram(cfg["program"])
    from r2v2_deploy.tasks.contact_reach_program import validate_finger_condition

    validate_finger_condition(cfg)
    c = validated_candidate(
        candidate,
        program,
        reach_finger_condition_protocol=cfg.get("reach_finger_condition_protocol"),
    )
    meta = {p.key: p.value for p in onnx.load(c["onnx"]).metadata_props}
    parity = json.loads(Path(c["parity"]).read_text())
    if (
        c.get("learning_protocol") != PROTOCOL
        or meta.get("learning_protocol") != PROTOCOL
        or parity.get("learning_protocol") != PROTOCOL
        or meta.get("reach_configuration_sha256") != c.get("reach_configuration_sha256")
    ):
        raise ValueError("Explicit matching Reach export identity required")
    exp = create_scene(cfg, c, output)
    if (
        exp.arm_assist
        or exp.virtual_upper
        or exp.balance_gain
        or hasattr(exp, "arm_follower")
    ):
        raise ValueError("Reach student cannot contain teacher controllers")
    from r2v2_deploy.tasks.contact_reach_program import (
        validate_execution,
        DemonstrationReachProgram,
        EXECUTION,
    )

    if validate_execution(cfg):
        if (
            c.get("reach_execution_protocol") != EXECUTION
            or meta.get("reach_execution_protocol") != EXECUTION
            or parity.get("reach_execution_protocol") != EXECUTION
        ):
            raise ValueError(
                "Candidate lacks explicit demonstration-clock execution identity"
            )
        DemonstrationReachProgram(exp, cfg)
    exp.keep_trace = False
    return exp


def scalar_measurements(exp):
    """CPU contact truth normalized to the shared scalar/tensor validity schema."""
    import mujoco

    row = dict(exp.current_metrics)
    d = exp.data
    m = exp.model
    q = d.qpos[exp.body_map.qpos]
    limits = m.jnt_range[exp.body_map.joints]
    row["body_joint_violation_rad"] = float(
        max(0.0, np.max(limits[:, 0] - q), np.max(q - limits[:, 1]))
    )
    row["parked_hand_contact"] = np.bool_(row["parked_hand_contact_count"] > 0)
    row["nonhand_object_contact"] = np.bool_(row["nonhand_object_contact_count"] > 0)
    row["robot_prop_penetration_m"] = max(
        [x["penetration_m"] for x in row["robot_prop_contacts"]] or [0.0]
    )
    row["robot_prop_force_N"] = 0.0
    row["nonfoot_ground_contact"] = np.bool_(False)
    for i, c in enumerate(exp.scratch.contact):
        if c.dist > 0:
            continue
        a, b = map(int, c.geom)
        if exp.floor_geom in (a, b):
            other = b if a == exp.floor_geom else a
            if other in exp.robot_geoms and other not in exp.foot_geoms:
                row["nonfoot_ground_contact"] = np.bool_(True)
        if (a in exp.robot_geoms and b in exp.table_geoms | exp.crate_geoms) or (
            b in exp.robot_geoms and a in exp.table_geoms | exp.crate_geoms
        ):
            force = np.zeros(6)
            mujoco.mj_contactForce(m, exp.scratch, i, force)
            row["robot_prop_force_N"] += float(np.linalg.norm(force[:3]))
    for key in ("finite_state", "floor_contact"):
        row[key] = np.bool_(row[key])
    return row


def local_test(
    cfg,
    candidate,
    output,
    snapshot,
    *,
    continuous,
    seed,
    episode_s=6.0,
    terminal_hold=False,
):
    from r2v2_deploy.data.delivery_state import restore_training
    from r2v2_deploy.tasks.physical_validity import (
        effective_hand_control,
        stable_table_contact,
    )
    from scipy.spatial.transform import Rotation

    if episode_s not in (6.0, 10.0):
        raise ValueError("Explicit bounded local CONTACT horizon required")
    exp = create_student(cfg, candidate, output)
    restore_training(exp, snapshot)
    tube = GoalTube(cfg["goal_manifest"], exp.program.sha256)
    if "reach_source_index" in snapshot:
        start = int(snapshot["reach_source_index"])
        if snapshot.get(
            "reach_source_data_sha256"
        ) != tube.data_sha256 or not 0 <= start < len(tube.times):
            raise ValueError("Student local test/reference clock binding differs")
    else:
        start = tube.index_for(exp.phase, float(exp.data.time - exp.phase_start))
    initial = float(exp.data.time)
    phase = exp.phase
    rng = np.random.default_rng(seed)
    narrow = phase in (
        "TOP_HOVER",
        "TOP_DESCEND",
        "PICK_CLOSE",
        "PICK_PROBE",
        "PICK_VERIFY",
    )
    jitter = rng.uniform(-1, 1, (2, 3)) * (0.001 if narrow else 0.002)
    jitter[1] *= 0.5
    rv = rng.uniform(-1, 1, (2, 3)) * math.radians(0.25 if narrow else 0.5)
    lost_since = None
    holding = 0.0
    best_hold = 0.0
    peak_foot = 0.0
    errors = []
    failure = None

    def stream():
        if cfg.get("execution_protocol"):
            from r2v2_deploy.tasks.contact_reach_program import clock_index

            index = clock_index(
                exp.data.time - initial if continuous else 0.0,
                start,
                len(tube.times),
                continuous=continuous,
                late=phase.startswith(
                    (
                        "DELIVERY_LOWER",
                        "DELIVERY_RELEASE",
                        "DELIVERY_RETREAT",
                        "DELIVERY_SETTLE",
                    )
                ),
            )
        else:
            index = start + round(
                source_clock(exp.data.time - initial, int(continuous)) / 0.02
            )
        targets, commands, source_phase = tube.sample_index(index)
        for i, side in enumerate(("left", "right")):
            q = targets[i, 3:]
            rot = Rotation.from_rotvec(rv[i]) * Rotation.from_quat(np.r_[q[1:], q[0]])
            xyzw = rot.as_quat()
            wxyz = np.r_[xyzw[3], xyzw[:3]]
            exp.policy.set_target_world(side, targets[i, :3] + jitter[i], wxyz)
            command = int(commands[i])
            controller = exp.hands.controllers[side]
            if command != controller.command and (
                cfg.get("execution_protocol")
                or command == 1
                or i == 1
                or stable_table_contact(exp.current_metrics)
            ):
                exp.hands.command(side, command)
        exp.reach_source_phase = str(source_phase)

    def safety():
        nonlocal lost_since, failure
        row = scalar_measurements(exp)
        now = float(exp.data.time)
        requires = getattr(exp, "reach_source_phase", phase) in (
            "PICK_VERIFY",
            "PICK_LIFT_OUT",
            "DELIVERY_TRANSFER",
            "DELIVERY_UPRIGHT",
            "DELIVERY_LOWER",
            "DELIVERY_RELEASE",
            "DELIVERY_RETREAT",
        )
        requires &= exp.hands.controllers["left"].command == 1
        lost = requires and not effective_hand_control(row, exp.weight_N)
        if lost and lost_since is None:
            lost_since = now
        if not lost:
            lost_since = None
        for name, bad in safety_conditions(
            row, 0.0 if lost_since is None else now - lost_since
        ):
            if bool(bad):
                failure = name
                exp.fail(name)
                break

    exp._stream = stream
    exp._gate = lambda: None
    exp._safety = safety
    started = time.monotonic()
    while not exp.done and exp.data.time - initial < episode_s - 1e-8:
        if time.monotonic() - started > 180:
            raise TimeoutError("CPU local Reach evaluation exceeded bound")
        exp.step()
        if exp.steps % 20 == 0:
            p = []
            a = []
            slow = True
            for side in ("left", "right"):
                actual, q = exp.policy.wrist_pose(exp.scratch, side)
                ref = exp.policy.references[side]
                p.append(float(np.linalg.norm(actual - ref.goal_position)))
                a.append(
                    float(2 * np.arccos(np.clip(abs(q @ ref.goal_quaternion), 0, 1)))
                )
                v, w = exp.policy.wrist_twist(exp.scratch, side)
                slow &= np.linalg.norm(v) < 0.02 and np.linalg.norm(w) < 0.15
            metrics = exp.current_metrics
            good = (
                max(p) < 0.005
                and max(a) < math.radians(3)
                and slow
                and metrics["root_speed_mps"] < 0.03
                and metrics["root_angular_speed_radps"] < 0.15
                and metrics["support_feet_count"] == 2
            )
            if terminal_hold and continuous:
                good &= exp.data.time - initial >= 8.4 - 1e-8
            holding = holding + 0.02 if good and not failure else 0.0
            best_hold = max(best_hold, holding)
            peak_foot = max(peak_foot, float(exp.current_metrics["foot_drift_m"]))
            errors.append((p, a))
    return dict(
        start_phase=phase,
        continuous=continuous,
        target_seed=seed,
        held_start=phase
        in (
            "PICK_VERIFY",
            "PICK_LIFT_OUT",
            "DELIVERY_TRANSFER",
            "DELIVERY_UPRIGHT",
            "DELIVERY_LOWER",
        ),
        success=bool(not failure and best_hold >= 0.3),
        failure=failure,
        duration_s=float(exp.data.time - initial),
        best_hold_s=best_hold,
        peak_foot_m=peak_foot,
        final_position_error_m=None if not errors else errors[-1][0],
        final_orientation_error_rad=None if not errors else errors[-1][1],
        position_rmse_m=None
        if not errors
        else np.sqrt(np.mean(np.square([e[0] for e in errors]), axis=0)).tolist(),
        orientation_rmse_rad=None
        if not errors
        else np.sqrt(np.mean(np.square([e[1] for e in errors]), axis=0)).tolist(),
        configured_horizon_s=episode_s,
        terminal_hold_required=terminal_hold,
        local_only=True,
        normal_start_full_success_claimed=False,
    )


def evaluate(cfg, candidate, output):
    cfg = {
        k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg.items()
    }
    from r2v2_deploy.configuration import prepare_run, digest
    from r2v2_deploy.evaluation.progress import contact_progress
    from r2v2_deploy.data.delivery_reset_references import load_references

    out = prepare_run(
        output,
        dict(protocol=PROTOCOL, candidate=str(candidate)),
        assets=[Path(candidate)],
    )
    result = dict(
        protocol=PROTOCOL,
        contact_only=True,
        air_ran=False,
        runtime_error=None,
        normal_standing_start=True,
        body_networks=1,
        runtime_teacher=False,
        runtime_ik=False,
    )
    try:
        exp = create_student(cfg, candidate, out / "full_assets")
        start = time.monotonic()
        while not exp.done and exp.data.time < 150:
            if time.monotonic() - start > 300:
                raise TimeoutError("Full CONTACT exceeded bound")
            exp.step()
        raw = exp.report()
        if cfg.get("execution_protocol"):
            outcome = exp.demonstration_reach_program.report()
            raw.update(
                demonstration_reach=outcome,
                evaluation_execution_protocol=cfg["execution_protocol"],
            )
            result.update(outcome)
        write_json(out / "full_contact.json", raw)
        progress = contact_progress(raw, exp.transitions)
        result.update(
            full_success=bool(
                exp.completed and exp.placement_completed and not exp.failure
            ),
            completed_phases=progress["completed_phases"],
            failure=exp.failure,
            failure_phase=exp.failure_phase,
            grasp_verified=exp.grasp_verified,
            extracted=exp.can_exit_verified,
            placement_completed=exp.placement_completed,
            released=exp.release_commanded,
            final_metrics=exp.current_metrics,
            full_report_sha256=digest(out / "full_contact.json"),
        )
        states = []
        # A denser TRAIN reset bank must not silently change the fixed tests.
        test_indexes = cfg.get(
            "evaluation_reset_reference_indexes",
            [
                cfg["reset_reference_index"],
                *cfg.get("additional_reset_reference_indexes", []),
            ],
        )
        for path in test_indexes:
            states.extend(load_references(os.path.expandvars(path), exp.program.sha256))
        results = []
        for wanted, continuous in [
            ("TOP_DESCEND", True),
            ("PICK_CLOSE", True),
            ("DELIVERY_TRANSFER", True),
            ("DELIVERY_LOWER", False),
            ("DELIVERY_RETREAT", False),
        ]:
            state = next(s for s in states if s["runtime"]["phase"] == wanted)
            results.append(
                local_test(
                    cfg,
                    candidate,
                    out / f"local_{len(results)}",
                    state,
                    continuous=continuous,
                    seed=100003 + len(results),
                )
            )
        if cfg.get("reach_evaluation_protocol") is not None:
            if cfg["reach_evaluation_protocol"] != "native_student_point_reach_v3":
                raise ValueError("Unknown point-Reach evaluation identity")
            student_states = []
            for path in cfg.get("student_evaluation_reference_indexes", []):
                student_states.extend(
                    load_references(os.path.expandvars(path), exp.program.sha256)
                )
            if not student_states:
                raise ValueError("New point tests require bound native student states")
            opening = min(
                student_states,
                key=lambda s: abs(float(s["policy"]["last_observation"][-1]) - 0.5),
            )
            opened = [
                s
                for s in student_states
                if float(s["policy"]["last_observation"][-1]) <= 0.05
            ]
            preopen = [s for s in student_states if s["hands"]["left"]["command"] == 1]
            if (
                not 0.05 < float(opening["policy"]["last_observation"][-1]) < 0.95
                or not opened
                or not preopen
            ):
                raise ValueError("Student bank lacks opening/open/pre-open coverage")
            cases = [
                ("student_opening_point", opening, False, 6.0),
                ("student_open_point", opened[0], False, 6.0),
                ("student_adjacent_reference", preopen[0], True, 10.0),
            ]
            for i, (name, state, continuous, horizon) in enumerate(cases):
                row = local_test(
                    cfg,
                    candidate,
                    out / name,
                    state,
                    continuous=continuous,
                    seed=110003 + i,
                    episode_s=horizon,
                    terminal_hold=True,
                )
                row.update(
                    case_id=name,
                    source_scope="same_nominal_student_training_family_diagnostic",
                    independent_generalization=False,
                )
                results.append(row)
            result["reach_evaluation_protocol"] = cfg["reach_evaluation_protocol"]
        result["local"] = results
        result["local_success_count"] = sum(r["success"] for r in results)
        result["fixed_scene_diagnostic_not_generalization"] = True
    except Exception as exc:
        result["runtime_error"] = repr(exc)
        raise
    finally:
        write_json(out / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    evaluate(json.loads(Path(args.config).read_text()), args.candidate, args.output)


if __name__ == "__main__":
    main()
