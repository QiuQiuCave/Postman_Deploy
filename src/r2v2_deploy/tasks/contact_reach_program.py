"""Demonstration-clock Reach; physical outcomes never gate reference commands."""

import copy
import numpy as np
from .contact_reach import GoalTube, safety_conditions

EXECUTION = "contact_reach_demonstration_clock_v2"
CRITIC = "reach_program_contact_context_v2"
FINGER_CONDITION = "continuous_reference_closure_v2"


def validate_finger_condition(cfg):
    """Explicit effective input contract; legacy command-zero semantics stay intact."""
    value = cfg.get("reach_finger_condition_protocol")
    if value is None:
        return False
    if value != FINGER_CONDITION or not validate_execution(cfg):
        raise ValueError(
            "Continuous finger reference requires matching Reach execution"
        )
    return True


def reference_closure(controller):
    """Command reference, available on hardware; NOT measured grasp feedback."""
    opened, closed = controller.poses[0], controller.poses[1]
    displacement = np.asarray(closed) - opened
    denominator = float(displacement @ displacement)
    if denominator <= 1e-12:
        raise ValueError("Distinct hand command poses required")
    return float(
        np.clip(
            (np.asarray(controller.reference.position) - opened)
            @ displacement
            / denominator,
            0.0,
            1.0,
        )
    )


def validate_execution(cfg):
    if cfg.get("execution_protocol") is None:
        return False
    if cfg["execution_protocol"] != EXECUTION or cfg.get("critic_profile") != CRITIC:
        raise ValueError(
            "Explicit matching demonstration-clock/Reach-context protocol required"
        )
    if cfg.get("late_continuous_motion_s") != 8.0 or cfg.get("late_episode_s") != 10.0:
        raise ValueError(
            "Explicit eight-second late motion/ten-second episode required"
        )
    return True


def clock_index(elapsed, start, count, *, continuous=False, late=False):
    clock = (
        min(max(float(elapsed) - 0.4, 0.0), 8.0 if late else 3.2)
        if continuous
        else float(elapsed)
    )
    return min(max(int(start) + round(clock / 0.02), 0), int(count) - 1)


class DemonstrationReachProgram:
    """Normal-start GOAL playback, never robot/object state or action playback."""

    def __init__(self, exp, cfg):
        if not validate_execution(cfg):
            raise ValueError("New execution protocol required")
        from r2v2_deploy.evaluation.contact_reach import scalar_measurements

        self.exp, self.measure = exp, scalar_measurements
        self.continuous_condition = validate_finger_condition(cfg)
        if self.continuous_condition:
            exp.policy.closure_provider = lambda: reference_closure(
                exp.hands.controllers["left"]
            )
        self.tube = GoalTube(cfg["goal_manifest"], exp.program.sha256)
        self.start, self.duration = float(exp.data.time), len(self.tube.times) * 0.02
        self.loss_since = self.settle_since = self.release_position = None
        self.table_touch = self.stable_touch = self.actual_release = self.withdrawn = (
            False
        )
        self.rows = {}
        exp._stream, exp._gate, exp._safety = self.stream, self.gate, self.safety
        exp.demonstration_reach_program = self

    def stream(self):
        from scipy.spatial.transform import Rotation

        e = self.exp
        index = clock_index(e.data.time - self.start, 0, len(self.tube.times))
        targets, commands, phase = self.tube.sample_index(index)
        phase = str(phase)
        if phase == "COMPLETE":
            phase = "DELIVERY_SETTLE"  # Source completion is not student success.
        if e.phase != phase:
            e.phase = phase
            e.transitions.append(
                dict(
                    time_s=float(e.data.time),
                    state=phase,
                    transition_source="demonstration_clock",
                )
            )
        e.phase_start = float(e.data.time - self.tube.elapsed[index])
        e.top_active = True
        for side, target, command in zip(("left", "right"), targets, commands):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(
                np.r_[target[4:], target[3]]
            ).as_matrix()
            transform[:3, 3] = target[:3]
            e.goal_wrist_transforms[side] = transform
            e.policy.set_target_world(side, target[:3], target[3:])
            if int(command) != e.hands.controllers[side].command:
                # No support, arrival, load or task-state guard.
                e.hands.command(side, int(command))
                if side == "left" and int(command) == 0:
                    e.release_commanded = True
                    e.release_snapshot = copy.deepcopy(e.current_metrics)
                    self.release_position = np.asarray(
                        e.current_metrics["T_world_wrist"]
                    )[:3, 3].copy()

    def safety(self):
        from .physical_validity import (
            effective_hand_control,
            stable_table_contact,
            robot_stable,
        )
        from r2v2_deploy.control.geometry import unsupported_load_conditions

        e, now = self.exp, float(self.exp.data.time)
        row = self.measure(e)
        requires = (
            e.phase
            in (
                "PICK_VERIFY",
                "PICK_LIFT_OUT",
                "DELIVERY_TRANSFER",
                "DELIVERY_UPRIGHT",
                "DELIVERY_LOWER",
                "DELIVERY_RELEASE",
                "DELIVERY_RETREAT",
            )
            and e.hands.controllers["left"].command == 1
        )
        lost = requires and not bool(effective_hand_control(row, e.weight_N))
        if lost and self.loss_since is None:
            self.loss_since = now
        if not lost:
            self.loss_since = None
        for name, bad in safety_conditions(
            row, 0.0 if self.loss_since is None else now - self.loss_since
        ):
            if bool(bad):
                e.fail(name)
                return
        # Observers only: none of these conditions changes goals or commands.
        if (
            requires
            and unsupported_load_conditions(row, e.weight_N)
            and not row["crate_object_contact"]
        ):
            if e.baseline_relation is None:
                e.baseline_relation = np.asarray(row["T_wrist_object"]).copy()
            e.grasp_verified = True
        if (
            e.grasp_verified
            and requires
            and effective_hand_control(row, e.weight_N)
            and row["above_highest_box_rim_m"] >= 0.01
        ):
            e.can_exit_verified = e.delivery_started = True
        placed = bool(stable_table_contact(row))
        self.table_touch |= bool(row["delivery_table_contact"])
        self.stable_touch |= bool(e.can_exit_verified and placed)
        self.actual_release |= bool(
            e.can_exit_verified
            and e.release_commanded
            and placed
            and row["active_hand_object_contact_count"] == 0
        )
        if self.release_position is not None:
            distance = np.linalg.norm(
                np.asarray(row["T_world_wrist"])[:3, 3] - self.release_position
            )
            self.withdrawn |= bool(
                self.actual_release
                and placed
                and distance >= 0.03
                and row["active_hand_object_contact_count"] == 0
            )
        settled = (
            e.phase == "DELIVERY_SETTLE"
            and self.withdrawn
            and placed
            and row["active_hand_object_contact_count"] == 0
            and robot_stable(row, None)
        )
        if settled and self.settle_since is None:
            self.settle_since = now
        if not settled:
            self.settle_since = None
        e.placement_completed = bool(settled and now - self.settle_since >= 1.0 - 1e-8)
        v = self.rows.setdefault(
            e.phase,
            dict(count=0, position_square=np.zeros(2), angle_square=np.zeros(2)),
        )
        v["count"] += 1
        for i, side in enumerate(("left", "right")):
            w = row["wrist_errors"][side]
            v["position_square"][i] += w["position_m"] ** 2
            v["angle_square"][i] += w["orientation_deg"] ** 2

    def gate(self):
        e = self.exp
        if float(e.data.time) - self.start < self.duration - 1e-8:
            return
        if not e.placement_completed:
            e.fail("Demonstration ended without stable real placement and withdrawal")
        else:
            e.completed = True
            e.phase = "COMPLETE"
            e.transitions.append(dict(time_s=float(e.data.time), state="COMPLETE"))

    def report(self):
        return dict(
            execution_protocol=EXECUTION,
            phase_count_is_timeline_only=True,
            reach_finger_condition_protocol=FINGER_CONDITION
            if self.continuous_condition
            else None,
            actual_table_touch=self.table_touch,
            actual_stable_touch=self.stable_touch,
            actual_release=self.actual_release,
            actual_withdrawal=self.withdrawn,
            commands_wait_for_contact=False,
            body_state_replay=False,
            reference_duration_s=self.duration,
            source_data_sha256=self.tube.data_sha256,
            tracking_by_phase={
                name: dict(
                    samples=v["count"],
                    position_rmse_mm=(
                        1000 * np.sqrt(v["position_square"] / v["count"])
                    ).tolist(),
                    orientation_rmse_deg=np.sqrt(
                        v["angle_square"] / v["count"]
                    ).tolist(),
                )
                for name, v in self.rows.items()
            },
        )
