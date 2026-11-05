"""Bounded Cartesian IK with posture priors and joint-target slew limits."""

from __future__ import annotations
import numpy as np
import mujoco as mujoco
from . import geometry


def sequence_plan(self, data, policy):
    if self.stage == "NATURAL_RETURN_RAISE_FOR_PICK":
        now = float(data.time)
        if (
            self.natural_return_smooth_plan_time is not None
            and abs(now - self.natural_return_smooth_plan_time) < 1e-09
        ):
            return self.plan_command.copy()
        before = self.cart_pose.copy()
        route_segment_plan(self, data, policy)
        limit = 0.003 if self.active_bottle == 2 else 0.006
        self.cart_pose[2:] = before[2:] + np.clip(
            self.cart_pose[2:] - before[2:], -limit, limit
        )
        low = self.original_upper_limits[2:, 0] + 0.02
        high = self.original_upper_limits[2:, 1] - 0.02
        self.plan_command = np.clip(
            self.cart_pose[2:]
            + data.qfrc_bias[self.upper_dofs[2:]] / self.policy.kp[14:],
            low,
            high,
        )
        if self.active_bottle == 2:
            prior = self.natural_return_last_arm_command
            self.plan_command = prior + np.clip(
                self.plan_command - prior, -limit, limit
            )
            self.cart_pose[2:] = (
                self.plan_command
                - data.qfrc_bias[self.upper_dofs[2:]] / self.policy.kp[14:]
            )
            self.natural_return_last_arm_command = self.plan_command.copy()
        self.natural_return_smooth_plan_time = now
        if self.arm_rows:
            self.arm_rows[-1]["arm_command_rad"] = self.plan_command.tolist()
            self.arm_rows[-1]["empty_raise_joint_target_slew_rad_per_20ms"] = 0.006
        return self.plan_command.copy()
    if self.stage == "NATURAL_EMPTY_LEFT_LOWER" and self.active_bottle >= 1:
        now = float(data.time)
        if (
            self.natural_empty_left_plan_time is not None
            and abs(now - self.natural_empty_left_plan_time) < 1e-09
        ):
            return self.plan_command.copy()
        command = route_segment_plan(self, data, policy).copy()
        self.natural_empty_left_qcommand += np.clip(
            self.natural_empty_left_qgoal - self.natural_empty_left_qcommand,
            -0.016,
            0.016,
        )
        low = self.original_upper_limits[2:9, 0] + 0.02
        high = self.original_upper_limits[2:9, 1] - 0.02
        command[:7] = np.clip(
            self.natural_empty_left_qcommand
            + self.bias_scale
            * data.qfrc_bias[self.upper_dofs[2:9]]
            / self.native.kp[14:21],
            low,
            high,
        )
        self.plan_command = command.copy()
        self.natural_empty_left_plan_time = now
        self.natural_empty_joint_rows.append(
            dict(
                time_s=now,
                active=self.active_bottle + 1,
                left_arm_pd_target_rad=command[:7].tolist(),
                right_arm_source="Original actual-state IK",
                fingers_source="Original independent Ruckig",
                physical_state_changed=False,
            )
        )
        if self.arm_rows:
            self.arm_rows[-1]["arm_command_rad"] = command.tolist()
            self.arm_rows[-1][
                "left_arm_source"
            ] = "Feasible empty-arm PD-target waypoint"
        return command
    if self.stage in ("NATURAL_EMPTY_LEFT_LOWER", "NATURAL_RIGHT_UPRIGHT_AT_SHELF"):
        return route_segment_plan(self, data, policy)
    return departure_plan(self, data, policy)


def departure_plan(self, data, policy):
    if self.open_back_active:
        return open_retreat_plan(self, data, policy)
    return confirmed_support_plan(self, data, policy)


def confirmed_support_plan(self, data, policy):
    if self.stage == "THIRD_MEASURED_AIR_UPRIGHT":
        return return_alignment_plan(self, data, policy)
    return place_ik_plan(self, data, policy)


def place_ik_plan(self, data, policy):
    if not (self.active_bottle == 2 and self.place_active and self.third_task_active):
        return return_alignment_plan(self, data, policy)
    now = float(data.time)
    if self.task_ik_time is not None and abs(now - self.task_ik_time) < 1e-09:
        return self.plan_command.copy()
    mj = mujoco
    solve = self.seeded_ik_step
    low = self.original_upper_limits[:, 0] + 0.02
    high = self.original_upper_limits[:, 1] - 0.02
    waist = data.qpos[self.upper_pos[:2]].copy()
    low[:2] = waist - 1e-05
    high[:2] = waist + 1e-05
    prior = data.qpos[self.upper_pos].copy()
    values = np.clip(
        prior if self.task_ik_values is None else self.task_ik_values, low, high
    )
    self.ik.qpos[:] = data.qpos
    self.ik.qvel[:] = data.qvel
    targets = [
        (policy.references[s].position.copy(), policy.references[s].quaternion.copy())
        for s in ("left", "right")
    ]
    for it in range(100):
        self.ik.qpos[self.upper_pos] = values
        mj.mj_kinematics(self.model, self.ik)
        mj.mj_comPos(self.model, self.ik)
        J = []
        error = []
        errors = []
        for side, (p, q) in zip(("left", "right"), targets):
            b = self.model.body(side + "_hand_roll_link").id
            ep = p - self.ik.xpos[b]
            er = geometry.rotation_error(q, self.ik.xquat[b])
            mj.mj_jacBody(self.model, self.ik, self.jp, self.jr, b)
            J.append(
                np.vstack(
                    (self.jp[:, self.upper_dofs], 0.2 * self.jr[:, self.upper_dofs])
                )
            )
            error.extend(np.r_[ep, 0.2 * er])
            errors.append(
                [float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er)))]
            )
        J = np.vstack(J)
        J[:, :2] = 0.0
        delta = solve(
            J,
            np.asarray(error),
            3e-05,
            prior - values,
            np.maximum(low - values, -0.08),
            np.minimum(high - values, 0.08),
        )
        values = np.clip(values + delta, low, high)
        if max((e[0] for e in errors)) < 0.0003 and max((e[1] for e in errors)) < 0.2:
            break
    self.task_ik_values = values.copy()
    if self.task_ik_pose is None:
        self.task_ik_pose = self.executed[12:].copy()
    self.task_ik_pose += np.clip(values - self.task_ik_pose, -0.016, 0.016)
    command = np.clip(
        self.task_ik_pose
        + self.bias_scale * data.qfrc_bias[self.upper_dofs] / self.native.kp[12:],
        low,
        high,
    )
    self.plan_command = command[2:].copy()
    self.task_ik_time = now
    self.task_ik_rows.append(
        dict(
            time_s=now,
            stage=self.stage,
            IK_residual=errors,
            arm_qdes_rad=self.plan_command.tolist(),
            measured_waist_only_in_private_FK=waist.tolist(),
            physical_state_changed=False,
        )
    )
    return self.plan_command.copy()


def return_alignment_plan(self, data, policy):
    if self.active_bottle != 2 or self.stage != "BOTTLE_PROGRAM":
        return grasp_stance_plan(self, data, policy)
    now = float(data.time)
    if self.plan_time is not None and abs(now - self.plan_time) < 1e-09:
        return self.plan_command.copy()
    mj = mujoco
    solve = self.seeded_ik_step
    low = self.original_upper_limits[:, 0] + 0.02
    high = self.original_upper_limits[:, 1] - 0.02
    waist = data.qpos[self.upper_pos[:2]].copy()
    low[:2] = waist - 1e-05
    high[:2] = waist + 1e-05
    preferred = data.qpos[self.upper_pos].copy()
    values = np.clip(
        preferred if self.source_cart_IK is None else self.source_cart_IK, low, high
    )
    self.ik.qpos[:] = data.qpos
    self.ik.qvel[:] = data.qvel
    targets = [
        (
            policy.references[side].position.copy(),
            policy.references[side].quaternion.copy(),
        )
        for side in ("left", "right")
    ]
    for iteration in range(100):
        self.ik.qpos[self.upper_pos] = values
        mj.mj_kinematics(self.model, self.ik)
        mj.mj_comPos(self.model, self.ik)
        blocks = []
        res = []
        errors = []
        for side, (p, q) in zip(("left", "right"), targets):
            body = self.model.body(side + "_hand_roll_link").id
            ep = p - self.ik.xpos[body]
            er = geometry.rotation_error(q, self.ik.xquat[body])
            mj.mj_jacBody(self.model, self.ik, self.jp, self.jr, body)
            blocks.append(
                np.vstack(
                    (self.jp[:, self.upper_dofs], 0.2 * self.jr[:, self.upper_dofs])
                )
            )
            res.extend(np.r_[ep, 0.2 * er])
            errors.append(
                [float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er)))]
            )
        J = np.vstack(blocks)
        J[:, :2] = 0.0
        delta = solve(
            J,
            np.asarray(res),
            3e-05,
            preferred - values,
            np.maximum(low - values, -0.08),
            np.minimum(high - values, 0.08),
        )
        values = np.clip(values + delta, low, high)
        if max((x[0] for x in errors)) < 0.0003 and max((x[1] for x in errors)) < 0.2:
            break
    self.source_cart_IK = values.copy()
    if self.upper_cmd is None:
        self.upper_pose = self.executed[12:].copy()
    self.upper_pose += np.clip(values - self.upper_pose, -0.016, 0.016)
    self.upper_cmd = np.clip(
        self.upper_pose
        + self.bias_scale * data.qfrc_bias[self.upper_dofs] / self.native.kp[12:],
        low,
        high,
    )
    self.plan_command = self.upper_cmd[2:].copy()
    self.plan_time = now
    self.source_cart_rows.append(
        dict(
            time_s=now,
            clock=self.native_clock,
            IK_residual=errors,
            arm_qdes_rad=self.plan_command.tolist(),
            measured_waist_fixed_only_in_private_FK=waist.tolist(),
            live_state_writes=False,
        )
    )
    return self.plan_command.copy()


def grasp_stance_plan(self, data, policy):
    if self.source_stance_active:
        return self.source_stance_arms.copy()
    return grasp_regularization_plan(self, data, policy)


def grasp_regularization_plan(self, data, policy):
    if (
        self.active_bottle == 1
        and self.stage == "BOTTLE_PROGRAM"
        and (self.native_clock >= 48.0)
    ):
        original = self.seeded_ik_step
        self.seeded_ik_step = (
            lambda J, error, regularization, posture, lower, upper: original(
                J, error, regularization * 0.1, posture, lower, upper
            )
        )
        try:
            return handoff_timing_plan(self, data, policy)
        finally:
            self.seeded_ik_step = original
    return handoff_timing_plan(self, data, policy)


def handoff_timing_plan(self, data, policy):
    if (
        self.active_bottle == 2
        and self.stage == "BOTTLE_PROGRAM"
        and (self.native_clock >= 48.0)
    ):
        original = self.seeded_ik_step
        self.seeded_ik_step = (
            lambda J, error, regularization, posture, lower, upper: original(
                J, error, regularization * 0.1, posture, lower, upper
            )
        )
        try:
            return open_retreat_plan(self, data, policy)
        finally:
            self.seeded_ik_step = original
    return open_retreat_plan(self, data, policy)


def open_retreat_plan(self, data, policy):
    if self.open_back_active:
        self.plan_time = float(data.time)
        self.plan_command = self.open_back_arms.copy()
        self.arm_rows.append(
            dict(
                time_s=float(data.time),
                stage=self.stage,
                arm_command_rad=self.plan_command.tolist(),
                body_source="Frozen Loco14 with actual executed arm PD14 held; both hands open",
            )
        )
        return self.plan_command.copy()
    return shelf_stance_plan(self, data, policy)


def shelf_stance_plan(self, data, policy):
    if self.stage in ["SECOND_FRONT_ROW_STANCE", "SECOND_FRONT_ROW_BRAKE"]:
        return self.row_arm_hold.copy()
    return left_approach_plan(self, data, policy)


def left_approach_plan(self, data, policy):
    if self.stage in ["THIRD_EXTRA_LEFT", "THIRD_EXTRA_LEFT_BRAKE"]:
        return self.third_extra_arms.copy()
    return right_handoff_plan(self, data, policy)


def right_handoff_plan(self, data, policy):
    if self.right_place_active and self.stage in [
        "RIGHT_SHELF_LOCO_STANCE",
        "RIGHT_SHELF_LOCO_BRAKE",
    ]:
        return self.right_loco_hold.copy()
    return carry_plan(self, data, policy)


def carry_plan(self, data, policy):
    if self.dual_pending:
        return self.dual_pending_arms.copy()
    if self.dual_mode:
        if self.stage in ["DUAL_BOTTLE_LOCO_TO_SHELF", "DUAL_BOTTLE_LOCO_BRAKE"]:
            return self.dual_arm_hold.copy()
        return route_segment_plan(self, data, policy)
    if self.shelf_regrip_active:
        old = self.regrasp_ik_step
        self.regrasp_ik_step = (
            lambda J, error, regularization, posture, lower, upper: old(
                J, error, regularization * 0.1, posture, lower, upper
            )
        )
        try:
            return repeat_plan(self, data, policy)
        finally:
            self.regrasp_ik_step = old
    return repeat_plan(self, data, policy)


def repeat_plan(self, data, policy):
    if self.empty_return or self.refine_active:
        return route_segment_plan(self, data, policy)
    if (
        self.side_route_active
        and self.stage in ["LOCO_TO_LEFT_SHORT_EDGE", "SIDE_SHELF_BRAKE"]
        and hasattr(self, "route_arm_hold")
        and (not hasattr(self, "route_world_R"))
    ):
        self.plan_time = float(data.time)
        self.plan_command = self.route_arm_hold.copy()
        return self.plan_command.copy()
    if (
        self.active_bottle > 0
        and self.stage == "BOTTLE_PROGRAM"
        and hasattr(self, "repeat_extraction_hold")
    ):
        self.plan_time = float(data.time)
        self.plan_command = self.repeat_extraction_hold.copy()
        details = []
        for side in ("left", "right"):
            (p, q) = policy.endpoint_pose(data, side)
            details.append(
                dict(
                    side=side,
                    reference_position_m=p.tolist(),
                    reference_quaternion_wxyz=q.tolist(),
                    actual_position_m=p.tolist(),
                    position_error_m=0.0,
                    orientation_error_deg=0.0,
                    reference_role="Measured-pose diagnostic only; actual arm PD targets are held from the physical clearance event",
                )
            )
        self.arm_rows.append(
            dict(
                time_s=float(data.time),
                phase=self.phase,
                stage=self.stage,
                arm_command_rad=self.plan_command.tolist(),
                arms=details,
                body_source="FrozenLoco14 plus held previously executed armPD14",
            )
        )
        return self.plan_command.copy()
    if (
        self.active_bottle > 0
        and (self.stage == "BOTTLE_PROGRAM" or self.stage.startswith("HANDOFF_"))
        and (not self.repeat_use_waist)
    ):
        original = self.seeded_ik_step
        self.seeded_ik_step = (
            lambda J, error, regularization, posture, lower, upper: original(
                J, error, regularization * 0.1, posture, lower, upper
            )
        )
        self.fixed_waist = True
        (old_times, old_positions) = (self.prior_times, self.prior_positions)
        if hasattr(self, "repeat_seed_positions"):
            self.prior_times = self.repeat_seed_times
            self.prior_positions = self.repeat_seed_positions
        if (self.native_clock >= 55.5 or self.stage.startswith("HANDOFF_")) and getattr(
            self, "upper_pose", None
        ) is not None:
            self.prior_times = np.array([self.native_clock])
            self.prior_positions = np.array([self.upper_pose.copy()])
        try:
            return shelf_pose_plan(self, data, policy)
        finally:
            self.seeded_ik_step = original
            (self.prior_times, self.prior_positions) = (old_times, old_positions)
    if (
        self.repeat_use_waist
        and self.active_bottle > 0
        and (self.stage == "BOTTLE_PROGRAM")
    ):
        if not hasattr(self, "repeat_waist_center"):
            self.repeat_waist_center = data.qpos[self.upper_pos[:2]].copy()
        self.upper_limits = self.original_upper_limits.copy()
        neighborhood = np.array([0.45, 0.18])
        self.upper_limits[:2, 0] = np.maximum(
            self.upper_limits[:2, 0], self.repeat_waist_center - neighborhood
        )
        self.upper_limits[:2, 1] = np.minimum(
            self.upper_limits[:2, 1], self.repeat_waist_center + neighborhood
        )
        self.fixed_waist = False
        return posture_ik_plan(self, data, policy)
    self.fixed_waist = True
    return shelf_pose_plan(self, data, policy)


def shelf_pose_plan(self, data, policy):
    if self.side_route_active and self.carry_mode == "cartesian":
        return route_segment_plan(self, data, policy)
    return shelf_plan(self, data, policy)


def shelf_plan(self, data, policy):
    if not self.place_active:
        return loaded_route_plan(self, data, policy)
    return route_segment_plan(self, data, policy)


def route_segment_plan(self, data, policy):
    if not self.extra_active or not self.cart_active:
        return loaded_route_plan(self, data, policy)
    now = float(data.time)
    if self.cart_time is not None and abs(now - self.cart_time) < 1e-09:
        return self.plan_command.copy()
    low = self.original_upper_limits[:, 0] + 0.02
    high = self.original_upper_limits[:, 1] - 0.02
    waist = data.qpos[self.upper_pos[:2]]
    low[:2] = waist - 1e-05
    high[:2] = waist + 1e-05
    values = np.clip(self.cart_values, low, high)
    self.ik.qpos[:] = data.qpos
    self.ik.qvel[:] = data.qvel
    targets = [
        (
            policy.references[side].position.copy(),
            policy.references[side].quaternion.copy(),
        )
        for side in ("left", "right")
    ]
    for iteration in range(60):
        self.ik.qpos[self.upper_pos] = values
        mujoco.mj_kinematics(self.model, self.ik)
        mujoco.mj_comPos(self.model, self.ik)
        blocks = []
        error = []
        errors = []
        for side, (p, q) in zip(("left", "right"), targets):
            b = self.model.body(side + "_hand_roll_link").id
            ep = p - self.ik.xpos[b]
            er = geometry.rotation_error(q, self.ik.xquat[b])
            mujoco.mj_jacBody(self.model, self.ik, self.jp, self.jr, b)
            blocks.append(
                np.vstack(
                    (self.jp[:, self.upper_dofs], 0.2 * self.jr[:, self.upper_dofs])
                )
            )
            error.extend(np.r_[ep, 0.2 * er])
            errors.append(
                [float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er)))]
            )
        J = np.vstack(blocks)
        J[:, :2] = 0.0
        delta = geometry.bounded_step(
            J,
            np.array(error),
            0.003,
            self.cart_prior - values,
            np.maximum(low - values, -0.08),
            np.minimum(high - values, 0.08),
        )
        values = np.clip(values + delta, low, high)
        if max((e[0] for e in errors)) < 0.0003 and max((e[1] for e in errors)) < 0.2:
            break
    self.cart_values = values.copy()
    self.cart_pose += np.clip(values - self.cart_pose, -0.016, 0.016)
    target = np.clip(
        self.cart_pose + data.qfrc_bias[self.upper_dofs] / self.policy.kp[12:],
        low,
        high,
    )[2:]
    f = geometry.smoothstep((now - self.carry_start) / 2.0)
    self.plan_command = (1 - f) * self.carry_initial_target + f * target
    self.cart_time = now
    details = []
    for side, (p, q) in zip(("left", "right"), targets):
        (actual, quat) = policy.endpoint_pose(data, side)
        details.append(
            dict(
                side=side,
                reference_position_m=p.tolist(),
                reference_quaternion_wxyz=q.tolist(),
                actual_position_m=actual.tolist(),
                position_error_m=float(np.linalg.norm(p - actual)),
                orientation_error_deg=float(
                    np.degrees(np.linalg.norm(geometry.rotation_error(q, quat)))
                ),
            )
        )
    self.arm_rows.append(
        dict(
            time_s=now,
            phase=self.phase,
            stage=self.stage,
            arm_command_rad=self.plan_command.tolist(),
            arms=details,
            body_raw_qdes=None,
            body_source="FrozenLoco14 plus external actual-state carry IK14",
        )
    )
    return self.plan_command.copy()


def loaded_route_plan(self, data, policy):
    if self.walk_start is None:
        return regrasp_plan(self, data, policy)
    now = float(data.time)
    if self.walk_plan_time is not None and abs(now - self.walk_plan_time) < 1e-09:
        return self.plan_command.copy()
    self.walk_plan_time = now
    self.plan_command = self.walk_arms.copy()
    details = []
    for side in ("left", "right"):
        (p, q) = policy.endpoint_pose(data, side)
        details.append(
            dict(
                side=side,
                reference_position_m=p.tolist(),
                reference_quaternion_wxyz=q.tolist(),
                actual_position_m=p.tolist(),
                position_error_m=0.0,
                orientation_error_deg=0.0,
                reference_role="Measured-pose diagnostic only; actual arm command is held joint PD target",
            )
        )
    self.arm_rows.append(
        dict(
            time_s=now,
            phase=self.phase,
            stage=self.stage,
            arm_command_rad=self.plan_command.tolist(),
            arms=details,
            body_raw_qdes=self.raw_body_command.tolist(),
            body_source="FrozenLoco14 plus held arm PD targets",
        )
    )
    return self.plan_command.copy()


def regrasp_plan(self, data, policy):
    if self.second_start is None:
        return arms_only_plan(self, data, policy)
    now = float(data.time)
    if self.second_plan_time is not None and abs(now - self.second_plan_time) < 1e-09:
        return self.plan_command.copy()
    preferred = self.second_initial_upper.copy()
    if self.stage != "LEFT_REGRASP_CLEAR_AND_RIGHT_RAISE":
        preferred = self.second_normal_seed.copy()
    waist = data.qpos[self.upper_pos[:2]].copy()
    low = self.original_upper_limits[:, 0] + 0.02
    high = self.original_upper_limits[:, 1] - 0.02
    low[:2] = waist - 1e-05
    high[:2] = waist + 1e-05
    values = np.clip(self.second_IK_values, low, high)
    self.ik.qpos[:] = data.qpos
    self.ik.qvel[:] = data.qvel
    targets = [
        (
            policy.references[side].position.copy(),
            policy.references[side].quaternion.copy(),
        )
        for side in ("left", "right")
    ]
    for iteration in range(70):
        self.ik.qpos[self.upper_pos] = values
        mujoco.mj_kinematics(self.model, self.ik)
        mujoco.mj_comPos(self.model, self.ik)
        blocks = []
        error = []
        errors = []
        for side, (p, q) in zip(("left", "right"), targets):
            b = self.model.body(side + "_hand_roll_link").id
            ep = p - self.ik.xpos[b]
            er = geometry.rotation_error(q, self.ik.xquat[b])
            mujoco.mj_jacBody(self.model, self.ik, self.jp, self.jr, b)
            blocks.append(
                np.vstack(
                    (self.jp[:, self.upper_dofs], 0.2 * self.jr[:, self.upper_dofs])
                )
            )
            error.extend(np.r_[ep, 0.2 * er])
            errors.append(
                [float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er)))]
            )
        J = np.vstack(blocks)
        J[:, :2] = 0.0
        delta = self.regrasp_ik_step(
            J,
            np.asarray(error),
            0.004,
            preferred - values,
            np.maximum(low - values, -0.08),
            np.minimum(high - values, 0.08),
        )
        values = np.clip(values + delta, low, high)
        if max((x[0] for x in errors)) < 0.0003 and max((x[1] for x in errors)) < 0.2:
            break
    self.second_IK_values = values.copy()
    self.second_upper_pose += np.clip(values - self.second_upper_pose, -0.016, 0.016)
    self.upper_cmd = np.clip(
        self.second_upper_pose + data.qfrc_bias[self.upper_dofs] / self.native.kp[12:],
        low,
        high,
    )
    self.plan_command = self.upper_cmd[2:].copy()
    self.second_plan_time = now
    details = []
    for side, (p, q) in zip(("left", "right"), targets):
        (actual, quat) = policy.endpoint_pose(data, side)
        details.append(
            dict(
                side=side,
                reference_position_m=p.tolist(),
                reference_quaternion_wxyz=q.tolist(),
                actual_position_m=actual.tolist(),
                position_error_m=float(np.linalg.norm(p - actual)),
                orientation_error_deg=float(
                    np.degrees(np.linalg.norm(geometry.rotation_error(q, quat)))
                ),
            )
        )
    self.arm_rows.append(
        dict(
            time_s=now,
            phase=self.phase,
            stage=self.stage,
            arm_command_rad=self.plan_command.tolist(),
            arms=details,
            body_raw_qdes=self.raw_body_command.tolist(),
            body_source="FrozenLoco14 plus second-transfer arm IK14",
        )
    )
    self.upper_rows.append(
        dict(
            time_s=now,
            source_posture_time_s=None,
            IK_posture_prior_positions_rad=preferred.tolist(),
            IK_solution_rad=values.tolist(),
            IK_residual=errors,
            commanded_upper_rad=self.upper_cmd.tolist(),
            live_physical_state_writes=False,
        )
    )
    return self.plan_command.copy()


def arms_only_plan(self, data, policy):
    waist = data.qpos[self.upper_pos[:2]].copy()
    self.upper_limits[:2, 0] = waist - 0.02001
    self.upper_limits[:2, 1] = waist + 0.02001
    clock = self.native_clock
    self.native_clock = max(6.7, clock)
    q = posture_ik_plan(self, data, policy)
    self.native_clock = clock
    return q


def posture_ik_plan(self, data, policy):
    now = float(data.time)
    if self.plan_time is not None and abs(now - self.plan_time) < 1e-09:
        return self.plan_command.copy()
    source = max(0.0, min(65.7, self.native_clock if self.native_active else 6.7))
    i = int(np.argmin(abs(self.prior_times - source)))
    preferred = self.prior_positions[i].copy()
    low = self.upper_limits[:, 0] + 0.02
    high = self.upper_limits[:, 1] - 0.02
    values = np.clip(preferred, low, high)
    self.ik.qpos[:] = data.qpos
    self.ik.qvel[:] = data.qvel
    targets = []
    for side in ("left", "right"):
        ref = policy.references[side]
        targets.append((ref.position.copy(), ref.quaternion.copy()))
    errors = []
    for iteration in range(60):
        self.ik.qpos[self.upper_pos] = values
        mujoco.mj_kinematics(self.model, self.ik)
        mujoco.mj_comPos(self.model, self.ik)
        blocks = []
        res = []
        errors = []
        for side, (p, q) in zip(("left", "right"), targets):
            body = self.model.body(side + "_hand_roll_link").id
            ep = p - self.ik.xpos[body]
            er = geometry.rotation_error(q, self.ik.xquat[body])
            mujoco.mj_jacBody(self.model, self.ik, self.jp, self.jr, body)
            blocks.append(
                np.vstack(
                    (self.jp[:, self.upper_dofs], 0.2 * self.jr[:, self.upper_dofs])
                )
            )
            res.extend(np.r_[ep, 0.2 * er])
            errors.append(
                (float(np.linalg.norm(ep)), float(np.degrees(np.linalg.norm(er))))
            )
        J = np.vstack(blocks)
        error = np.asarray(res)
        if getattr(self, "fixed_waist", False):
            J[:, :2] = 0.0
        delta = self.seeded_ik_step(
            J,
            error,
            0.003,
            preferred - values,
            np.maximum(low - values, -0.08),
            np.minimum(high - values, 0.08),
        )
        values = np.clip(values + delta, low, high)
        if max((x[0] for x in errors)) < 0.0003 and max((x[1] for x in errors)) < 0.2:
            break
    if self.upper_cmd is None:
        self.upper_pose = self.executed[12:].copy()
    self.upper_pose += np.clip(values - self.upper_pose, -0.016, 0.016)
    self.upper_cmd = (
        self.upper_pose
        + self.bias_scale * data.qfrc_bias[self.upper_dofs] / self.native.kp[12:]
    )
    self.upper_cmd = np.clip(self.upper_cmd, low, high)
    self.plan_command = self.upper_cmd[2:].copy()
    self.plan_time = now
    details = []
    for side, (p, q) in zip(("left", "right"), targets):
        (actual, quat) = policy.endpoint_pose(data, side)
        details.append(
            dict(
                side=side,
                reference_position_m=p.tolist(),
                reference_quaternion_wxyz=q.tolist(),
                actual_position_m=actual.tolist(),
                position_error_m=float(np.linalg.norm(p - actual)),
                orientation_error_deg=float(
                    np.degrees(np.linalg.norm(geometry.rotation_error(q, quat)))
                ),
            )
        )
    self.arm_rows.append(
        dict(
            time_s=now,
            phase=self.phase,
            stage=self.stage,
            arm_command_rad=self.plan_command.tolist(),
            arms=details,
            body_raw_qdes=(
                None
                if self.raw_body_command is None
                else self.raw_body_command.tolist()
            ),
            body_source=self.owner,
        )
    )
    self.upper_rows.append(
        dict(
            time_s=now,
            source_posture_time_s=float(self.prior_times[i]),
            IK_posture_prior_positions_rad=preferred.tolist(),
            IK_solution_rad=values.tolist(),
            IK_residual=errors,
            commanded_upper_rad=self.upper_cmd.tolist(),
            live_physical_state_writes=False,
        )
    )
    return self.plan_command.copy()
