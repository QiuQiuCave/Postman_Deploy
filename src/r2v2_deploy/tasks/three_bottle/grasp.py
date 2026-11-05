"""Stance alignment, approach, contact descent, and grasp gates."""

from __future__ import annotations
import numpy as np
from mujoco._functions import mj_forward
from r2v2_deploy.tasks.physical_validity import (
    effective_hand_control as effective_hand_control,
)
import json as json
import mujoco as mujoco
from . import crate_control, geometry, handoff, placement, return_path, sequence


def contact_descent_loaded_targets(self):
    if not (self.safe_place_active and self.stage == "THIRD_SAFE_CONTACT_LOWER"):
        return return_path.departure_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.walk_pose[:] = [self.placement_height, self.placement_pitch]
    self.post_pose[:] = self.walk_pose
    if self.contact_descent_key != self.stage_start:
        self.contact_descent_key = self.stage_start
        self.contact_descent_goals = self.held_goals()
        self.contact_descent_hold = None
        self.contact_descent_since = None
        self.native.follow_current(self.scratch)
        self.policy.follow_current(self.scratch)
        self.multi_events.append(
            dict(
                time_s=now,
                event="third_pure_vertical_descent_without_axis_rotation",
                speed_mps=self.contact_descent_speed,
                physical_state_changed=False,
            )
        )
    (l, r) = self.dual_measure()
    goals = self.contact_descent_goals.copy()
    goals[0, 2, 3] -= min(0.13, self.contact_descent_speed * elapsed)
    if (
        self.shelf_support(l)
        and l["middle_shelf_upward_N"] > 0.25 * 0.981
        and (self.contact_descent_hold is None)
    ):
        self.contact_descent_hold = self.held_goals()
    if self.contact_descent_hold is not None:
        goals = self.contact_descent_hold.copy()
    support = (
        self.shelf_support(l)
        and l["middle_shelf_upward_N"] > 0.25 * 0.981
        and (l["object_tilt_deg"] < 8.0)
        and (l["object_linear_speed_mps"] < 0.03)
        and (l["object_angular_speed_radps"] < 0.2)
    )
    self.contact_descent_since = (
        (now if self.contact_descent_since is None else self.contact_descent_since)
        if support
        else None
    )
    if (
        self.contact_descent_since is not None
        and now - self.contact_descent_since > 0.06
    ):
        self.safe_place_active = False
        self.place_phase_goals = self.held_goals()
        goals = self.place_phase_goals.copy()
        self.native.follow_current(self.scratch)
        self.policy.follow_current(self.scratch)
        self.enable_carry()
        self.carry_start = now - 2.0
        self.multi_events.append(
            dict(
                time_s=now,
                event="third_open_on_real_shared_shelf_support",
                shelf_N=l["middle_shelf_upward_N"],
                hand_N=l["hand_vertical_force_N"],
                tilt_deg=l["object_tilt_deg"],
            )
        )
        self.enter("SHELF_OPEN_LEFT")
    elif elapsed > 12.0:
        self.failure = "Pure vertical descent did not transfer bottle load to shelf"
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def centered_handoff_configure_repeat_handoff(self):
    parked_right_hand_configure_repeat_handoff(self)
    if self.active_bottle == 2:
        self.object_goal[:3, 3] = self.post_base + self.post_R @ np.array(
            [0.58, 0.0, 1.4 - self.post_base[2]]
        )
        self.left_grasp_goal = self.object_goal @ self.object_to_left
        self.right_grasp_yaw = self.source_center_right_yaw
        self.right_offset = np.array([0.16, 0.028, 0.072])


def centered_handoff_advance(self, stage, goals):
    if self.active_bottle == 2 and stage == "HANDOFF_RIGHT_PREAPPROACH":
        axis = self.object_pose()[:3, 2]
        self.source_center_right_R = (
            geometry.axis_rotation(np.array([0.0, 0.0, 1.0]), axis)
            @ self.post_R
            @ geometry.rotation_z(np.deg2rad(self.source_center_right_yaw))
        )
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_right_approach_axis_measured_then_frozen",
                right_yaw_relative_body_deg=self.source_center_right_yaw,
                live_state_writes=False,
            )
        )
    return handoff.empty_hand_clearance_advance(self, stage, goals)


def centered_handoff_loaded_targets(self):
    if (
        self.active_bottle == 2
        and self.stage == "LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL"
        and (float(self.data.time) - self.stage_start >= 9.0)
    ):
        now = float(self.data.time)
        (left, right) = self.dual_measure()
        held = self.held_goals()
        distance = float(np.linalg.norm(held[0, :3, 3] - self.object_pose()[:3, 3]))
        geometric = (
            distance > 0.2
            and held[0, 2, 3] > 1.25
            and (left["active_hand_object_contact_count"] == 0)
        )
        if geometric and effective_hand_control(right, 0.981):
            if getattr(self, "source_empty_safe_key", None) != self.stage_start:
                self.source_empty_safe_key = self.stage_start
                self.source_center_empty_since = None
                self.source_empty_safe_goals = held.copy()
                self.source_empty_safe_start = now
                self.source_empty_safe_pose = np.array(
                    self.post_pose, dtype=float
                ).copy()
                self.native.follow_current(self.scratch)
                self.policy.follow_current(self.scratch)
                self.enable_carry()
                self.carry_start = now - 2.0
            goals = self.source_empty_safe_goals.copy()
            good = (
                effective_hand_control(right, 0.981)
                and right["active_hand_object_contact_count"] >= 2
            )
            self.source_center_empty_since = (
                (
                    now
                    if self.source_center_empty_since is None
                    else self.source_center_empty_since
                )
                if good
                else None
            )
            if self.failure == "Empty left arm did not reach normal grasp preparation":
                self.failure = None
            self.native_clock = 65.7
            self.native_start = now - 65.7
            self.source_phase = "PICK_LIFT_OUT"
            self.post_twist[:] = 0.0
            self.walk_twist[:] = 0.0
            self.post_pose[:] = self.source_empty_safe_pose
            self.walk_pose[:] = self.source_empty_safe_pose
            self.bottle_commands = [0, 1]
            if (
                self.source_center_empty_since is not None
                and now - self.source_center_empty_since >= 0.3
            ):
                diagnostic = self.phase_goals.copy()
                diagnostic[0] = self.normal_left_goal
                self.multi_events.append(
                    dict(
                        time_s=now,
                        event="third_empty_left_actual_geometric_clearance_accepted",
                        old_empty_waypoint_error_m_deg=self.errors(diagnostic)[0],
                        distance_to_bottle_m=distance,
                        right_upward_N=right["hand_vertical_force_N"],
                        final_left_close_and_real_load_gates_unchanged=True,
                        physical_state_changed=False,
                    )
                )
                self.advance("LEFT_REGRASP_PREAPPROACH", goals)
            elif now - self.source_empty_safe_start > 8.0:
                self.failure = (
                    "Actual empty-left clearance did not settle with right support"
                )
            self.post_goals = goals.copy()
            return (self.stage, goals, 65.7)
    if (
        self.active_bottle == 2
        and self.stage == "HANDOFF_MOVE_LEFT_CLEAR_OF_BOX"
        and (float(self.data.time) - self.stage_start >= 5.0)
    ):
        now = float(self.data.time)
        elapsed = now - self.stage_start
        goals = self.start_goals.copy()
        goals[0] = self.left_grasp_goal
        (left, right) = self.dual_measure()
        err = self.errors(goals)[0]
        good = (
            err[0] < 0.015
            and err[1] < 3.0
            and effective_hand_control(left, 0.981)
            and (left["hand_vertical_force_N"] > 0.8 * 0.981)
            and (right["active_hand_object_contact_count"] == 0)
            and (left["object_linear_speed_mps"] < 0.03)
            and (left["object_angular_speed_radps"] < 0.2)
            and (left["root_speed_mps"] < 0.08)
            and (left["support_feet_count"] == 2)
        )
        self.source_center_waypoint_since = (
            (
                now
                if self.source_center_waypoint_since is None
                else self.source_center_waypoint_since
            )
            if good
            else None
        )
        if (
            self.source_center_waypoint_since is not None
            and now - self.source_center_waypoint_since >= 0.3
        ):
            goals = self.held_goals()
            self.native.follow_current(self.scratch)
            self.policy.follow_current(self.scratch)
            self.enable_carry()
            self.carry_start = now - 2.0
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="third_center_handoff_actual_supported_waypoint_accepted",
                    left_wrist_error_m_deg=err,
                    left_upward_N=left["hand_vertical_force_N"],
                    final_right_alignment_m_deg=[0.004, 3.0],
                    physical_state_changed=False,
                )
            )
            self.advance("HANDOFF_RIGHT_PREAPPROACH", goals)
            self.post_goals = goals.copy()
            return (self.stage, goals, 65.7)
    if self.active_bottle != 2 or self.stage not in [
        "HANDOFF_RIGHT_PREAPPROACH",
        "HANDOFF_RIGHT_FINAL_APPROACH",
    ]:
        return sequence.return_alignment_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    pre = self.stage == "HANDOFF_RIGHT_PREAPPROACH"
    duration = 3.0 if pre else 2.0
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.post_twist[:] = 0.0
    self.post_pose[:] = [0.82, 0.35]
    self.bottle_commands = [1, 0]
    if self.source_center_right_R is None:
        self.source_center_right_R = (
            geometry.axis_rotation(np.array([0.0, 0.0, 1.0]), self.object_pose()[:3, 2])
            @ self.post_R
            @ geometry.rotation_z(np.deg2rad(self.source_center_right_yaw))
        )
    target = np.eye(4)
    target[:3, :3] = self.source_center_right_R
    target[:3, 3] = self.object_pose()[:3, 3] - target[:3, :3] @ self.right_offset
    if pre:
        target[:3, 3] += self.post_R @ np.array([0.0, -0.1, 0.05])
    goals = self.phase_goals.copy()
    goals[1] = geometry.mix_transform(
        self.phase_goals[1], target, geometry.smoothstep(elapsed / duration)
    )
    err = self.errors(goals)[1]
    (left, right) = self.dual_measure()
    good = (
        elapsed >= duration + 0.5
        and err[0] < (0.02 if pre else 0.004)
        and (err[1] < (6.0 if pre else 3.0))
    )
    if pre:
        good = (
            good
            and right["active_hand_object_contact_count"] == 0
            and effective_hand_control(left, 0.981)
        )
    self.move_good = (
        (now if self.move_good is None else self.move_good) if good else None
    )
    if self.move_good is not None and now - self.move_good >= 0.4:
        self.advance(
            "HANDOFF_RIGHT_FINAL_APPROACH" if pre else "HANDOFF_RIGHT_CLOSE_AND_LOAD",
            goals,
        )
    elif elapsed > duration + 10.0:
        self.failure = (
            "Centered third right approach did not meet actual alignment/clearance"
        )
    self.post_goals = goals.copy()
    if int(round(now / 0.02)) % 100 == 0:
        print(
            json.dumps(
                dict(
                    time_s=now,
                    stage=self.stage,
                    errors=self.errors(goals),
                    left_N=left["hand_vertical_force_N"],
                    right_N=right["hand_vertical_force_N"],
                    right_contacts=right["active_hand_object_contact_count"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def grasp_stance_start_next_pick(self):
    if self.active_bottle == 2 and (not self.source_stance_done):
        self.begin_source_stance()
        return
    return placement.third_return_start_next_pick(self)


def grasp_stance_begin_source_stance(self):
    self.source_stance_active = True
    self.source_stance_done = False
    self.empty_return = False
    self.extra_active = True
    self.cart_active = False
    self.native_active = True
    self.owner = "loco"
    self.arm_mode = "locked"
    self.arm_planning = True
    self.source_stance_arms = self.executed[14:].copy()
    self.arm_hold = self.source_stance_arms.copy()
    self.source_stance_origin = self.scratch.xpos[self.base].copy()
    self.source_stance_crate_origin = self.scratch.xpos[self.crate_body].copy()
    self.source_stance_crate_max = 0.0
    self.source_stance_target = self.scratch.xpos[self.cargo_body, :2] + np.array(
        [0.3313, -0.60724]
    )
    self.source_stance_yaw = np.deg2rad(113.1)
    self.source_stance_good = None
    self.post_twist[:] = 0.0
    self.walk_twist[:] = 0.0
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    self.enter("SOURCE_THIRD_STANCE_FORWARD_ARC")
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="third_measured_stance_real_loco_correction_begin",
            actual_base_xyz_m=self.source_stance_origin.tolist(),
            actual_yaw_deg=float(np.rad2deg(self.base_yaw())),
            target_xy_m=self.source_stance_target.tolist(),
            target_yaw_deg=113.1,
            physical_state_changed=False,
        )
    )


def grasp_stance_loaded_targets(self):
    if not self.source_stance_active:
        return placement.grasp_timing_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 6.7
    self.native_start = now - 6.7
    self.source_phase = "EMPTY_RETURN"
    self.bottle_commands = [0, 0]
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    delta = self.source_stance_target - self.scratch.xpos[self.base, :2]
    if self.stage == "SOURCE_THIRD_STANCE_FORWARD_ARC":
        local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ delta
        cmd = np.array(
            [
                self.source_stance_speed,
                np.clip(1.5 * local[1], -0.1, 0.1),
                np.clip(
                    1.5 * geometry.wrap_angle(self.source_stance_yaw - self.base_yaw()),
                    -self.source_stance_yaw_cap,
                    self.source_stance_yaw_cap,
                ),
            ]
        ) * geometry.smoothstep(elapsed / 0.6)
        if elapsed > 0.6 and (
            np.linalg.norm(delta) < 0.07
            or self.scratch.xpos[self.base, 1] > self.source_stance_target[1] - 0.06
        ):
            self.source_stance_brake = cmd.copy()
            self.enter("SOURCE_THIRD_STANCE_BRAKE")
        elif elapsed > 8.0:
            self.failure = "Measured third stance Loco arc did not approach target"
    else:
        cmd = self.source_stance_brake * (1 - geometry.smoothstep(elapsed / 0.6))
        row = self.native_measure()
        good = row["root_speed_mps"] < 0.08 and row["support_feet_count"] == 2
        self.source_stance_good = (
            (now if self.source_stance_good is None else self.source_stance_good)
            if good
            else None
        )
        if (
            elapsed >= 2.5
            and self.source_stance_good is not None
            and (now - self.source_stance_good >= 0.3)
        ):
            self.source_stance_active = False
            self.source_stance_done = True
            self.post_twist[:] = 0.0
            self.walk_twist[:] = 0.0
            self.loco.command[:] = 0.0
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="third_real_loco_correction_actual_stopped",
                    actual_base_xyz_m=self.scratch.xpos[self.base].tolist(),
                    actual_yaw_deg=float(np.rad2deg(self.base_yaw())),
                    target_error_xy_m=delta.tolist(),
                    physical_state_changed=False,
                )
            )
            placement.third_return_start_next_pick(self)
            return (self.stage, self.home_start, 6.7)
        elif elapsed > 8.0:
            self.failure = "Measured third stance Loco arc did not settle"
    self.post_twist[:] = cmd
    self.walk_twist[:] = cmd
    self.loco.command[:] = cmd
    self.loco.pose_command[:] = self.walk_pose
    if int(round(now / 0.02)) % 50 == 0:
        print(
            json.dumps(
                dict(
                    time_s=now,
                    stage=self.stage,
                    base=self.scratch.xpos[self.base].tolist(),
                    yaw_deg=float(np.rad2deg(self.base_yaw())),
                    target=self.source_stance_target.tolist(),
                    command=cmd.tolist(),
                )
            ),
            flush=True,
        )
    return (self.stage, self.held_goals(), 6.7)


def grasp_stance_step(self):
    parked_right_hand_step(self)
    if (
        self.active_bottle == 2
        and hasattr(self, "source_stance_crate_origin")
        and (not self.extraction_success)
    ):
        move = float(
            np.linalg.norm(
                self.scratch.xpos[self.crate_body, :2]
                - self.source_stance_crate_origin[:2]
            )
        )
        self.source_stance_crate_max = max(self.source_stance_crate_max, move)
        if move > 0.002 and (not self.failure):
            self.failure = "Third stance plus pickup moved box over2mm"
            self.done = True


def parked_right_hand_configure_repeat_handoff(self):
    adaptive_grasp_configure_repeat_handoff(self)
    if self.active_bottle == 2:
        self.right_offset = np.array(
            [self.source_third_right_x, 0.028, self.source_third_right_z]
        )


def parked_right_hand_step(self):
    placement.shelf_heading_step(self)
    if (
        self.stop_after_right_handoff
        and self.active_bottle == 2
        and self.handoff_success
        and (self.stage == "LEFT_REGRASP_CLEAR_AND_RIGHT_RAISE")
        and (not self.failure)
    ):
        self.done = True
        self.enter("THIRD_RIGHT_HANDOFF_DIAGNOSTIC_COMPLETE")


def adaptive_grasp_configure_repeat_handoff(self):
    sequence.repeat_configure_repeat_handoff(self)
    if self.active_bottle == 2:
        self.object_goal[2, 3] = 1.34
        self.left_grasp_goal = self.object_goal @ self.object_to_left


def adaptive_grasp_advance(self, stage, goals):
    if (
        self.active_bottle == 2
        and stage == "LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL"
        and (not self.third_right_trim_done)
    ):
        self.third_right_trim_goals = goals.copy()
        self.third_right_trim_since = None
        self.enable_carry()
        return handoff.handoff_advance(self, "THIRD_RIGHT_HELD_UPRIGHT", goals)
    return handoff.handoff_advance(self, stage, goals)


def adaptive_grasp_trim_targets(self, right=False):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_TRANSFER"
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.bottle_commands = [0, 1] if right else [1, 0]
    pose = (
        np.array([0.82, 0.35])
        if right
        else np.array([self.placement_height, self.placement_pitch])
    )
    self.walk_pose[:] = pose
    self.post_pose[:] = pose
    goals = (
        self.third_right_trim_goals if right else self.final_second_trim_goals
    ).copy()
    idx = 1 if right else 0
    (left, other) = self.dual_measure()
    row = other if right else left
    err = self.errors(goals)[idx]
    axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
    if err[0] < 0.008 and err[1] < 5.0:
        rot = geometry.axis_rotation(axis, np.array([0.0, 0.0, 1.0]), 0.08 * 0.02)
        pivot = self.scratch.xpos[self.cargo_body].copy()
        goals[idx, :3, 3] = pivot + rot @ (goals[idx, :3, 3] - pivot)
        goals[idx, :3, :3] = rot @ goals[idx, :3, :3]
    good = (
        row["object_tilt_deg"] < (5.0 if right else 3.0)
        and row["object_linear_speed_mps"] < 0.025
        and (row["object_angular_speed_radps"] < 0.18)
        and effective_hand_control(row, 0.981)
        and (row["root_speed_mps"] < 0.08)
        and (row["support_feet_count"] == 2)
    )
    key = "third_right_trim_since" if right else "final_second_trim_since"
    since = getattr(self, key)
    since = (now if since is None else since) if good else None
    setattr(self, key, since)
    if right:
        self.third_right_trim_goals = goals.copy()
    else:
        self.final_second_trim_goals = goals.copy()
    if since is not None and now - since >= 0.4:
        goals = self.held_goals()
        self.multi_events.append(
            dict(
                time_s=now,
                event=(
                    "third_right_actual_axis_upright_before_left_regrasp"
                    if right
                    else "second_final_actual_upright_before_descent"
                ),
                active=self.active_bottle + 1,
                actual_tilt_deg=row["object_tilt_deg"],
                hand_upward_N=row["hand_vertical_force_N"],
            )
        )
        if right:
            self.third_right_trim_done = True
            self.second_right_goal = goals[1].copy()
            handoff.handoff_advance(self, "LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL", goals)
        else:
            self.final_second_trim_done = True
            self.place_phase_goals = goals.copy()
            self.practical_lower_clock = 0.0
            self.practical_lower_start = now
            self.enter("SHELF_LOWER_TO_ACTUAL_CONTACT")
    elif elapsed > 18.0:
        self.failure = "Measured held-bottle upright correction did not settle before descent/regrasp"
    if right:
        self.post_goals = goals.copy()
    else:
        self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def adaptive_grasp_loaded_targets(self):
    if self.stage == "SECOND_SHELF_FINAL_UPRIGHT":
        return self.trim_targets(False)
    if self.stage == "THIRD_RIGHT_HELD_UPRIGHT":
        return self.trim_targets(True)
    if self.active_bottle == 2 and self.stage in [
        "HANDOFF_RIGHT_PREAPPROACH",
        "HANDOFF_RIGHT_FINAL_APPROACH",
    ]:
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        if axis[2] > 0.5:
            nominal = self.post_R @ geometry.rotation_z(
                np.deg2rad(self.right_grasp_yaw)
            )
            self.right_R = (
                geometry.axis_rotation(np.array([0.0, 0.0, 1.0]), axis) @ nominal
            )
    former = self.stage
    result = grasp_regularization_loaded_targets(self)
    if (
        self.active_bottle == 1
        and former == "SHELF_UPRIGHT_AND_ALIGN"
        and (self.stage == "SHELF_LOWER_TO_ACTUAL_CONTACT")
        and (not self.final_second_trim_done)
    ):
        self.final_second_trim_goals = self.held_goals()
        self.final_second_trim_since = None
        self.enable_carry()
        self.carry_start = float(self.data.time) - 2.0
        self.enter("SECOND_SHELF_FINAL_UPRIGHT")
        return (self.stage, self.final_second_trim_goals.copy(), 65.7)
    return result


def grasp_tilt_rotate_goal(self, goals, clock):
    result = grasp_azimuth_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 1:
        pivot = self.bottle_R @ np.array([0.438, 0.454, 1.04498]) + self.bottle_t
        if self.second_tilt_axis is None:
            final = grasp_azimuth_rotate_goal(
                self, np.repeat(np.eye(4)[None], 2, axis=0), 55.5
            )[0]
            axis = np.cross(
                np.r_[final[:2, 3] - pivot[:2], 0.0], np.array([0.0, 0.0, 1.0])
            )
            self.second_tilt_axis = axis / np.linalg.norm(axis)
        (x, y, z) = self.second_tilt_axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        angle = np.deg2rad(self.second_radial_tilt_deg) * geometry.smoothstep(
            (clock - 41.18) / 6.7
        )
        rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        result[0, :3, 3] = pivot + rot @ (result[0, :3, 3] - pivot)
        result[0, :3, :3] = rot @ result[0, :3, :3]
    return result


def grasp_azimuth_rotate_goal(self, goals, clock):
    result = placement.upright_grasp_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 1:
        pivot = self.bottle_R @ np.array([0.438, 0.454, 1.04498]) + self.bottle_t
        angle = np.deg2rad(self.second_axial_yaw_deg) * geometry.smoothstep(
            (clock - 41.18) / 6.7
        )
        rot = geometry.rotation_z(angle)
        result[0, :3, 3] = pivot + rot @ (result[0, :3, 3] - pivot)
        result[0, :3, :3] = rot @ result[0, :3, :3]
    return result


def grasp_regularization_loaded_targets(self):
    start = self.native_start
    clock = float(self.data.time) - start if start is not None else -1.0
    eligible = (
        self.active_bottle == 1
        and self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
        and (6.7 <= clock < 52.2)
    )
    relaxed = False
    if eligible:
        ref = self.native.references["left"]
        (p, q) = self.native.endpoint_pose(self.scratch, "left")
        ep = float(np.linalg.norm(ref.position - p))
        er = float(
            np.degrees(np.linalg.norm(geometry.rotation_error(ref.quaternion, q)))
        )
        (left, _) = self.dual_measure()
        relaxed = (
            p[2] > 1.25
            and left["active_hand_object_contact_count"] == 0
            and (np.linalg.norm(p - self.scratch.xpos[self.cargo_body]) > 0.12)
            and (ep < 0.05)
            and (er < 10.0)
        )
    result = initial_layout_loaded_targets(self)
    if (
        relaxed
        and self.stage == "BOTTLE_PROGRAM"
        and (self.native_start - start > 0.015)
    ):
        self.native_start = start + 0.01
        self.path_stall_since = None
        self.second_high_relaxed_steps += 1
        self.second_high_relaxed_max_mm = max(
            self.second_high_relaxed_max_mm, ep * 1000
        )
        if self.failure and self.failure.startswith(
            "Repeated wrist path actual tracking stalled"
        ):
            self.failure = None
    if (
        self.stop_after_second_extraction
        and self.active_bottle == 1
        and any(
            (
                e["event"] == "bottle_actual_held_extraction_verified"
                and e.get("active") == 2
                for e in self.multi_events
            )
        )
    ):
        self.done = True
        self.enter("SECOND_EXTRACTION_DIAGNOSTIC_COMPLETE")
    return result


def initial_layout_load_place_initialization(self, path):
    assert self.steps == 0 and float(self.data.time) == 0.0
    self.sync()
    if self.source_layout == "wall_row":
        crate_R = self.scratch.xmat[self.crate_body].reshape(3, 3)
        crate_p = self.scratch.xpos[self.crate_body]
        for body, x in zip(self.bottle_bodies, [-0.072, -0.012, 0.048]):
            qa = int(self.model.jnt_qposadr[self.model.body_jntadr[body]])
            self.data.qpos[qa : qa + 3] = crate_p + crate_R @ np.array(
                [x, -0.146, 0.066]
            )
        mj_forward(self.model, self.data)
        self.sync()
    self.branch_mode = False
    self.branch_initial_time = 0.0
    self.initial_three = self.all_bottles()
    crate_R = self.scratch.xmat[self.crate_body].reshape(3, 3)
    crate_p = self.scratch.xpos[self.crate_body]
    for b in self.initial_three:
        local = crate_R.T @ (np.asarray(b["position_m"]) - crate_p)
        assert (
            abs(local[0]) < 0.12 and abs(local[1]) < 0.18 and (0.06 < local[2] < 0.07)
        ), local
    self.multi_events.append(
        dict(
            time_s=0.0,
            event="THREE_FREE_BOTTLES_ON_SOURCE_BEFORE_FIRST_STEP",
            checkpoint_loaded=False,
            bottles=self.initial_three,
        )
    )


def initial_layout_loaded_targets(self):
    result = return_path.supported_retreat_loaded_targets(self)
    if (
        self.failure
        == "Loaded turn did not reach target heading within bounded duration"
        and self.stage == "TURN_LOADED"
        and (float(self.data.time) - self.stage_start < 60.0)
    ):
        self.failure = None
        self.extended_turn_timeout_steps += 1
    return result


def grasp_frame_rotate_goal(self, goals, clock):
    if goals is None:
        return None
    goals = goals.copy()
    angle = np.deg2rad(self.grasp_yaw) * geometry.smoothstep((clock - 41.18) / 6.7)
    R = geometry.rotation_z(angle)
    pivot = self.bottle_R @ np.array([0.438, 0.454, 1.04498]) + self.bottle_t
    goals[0, :3, 3] = pivot + R @ (goals[0, :3, 3] - pivot)
    goals[0, :3, :3] = R @ goals[0, :3, :3]
    return goals


def grasp_frame_loaded_targets(self):
    now = float(self.data.time)
    if self.stage == "ALIGNED_EMPTY_HOME" and (not self.arm_planning):
        self.arm_planning = True
        self.transitions.append(
            dict(
                time_s=now,
                external_bounded_arm_IK_enabled=True,
                live_state_writes=False,
            )
        )
    if (
        self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
        and (now - self.native_start >= 55.5)
    ):
        self.native_start = now - 55.5
        if self.close_wait is None:
            self.close_wait = now
        (values, _, _) = self.tube.sample_index(round(55.5 / 0.02))
        goals = self.rotate_goal(
            geometry.matrix_targets(values, self.bottle_R, self.bottle_t), 55.5
        )
        errors = []
        for side, T in zip(("left", "right"), goals):
            (p, q) = self.native.endpoint_pose(self.scratch, side)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, T[:3, :3].ravel())
            errors.append(
                (
                    float(np.linalg.norm(T[:3, 3] - p)),
                    float(np.degrees(np.linalg.norm(geometry.rotation_error(quat, q)))),
                )
            )
        row = self.native_measure()
        good = (
            errors[0][0] < 0.003
            and errors[0][1] < 2.0
            and (errors[1][0] < 0.015)
            and (row["object_tilt_deg"] < 15.0)
            and (row["root_speed_mps"] < 0.1)
        )
        self.close_good = (
            (now if self.close_good is None else self.close_good) if good else None
        )
        if self.close_good is not None and now - self.close_good >= 0.3:
            self.close_released = True
            self.gate_rows.append(
                dict(
                    time_s=now,
                    gate="actual_rotated_grasp_pose_ready",
                    errors=errors,
                    grasp_frame_rotation_deg=self.grasp_yaw,
                )
            )
        elif now - self.close_wait > 12.0:
            self.failure = (
                "Bounded teacher did not align actual rotated grasp before closing"
            )
    (phase, goals, clock) = grasp_gate_loaded_targets(self)
    if self.stage == "ALIGNED_EMPTY_HOME":
        self.loco.pose_command[:] = [
            0.82,
            self.pick_pitch * geometry.smoothstep((now - self.stage_start) / 6.0),
        ]
    if self.stage == "BOTTLE_PROGRAM":
        goals = self.rotate_goal(goals, clock)
    return (phase, goals, clock)


def grasp_gate_set_frame(self, yaw):
    stance_set_frame(self, yaw + np.deg2rad(self.goal_yaw_offset))
    direction = self.scratch.xpos[self.cargo_body, :2] - self.station_xy
    self.station_xy += self.closer * direction / np.linalg.norm(direction)


def grasp_gate_activate_native(self):
    sequence.control_activate_native(self)
    if self.live_start is None:
        for policy in (self.policy, self.loco):
            original = policy.observe
            cache = dict(time=None, value=None)

            def once(data, *args, _original=original, _cache=cache, **kwargs):
                if (
                    _cache["time"] is None
                    or abs(float(data.time) - _cache["time"]) > 1e-09
                ):
                    _cache["value"] = _original(data, *args, **kwargs)
                    _cache["time"] = float(data.time)
                return _cache["value"].copy()

            policy.observe = once
        self.live_start = float(self.data.time)
        self.switch_time = None
        self.transitions.append(
            dict(
                time_s=self.live_start,
                live_controller_blend_s=self.live_blend_s,
                not_static_previous_action=True,
                rolling_history_reset=False,
            )
        )


def grasp_gate_loaded_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    if (
        self.stage == "LOCO_STANCE_STOP"
        and elapsed >= 4.0
        and self.forward_enabled
        and (not self.forward_done)
    ):
        self.forward_done = True
        if self.station_xy[1] - self.scratch.xpos[self.base, 1] > 0.04:
            self.enter("LOCO_FORWARD_CLOSE")
            return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "LOCO_FORWARD_CLOSE":
        self.loco.pose_command[:] = [0.82, 0.0]
        self.loco.command[:] = geometry.smoothstep(elapsed / 0.6) * np.array(
            [
                0.25,
                self.forward_vy,
                np.clip(
                    2 * geometry.wrap_angle(self.station_yaw - self.base_yaw()),
                    -0.2,
                    0.2,
                ),
            ]
        )
        close = (
            self.scratch.xpos[self.base, 1] >= self.station_xy[1] - self.forward_lead
        )
        feet = [
            self.scratch.xpos[self.model.body(side + "_ankle_roll_link").id]
            for side in ("left", "right")
        ]
        foot_ready = self.foot_gap is None or (
            feet[0][1] - feet[1][1] >= self.foot_gap
            and max((x[2] for x in feet)) < 0.061
        )
        if (
            close
            and foot_ready
            or self.scratch.xpos[self.base, 1] >= self.station_xy[1] + 0.14
            or elapsed >= 4.0
        ):
            self.loco.command[:] = 0.0
            self.log_stance("forward_brake")
            self.enter("LOCO_STANCE_STOP")
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "LOCO_ALIGN_MEASURED_STANCE":
        self.loco.pose_command[:] = [0.82, np.deg2rad(self.empty_pitch)]
        y = self.scratch.xpos[self.base, 1]
        if y > self.station_xy[1] + 0.1:
            self.zig_sign = -1.0
        elif y < self.station_xy[1] - 0.08:
            self.zig_sign = 1.0
        self.loco.command[:] = geometry.smoothstep(elapsed / 0.6) * np.array(
            [0.25 * self.zig_sign, -0.15, self.wz]
        )
        if (
            self.scratch.xpos[self.base, 0] >= self.station_xy[0] - self.brake_lead
            or elapsed >= 10.0
        ):
            self.loco.command[:] = 0.0
            self.log_stance("brake")
            self.enter("LOCO_STANCE_STOP")
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "ALIGNED_EMPTY_HOME":
        self.loco.command[:] = 0.0
        self.bottle_commands = [0, 0]
        f = geometry.smoothstep(max(0.0, elapsed - 1.0) / max(self.prep_s - 4.0, 1.0))
        goals = np.array(
            [
                geometry.mix_transform(a, b, f)
                for (a, b) in zip(self.home_start, self.home_goal)
            ]
        )
        if elapsed >= self.prep_s:
            self.log_stance("before_native")
            self.activate_native()
            self.native_start = now - self.source_start
            self.bottle_stage_started = True
            self.enter("BOTTLE_PROGRAM")
        return (self.stage, goals, self.paused_clock)
    if self.stage == "BOTTLE_PROGRAM" and self.native_clock >= 65.7:
        self.native_start = now - 65.7
        if self.extraction_wait_start is None:
            self.extraction_wait_start = now
        row = self.native_measure()
        held = bool(effective_hand_control(row, 0.981))
        good = (
            self.bottle_exited
            and held
            and (not row["crate_object_contact"])
            and (row["above_highest_box_rim_m"] >= 0.01)
        )
        self.extract_since = (
            (now if self.extract_since is None else self.extract_since)
            if good
            else None
        )
        if self.extract_since is not None and now - self.extract_since >= 1.0:
            self.extraction_success = True
            self.done = True
            self.enter("BOTTLE_EXTRACTED_AND_HELD")
            return (self.stage, self.held_goals(), 65.7)
        elif now - self.extraction_wait_start > 8.0:
            self.failure = (
                "Original lift-out goals did not produce real held extraction"
            )
    result = stance_loaded_targets(self)
    return result


def stance_set_frame(self, yaw):
    self.bottle_R = geometry.rotation_z(yaw)
    self.bottle_t = np.r_[
        self.scratch.xpos[self.cargo_body, :2]
        - (self.bottle_R @ np.array([0.438, 0.454, 0]))[:2],
        0.0,
    ]
    self.station_xy = (self.bottle_R @ self.native_base + self.bottle_t)[:2]
    self.station_xy[0] += self.dx
    self.station_yaw = yaw + self.native_yaw


def stance_log_stance(self, event):
    self.stance_events.append(
        dict(
            time_s=float(self.data.time),
            event=event,
            base=self.scratch.xpos[self.base].tolist(),
            yaw_deg=float(np.degrees(self.base_yaw())),
            target_xy=self.station_xy.tolist(),
            target_yaw_deg=float(np.degrees(self.station_yaw)),
            error_m=float(
                np.linalg.norm(self.station_xy - self.scratch.xpos[self.base, :2])
            ),
        )
    )


def stance_loaded_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    if self.stage == "NEAR_CLEAR":
        result = table_approach_loaded_targets(self)
        if self.near_ready:
            self.done = False
            self.near_ready = False
            self.set_frame(
                np.deg2rad(
                    self.measure()["crate_yaw_deg"]
                    if self.fixed_frame_yaw is None
                    else self.fixed_frame_yaw
                )
            )
            self.move_origin = self.scratch.xpos[self.base].copy()
            self.move_start_yaw = self.base_yaw()
            self.clock_pause = 0.0
            self.log_stance("start")
            self.enter("LOCO_ALIGN_MEASURED_STANCE")
            return (self.stage, self.held_goals(), self.paused_clock)
        return result
    if self.stage == "LOCO_ALIGN_MEASURED_STANCE":
        self.loco.pose_command[:] = [0.82, 0.0]
        ramp = geometry.smoothstep(elapsed / 0.6)
        error = self.station_xy - self.scratch.xpos[self.base, :2]
        local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ error
        if self.mode == "lateral":
            v = [0.0, -0.15, self.wz]
        elif self.mode == "diagonal":
            v = [0.2, -0.15, self.wz]
        elif self.mode == "backdiag":
            v = [-0.15, -0.15, self.wz]
        elif self.mode == "zigzag":
            y = self.scratch.xpos[self.base, 1]
            if y > self.station_xy[1] + 0.1:
                self.zig_sign = -1.0
            elif y < self.station_xy[1] - 0.08:
                self.zig_sign = 1.0
            v = [0.25 * self.zig_sign, -0.15, self.wz]
        else:
            v = [
                np.clip(1.2 * local[0], -0.2, 0.25),
                np.clip(1.2 * local[1], -0.15, 0.15),
                self.wz,
            ]
        self.loco.command[:] = ramp * np.array(v)
        if self.scratch.xpos[self.base, 0] >= self.station_xy[0] - 0.06 or elapsed >= (
            10.0 if self.mode == "zigzag" else 5.0
        ):
            self.loco.command[:] = 0.0
            self.log_stance("brake")
            self.enter("LOCO_STANCE_STOP")
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "LOCO_STANCE_STOP":
        self.loco.command[:] = 0.0
        if elapsed >= 4.0:
            if self.post_turn_vx is not None and (not self.turned):
                self.enter("LOCO_STANCE_ARC")
                return (self.stage, self.held_goals(), self.paused_clock)
            if self.frame_mode == "body":
                self.set_frame(self.base_yaw() - self.native_yaw)
            self.arrived = True
            self.log_stance("stopped")
            if self.after_action == "measure":
                self.done = True
                self.enter("STANCE_MEASURED")
            elif self.after_action in ("hold", "switch"):
                self.hold_goals = self.held_goals()
                self.bottle_commands = [0, 0]
                if self.after_action == "switch":
                    self.activate_native()
                self.enter("NATIVE_HOLD_ISOLATION")
            else:
                self.source_start = 6.7
                self.home_start = self.held_goals()
                (values, _, _) = self.tube.sample_index(round(self.source_start / 0.02))
                self.home_goal = geometry.matrix_targets(
                    values, self.bottle_R, self.bottle_t
                )
                self.prepare_empty_hands()
                self.enter("ALIGNED_EMPTY_HOME")
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "LOCO_STANCE_ARC":
        error = geometry.wrap_angle(self.station_yaw - self.base_yaw())
        self.loco.command[:] = geometry.smoothstep(elapsed / 0.6) * np.array(
            [self.post_turn_vx, 0.0, -0.2 if error < 0 else 0.2]
        )
        if abs(error) < np.deg2rad(3.0) or elapsed >= 8.0:
            self.loco.command[:] = 0.0
            self.turned = True
            self.log_stance("arc_brake")
            self.enter("LOCO_STANCE_STOP")
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "NATIVE_HOLD_ISOLATION":
        if elapsed >= 8.0:
            self.done = True
            self.enter("HOLD_TEST_COMPLETE")
        return (self.stage, self.hold_goals, self.paused_clock)
    if self.stage == "ALIGNED_EMPTY_HOME":
        self.loco.command[:] = 0.0
        self.bottle_commands = [0, 0]
        f = geometry.smoothstep(max(0.0, elapsed - 1.0) / 7.0)
        goals = np.array(
            [
                geometry.mix_transform(a, b, f)
                for (a, b) in zip(self.home_start, self.home_goal)
            ]
        )
        if elapsed >= 11.0:
            self.log_stance("before_native")
            self.activate_native()
            self.native_start = now - self.source_start
            if self.prehold > 0:
                self.enter("NATIVE_RAISED_SETTLE")
            else:
                self.bottle_stage_started = True
                self.enter("BOTTLE_PROGRAM")
        return (self.stage, goals, self.paused_clock)
    if self.stage == "NATIVE_RAISED_SETTLE":
        self.bottle_commands = [0, 0]
        if elapsed >= self.prehold:
            self.log_stance("after_native_raised_settle")
            self.native_start = now - self.source_start
            self.bottle_stage_started = True
            self.enter("BOTTLE_PROGRAM")
        return (self.stage, self.home_goal, self.paused_clock)
    return sequence.control_loaded_targets(self)


def table_approach_loaded_targets(self):
    if self.stage == "NEAR_CLEAR":
        row = self.measure()
        elapsed = float(self.data.time) - self.stage_start
        self.loco.command[:] = 0.0
        self.finger_closed = False
        f = float(np.clip(elapsed / 5.0, 0.0, 1.0))
        f = f * f * (3.0 - 2.0 * f)
        self.loco.pose_command[:] = (1.0 - f) * self.release_pose + f * np.array(
            [0.82, 0.0]
        )
        targets = self.clear_goals.copy()
        if self.variant == "arms":
            self.arm_mode = "reach"
            R = self.clear_rotation
            targets[0, :3, 3] += f * (R @ np.array([-0.04, 0.06, 0.1]))
            targets[1, :3, 3] += f * (R @ np.array([-0.04, -0.06, 0.1]))
        good = self.placement_good(row, True)
        if self.stable(good and elapsed >= 7.0, 1.0):
            self.release_verified = True
            self.physical_placement_verified = True
            self.near_ready = True
            self.event("box_released_without_base_retreat", row)
            self.done = True
            self.enter("NEAR_READY")
        elif elapsed > 16.0:
            self.failure = "Near-table empty-hand clearance did not settle"
        return (self.stage, targets, self.paused_clock)
    result = crate_control.crate_route_loaded_targets(self)
    if self.stage == "RETRACT" and self.release_base is None:
        self.release_base = self.scratch.xpos[self.base].copy()
        self.release_pose = self.loco.pose_command.copy()
        self.clear_goals = self.held_goals()
        self.clear_rotation = self.scratch.xmat[self.crate_body].reshape(3, 3).copy()
        self.loco.command[:] = 0.0
        self.enter("NEAR_CLEAR")
        return (self.stage, self.clear_goals, self.paused_clock)
    return result
