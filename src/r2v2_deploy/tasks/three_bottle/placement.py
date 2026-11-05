"""Shelf approach, lowering, support confirmation, and release."""

from __future__ import annotations
import numpy as np
from r2v2_deploy.control.hand_control import HandReference as HandReference
from r2v2_deploy.tasks.physical_validity import (
    effective_hand_control as effective_hand_control,
)
import json as json
import mujoco as mujoco
from . import geometry, grasp, handoff, return_path, sequence


def support_feedback_loaded_targets(self):
    (stage, goals, clock) = right_release_loaded_targets(self)
    if self.safe_place_active:
        nominal = goals[0].copy()
        if self.safe_feedback_stage != stage:
            self.safe_feedback_stage = stage
            self.safe_feedback_live = self.held_goals()[0].copy()
            self.safe_feedback_nominal = nominal.copy()
        self.safe_feedback_live[:3, 3] += (
            nominal[:3, 3] - self.safe_feedback_nominal[:3, 3]
        )
        self.safe_feedback_nominal = nominal.copy()
        err = self.errors(np.array([self.safe_feedback_live, goals[1]]))[0]
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        vector = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        angle = 0.0
        if np.linalg.norm(vector) > 1e-08 and err[0] < 0.008 and (err[1] < 5.0):
            angle = min(float(np.arctan2(np.linalg.norm(vector), axis[2])), 0.12 * 0.02)
            (x, y, z) = vector / np.linalg.norm(vector)
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            pivot = self.scratch.xpos[self.cargo_body].copy()
            self.safe_feedback_live[:3, 3] = pivot + rot @ (
                self.safe_feedback_live[:3, 3] - pivot
            )
            self.safe_feedback_live[:3, :3] = rot @ self.safe_feedback_live[:3, :3]
        goals[0] = self.safe_feedback_live.copy()
        self.place_goals = goals.copy()
        if int(round(float(self.data.time) / 0.02)) % 25 == 0:
            self.safe_feedback_rows.append(
                dict(
                    time_s=float(self.data.time),
                    stage=stage,
                    measured_axis=axis.tolist(),
                    correction_rad=angle,
                    actual_wrist_error=err,
                )
            )
    return (stage, goals, clock)


def right_release_station_object_frame(self):
    obj = shelf_alignment_station_object_frame(self)
    if self.active_bottle == 2 and getattr(self, "shelf_regrip_active", False):
        obj[:3, 3] -= self.safe_grip_lower * obj[:3, 2]
    return obj


def right_release_begin_safe_place(self):
    self.safe_place_initialized = True
    self.safe_place_active = True
    self.safe_place_goals = self.held_goals()
    self.safe_place_start_goals = self.safe_place_goals.copy()
    self.safe_place_object = self.object_pose().copy()
    self.safe_place_since = None
    self.safe_place_pose_start = self.walk_pose.copy()
    away = self.safe_place_goals[1, :2, 3] - self.safe_place_object[:2, 3]
    self.safe_place_right = self.safe_place_goals[1].copy()
    self.safe_place_right[:2, 3] += 0.18 * away / max(np.linalg.norm(away), 1e-09)
    self.safe_place_right[2, 3] -= 0.1
    self.placement_height = self.safe_place_height
    self.placement_pitch = 0.25 if self.safe_place_height < 0.8 else 0.12
    self.third_task_target = np.array([self.safe_place_x, 2.15])
    self.target_xy = self.third_task_target.copy()
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="third_empty_right_separate_before_placement",
            actual_object_xyz_m=self.safe_place_object[:3, 3].tolist(),
            right_wrist_target_xyz_m=self.safe_place_right[:3, 3].tolist(),
            physical_state_changed=False,
        )
    )
    self.enter("THIRD_SAFE_CLEAR_RIGHT")


def right_release_loaded_targets(self):
    if not self.safe_place_active:
        result = place_ik_loaded_targets(self)
        if (
            self.active_bottle == 2
            and self.stage == "THIRD_FRONT_ALIGNMENT"
            and (not self.safe_place_initialized)
        ):
            self.begin_safe_place()
        else:
            return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    (l, r) = self.dual_measure()
    goals = self.safe_place_goals.copy()
    if self.stage == "THIRD_SAFE_CLEAR_RIGHT":
        f = geometry.smoothstep(elapsed / 6.0)
        goals[1] = geometry.mix_transform(
            self.safe_place_start_goals[1], self.safe_place_right, f
        )
        self.walk_pose[:] = (1 - f) * self.safe_place_pose_start + f * np.array(
            [self.placement_height, self.placement_pitch]
        )
        good = (
            r["active_hand_object_contact_count"] == 0
            and effective_hand_control(l, 0.981)
            and (l["object_linear_speed_mps"] < 0.02)
            and (l["object_tilt_deg"] < 8.0)
            and (self.errors(goals)[1][0] < 0.015)
        )
        self.safe_place_since = (
            (now if self.safe_place_since is None else self.safe_place_since)
            if good and elapsed > 6.0
            else None
        )
        if self.safe_place_since is not None and now - self.safe_place_since > 0.5:
            self.safe_place_goals = self.held_goals()
            self.safe_place_object = self.object_pose().copy()
            self.safe_place_destination = self.safe_place_goals.copy()
            self.safe_place_destination[0, :3, 3] += (
                np.array([self.safe_place_x, 2.15, self.middle_height + 0.105])
                - self.safe_place_object[:3, 3]
            )
            self.safe_place_since = None
            self.enter("THIRD_SAFE_FRONT_MOVE")
        elif elapsed > 16.0:
            self.failure = (
                "Empty right hand did not clear while third remained stably held"
            )
    elif self.stage == "THIRD_SAFE_FRONT_MOVE":
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        goals[0] = geometry.mix_transform(
            self.safe_place_goals[0],
            self.safe_place_destination[0],
            geometry.smoothstep(elapsed / self.safe_place_move_s),
        )
        actual = self.scratch.xpos[self.cargo_body]
        good = (
            abs(actual[0] - self.safe_place_x) < 0.08
            and abs(actual[1] - 2.15) < 0.04
            and (actual[1] > 2.125)
            and (l["object_tilt_deg"] < 8.0)
            and (l["object_linear_speed_mps"] < 0.025)
            and effective_hand_control(l, 0.981)
        )
        self.safe_place_since = (
            (now if self.safe_place_since is None else self.safe_place_since)
            if good and elapsed > self.safe_place_move_s
            else None
        )
        if self.safe_place_since is not None and now - self.safe_place_since > 0.4:
            self.safe_place_goals = self.held_goals()
            self.safe_place_since = None
            self.enter("THIRD_SAFE_CONTACT_LOWER")
        elif elapsed > self.safe_place_move_s + 12.0:
            self.failure = (
                "Safe third translation did not reach shelf region with actual control"
            )
    else:
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        goals[0, 2, 3] -= min(0.13, elapsed * 0.008)
        support = (
            self.shelf_support(l)
            and l["middle_shelf_upward_N"] > 0.25 * 0.981
            and (l["object_tilt_deg"] < 8.0)
            and (l["object_linear_speed_mps"] < 0.03)
            and (l["object_angular_speed_radps"] < 0.2)
        )
        self.safe_place_since = (
            (now if self.safe_place_since is None else self.safe_place_since)
            if support
            else None
        )
        if self.safe_place_since is not None and now - self.safe_place_since > 0.06:
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
        elif elapsed > 20.0:
            self.failure = "Safe third lowering did not establish real shelf load"
    self.post_pose[:] = self.walk_pose
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def place_ik_loaded_targets(self):
    if self.active_bottle == 2 and self.task_ik_hold_pose and self.place_active:
        self.placement_height = 0.82
        self.placement_pitch = 0.12
    return grasp.centered_handoff_loaded_targets(self)


def workspace_safe_third_station(self):
    target = self.object_pose().copy()
    point = (
        self.scratch.xpos[self.base, :2]
        + geometry.rotation_z(self.base_yaw())[:2, :2] @ self.body_workpoint
    )
    target[:3, 3] = [*point, 1.3]
    self.second_right_goal = target @ self.right_relation
    self.body_station_world = target[:3, 3].copy()


def workspace_start_shelf_regrip(self):
    shelf_workspace_start_shelf_regrip(self)
    if self.active_bottle == 2:
        for event in reversed(self.multi_events):
            if event["event"] == "third_right_supported_workpoint_clear_of_rack_post":
                event["inherited_station_goal_xyz_m"] = event["object_goal_xyz_m"]
                event["object_goal_xyz_m"] = self.body_station_world.tolist()
                break


def workspace_switch_active(self, index):
    shelf_entry_switch_active(self, index)
    if index == 2:
        self.third_task_target = np.array([self.body_place_x, 2.15])
        self.target_xy = self.third_task_target.copy()


def shelf_workspace_start_shelf_regrip(self):
    station_heading_start_shelf_regrip(self)
    if self.active_bottle == 2:
        for event in reversed(self.multi_events):
            if event["event"] == "third_right_supported_workpoint_clear_of_rack_post":
                event["object_goal_xyz_m"] = [self.source_third_station_x, 2.05, 1.3]
                break


def grasp_timing_loaded_targets(self):
    if self.active_bottle != 2 or self.stage not in [
        "HANDOFF_MOVE_LEFT_CLEAR_OF_BOX",
        "HANDOFF_RIGHT_PREAPPROACH",
        "HANDOFF_RIGHT_FINAL_APPROACH",
    ]:
        return handoff.empty_hand_clearance_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.post_twist[:] = 0.0
    self.post_pose[:] = [0.82, 0.35]
    self.bottle_commands = [1, 0]
    if self.stage == "HANDOFF_MOVE_LEFT_CLEAR_OF_BOX":
        goals = self.start_goals.copy()
        goals[0] = geometry.mix_transform(
            self.start_goals[0],
            self.left_grasp_goal,
            geometry.smoothstep(elapsed / 4.0),
        )
        err = self.errors(goals)[0]
        good = elapsed >= 4.5 and err[0] < 0.005 and (err[1] < 3.0)
        self.move_good = (
            (now if self.move_good is None else self.move_good) if good else None
        )
        if self.move_good is not None and now - self.move_good >= 0.3:
            self.advance("HANDOFF_RIGHT_PREAPPROACH", goals)
        elif elapsed > 12.0:
            self.failure = (
                "Third left held bottle did not reach shortened safe workpoint"
            )
    else:
        pre = self.stage == "HANDOFF_RIGHT_PREAPPROACH"
        duration = 3.0 if pre else 2.0
        obj = self.object_pose()
        self.right_R = (
            geometry.axis_rotation(np.array([0.0, 0.0, 1.0]), obj[:3, 2])
            @ self.post_R
            @ geometry.rotation_z(np.deg2rad(self.right_grasp_yaw))
        )
        target = np.eye(4)
        target[:3, :3] = self.right_R
        target[:3, 3] = obj[:3, 3] - self.right_R @ self.right_offset
        if pre:
            target[:3, 3] += self.post_R @ np.array([0.0, -0.065, 0.03])
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
                (
                    "HANDOFF_RIGHT_FINAL_APPROACH"
                    if pre
                    else "HANDOFF_RIGHT_CLOSE_AND_LOAD"
                ),
                goals,
            )
        elif elapsed > duration + 10.0:
            self.failure = (
                "Shortened handoff staging did not meet actual wrist alignment"
            )
    self.post_goals = goals.copy()
    if int(round(now / 0.02)) % 100 == 0:
        (l, r) = self.dual_measure()
        print(
            json.dumps(
                dict(
                    time_s=now,
                    stage=self.stage,
                    errors=self.errors(goals),
                    left_N=l["hand_vertical_force_N"],
                    right_N=r["hand_vertical_force_N"],
                    left_contacts=l["active_hand_object_contact_count"],
                    right_contacts=r["active_hand_object_contact_count"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def station_heading_start_shelf_regrip(self):
    station_depth_start_shelf_regrip(self)
    if self.active_bottle == 2:
        for event in self.multi_events[-3:]:
            if event["event"] == "third_right_supported_workpoint_clear_of_rack_post":
                event["inherited_station_goal_xyz_m"] = event["object_goal_xyz_m"]
                event["object_goal_xyz_m"] = [-1.05, 2.05, 1.3]


def station_heading_loaded_targets(self):
    former = self.stage
    started = self.stage_start
    result = grasp_approach_loaded_targets(self)
    if (
        self.active_bottle == 2
        and self.shelf_regrip_active
        and (former == "LEFT_REGRASP_CLEAR_AND_RIGHT_RAISE")
        and (self.stage == former)
    ):
        now = float(self.data.time)
        (phase, goals, clock) = result
        (left, right) = self.dual_measure()
        err = self.errors(goals)[1]
        good = (
            now - started >= 8.0
            and err[0] < 0.012
            and (err[1] < 3.0)
            and (left["active_hand_object_contact_count"] == 0)
            and effective_hand_control(right, 0.981)
            and (right["hand_vertical_force_N"] > 0.8 * 0.981)
            and (right["object_linear_speed_mps"] < 0.03)
            and (right["object_angular_speed_radps"] < 0.2)
        )
        if good:
            if self.failure == "Right held bottle did not reach regrasp station":
                self.failure = None
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="third_supported_station_actual_waypoint_accepted",
                    actual_right_wrist_error_m_deg=err,
                    final_left_close_alignment_unchanged=True,
                )
            )
            goals = self.held_goals()
            self.advance("LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL", goals)
            return (self.stage, goals, clock)
    return result


def grasp_approach_loaded_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    if self.active_bottle == 2 and self.stage in [
        "HANDOFF_RIGHT_CLOSE_AND_LOAD",
        "HANDOFF_LEFT_OPEN",
    ]:
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.post_twist[:] = 0.0
        self.post_pose[:] = [0.82, 0.35]
        goals = self.phase_goals.copy()
        (left, right) = self.dual_measure()
        if self.stage == "HANDOFF_RIGHT_CLOSE_AND_LOAD":
            self.bottle_commands = [1, 1]
            goals[1, 2, 3] += 0.007 * geometry.smoothstep((elapsed - 3.0) / 2.0)
            good = (
                right["active_hand_object_contact_count"] >= 2
                and right["hand_vertical_force_N"] > 0.8 * 0.981
                and (right["object_linear_speed_mps"] < 0.04)
                and (right["object_angular_speed_radps"] < 0.4)
                and (
                    not any(
                        (
                            right[k]
                            for k in [
                                "crate_object_contact",
                                "table_contact",
                                "delivery_table_contact",
                                "floor_contact",
                            ]
                        )
                    )
                )
            )
            self.gradual_right_load_since = (
                (
                    now
                    if self.gradual_right_load_since is None
                    else self.gradual_right_load_since
                )
                if good
                else None
            )
            if (
                elapsed >= 5.0
                and self.gradual_right_load_since is not None
                and (now - self.gradual_right_load_since >= 1.0)
            ):
                self.multi_events.append(
                    dict(
                        time_s=now,
                        event="third_right_stronger_load_before_binary_left_release",
                        right_contacts=right["active_hand_object_contact_count"],
                        right_upward_N=right["hand_vertical_force_N"],
                    )
                )
                self.handoff_owner = "right"
                self.advance("HANDOFF_LEFT_OPEN", goals)
            elif elapsed > 16.0:
                self.failure = "Third right grasp did not sustain stronger measured load; left kept closed"
        else:
            self.bottle_commands = [0, 1]
            if elapsed > 3.5:
                if left[
                    "active_hand_object_contact_count"
                ] == 0 and effective_hand_control(right, 0.981):
                    self.advance("HANDOFF_LEFT_WITHDRAW", goals)
                elif elapsed > 8.0:
                    self.failure = (
                        "Left release did not leave actual right-only control"
                    )
        self.post_goals = goals.copy()
        if int(round(now / 0.02)) % 100 == 0:
            print(
                json.dumps(
                    dict(
                        time_s=now,
                        stage=self.stage,
                        left_contacts=left["active_hand_object_contact_count"],
                        right_contacts=right["active_hand_object_contact_count"],
                        left_N=left["hand_vertical_force_N"],
                        right_N=right["hand_vertical_force_N"],
                        bottle_tilt_deg=right["object_tilt_deg"],
                    )
                ),
                flush=True,
            )
        return (self.stage, goals, 65.7)
    return handoff.release_clearance_loaded_targets(self)


def shelf_heading_start_practical_place(self):
    self.second_yaw = self.source_second_place_yaw
    return front_shelf_start_practical_place(self)


def shelf_heading_step(self):
    return_path.supported_retreat_step(self)
    if (
        self.stop_after_second_shelf
        and 2 in self.completed_bottles
        and (not self.failure)
    ):
        self.done = True
        self.enter("SECOND_SHELF_DIAGNOSTIC_COMPLETE")


def station_depth_adjust_station_path(self):
    if self.station_path_adjusted:
        return
    self.station_path = {k: v.copy() for (k, v) in self.station_path.items()}
    radial = self.station_grasp[:2, 3]
    offset = -self.station_inward * radial / np.linalg.norm(radial)
    times = self.station_path["final_time_s"]
    path = self.station_path["final_T_object_wrist"]
    for i, t in enumerate(times):
        path[i, :2, 3] += offset * geometry.smoothstep(
            float(t) / max(float(times[-1]), 1e-09)
        )
    self.station_path_adjusted = True


def station_depth_start_shelf_regrip(self):
    station_clearance_start_shelf_regrip(self)
    if self.active_bottle == 2:
        self.adjust_station_path()


def station_clearance_start_shelf_regrip(self):
    handoff.carry_start_shelf_regrip(self)
    if self.active_bottle == 2:
        self.safe_third_station()
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_right_supported_workpoint_clear_of_rack_post",
                actual_object_xyz_m=self.scratch.xpos[self.cargo_body].tolist(),
                object_goal_xyz_m=[-1.3, 2.02, 1.3],
                physical_state_changed=False,
            )
        )


def shelf_alignment_station_object_frame(self):
    if self.active_bottle == 2 and getattr(self, "shelf_regrip_active", False):
        obj = self.object_pose().copy()
        target = obj @ self.station_grasp
        heading = np.arctan2(target[1, 0], target[0, 0])
        desired = np.deg2rad(np.clip(np.degrees(self.base_yaw()) - 60.0, 15.0, 55.0))
        obj[:3, :3] = obj[:3, :3] @ geometry.rotation_z(
            geometry.wrap_angle(desired - heading)
        )
        return obj
    return shelf_entry_station_object_frame(self)


def shelf_entry_switch_active(self, index):
    third_place_switch_active(self, index)
    if index == 2:
        self.third_task_target = np.array([-1.23, 2.15])
        self.target_xy[:] = self.third_task_target


def shelf_entry_station_object_frame(self):
    if self.active_bottle == 2 and getattr(self, "shelf_regrip_active", False):
        obj = self.object_pose().copy()
        target = obj @ self.station_grasp
        heading = np.arctan2(target[1, 0], target[0, 0])
        obj[:3, :3] = obj[:3, :3] @ geometry.rotation_z(
            geometry.wrap_angle(np.deg2rad(self.third_world_grasp_yaw) - heading)
        )
        return obj
    return handoff.carry_station_object_frame(self)


def shelf_entry_loaded_targets(self):
    old = self.native_start
    eligible = (
        self.active_bottle == 1
        and self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
        and (old is not None)
        and (6.7 <= float(self.data.time) - old < 48.0)
    )
    result = return_route_loaded_targets(self)
    if (
        eligible
        and self.stage == "BOTTLE_PROGRAM"
        and (abs(self.native_start - old + 0.004) < 1e-07)
    ):
        self.native_start = old + 0.01
    return result


def upright_grasp_rotate_goal(self, goals, clock):
    result = third_return_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 2:
        pivot = self.bottle_R @ np.array([0.438, 0.454, 1.04498]) + self.bottle_t
        if self.tilt_axis is None:
            final = third_return_rotate_goal(
                self, np.repeat(np.eye(4)[None], 2, axis=0), 55.5
            )[0]
            axis = np.cross(
                np.r_[final[:2, 3] - pivot[:2], 0.0], np.array([0.0, 0.0, 1.0])
            )
            self.tilt_axis = axis / np.linalg.norm(axis)
        (x, y, z) = self.tilt_axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        angle = np.deg2rad(self.third_tilt) * geometry.smoothstep((clock - 41.18) / 6.7)
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        result[0, :3, 3] = pivot + R @ (result[0, :3, 3] - pivot)
        result[0, :3, :3] = R @ result[0, :3, :3]
    return result


def third_return_start_next_pick(self):
    handoff.carry_start_next_pick(self)
    if self.active_bottle == 2:
        self.no_push_origin = self.scratch.xpos[self.crate_body].copy()
        self.no_push_max_m = 0.0
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_pick_no_push_reference",
                crate_position_m=self.no_push_origin.tolist(),
                guard_m=0.002,
            )
        )


def third_return_rotate_goal(self, goals, clock):
    result = handoff_timing_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 2:
        f = geometry.smoothstep((clock - 52.2) / 3.3) * (
            1 - geometry.smoothstep((clock - 60.0) / 4.0)
        )
        result[0, 2, 3] += self.third_shallow * f
    return result


def third_return_step(self):
    return_path.open_retreat_step(self)
    if (
        self.active_bottle == 2
        and self.no_push_origin is not None
        and (not self.extraction_success)
    ):
        move = float(
            np.linalg.norm(
                self.scratch.xpos[self.crate_body, :2] - self.no_push_origin[:2]
            )
        )
        self.no_push_max_m = max(self.no_push_max_m, move)
        if move > 0.002 and (not self.failure):
            self.failure = "Third pickup no-push task constraint: box moved over2mm"
            self.done = True
    if (
        self.stop_after_third
        and self.active_bottle == 2
        and self.extraction_success
        and (not self.failure)
    ):
        self.done = True
        self.enter("LOCAL_THIRD_EXTRACTION_NO_PUSH_COMPLETE")


def handoff_timing_rotate_goal(self, goals, clock):
    result = return_route_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 2 and (clock < 55.5):
        old = (
            geometry.smoothstep((clock - 8.0) / 5.0)
            * (1 - geometry.smoothstep((clock - 48.0) / 4.2))
            if clock < 52.2
            else 0.0
        )
        new = geometry.smoothstep((clock - 8.0) / 5.0) * (
            1 - geometry.smoothstep((clock - 52.2) / 3.3)
        )
        result[0, 2, 3] += self.third_clearance_raise * (new - old)
    return result


def return_route_start_return(self):
    return_path.third_approach_start_return(self)
    if self.active_bottle == 2:
        self.empty_target += self.third_stance_delta
        self.empty_brake_target += self.third_stance_delta
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_actual_loco_return_stance_optimization",
                relative_to_v133_xy_m=self.third_stance_delta.tolist(),
                target_xy_m=self.empty_target.tolist(),
                physical_state_changed=False,
            )
        )


def return_route_rotate_goal(self, goals, clock):
    result = handoff.handoff_clearance_rotate_goal(self, goals, clock)
    if result is not None and self.active_bottle == 2 and (clock < 52.2):
        f = geometry.smoothstep((clock - 8.0) / 5.0) * (
            1 - geometry.smoothstep((clock - 48.0) / 4.2)
        )
        result[0, 2, 3] += self.third_clearance_raise * f
    return result


def return_route_loaded_targets(self):
    now = float(self.data.time)
    if self.stage == "EFFICIENT_KEEP_BOTH_HANDS_LOADED":
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.bottle_commands = [1, 1]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [0.82, 0.35]
        self.post_pose[:] = self.walk_pose
        (l, r) = self.dual_measure()
        good = self.both_ready()
        self.fast_shared_since = (
            (now if self.fast_shared_since is None else self.fast_shared_since)
            if good
            else None
        )
        if self.fast_shared_since is not None and now - self.fast_shared_since >= 0.4:
            self.shortcut_count += 1
            self.shelf_regrip_active = False
            self.handoff_success = True
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="skip_redundant_regrasp_carry_existing_actual_two_hand_grip",
                    active=self.active_bottle + 1,
                    left_N=l["hand_vertical_force_N"],
                    right_N=r["hand_vertical_force_N"],
                )
            )
            self.begin_dual()
            return (self.stage, self.dual_goals, 65.7)
        if now - self.stage_start > 7.0:
            self.failure = "Existing reverse-left/normal-right grip did not share actual load before direct carry"
        return (self.stage, self.efficient_shared_goals.copy(), 65.7)
    former = self.stage
    before = self.native_start
    clock = now - before if before is not None else -1.0
    eligible = (
        self.active_bottle > 0
        and self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
        and (6.7 <= clock < 48.0)
    )
    result = return_path.open_retreat_loaded_targets(self)
    if (
        self.shortcut_enabled
        and former == "HANDOFF_RIGHT_CLOSE_AND_LOAD"
        and (self.stage == "HANDOFF_LEFT_OPEN")
    ):
        self.efficient_shared_goals = self.held_goals()
        self.fast_shared_since = None
        self.bottle_commands = [1, 1]
        self.enter("EFFICIENT_KEEP_BOTH_HANDS_LOADED")
        return (self.stage, self.efficient_shared_goals.copy(), 65.7)
    if (
        eligible
        and self.stage == "BOTTLE_PROGRAM"
        and (abs(self.native_start - before - 0.01) < 1e-07)
    ):
        self.native_start -= (self.efficient_pick_rate - 0.5) * 0.02
        self.accelerated_empty_steps += 1
    return result


def third_place_switch_active(self, index):
    front_shelf_switch_active(self, index)
    if index == 2:
        self.target_xy[:] = getattr(self, "third_task_target", [-1.45, 2.15])


def third_place_begin_walk(self):
    shelf_lowering_begin_walk(self)
    if self.active_bottle == 2 and (self.place_active or self.refine_active):
        self.start_third_functional()


def third_place_start_third_functional(self):
    self.refine_active = False
    self.place_active = True
    self.third_task_active = True
    self.third_task_goals = self.held_goals()
    self.third_task_right = np.linalg.inv(self.torso_pose()) @ self.third_task_goals[1]
    self.third_task_since = None
    self.target_xy[:] = self.third_task_target
    self.enable_carry()
    self.carry_start = float(self.data.time) - 2.0
    self.enter("THIRD_MEASURED_AIR_UPRIGHT")


def third_place_loaded_targets(self):
    if (
        self.active_bottle != 2
        or not self.third_task_active
        or self.stage
        not in [
            "THIRD_MEASURED_AIR_UPRIGHT",
            "THIRD_FRONT_ALIGNMENT",
            "THIRD_FRONT_CONTACT_LOWER",
        ]
    ):
        return staging_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    (l, r) = self.dual_measure()
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    if self.stage == "THIRD_MEASURED_AIR_UPRIGHT":
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        goals = self.third_task_goals.copy()
        err = self.errors(goals)[0]
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        vector = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(vector) > 1e-08 and err[0] < 0.008 and (err[1] < 5.0):
            angle = min(float(np.arctan2(np.linalg.norm(vector), axis[2])), 0.1 * 0.02)
            (x, y, z) = vector / np.linalg.norm(vector)
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            pivot = self.scratch.xpos[self.cargo_body].copy()
            goals[0, :3, 3] = pivot + rot @ (goals[0, :3, 3] - pivot)
            goals[0, :3, :3] = rot @ goals[0, :3, :3]
        self.third_task_goals = goals.copy()
        good = (
            l["object_tilt_deg"] < 4.0
            and l["object_linear_speed_mps"] < 0.02
            and (l["object_angular_speed_radps"] < 0.15)
            and effective_hand_control(l, 0.981)
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.third_task_since = (
            (now if self.third_task_since is None else self.third_task_since)
            if good
            else None
        )
        if self.third_task_since is not None and now - self.third_task_since > 0.4:
            shelf_begin_place(self)
            self.place_phase_goals = self.held_goals()
            self.third_task_right = (
                np.linalg.inv(self.torso_pose()) @ self.place_phase_goals[1]
            )
            self.enter("THIRD_FRONT_ALIGNMENT")
            self.third_task_since = None
            goals = self.place_phase_goals.copy()
        elif elapsed > 14.0:
            self.failure = "Third held bottle did not become actually upright"
    else:
        f = geometry.smoothstep((now - self.place_start) / 5.0)
        self.walk_pose[:] = [
            (1 - f) * 0.82 + f * self.placement_height,
            (1 - f) * 0.12 + f * self.placement_pitch,
        ]
        self.post_pose[:] = self.walk_pose
        goals = self.place_phase_goals.copy()
        if self.stage == "THIRD_FRONT_ALIGNMENT":
            obj = self.object_initial.copy()
            obj[:3, :3] = geometry.rotation_z(np.arctan2(obj[1, 0], obj[0, 0]))
            obj[:3, 3] = [*self.third_task_target, self.middle_height + 0.105]
            target = obj @ self.object_wrist
            goals[0] = geometry.mix_transform(
                self.place_phase_goals[0], target, geometry.smoothstep(elapsed / 5.0)
            )
            actual = self.scratch.xpos[self.cargo_body]
            good = (
                abs(actual[0] - self.third_task_target[0]) < 0.08
                and abs(actual[1] - self.third_task_target[1]) < 0.04
                and (actual[1] > 2.125)
                and (l["object_tilt_deg"] < 10.0)
                and (l["object_linear_speed_mps"] < 0.05)
                and (l["object_angular_speed_radps"] < 0.25)
                and effective_hand_control(l, 0.981)
            )
            if elapsed > 5.5 and good:
                self.place_phase_goals = self.held_goals()
                goals = self.place_phase_goals.copy()
                self.third_task_lower_start = now
                self.enter("THIRD_FRONT_CONTACT_LOWER")
            elif elapsed > 16.0:
                self.failure = "Third did not enter actual front-row placement region"
        else:
            goals[0, 2, 3] -= min(0.13, elapsed * 0.012)
            support = (
                self.shelf_support(l)
                and l["middle_shelf_upward_N"] > 0.25 * 0.981
                and (l["object_tilt_deg"] < 8.0)
                and (l["object_linear_speed_mps"] < 0.03)
                and (l["object_angular_speed_radps"] < 0.2)
            )
            self.third_task_since = (
                (now if self.third_task_since is None else self.third_task_since)
                if support
                else None
            )
            if self.third_task_since is not None and now - self.third_task_since > 0.06:
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
                self.failure = (
                    "Third front-row lowering did not establish real shelf support"
                )
    goals[1] = self.torso_pose() @ self.third_task_right
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def staging_station_grasp_targets(self):
    phase0 = self.stage
    start0 = self.stage_start
    result = handoff.carry_station_grasp_targets(self)
    if (
        self.active_bottle == 2
        and phase0 in ["LEFT_REGRASP_PREAPPROACH", "LEFT_REGRASP_FINAL_APPROACH"]
        and (self.stage == phase0)
    ):
        now = float(self.data.time)
        elapsed = now - start0
        pre = phase0 == "LEFT_REGRASP_PREAPPROACH"
        key = (self.active_bottle, phase0, start0)
        if self.soft_staging_key != key:
            self.soft_staging_key = key
            self.soft_staging_since = None
        (phase, goals, clock) = result
        error = self.errors(goals)[0]
        (left, right) = self.dual_measure()
        times = self.station_path[("pre" if pre else "final") + "_time_s"]
        duration = (2.0 if pre else 0.0) + times[-1] / 0.7
        good = (
            elapsed > duration + 0.4
            and error[0] < (0.012 if pre else 0.003)
            and (error[1] < (6.0 if pre else 2.0))
            and effective_hand_control(right, 0.981)
            and (right["hand_vertical_force_N"] > 0.8 * 0.981)
        )
        self.soft_staging_since = (
            (now if self.soft_staging_since is None else self.soft_staging_since)
            if good
            else None
        )
        if (
            good
            and self.failure == "Object-relative known-good wrist path did not align"
        ):
            self.failure = None
        if self.soft_staging_since is not None and now - self.soft_staging_since >= 0.4:
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="actual_auxiliary_shelf_grasp_waypoint_accepted",
                    active=self.active_bottle + 1,
                    preapproach=pre,
                    error=error,
                    right_support_N=right["hand_vertical_force_N"],
                    left_load_release_gate_unchanged=True,
                )
            )
            self.advance(
                "LEFT_REGRASP_FINAL_APPROACH" if pre else "LEFT_REGRASP_CLOSE_AND_LOAD",
                goals,
            )
            return (self.stage, goals, clock)
    return result


def staging_loaded_targets(self):
    if (
        self.active_bottle == 2
        and self.place_active
        and (self.stage == "SHELF_WITHDRAW_LEFT")
    ):
        now = float(self.data.time)
        key = (self.active_bottle, self.stage_start)
        if self.third_clear_key != key:
            self.third_clear_key = key
            self.third_clear_start = now
            self.third_clear_goals = self.held_goals()
            self.third_clear_origin = self.third_clear_goals[0, :3, 3].copy()
            self.third_clear_since = None
            self.enable_carry()
            self.carry_start = now - 2.0
        elapsed = now - self.third_clear_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_RETREAT"
        self.bottle_commands = [0, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        self.post_pose[:] = self.walk_pose
        goals = self.third_clear_goals.copy()
        goals[0, :3, 3] += geometry.smoothstep(elapsed / 4.0) * np.array(
            [-0.035, -0.1, 0.025]
        )
        (l, r) = self.dual_measure()
        delta = self.held_goals()[0, :3, 3] - self.third_clear_origin
        good = (
            self.shelf_stable(l)
            and l["active_hand_object_contact_count"] == 0
            and (r["active_hand_object_contact_count"] == 0)
            and (np.linalg.norm(delta) > 0.04)
            and (delta[1] < -0.03)
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.third_clear_since = (
            (now if self.third_clear_since is None else self.third_clear_since)
            if good
            else None
        )
        if (
            elapsed >= 4.5
            and self.third_clear_since is not None
            and (now - self.third_clear_since >= 0.5)
        ):
            self.place_phase_goals = self.held_goals()
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="third_empty_hand_actual_clearance_verified",
                    actual_wrist_delta_xyz_m=delta.tolist(),
                    wrist_error_diagnostic=self.errors(goals),
                    hands_clear=True,
                    shelf_N=l["middle_shelf_upward_N"],
                )
            )
            self.enter("SHELF_SETTLE")
        elif elapsed > 10.0:
            self.failure = "Third actual empty-hand clearance did not occur"
        self.place_goals = goals.copy()
        return (self.stage, goals, 65.7)
    result = sequence.bottle_sequence_loaded_targets(self)
    if self.active_bottle == 2 and self.place_active:
        key = (self.active_bottle, self.place_start)
        if self.third_right_park_key != key:
            self.third_right_park_key = key
            self.third_right_park_relative = (
                np.linalg.inv(self.torso_pose()) @ self.held_goals()[1]
            )
        (phase, goals, clock) = result
        goals = goals.copy()
        goals[1] = self.torso_pose() @ self.third_right_park_relative
        self.place_goals = goals.copy()
        return (phase, goals, clock)
    return result


def front_shelf_switch_active(self, index):
    shelf_stance_switch_active(self, index)
    self.target_xy[:] = [[-0.49, 2.17], [-0.9, 2.15], [-1.25, 2.15]][index]


def front_shelf_torso_pose(self):
    b = self.model.body("waist_pitch_link").id
    t = np.eye(4)
    t[:3, :3] = self.scratch.xmat[b].reshape(3, 3)
    t[:3, 3] = self.scratch.xpos[b]
    return t


def front_shelf_start_practical_place(self):
    self.right_park_relative_place = (
        np.linalg.inv(self.torso_pose()) @ self.held_goals()[1]
    )
    self.second_x = -0.9
    self.near_y = 2.15
    shelf_stance_start_practical_place(self)


def front_shelf_loaded_targets(self):
    result = shelf_stance_loaded_targets(self)
    if (
        self.active_bottle == 1
        and self.practical_active
        and (self.right_park_relative_place is not None)
    ):
        (phase, goals, clock) = result
        goals = goals.copy()
        goals[1] = self.torso_pose() @ self.right_park_relative_place
        self.place_goals = goals.copy()
        return (phase, goals, clock)
    return result


def shelf_stance_switch_active(self, index):
    shelf_lowering_switch_active(self, index)
    self.target_xy[:] = [-0.49 - 0.25 * index, 2.17]


def shelf_stance_start_practical_place(self):
    shelf_lowering_start_practical_place(self)
    if not getattr(self, "axis_trim_done", False):
        self.axis_trim_active = True
        self.axis_trim_goals = self.held_goals()
        self.axis_trim_since = None
        self.enter("SECOND_MEASURED_AIR_UPRIGHT")


def shelf_stance_dual_targets(self):
    if self.stage in ["SECOND_FRONT_ROW_STANCE", "SECOND_FRONT_ROW_BRAKE"]:
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.bottle_commands = [1, 1]
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        if self.stage == "SECOND_FRONT_ROW_STANCE":
            world = np.array(
                [
                    -0.18,
                    np.clip(
                        1.8 * (self.second_stand_y - self.scratch.xpos[self.base, 1]),
                        -0.12,
                        0.12,
                    ),
                ]
            ) * geometry.smoothstep(elapsed / 0.8)
            cmd = np.r_[
                geometry.rotation_z(self.base_yaw())[:2, :2].T @ world,
                np.clip(geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12),
            ]
            if self.scratch.xpos[self.base, 0] <= self.second_stand_x + 0.025:
                self.row_brake = cmd.copy()
                self.enter("SECOND_FRONT_ROW_BRAKE")
            elif elapsed > 10.0:
                self.failure = "Second front-row held stance did not advance"
        else:
            cmd = self.row_brake * (1 - geometry.smoothstep(elapsed / 0.8))
            (l, r) = self.dual_measure()
            good = (
                (effective_hand_control(l, 0.981) or effective_hand_control(r, 0.981))
                and l["root_speed_mps"] < 0.08
                and (l["support_feet_count"] == 2)
            )
            self.row_stopped = (
                (now if self.row_stopped is None else self.row_stopped)
                if good
                else None
            )
            if (
                elapsed > 2.5
                and self.row_stopped is not None
                and (now - self.row_stopped > 0.3)
            ):
                self.enable_carry()
                self.dual_station_goals = self.held_goals()
                self.second_stance_done = True
                self.multi_events.append(
                    dict(
                        time_s=now,
                        event="second_front_row_real_loco_stance_complete",
                        actual_delta_xyz=(
                            self.scratch.xpos[self.base] - self.row_origin
                        ).tolist(),
                        actual_base=self.scratch.xpos[self.base].tolist(),
                    )
                )
                self.enter("DUAL_STATION_OPEN_LEFT_FOR_REGRIP")
            elif elapsed > 8.0:
                self.failure = "Second front-row held stance did not settle"
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        return (self.stage, self.held_goals(), 65.7)
    result = left_approach_dual_targets(self)
    if (
        self.active_bottle == 1
        and self.stage == "DUAL_STATION_OPEN_LEFT_FOR_REGRIP"
        and (not self.second_stance_done)
    ):
        self.row_origin = self.scratch.xpos[self.base].copy()
        self.row_arm_hold = self.executed[14:].copy()
        self.row_stopped = None
        self.bottle_commands = [1, 1]
        self.enter("SECOND_FRONT_ROW_STANCE")
        return (self.stage, self.held_goals(), 65.7)
    return result


def shelf_stance_loaded_targets(self):
    if self.active_bottle == 1 and getattr(self, "axis_trim_active", False):
        now = float(self.data.time)
        elapsed = now - self.stage_start
        (l, r) = self.dual_measure()
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_TRANSFER"
        self.bottle_commands = [1, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        goals = self.axis_trim_goals.copy()
        err = self.errors(goals)[0]
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        vector = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(vector) > 1e-08 and err[0] < 0.008 and (err[1] < 5.0):
            angle = min(float(np.arctan2(np.linalg.norm(vector), axis[2])), 0.1 * 0.02)
            (x, y, z) = vector / np.linalg.norm(vector)
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            pivot = self.scratch.xpos[self.cargo_body].copy()
            goals[0, :3, 3] = pivot + rot @ (goals[0, :3, 3] - pivot)
            goals[0, :3, :3] = rot @ goals[0, :3, :3]
        self.axis_trim_goals = goals.copy()
        good = (
            l["object_tilt_deg"] < 4.0
            and l["object_linear_speed_mps"] < 0.02
            and (l["object_angular_speed_radps"] < 0.15)
            and effective_hand_control(l, 0.981)
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.axis_trim_since = (
            (now if self.axis_trim_since is None else self.axis_trim_since)
            if good
            else None
        )
        if self.axis_trim_since is not None and now - self.axis_trim_since > 0.4:
            self.axis_trim_done = True
            self.axis_trim_active = False
            shelf_lowering_start_practical_place(self)
            return (self.stage, self.place_goals, 65.7)
        elif elapsed > 14.0:
            self.failure = "Second front-row held bottle did not become upright"
        return (self.stage, goals, 65.7)
    return shelf_lowering_loaded_targets(self)


def shelf_lowering_switch_active(self, index):
    third_place_clearance_switch_active(self, index)
    if index == 1:
        self.target_xy[:] = [
            getattr(self, "second_x", -0.65),
            getattr(self, "near_y", 2.17),
        ]
        self.object_yaw_offset = getattr(self, "second_yaw", 75.0)


def shelf_lowering_start_practical_place(self):
    self.refine_active = False
    self.place_active = False
    self.target_xy[:] = [self.second_x, self.near_y]
    self.object_yaw_offset = self.second_yaw
    shelf_begin_place(self)
    self.practical_active = True
    self.practical_contact_since = None
    self.practical_lower_clock = 0.0
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="second_near_front_contact_placement_begin",
            target_xy=self.target_xy.tolist(),
            actual_base=self.scratch.xpos[self.base].tolist(),
            wrist_precision_not_completion_gate=True,
        )
    )


def shelf_lowering_begin_walk(self):
    handoff.carry_begin_walk(self)
    if self.active_bottle == 1 and (self.place_active or self.refine_active):
        self.start_practical_place()


def shelf_lowering_loaded_targets(self):
    if (
        self.active_bottle != 1
        or not self.practical_active
        or self.stage
        not in [
            "SHELF_UPRIGHT_AND_ALIGN",
            "SHELF_LOWER_TO_ACTUAL_CONTACT",
            "SHELF_LOAD_HOLD",
        ]
    ):
        return first_release_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    (left, right) = self.dual_measure()
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    f = geometry.smoothstep((now - self.place_start) / 2.0)
    self.walk_pose[:] = [
        (1 - f) * 0.82 + f * self.placement_height,
        (1 - f) * self.route_pitch + f * self.placement_pitch,
    ]
    self.post_pose[:] = self.walk_pose
    goals = self.place_phase_goals.copy()
    support = (
        self.shelf_support(left)
        and left["middle_shelf_upward_N"] > 0.25 * 0.981
        and (left["object_tilt_deg"] < 8.0)
        and (left["object_linear_speed_mps"] < 0.03)
        and (left["object_angular_speed_radps"] < 0.2)
    )
    self.practical_contact_since = (
        (now if self.practical_contact_since is None else self.practical_contact_since)
        if support
        else None
    )
    if (
        self.practical_contact_since is not None
        and now - self.practical_contact_since >= 0.06
    ):
        goals = self.held_goals()
        self.place_phase_goals = goals.copy()
        self.native.follow_current(self.scratch)
        self.policy.follow_current(self.scratch)
        self.enable_carry()
        self.carry_start = now - 2.0
        self.multi_events.append(
            dict(
                time_s=now,
                event="second_open_on_real_shared_shelf_support",
                shelf_N=left["middle_shelf_upward_N"],
                hand_N=left["hand_vertical_force_N"],
                tilt_deg=left["object_tilt_deg"],
                wrist_error_diagnostic=self.errors(goals),
            )
        )
        self.enter("SHELF_OPEN_LEFT")
        return (self.stage, goals, 65.7)
    if self.stage == "SHELF_UPRIGHT_AND_ALIGN":
        obj = self.object_initial.copy()
        yaw = np.arctan2(obj[1, 0], obj[0, 0])
        obj[:3, :3] = geometry.rotation_z(yaw + np.deg2rad(self.second_yaw))
        obj[:3, 3] = [*self.target_xy, self.middle_height + 0.1]
        target = obj @ self.object_wrist
        goals[0] = geometry.mix_transform(
            self.place_phase_goals[0], target, geometry.smoothstep(elapsed / 3.0)
        )
        actual = self.scratch.xpos[self.cargo_body]
        good = (
            np.linalg.norm(actual[:2] - self.target_xy) < 0.055
            and left["object_tilt_deg"] < 12
            and (left["object_linear_speed_mps"] < 0.05)
            and (left["object_angular_speed_radps"] < 0.25)
            and effective_hand_control(left, 0.981)
        )
        if elapsed > 3.5 and good:
            self.place_phase_goals = self.held_goals()
            goals = self.place_phase_goals.copy()
            self.practical_lower_start = now
            self.enter("SHELF_LOWER_TO_ACTUAL_CONTACT")
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="second_lower_from_actual_reachable_pose",
                    actual_bottle=actual.tolist(),
                    wrist_error_diagnostic=self.errors(np.array([target, goals[1]])),
                )
            )
        elif elapsed > 12.0:
            self.failure = (
                "Near-front bottle did not enter actual safe placement region"
            )
    else:
        self.practical_lower_clock += 0.02
        goals[0, 2, 3] -= min(0.13, self.practical_lower_clock * 0.012)
        if now - self.practical_lower_start > 12.0:
            self.failure = "Near-front descent did not establish real shelf support; fingers kept closed"
    self.place_goals = goals.copy()
    if int(round((now - self.place_start) / 0.02)) % 50 == 0:
        print(
            json.dumps(
                dict(
                    practical_t=now - self.place_start,
                    stage=self.stage,
                    bottle=self.scratch.xpos[self.cargo_body].tolist(),
                    tilt=left["object_tilt_deg"],
                    shelf_N=left["middle_shelf_upward_N"],
                    hand_N=left["hand_vertical_force_N"],
                    wrist_error=self.errors(goals),
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def shelf_lowering_start_return(self):
    self.practical_active = False
    return sequence.repeat_start_return(self)


def third_place_clearance_switch_active(self, index):
    second_depth_switch_active(self, index)
    if index == 2:
        self.object_yaw_offset = getattr(self, "third_place_yaw", 75.0)


def third_place_clearance_begin_place(self):
    if self.active_bottle == 2:
        self.object_yaw_offset = getattr(self, "third_place_yaw", 75.0)
    return sequence.repeat_begin_place(self)


def left_approach_dual_targets(self):
    if self.stage in ["THIRD_EXTRA_LEFT", "THIRD_EXTRA_LEFT_BRAKE"]:
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.bottle_commands = [1, 1]
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        if self.stage == "THIRD_EXTRA_LEFT":
            world = np.array(
                [
                    -0.2,
                    np.clip(
                        1.5
                        * (
                            self.third_extra_origin[1] - self.scratch.xpos[self.base, 1]
                        ),
                        -0.1,
                        0.1,
                    ),
                ]
            ) * geometry.smoothstep(elapsed / 0.8)
            cmd = np.r_[
                geometry.rotation_z(self.base_yaw())[:2, :2].T @ world,
                np.clip(
                    1.2 * geometry.wrap_angle(self.third_extra_yaw - self.base_yaw()),
                    -0.12,
                    0.12,
                ),
            ]
            progress = self.third_extra_origin[0] - self.scratch.xpos[self.base, 0]
            if progress >= max(0.03, self.extra_left_distance - 0.05):
                self.third_extra_brake = cmd.copy()
                self.enter("THIRD_EXTRA_LEFT_BRAKE")
            elif elapsed > 10.0:
                self.failure = "Extra third-bottle Loco left shift did not advance"
        else:
            cmd = self.third_extra_brake * (1 - geometry.smoothstep(elapsed / 0.6))
            (l, r) = self.dual_measure()
            good = (
                (effective_hand_control(l, 0.981) or effective_hand_control(r, 0.981))
                and l["root_speed_mps"] < 0.08
                and (l["support_feet_count"] == 2)
            )
            self.third_extra_stopped = (
                (now if self.third_extra_stopped is None else self.third_extra_stopped)
                if good
                else None
            )
            if (
                elapsed >= 2.5
                and self.third_extra_stopped is not None
                and (now - self.third_extra_stopped >= 0.3)
            ):
                self.enable_carry()
                self.dual_station_goals = self.held_goals()
                self.third_extra_done = True
                self.multi_events.append(
                    dict(
                        time_s=now,
                        event="third_actual_extra_left_shift_complete",
                        requested_m=self.extra_left_distance,
                        actual_delta_xyz_m=(
                            self.scratch.xpos[self.base] - self.third_extra_origin
                        ).tolist(),
                        bottles=self.all_bottles(),
                    )
                )
                self.enter("DUAL_STATION_OPEN_LEFT_FOR_REGRIP")
            elif elapsed > 8.0:
                self.failure = (
                    "Extra third-bottle Loco shift did not brake with actual support"
                )
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        return (self.stage, self.held_goals(), 65.7)
    result = handoff.carry_dual_targets(self)
    if (
        self.active_bottle == 2
        and self.stage == "DUAL_STATION_OPEN_LEFT_FOR_REGRIP"
        and (not self.third_extra_done)
    ):
        self.third_extra_origin = self.scratch.xpos[self.base].copy()
        self.third_extra_yaw = self.base_yaw()
        self.third_extra_arms = self.executed[14:].copy()
        self.third_extra_stopped = None
        self.bottle_commands = [1, 1]
        self.enter("THIRD_EXTRA_LEFT")
        return (self.stage, self.held_goals(), 65.7)
    return result


def first_release_loaded_targets(self):
    if (
        self.active_bottle == 0
        and self.place_active
        and (self.stage == "SHELF_LOAD_HOLD")
    ):
        now = float(self.data.time)
        (l, r) = self.dual_measure()
        good = (
            self.shelf_support(l)
            and l["middle_shelf_upward_N"] > 0.25 * 0.981
            and (l["object_tilt_deg"] < 8.0)
            and (l["object_linear_speed_mps"] < 0.02)
            and (l["object_angular_speed_radps"] < 0.2)
        )
        self.first_contact_release_since = (
            (
                now
                if self.first_contact_release_since is None
                else self.first_contact_release_since
            )
            if good
            else None
        )
        if (
            self.first_contact_release_since is not None
            and now - self.first_contact_release_since >= 0.04
        ):
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="first_open_after_measured_shared_shelf_load",
                    shelf_N=l["middle_shelf_upward_N"],
                    hand_N=l["hand_vertical_force_N"],
                    tilt_deg=l["object_tilt_deg"],
                    original_physical_guards_unchanged=True,
                )
            )
            self.place_phase_goals = self.held_goals()
            self.native.follow_current(self.scratch)
            self.policy.follow_current(self.scratch)
            self.enable_carry()
            self.carry_start = now - 2.0
            self.enter("SHELF_OPEN_LEFT")
    return first_upright_loaded_targets(self)


def first_upright_loaded_targets(self):
    if (
        self.active_bottle == 0
        and self.place_active
        and (self.stage == "SHELF_UPRIGHT_AND_ALIGN")
        and (not self.first_trim_done)
    ):
        self.first_trim_active = True
        self.first_trim_goals = self.held_goals()
        self.first_trim_since = None
        self.enable_carry()
        self.carry_start = float(self.data.time) - 2.0
        self.enter("FIRST_AIR_UPRIGHT")
    if self.first_trim_active:
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_TRANSFER"
        self.bottle_commands = [1, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        goal = self.first_trim_goals.copy()
        (l, r) = self.dual_measure()
        e = self.errors(goal)[0]
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        v = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(v) > 1e-08 and e[0] < 0.005 and (e[1] < 3.0):
            angle = min(float(np.arctan2(np.linalg.norm(v), axis[2])), 0.12 * 0.02)
            (x, y, z) = v / np.linalg.norm(v)
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            pivot = self.scratch.xpos[self.cargo_body].copy()
            goal[0, :3, 3] = pivot + rot @ (goal[0, :3, 3] - pivot)
            goal[0, :3, :3] = rot @ goal[0, :3, :3]
        self.first_trim_goals = goal.copy()
        good = (
            l["object_tilt_deg"] < 5.0
            and l["object_linear_speed_mps"] < 0.02
            and (l["object_angular_speed_radps"] < 0.15)
            and effective_hand_control(l, 0.981)
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.first_trim_since = (
            (now if self.first_trim_since is None else self.first_trim_since)
            if good
            else None
        )
        if self.first_trim_since is not None and now - self.first_trim_since >= 0.4:
            self.first_trim_active = False
            self.first_trim_done = True
            self.object_yaw_offset = 45.0
            self.place_active = False
            shelf_begin_place(self)
            return (self.stage, self.place_goals, 65.7)
        elif elapsed > 12.0:
            self.failure = "First bottle did not become measurably upright while held before shelf placement"
        if int(round(elapsed / 0.02)) % 100 == 0:
            print(
                dict(
                    first_trim_t=elapsed,
                    tilt=l["object_tilt_deg"],
                    left_N=l["hand_vertical_force_N"],
                    errors=e,
                ),
                flush=True,
            )
        return (self.stage, goal, 65.7)
    return left_stance_loaded_targets(self)


def left_stance_start_left_after_stance(self):
    self.right_place_active = False
    self.place_active = False
    handoff.carry_start_shelf_regrip(self)


def left_stance_loaded_targets(self):
    if self.left_after_stance_pending:
        self.left_after_stance_pending = False
        self.start_left_after_stance()
    if (
        self.active_bottle == 0
        and self.shelf_regrip_active
        and (self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD")
    ):
        (l, r) = self.dual_measure()
        if l["hand_vertical_force_N"] < 0.6 * 0.981:
            self.left_load_since = None
        if (
            l["hand_vertical_force_N"] < 0.8 * 0.981
            and float(self.data.time) - self.stage_start > 5.0
        ):
            self.phase_goals[0, 2, 3] += 0.002 * 0.02
    value = handoff.right_handoff_loaded_targets(self)
    if (
        self.active_bottle == 0
        and self.right_place_active
        and (self.stage == "RIGHT_SHELF_ALIGN_AIR")
    ):
        self.start_left_after_stance()
        return (self.stage, self.second_start_goals, 65.7)
    return value


def second_depth_switch_active(self, index):
    sequence.repeat_switch_active(self, index)
    self.target_xy[1] = 2.29 if index == 1 else 2.1700333121935995


def rack_begin_walk(self):
    loaded_route_begin_walk(self)
    self.begin_route()


def rack_begin_route(self):
    shelf_pose_begin_route(self)
    self.right_lower_origin = self.carry_goals[1].copy()


def rack_right_rest_goal(self, original):
    if not self.lower_right:
        return original
    goal = original.copy()
    body = self.scratch.xpos[self.base]
    rotation = geometry.rotation_z(self.base_yaw())
    goal[:3, 3] = body + rotation @ np.array([0.03, -0.28, -0.06])
    local_orientation = self.right_lower_origin[:3, :3]
    goal[:3, :3] = rotation @ geometry.rotation_z(-self.carry_yaw) @ local_orientation
    f = geometry.smoothstep((float(self.data.time) - self.route_start) / 3.0)
    return geometry.mix_transform(original, goal, f)


def rack_extra_goals(self):
    goals = shelf_pose_extra_goals(self)
    goals[1] = self.right_rest_goal(goals[1])
    return goals


def rack_loaded_targets(self):
    if not self.place_active or self.stage != "SHELF_UPRIGHT_AND_ALIGN":
        return shelf_pose_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    f = geometry.smoothstep((now - self.place_start) / 2.0)
    self.walk_pose[:] = [
        (1 - f) * 0.82 + f * self.placement_height,
        (1 - f) * self.route_pitch + f * self.placement_pitch,
    ]
    self.post_pose[:] = self.walk_pose
    obj = self.object_initial.copy()
    yaw = np.arctan2(obj[1, 0], obj[0, 0])
    upright = obj.copy()
    upright[:3, :3] = geometry.rotation_z(yaw + np.deg2rad(self.object_yaw_offset))
    obj = geometry.mix_transform(self.object_initial, upright, self.upright_fraction)
    obj[:3, 3] = [*self.target_xy, self.middle_height + 0.068]
    self.place_object_goal = obj
    target = obj @ self.object_wrist
    goals = self.place_phase_goals.copy()
    goals[0] = geometry.mix_transform(
        self.place_phase_goals[0], target, geometry.smoothstep(elapsed / self.move_s)
    )
    if elapsed >= self.move_s + 0.3:
        error = self.errors(goals)[0]
        if error[0] < 0.008 and error[1] < 4.0:
            self.place_phase_goals = goals.copy()
            self.place_lower_goal = goals.copy()
            self.place_lower_start = now
            self.enter("SHELF_LOWER_TO_ACTUAL_CONTACT")
        elif elapsed > self.move_s + 6.0:
            self.failure = "Shelf wrist alignment did not reach actual pose"
    if now - self.place_start > 30.0:
        self.failure = "Shelf placement attempt time limit"
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def shelf_pose_begin_route(self):
    self.done = False
    self.side_route_active = True
    self.place_active = False
    self.extra_active = self.carry_mode == "cartesian"
    self.route_start = float(self.data.time)
    self.route_origin = self.scratch.xpos[self.base].copy()
    self.walk_arms = self.executed[14:].copy()
    self.arm_hold = self.walk_arms.copy()
    self.route_initial_pose = self.walk_pose.copy()
    self.route_twist = np.zeros(3)
    self.enter("SIDE_SHELF_POSTURE_PREP")
    if self.carry_mode == "cartesian":
        self.enable_carry()
        self.carry_object_center = self.scratch.xpos[self.cargo_body].copy()


def shelf_pose_extra_goals(self):
    goals = self.carry_goals.copy()
    f = geometry.smoothstep((float(self.data.time) - self.route_start) / 2.0)
    goals[:, :3, 3] += geometry.rotation_z(self.carry_yaw) @ np.array(
        [-self.carry_pullback * f, 0.0, 0.0]
    )
    (a, b) = np.deg2rad([self.carry_tilt_x, self.carry_tilt_y]) * f
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    tilt = (
        geometry.rotation_z(self.carry_yaw)
        @ Rx
        @ Ry
        @ geometry.rotation_z(-self.carry_yaw)
    )
    goals[0, :3, 3] = self.carry_object_center + tilt @ (
        goals[0, :3, 3] - self.carry_object_center
    )
    goals[0, :3, :3] = tilt @ goals[0, :3, :3]
    rotation = geometry.rotation_z(
        geometry.wrap_angle(self.base_yaw() - self.carry_yaw)
    )
    base = self.scratch.xpos[self.base]
    for i in range(2):
        goals[i, :2, 3] = base[:2] + rotation[:2, :2] @ (
            goals[i, :2, 3] - self.carry_origin[:2]
        )
        goals[i, :3, :3] = rotation @ goals[i, :3, :3]
    return goals


def shelf_pose_loaded_targets(self):
    if self.place_active:
        value = shelf_loaded_targets(self)
        f = geometry.smoothstep((float(self.data.time) - self.place_start) / 2.0)
        self.walk_pose[0] = (1 - f) * 0.82 + f * self.placement_height
        self.walk_pose[1] = (1 - f) * self.route_pitch + f * self.placement_pitch
        self.post_pose[:] = self.walk_pose
        return value
    if not self.side_route_active:
        result = loaded_route_loaded_targets(self)
        if self.done and self.walk_success:
            self.begin_route()
        else:
            return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    command = np.zeros(3)
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [1, 0]
    f = geometry.smoothstep((now - self.route_start) / 2.0)
    self.walk_pose[:] = (1 - f) * self.route_initial_pose + f * np.array(
        [0.82, self.route_pitch]
    )
    self.post_pose[:] = self.walk_pose
    if self.stage == "SIDE_SHELF_POSTURE_PREP":
        if elapsed >= 2.2:
            self.enter("LOCO_TO_LEFT_SHORT_EDGE")
    elif self.stage == "LOCO_TO_LEFT_SHORT_EDGE":
        local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ (
            self.route_goal - self.scratch.xpos[self.base, :2]
        )
        command = np.array(
            [
                (
                    np.sign(local[0]) * max(0.2, min(0.25, 0.9 * abs(local[0])))
                    if abs(local[0]) > 0.12
                    else 0.0
                ),
                (
                    np.sign(local[1])
                    * (
                        0.2
                        if self.slow_near and abs(local[1]) < 0.3
                        else self.route_lateral
                    )
                    if abs(local[1]) > 0.07
                    else 0.0
                ),
                np.clip(
                    0.8 * geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12
                ),
            ]
        ) * geometry.smoothstep(elapsed / 1.2)
        if np.linalg.norm(
            self.scratch.xpos[self.base, :2] - self.route_goal
        ) < 0.1 and abs(geometry.wrap_angle(np.pi / 2 - self.base_yaw())) < np.deg2rad(
            10
        ):
            self.route_brake = command.copy()
            self.enter("SIDE_SHELF_BRAKE")
        elif elapsed > 45.0:
            self.failure = "Held Loco did not reach shelf stance"
    elif self.stage == "SIDE_SHELF_BRAKE":
        command = self.route_brake * (1 - geometry.smoothstep(elapsed / 1.5))
        (l, rr) = self.dual_measure()
        good = (
            bool(effective_hand_control(l, 0.981))
            and rr["active_hand_object_contact_count"] == 0
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.route_stop_since = (
            (now if self.route_stop_since is None else self.route_stop_since)
            if good
            else None
        )
        if (
            elapsed >= 2.5
            and self.route_stop_since is not None
            and (now - self.route_stop_since >= 0.2)
        ):
            self.route_arrived = True
            self.route_event.append(
                dict(
                    time_s=now,
                    event="actual_shelf_stance_reached",
                    base=self.scratch.xpos[self.base].tolist(),
                )
            )
            self.begin_place()
            self.side_route_active = False
            return (self.stage, self.place_goals, 65.7)
        elif elapsed > 9.0:
            self.failure = "Held Loco did not settle at shelf stance"
    self.walk_twist[:] = command
    self.post_twist[:] = command
    if int(round((now - self.route_start) / 0.02)) % 100 == 0:
        print(
            json.dumps(
                dict(
                    route_t=now - self.route_start,
                    stage=self.stage,
                    base=self.scratch.xpos[self.base].tolist(),
                    yaw_deg=float(np.degrees(self.base_yaw())),
                    command=command.tolist(),
                )
            ),
            flush=True,
        )
    goals = self.extra_goals() if self.carry_mode == "cartesian" else self.held_goals()
    return (self.stage, goals, 65.7)


def shelf_progress_back(self):
    value = loaded_route_progress_back(self)
    return value + (0.45 - self.back_distance) if not self.place_active else value


def shelf_begin_place(self):
    self.done = False
    self.extra_active = True
    self.place_active = True
    self.place_start = float(self.data.time)
    self.place_source_lower = self.executed[:14].copy()
    self.enable_carry()
    self.carry_start = float(self.data.time) - 1.0
    self.place_goals = self.held_goals()
    self.object_initial = self.object_pose()
    self.object_wrist = np.linalg.inv(self.object_initial) @ self.place_goals[0]
    self.place_phase_goals = self.place_goals.copy()
    self.bottle_commands = [1, 0]
    self.bottle_loss_since = None
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.place_initial_pose = self.walk_pose.copy()
    self.place_contact_since = None
    self.place_hold_since = None
    self.place_object_goal = self.object_initial.copy()
    self.enter("SHELF_UPRIGHT_AND_ALIGN")
    self.place_events.append(
        dict(
            time_s=self.place_start,
            event="begin_shelf_placement",
            base=self.scratch.xpos[self.base].tolist(),
            bottle=self.object_initial[:3, 3].tolist(),
            initialization=(
                "Checkpoint before first step"
                if self.branch_mode
                else "Continuous original world"
            ),
        )
    )


def shelf_loaded_targets(self):
    if not self.place_active:
        result = loaded_route_loaded_targets(self)
        if self.done and self.walk_success:
            self.begin_place()
            return (self.stage, self.place_goals, 65.7)
        return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.bottle_commands = [1, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.walk_pose[:] = [0.82, self.place_pitch]
    self.post_pose[:] = self.walk_pose
    goals = self.place_phase_goals.copy()
    (left, right) = self.dual_measure()
    if self.stage == "SHELF_UPRIGHT_AND_ALIGN":
        obj = self.object_initial.copy()
        yaw = np.arctan2(obj[1, 0], obj[0, 0])
        upright = obj.copy()
        upright[:3, :3] = geometry.rotation_z(yaw + np.deg2rad(self.object_yaw_offset))
        obj = geometry.mix_transform(
            self.object_initial, upright, self.upright_fraction
        )
        obj[:3, 3] = [
            self.place_edge - 0.12,
            self.place_target_y,
            self.middle_height + 0.068,
        ]
        if self.adaptive_xy:
            obj[0, 3] = np.clip(
                self.object_initial[0, 3],
                self.place_edge - 0.4 + 0.055,
                self.place_edge - 0.055,
            )
            obj[1, 3] = np.clip(
                self.object_initial[1, 3],
                self.shelf_layout["center"][1] - self.shelf_layout["width"] / 2 + 0.065,
                self.shelf_layout["center"][1] + self.shelf_layout["width"] / 2 - 0.065,
            )
        self.place_object_goal = obj
        target = obj @ self.object_wrist
        goals[0] = geometry.mix_transform(
            self.place_phase_goals[0],
            target,
            geometry.smoothstep(elapsed / self.move_s),
        )
        if elapsed >= self.move_s + 0.3:
            error = self.errors(goals)[0]
            if error[0] < 0.008 and error[1] < 4.0:
                self.place_phase_goals = goals.copy()
                self.place_lower_goal = goals.copy()
                self.place_lower_start = now
                self.enter("SHELF_LOWER_TO_ACTUAL_CONTACT")
            elif elapsed > self.move_s + 6.0:
                self.failure = "Shelf wrist alignment did not reach actual pose"
    elif self.stage == "SHELF_LOWER_TO_ACTUAL_CONTACT":
        goals[0, 2, 3] -= min(0.075, elapsed * 0.018)
        if self.shelf_support(left) and left["middle_shelf_upward_N"] > 0.7 * 0.981:
            self.place_phase_goals = goals.copy()
            self.place_contact_since = None
            self.enter("SHELF_LOAD_HOLD")
        if elapsed > 7.0:
            self.failure = (
                "No stable real middle-shelf support before release; hand kept closed"
            )
    elif self.stage == "SHELF_LOAD_HOLD":
        good = (
            self.shelf_support(left)
            and left["middle_shelf_upward_N"] > 0.7 * 0.981
            and (left["object_tilt_deg"] < 18.0)
            and (left["object_linear_speed_mps"] < 0.03)
            and (left["object_angular_speed_radps"] < 0.35)
        )
        self.place_contact_since = (
            (now if self.place_contact_since is None else self.place_contact_since)
            if good
            else None
        )
        if (
            self.place_contact_since is not None
            and now - self.place_contact_since >= 0.2
        ):
            self.place_events.append(
                dict(
                    time_s=now,
                    event="middle_shelf_actual_load_before_hand_open",
                    upward_N=left["middle_shelf_upward_N"],
                    tilt_deg=left["object_tilt_deg"],
                )
            )
            self.place_phase_goals = goals.copy()
            self.enter("SHELF_OPEN_LEFT")
        elif elapsed > 4.0:
            self.failure = "Middle shelf load did not settle before release"
    elif self.stage == "SHELF_OPEN_LEFT":
        self.bottle_commands = [0, 0]
        if elapsed >= 2.0:
            if (
                self.shelf_stable(left)
                and self.hands.controllers["left"].finished
                and (self.hands.controllers["left"].command == 0)
                and (left["hand_vertical_force_N"] < 0.2 * 0.981)
            ):
                self.place_events.append(
                    dict(
                        time_s=now,
                        event="fingers_open_shelf_bears_weight_begin_withdrawal",
                        left_contacts=left["active_hand_object_contact_count"],
                        shelf_force_N=left["middle_shelf_upward_N"],
                    )
                )
                self.place_phase_goals = goals.copy()
                self.enter("SHELF_WITHDRAW_LEFT")
            elif elapsed > 6.0:
                self.failure = "Shelf release did not produce actual hand-free support"
    elif self.stage == "SHELF_WITHDRAW_LEFT":
        self.bottle_commands = [0, 0]
        goals[0, :3, 3] += geometry.smoothstep(elapsed / 2.5) * np.array(
            [-0.02, -0.12, 0.08]
        )
        if elapsed >= 3.0:
            self.place_phase_goals = goals.copy()
            self.enter("SHELF_SETTLE")
    elif self.stage == "SHELF_SETTLE":
        self.bottle_commands = [0, 0]
        good = (
            self.shelf_stable(left)
            and left["active_hand_object_contact_count"] == 0
            and (right["active_hand_object_contact_count"] == 0)
            and (left["root_speed_mps"] < self.completion_root_limit)
            and (left["support_feet_count"] == 2)
        )
        self.place_hold_since = (
            (now if self.place_hold_since is None else self.place_hold_since)
            if good
            else None
        )
        if self.place_hold_since is not None and now - self.place_hold_since >= 1.5:
            self.place_success = True
            self.done = True
            self.enter("SECOND_SHELF_PLACEMENT_COMPLETE")
        elif elapsed > 7.0:
            self.failure = (
                "Shelf bottle and robot did not settle after actual withdrawal"
            )
    if now - self.place_start > 30.0:
        self.failure = "Shelf placement attempt time limit"
    self.place_goals = goals.copy()
    if int(round((now - self.place_start) / 0.02)) % 50 == 0:
        print(
            json.dumps(
                dict(
                    place_t=now - self.place_start,
                    stage=self.stage,
                    errors=self.errors(goals),
                    left_force=left["hand_vertical_force_N"],
                    shelf_force=left["middle_shelf_upward_N"],
                    tilt=left["object_tilt_deg"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def shelf_step(self):
    float(self.data.time)
    len(self.trace)
    route_segment_step(self)
    if self.place_active and self.place_body == "hold":
        pass


def route_segment_enable_carry(self):
    self.carry_origin = self.scratch.xpos[self.base].copy()
    self.carry_yaw = self.base_yaw()
    self.carry_goals = self.held_goals()
    self.carry_initial_target = self.executed[14:].copy()
    self.carry_start = float(self.data.time)
    self.cart_active = self.carry_kind == "cartesian"
    self.cart_time = None
    self.cart_values = self.scratch.qpos[self.upper_pos].copy()
    self.cart_pose = self.cart_values.copy()
    self.cart_prior = self.cart_values.copy()
    self.arm_mode = "reach" if self.cart_active else "locked"
    self.arm_hold = self.carry_initial_target.copy()
    self.arm_planning = True


def route_segment_step(self):
    count = len(self.trace)
    t = float(self.data.time)
    sequence.control_step(self)
    if self.extra_active and len(self.trace) > count:
        self.extra_controls.append(
            dict(
                time_s=t,
                executed_qdes_rad=self.executed.tolist(),
                loco_qdes_rad=self.loco.q_des.tolist(),
                arm_mode=self.carry_kind,
                twist=self.walk_twist.tolist(),
            )
        )


def loaded_route_begin_walk(self):
    self.done = False
    self.walk_start = float(self.data.time)
    self.walk_origin = self.scratch.xpos[self.base].copy()
    self.walk_yaw = self.base_yaw()
    self.walk_R = geometry.rotation_z(self.walk_yaw)
    self.walk_twist = np.zeros(3)
    self.walk_pose = np.array([0.82, 0.35])
    self.walk_arms = self.executed[14:].copy()
    self.walk_last_command = np.zeros(3)
    self.walk_plan_time = None
    self.handoff_owner = "left"
    self.held_walk_started = True
    self.walk_boundary = dict(
        time_s=self.walk_start,
        qpos=self.data.qpos.tolist(),
        qvel=self.data.qvel.tolist(),
    )
    self.enter("LEFT_HELD_LOCO_POSTURE_PREP")
    self.walk_events.append(
        dict(
            time_s=self.walk_start,
            event="start_left_normal_grasp_walk",
            histories_reset=False,
            arms="Hold actually executed PD targets",
            lower="FrozenLoco14",
        )
    )


def loaded_route_progress_back(self):
    return float(
        -(self.walk_R.T @ (self.scratch.xpos[self.base] - self.walk_origin))[0]
    )


def loaded_route_begin_brake(self, next_stage):
    self.brake_start = float(self.data.time)
    self.brake_command = self.walk_twist.copy()
    self.after_brake = next_stage
    self.walk_settle = None
    self.enter("LEFT_HELD_LOCO_BRAKE")


def loaded_route_goal_command(self, xy, yaw):
    local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ (
        np.asarray(xy) - self.scratch.xpos[self.base, :2]
    )
    err = geometry.wrap_angle(yaw - self.base_yaw())
    return np.array(
        [
            np.clip(0.9 * local[0], -0.2, 0.2),
            np.clip(0.9 * local[1], -0.12, 0.12),
            np.clip(0.8 * err, -0.18, 0.18),
        ]
    )


def loaded_route_loaded_targets(self):
    if self.walk_start is None:
        result = handoff.regrasp_loaded_targets(self)
        if self.done and self.left_normal_success:
            self.begin_walk()
            return (self.stage, self.held_goals(), 65.7)
        return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    total = now - self.walk_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [1, 0]
    self.post_twist[:] = self.walk_twist
    f = geometry.smoothstep(total / 5.0)
    self.walk_pose[:] = (1 - f) * np.array([0.82, 0.35]) + f * np.array(
        [self.walk_height, self.walk_pitch]
    )
    desired = np.zeros(3)
    if self.stage == "LEFT_HELD_LOCO_POSTURE_PREP":
        if elapsed >= 5.5:
            self.enter("LEFT_HELD_LOCO_BACK_CLEAR")
    elif self.stage == "LEFT_HELD_LOCO_BACK_CLEAR":
        desired = np.array(
            [
                -self.route_speed * geometry.smoothstep(elapsed / 1.5),
                0.0,
                np.clip(
                    0.8 * geometry.wrap_angle(self.walk_yaw - self.base_yaw()),
                    -0.12,
                    0.12,
                ),
            ]
        )
        if self.progress_back() >= (0.45 if self.route_kind == "back" else 0.7):
            self.begin_brake(
                "COMPLETE"
                if self.route_kind == "back"
                else "TURN" if self.route_kind == "perpendicular" else "LATERAL"
            )
        elif elapsed > 15.0:
            self.failure = "Left-held backward clearance did not make progress"
    elif self.stage == "LEFT_HELD_LOCO_TURN_RIGHT_90":
        error = geometry.wrap_angle(-self.base_yaw())
        rate = np.clip(0.9 * error, -0.18, 0.18)
        if abs(error) > np.deg2rad(4.0):
            rate = np.sign(rate) * max(abs(rate), 0.12)
        desired = geometry.smoothstep(elapsed / 1.2) * np.array([0.12, 0.0, rate])
        if abs(error) < np.deg2rad(3.0):
            self.begin_brake("APPROACH")
        elif elapsed > 35.0:
            self.failure = "Left-held forward arc did not reach east heading"
    elif self.stage == "LEFT_HELD_LOCO_LATERAL_CORRIDOR":
        target = [self.shelf_layout["stance"][0], self.lateral_start_y]
        desired = self.goal_command(target, np.pi / 2) * geometry.smoothstep(
            elapsed / 1.2
        )
        if abs(self.scratch.xpos[self.base, 0] - target[0]) < 0.1:
            self.begin_brake("APPROACH")
        elif elapsed > 30.0:
            self.failure = "Left-held lateral corridor did not make progress"
    elif self.stage == "LEFT_HELD_LOCO_APPROACH_RACK":
        target = self.shelf_layout["stance"]
        heading = np.deg2rad(self.shelf_layout["heading_deg"])
        desired = self.goal_command(target, heading) * geometry.smoothstep(
            elapsed / 1.2
        )
        distance = np.linalg.norm(self.scratch.xpos[self.base, :2] - target)
        error = abs(geometry.wrap_angle(heading - self.base_yaw()))
        if distance < 0.1 and error < np.deg2rad(5.0):
            self.begin_brake("COMPLETE")
        elif elapsed > 35.0:
            self.failure = "Left-held approach did not reach fixed rack staging point"
    elif self.stage == "LEFT_HELD_LOCO_BRAKE":
        desired = self.brake_command * (1.0 - geometry.smoothstep(elapsed / 1.5))
        (left, right) = self.dual_measure()
        good = (
            bool(effective_hand_control(left, 0.981))
            and (not left["delivery_table_contact"])
            and (not left.get("shelf_bottle_contact", False))
            and (right["active_hand_object_contact_count"] == 0)
            and (left["root_speed_mps"] < 0.03)
            and (left["support_feet_count"] == 2)
        )
        self.walk_settle = (
            (now if self.walk_settle is None else self.walk_settle) if good else None
        )
        if (
            elapsed >= 3.0
            and self.walk_settle is not None
            and (now - self.walk_settle >= 1.0)
        ):
            self.walk_events.append(
                dict(
                    time_s=now,
                    event="stopped",
                    base=self.scratch.xpos[self.base].tolist(),
                    yaw_deg=float(np.degrees(self.base_yaw())),
                    next=self.after_brake,
                )
            )
            if self.after_brake == "COMPLETE":
                self.walk_success = True
                self.done = True
                self.enter("LEFT_HELD_LOCO_ROUTE_COMPLETE")
            elif self.after_brake == "TURN":
                self.enter("LEFT_HELD_LOCO_TURN_RIGHT_90")
            elif self.after_brake == "LATERAL":
                self.lateral_start_y = float(self.scratch.xpos[self.base, 1])
                self.enter("LEFT_HELD_LOCO_LATERAL_CORRIDOR")
            else:
                self.enter("LEFT_HELD_LOCO_APPROACH_RACK")
        elif elapsed > 9.0:
            self.failure = (
                "Left-held route could not stop with actual left-only bottle control"
            )
    self.walk_twist[:] = desired
    self.post_twist[:] = desired
    self.post_pose[:] = self.walk_pose
    if int(round(total / 0.02)) % 100 == 0:
        print(
            json.dumps(
                dict(
                    walk_t=total,
                    stage=self.stage,
                    base=self.scratch.xpos[self.base].tolist(),
                    yaw_deg=float(np.degrees(self.base_yaw())),
                    back_m=self.progress_back(),
                    twist=self.walk_twist.tolist(),
                )
            ),
            flush=True,
        )
    if total > 110.0:
        self.failure = "Held-bottle route time limit"
    return (self.stage, self.held_goals(), 65.7)


def grip_geometry_start_second(self):
    handoff.regrasp_start_second(self)
    self.left_offset[:] = self.grip_offset
    self.second_events[0]["left_grip_offset_wrist_m"] = self.left_offset.tolist()
