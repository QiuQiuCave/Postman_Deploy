"""Left/right handover, loaded transport, and shelf regrasp."""

from __future__ import annotations
import numpy as np
from r2v2_deploy.tasks.physical_validity import (
    effective_hand_control as effective_hand_control,
)
import json as json
import mujoco as mujoco
from . import geometry, grasp, placement, sequence


def empty_hand_clearance_advance(self, stage, goals):
    if self.active_bottle == 2 and stage == "HANDOFF_LEFT_OPEN":
        axis = self.object_pose()[:3, 2]
        away = goals[0, :3, 3] - self.object_pose()[:3, 3]
        away /= np.linalg.norm(away)
        self.source_left_open_vector = 0.08 * axis + 0.035 * away
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_left_open_with_actual_axis_clearance",
                wrist_goal_delta_xyz_m=self.source_left_open_vector.tolist(),
                binary_finger_configuration_unchanged=True,
            )
        )
    return grasp.adaptive_grasp_advance(self, stage, goals)


def empty_hand_clearance_loaded_targets(self):
    if (
        self.active_bottle == 2
        and self.stage == "HANDOFF_LEFT_OPEN"
        and (self.source_left_open_vector is not None)
    ):
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.post_twist[:] = 0.0
        self.post_pose[:] = [0.82, 0.35]
        self.bottle_commands = [0, 1]
        goals = self.phase_goals.copy()
        goals[0, :3, 3] += (
            geometry.smoothstep(elapsed / 3.0) * self.source_left_open_vector
        )
        (left, right) = self.dual_measure()
        if elapsed > 4.0:
            if left["active_hand_object_contact_count"] == 0 and effective_hand_control(
                right, 0.981
            ):
                self.advance("HANDOFF_LEFT_WITHDRAW", goals)
            elif elapsed > 8.0:
                self.failure = "Left simultaneous open/withdraw did not leave real right-only control"
        self.post_goals = goals.copy()
        return (self.stage, goals, 65.7)
    return placement.station_heading_loaded_targets(self)


def release_clearance_loaded_targets(self):
    now = float(self.data.time)
    if (
        self.active_bottle == 1
        and self.practical_active
        and (self.stage in ["SHELF_LOWER_TO_ACTUAL_CONTACT", "SHELF_LOAD_HOLD"])
    ):
        (left, right) = self.dual_measure()
        good = (
            self.shelf_support(left)
            and left["middle_shelf_upward_N"] > 0.25 * 0.981
            and (left["object_tilt_deg"] < 12.0)
            and (left["object_linear_speed_mps"] < 0.03)
            and (left["object_angular_speed_radps"] < 0.2)
        )
        self.second_early_share_since = (
            (
                now
                if self.second_early_share_since is None
                else self.second_early_share_since
            )
            if good
            else None
        )
        if (
            self.second_early_share_since is not None
            and now - self.second_early_share_since >= 0.06
        ):
            goals = self.held_goals()
            self.place_phase_goals = goals.copy()
            self.native.follow_current(self.scratch)
            self.policy.follow_current(self.scratch)
            self.enable_carry()
            self.carry_start = now - 2.0
            self.second_early_release_used = True
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="second_release_on_actual_shelf_load_before_lean_grows",
                    actual_tilt_deg=left["object_tilt_deg"],
                    shelf_upward_N=left["middle_shelf_upward_N"],
                    left_upward_N=left["hand_vertical_force_N"],
                    auxiliary_release_tilt_gate_deg=12.0,
                    final_upright_hand_free_gate_unchanged=True,
                )
            )
            self.enter("SHELF_OPEN_LEFT")
            return (self.stage, goals, 65.7)
    if (
        self.active_bottle == 2
        and self.stage == "HANDOFF_RIGHT_PREAPPROACH"
        and (now - self.stage_start >= 7.0)
    ):
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        if axis[2] > 0.5:
            self.right_R = (
                geometry.axis_rotation(np.array([0.0, 0.0, 1.0]), axis)
                @ self.post_R
                @ geometry.rotation_z(np.deg2rad(self.right_grasp_yaw))
            )
        obj = self.object_pose()
        goals = self.phase_goals.copy()
        goals[1, :3, :3] = self.right_R
        goals[1, :3, 3] = (
            obj[:3, 3]
            - self.right_R @ self.right_offset
            + self.post_R @ np.array([0.0, -0.065, 0.03])
        )
        err = self.errors(goals)[1]
        (left, right) = self.dual_measure()
        distance = float(
            np.linalg.norm(
                self.scratch.xpos[self.model.body("right_hand_roll_link").id]
                - obj[:3, 3]
            )
        )
        if (
            err[0] < getattr(self, "source_right_pre_gate_m", 0.015)
            and err[1] < 6.0
            and (right["active_hand_object_contact_count"] == 0)
            and (distance > 0.12)
            and effective_hand_control(left, 0.981)
        ):
            self.third_soft_right_pre_count += 1
            goals = self.held_goals()
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="third_empty_right_preapproach_actual_clearance_accepted",
                    position_error_m=err[0],
                    orientation_error_deg=err[1],
                    actual_wrist_bottle_distance_m=distance,
                    final_right_close_alignment_gate_unchanged=True,
                )
            )
            self.advance("HANDOFF_RIGHT_FINAL_APPROACH", goals)
            return (self.stage, goals, 65.7)
    return grasp.adaptive_grasp_loaded_targets(self)


def handoff_clearance_rotate_goal(self, goals, clock):
    result = carry_rotate_goal(self, goals, clock)
    if self.active_bottle == 1 and result is not None:
        f = geometry.smoothstep((clock - 30.0) / 4.0) * (
            1 - geometry.smoothstep((clock - 48.0) / 4.0)
        )
        result[0, 2, 3] += self.flip_height * f
        result[0, 1, 3] -= self.flip_back * f
        result[0, :3, :3] = (
            geometry.rotation_z(self.flip_yaw * 3.141592653589793 / 180.0 * f)
            @ result[0, :3, :3]
        )
    return result


def right_handoff_loaded_targets(self):
    if not self.right_place_active:
        return carry_modes_loaded_targets(self)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_LOWER"
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    self.bottle_commands = [0, 1]
    goals = self.right_place_from.copy()
    if self.stage == "RIGHT_SHELF_LOCO_STANCE":
        delta = self.right_loco_goal - self.scratch.xpos[self.base, :2]
        world = np.array(
            [
                np.sign(delta[0]) * 0.2 if abs(delta[0]) > 0.025 else 0.0,
                np.sign(delta[1]) * 0.2 if abs(delta[1]) > 0.025 else 0.0,
            ]
        ) * geometry.smoothstep(elapsed / 1.5)
        cmd = np.r_[
            geometry.rotation_z(self.base_yaw())[:2, :2].T @ world,
            np.clip(
                0.8 * geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12
            ),
        ]
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        if np.linalg.norm(delta) < 0.04:
            self.right_loco_brake = cmd.copy()
            self.enter("RIGHT_SHELF_LOCO_BRAKE")
        elif elapsed > 15.0:
            self.failure = "Right-held Loco stance did not reach the shelf approach"
        goals = self.held_goals()
    elif self.stage == "RIGHT_SHELF_LOCO_BRAKE":
        self.walk_twist[:] = self.right_loco_brake * (
            1 - geometry.smoothstep(elapsed / 1.2)
        )
        self.post_twist[:] = self.walk_twist
        (l, r) = self.dual_measure()
        good = (
            effective_hand_control(r, 0.981)
            and r["root_speed_mps"] < 0.08
            and (r["support_feet_count"] == 2)
        )
        self.right_loco_stopped = (
            (now if self.right_loco_stopped is None else self.right_loco_stopped)
            if good
            else None
        )
        goals = self.held_goals()
        if (
            elapsed > 2.5
            and self.right_loco_stopped is not None
            and (now - self.right_loco_stopped >= 0.3)
        ):
            self.enable_carry()
            self.right_place_from = self.held_goals()
            obj = self.object_pose()
            self.right_place_relation = np.linalg.inv(obj) @ self.right_place_from[1]
            target = obj.copy()
            target[:3, :3] = geometry.rotation_z(np.arctan2(obj[1, 0], obj[0, 0]))
            target[:3, 3] = [*self.target_xy, self.middle_height + 0.095]
            self.right_place_goal = target @ self.right_place_relation
            self.enter("RIGHT_SHELF_ALIGN_AIR")
        elif elapsed > 8.0:
            self.failure = "Right-held Loco stance did not brake with actual support"
    elif self.stage == "RIGHT_SHELF_ALIGN_AIR":
        goals[1] = geometry.mix_transform(
            self.right_place_from[1],
            self.right_place_goal,
            geometry.smoothstep(elapsed / 7.0),
        )
        (l, r) = self.dual_measure()
        self.errors(goals)[1]
        gap = r["bottom_height_m"] - self.middle_height
        good = (
            elapsed >= 8.0
            and np.linalg.norm(self.object_pose()[:2, 3] - self.target_xy) < 0.035
            and (0.015 < gap < 0.075)
            and (r["object_tilt_deg"] < 8.0)
            and (r["object_angular_speed_radps"] < 0.2)
            and effective_hand_control(r, 0.981)
            and (r["object_linear_speed_mps"] < 0.02)
            and (r["root_speed_mps"] < 0.08)
        )
        self.right_place_good = (
            (now if self.right_place_good is None else self.right_place_good)
            if good
            else None
        )
        if self.right_place_good is not None and now - self.right_place_good >= 0.3:
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="right_release_from_measured_small_clearance",
                    actual_gap_m=gap,
                    actual_bottle=self.all_bottles()[0],
                )
            )
            self.right_place_from = goals.copy()
            self.enter("RIGHT_SHELF_OPEN_ABOVE_BOARD")
        elif elapsed > 17.0:
            self.failure = "Right-held bottle did not align over shelf"
    elif self.stage == "RIGHT_SHELF_OPEN_ABOVE_BOARD":
        self.bottle_commands = [0, 0]
        (l, r) = self.dual_measure()
        if (
            elapsed > 2.0
            and self.hands.controllers["right"].finished
            and self.shelf_support(r)
        ):
            self.right_place_from = goals.copy()
            self.enter("RIGHT_SHELF_WITHDRAW")
        elif elapsed > 7.0:
            self.failure = "Right release did not settle onto real shelf"
    elif self.stage == "RIGHT_SHELF_WITHDRAW":
        self.bottle_commands = [0, 0]
        goals[1, :3, 3] += geometry.smoothstep(elapsed / 3.0) * np.array(
            [0.02, -0.12, 0.08]
        )
        if elapsed > 3.2:
            self.right_place_from = goals.copy()
            self.enter("RIGHT_SHELF_SETTLE")
    elif self.stage == "RIGHT_SHELF_SETTLE":
        self.bottle_commands = [0, 0]
        (l, r) = self.dual_measure()
        good = (
            self.shelf_stable(r)
            and (not l["active_hand_object_contact_count"])
            and (not r["active_hand_object_contact_count"])
            and (r["root_speed_mps"] < 0.08)
            and (r["support_feet_count"] == 2)
        )
        self.right_place_stable = (
            (now if self.right_place_stable is None else self.right_place_stable)
            if good
            else None
        )
        if self.right_place_stable is not None and now - self.right_place_stable >= 1.5:
            self.place_success = True
            self.done = True
            self.completed_bottles.append(1)
            self.right_place_active = False
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="first_right_hand_actual_shelf_release_and_withdrawal",
                    bottles=self.all_bottles(),
                )
            )
            if self.max_objects > 1:
                self.switch_active(1)
                self.start_return()
                self.done = False
        elif elapsed > 8.0:
            self.failure = (
                "Right-first shelf placement did not remain upright and hands clear"
            )
    self.place_goals = goals.copy()
    self.right_place_goals = goals.copy()
    if int(round((now - self.right_place_start) / 0.02)) % 100 == 0:
        (l, r) = self.dual_measure()
        print(
            dict(
                right_place_t=now - self.right_place_start,
                stage=self.stage,
                tilt=r["object_tilt_deg"],
                right_N=r["hand_vertical_force_N"],
                shelf_N=r["middle_shelf_upward_N"],
                errors=self.errors(goals),
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def carry_modes_loaded_targets(self):
    if (
        self.active_bottle == 0
        and self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD"
        and (not self.shelf_regrip_active)
        and (not self.dual_mode)
    ):
        now = float(self.data.time)
        self.first_shared_since = (
            (now if self.first_shared_since is None else self.first_shared_since)
            if self.both_ready()
            else None
        )
        if self.first_shared_since is not None and now - self.first_shared_since >= 0.4:
            self.begin_dual()
            return (self.stage, self.dual_goals, 65.7)
    return carry_loaded_targets(self)


def carry_enable_carry(self):
    placement.route_segment_enable_carry(self)
    self.carry_object_center = self.scratch.xpos[self.cargo_body].copy()


def carry_both_ready(self):
    (l, r) = self.dual_measure()
    return (
        effective_hand_control(l, 0.981)
        and effective_hand_control(r, 0.981)
        and (l["hand_vertical_force_N"] > 0.2 * 0.981)
        and (r["hand_vertical_force_N"] > 0.2 * 0.981)
        and (l["object_linear_speed_mps"] < 0.03)
        and (l["object_angular_speed_radps"] < 0.2)
        and (l["root_speed_mps"] < 0.08)
        and (l["support_feet_count"] == 2)
    )


def carry_begin_dual(self):
    self.dual_pending = False
    self.dual_mode = True
    self.done = False
    self.failure = None
    self.side_route_active = False
    self.refine_active = False
    self.place_active = False
    self.extra_active = True
    self.enable_carry()
    self.dual_origin = self.scratch.xpos[self.base].copy()
    self.dual_yaw = self.base_yaw()
    self.dual_R = geometry.rotation_z(self.dual_yaw)
    self.dual_goals = self.held_goals()
    self.dual_pose = self.post_pose.copy()
    self.dual_started = float(self.data.time)
    self.dual_nominal_x = self.shelf_base_x - 0.16 * self.active_bottle
    self.dual_stopped_since = None
    self.dual_left_since = None
    self.dual_solo_since = None
    self.__dict__.pop("route_world_R", None)
    self.__dict__.pop("route_arm_hold", None)
    self.bottle_commands = [1, 1]
    self.handoff_owner = "left"
    self.enter("DUAL_BOTTLE_CARRY_PREP")
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="two_real_hands_carry_begin",
            active=self.active_bottle + 1,
            bottles=self.all_bottles(),
        )
    )


def carry_dual_carry_goals(self, elapsed):
    f = geometry.smoothstep(elapsed / 4.0)
    goals = self.dual_goals.copy()
    rotation = geometry.rotation_z(geometry.wrap_angle(self.base_yaw() - self.dual_yaw))
    shift = self.dual_R @ np.array([-0.22 * f, 0.0, 0.0])
    for i in range(2):
        goals[i, :2, 3] = self.scratch.xpos[self.base, :2] + rotation[:2, :2] @ (
            goals[i, :2, 3] + shift[:2] - self.dual_origin[:2]
        )
        goals[i, :3, :3] = rotation @ goals[i, :3, :3]
    return goals


def carry_dual_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [1, 1]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    f = geometry.smoothstep((now - self.dual_started) / 4.0)
    self.walk_pose[:] = (1 - f) * self.dual_pose + f * np.array([0.82, 0.12])
    self.post_pose[:] = self.walk_pose
    goals = self.dual_carry_goals(now - self.dual_started)
    if self.stage == "DUAL_BOTTLE_CARRY_PREP":
        if elapsed >= 4.5:
            self.dual_arm_hold = self.executed[14:].copy()
            self.enter("DUAL_BOTTLE_LOCO_TO_SHELF")
    elif self.stage == "DUAL_BOTTLE_LOCO_TO_SHELF":
        world = np.array(
            [-0.2, np.clip(1.8 * (1.53 - self.scratch.xpos[self.base, 1]), -0.15, 0.1)]
        )
        local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ world
        cmd = np.r_[
            local,
            np.clip(
                0.8 * geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12
            ),
        ] * geometry.smoothstep(elapsed / 2.0)
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        if self.scratch.xpos[self.base, 0] <= self.dual_nominal_x + 0.2:
            self.dual_brake = cmd.copy()
            self.enter("DUAL_BOTTLE_LOCO_BRAKE")
        elif elapsed > 30.0:
            self.failure = "Dual-hand Loco did not reach shelf region"
    elif self.stage == "DUAL_BOTTLE_LOCO_BRAKE":
        cmd = self.dual_brake * (1 - geometry.smoothstep(elapsed / 1.5))
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        (l, r) = self.dual_measure()
        good = (
            (effective_hand_control(l, 0.981) or effective_hand_control(r, 0.981))
            and l["root_speed_mps"] < 0.08
            and (l["support_feet_count"] == 2)
        )
        self.dual_stopped_since = (
            (now if self.dual_stopped_since is None else self.dual_stopped_since)
            if good
            else None
        )
        if (
            elapsed > 2.5
            and self.dual_stopped_since is not None
            and (now - self.dual_stopped_since >= 0.3)
        ):
            self.enable_carry()
            self.dual_station_goals = self.held_goals()
            self.enter("DUAL_STATION_OPEN_LEFT_FOR_REGRIP")
        elif elapsed > 9.0:
            self.failure = "Dual-hand Loco did not stop with real support"
    elif self.stage == "DUAL_STATION_OPEN_LEFT_FOR_REGRIP":
        goals = self.dual_station_goals.copy()
        self.bottle_commands = [0, 1]
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        (l, r) = self.dual_measure()
        if (
            elapsed >= 4.0
            and abs(l["hand_vertical_force_N"]) < 0.1 * 0.981
            and self.hands.controllers["left"].finished
            and effective_hand_control(r, 0.981)
            and (r["hand_vertical_force_N"] > 0.8 * 0.981)
            and (r["root_speed_mps"] < 0.08)
        ):
            self.dual_station_goals = self.held_goals()
            self.enter("DUAL_STATION_CLEAR_EMPTY_LEFT")
        elif elapsed > 8.0:
            self.failure = (
                "Right did not retain independent support for shelf-side regrip"
            )
    elif self.stage == "DUAL_STATION_CLEAR_EMPTY_LEFT":
        goals = self.dual_station_goals.copy()
        self.bottle_commands = [0, 1]
        goals[0, :3, 3] += geometry.smoothstep(elapsed / 3.0) * (
            geometry.rotation_z(self.base_yaw()) @ np.array([-0.1, 0.18, 0.1])
        )
        (l, r) = self.dual_measure()
        if (
            elapsed >= 4.0
            and effective_hand_control(r, 0.981)
            and (l["active_hand_object_contact_count"] == 0)
        ):
            self.start_shelf_regrip()
            return (self.stage, self.second_start_goals, 65.7)
        elif elapsed > 8.0:
            self.failure = "Empty left did not clear the right-held bottle at shelf"
    elif self.stage == "DUAL_STATION_LEFT_PRELOAD":
        goals = self.dual_station_goals.copy()
        goals[0, 2, 3] += 0.01 * geometry.smoothstep(elapsed / 3.0)
        (l, r) = self.dual_measure()
        good = (
            effective_hand_control(l, 0.981)
            and l["hand_vertical_force_N"] > 0.6 * 0.981
        )
        self.dual_left_since = (
            (now if self.dual_left_since is None else self.dual_left_since)
            if good
            else None
        )
        if (
            elapsed > 3.0
            and self.dual_left_since is not None
            and (now - self.dual_left_since >= 0.4)
        ):
            self.dual_station_goals = goals.copy()
            self.enter("DUAL_STATION_OPEN_RIGHT")
        elif elapsed > 9.0:
            self.failure = "Left hand did not preload before right release at shelf"
    elif self.stage == "DUAL_STATION_OPEN_RIGHT":
        goals = self.dual_station_goals.copy()
        self.bottle_commands = [1, 0]
        (l, r) = self.dual_measure()
        if elapsed >= 3.0 and effective_hand_control(l, 0.981):
            self.dual_station_goals = goals.copy()
            self.dual_right_retreat = self.dual_R @ np.array([-0.05, -0.12, -0.08])
            self.enter("DUAL_STATION_WITHDRAW_RIGHT")
        elif elapsed > 7.0:
            self.failure = "Left did not retain real support as right opened"
    elif self.stage == "DUAL_STATION_WITHDRAW_RIGHT":
        goals = self.dual_station_goals.copy()
        self.bottle_commands = [1, 0]
        goals[1, :3, 3] += self.dual_right_retreat * geometry.smoothstep(elapsed / 3.0)
        (l, r) = self.dual_measure()
        clearance = geometry.bottle_bottom_clearance(
            self.model, self.scratch, "left", self.model.body(self.cargo_body).name
        )
        good = (
            effective_hand_control(l, 0.981)
            and r["active_hand_object_contact_count"] == 0
            and (l["object_tilt_deg"] < 30)
            and (clearance["hand_above_bottle_bottom_m"] >= 0.015)
        )
        self.dual_solo_since = (
            (now if self.dual_solo_since is None else self.dual_solo_since)
            if good
            else None
        )
        if (
            elapsed > 3.2
            and self.dual_solo_since is not None
            and (now - self.dual_solo_since >= 1.0)
        ):
            self.dual_mode = False
            self.left_normal_success = True
            self.walk_success = True
            self.place_goals = goals.copy()
            self.route_start = now
            self.carry_tilt_x = 0.0
            self.carry_tilt_y = 0.0
            self.route_arm_hold = self.executed[14:].copy()
            self.route_hold_rebased = False
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="left_only_bottom_exposed_after_dual_transport",
                    active=self.active_bottle + 1,
                    clearance=clearance,
                    bottles=self.all_bottles(),
                )
            )
            sequence.repeat_begin_place(self)
            return (self.stage, self.place_goals, 65.7)
        elif elapsed > 9.0:
            self.failure = (
                "Left-only exposed-bottom grip did not qualify after dual transport"
            )
    self.dual_current_goals = goals.copy()
    if int(round((now - self.dual_started) / 0.02)) % 100 == 0:
        (l, r) = self.dual_measure()
        print(
            json.dumps(
                dict(
                    dual_t=now - self.dual_started,
                    stage=self.stage,
                    base=self.scratch.xpos[self.base].tolist(),
                    left_N=l["hand_vertical_force_N"],
                    right_N=r["hand_vertical_force_N"],
                    tilt=l["object_tilt_deg"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def carry_loaded_targets(self):
    if self.dual_mode:
        return self.dual_targets()
    if (
        self.active_bottle == 2
        and self.place_active
        and (self.stage == "SHELF_OPEN_LEFT")
        and (not getattr(self, "third_withdraw_enabled", False))
    ):
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_RELEASE"
        self.bottle_commands = [0, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        self.post_pose[:] = self.walk_pose
        bottles = self.all_bottles()
        (l, r) = self.dual_measure()
        good = all(
            (
                b["upward_N"]["shelf"] > 0.8 * 0.981
                and b["tilt_deg"] < 8.0
                and (b["linear_speed_mps"] < 0.02)
                and (b["angular_speed_radps"] < 0.15)
                and all(
                    (n == self.model.geom(self.middle_geom).name for n in b["contacts"])
                )
                for b in bottles
            )
        )
        good = (
            good
            and l["root_speed_mps"] < 0.08
            and (l["support_feet_count"] == 2)
            and all((self.hands.controllers[k].finished for k in ["left", "right"]))
        )
        self.final_open_hold_since = (
            (
                now
                if getattr(self, "final_open_hold_since", None) is None
                else self.final_open_hold_since
            )
            if good and elapsed >= 2.0
            else None
        )
        if (
            self.final_open_hold_since is not None
            and now - self.final_open_hold_since >= 1.5
        ):
            self.completed_bottles = sorted(set(self.completed_bottles + [3]))
            self.place_success = True
            self.done = True
            self.global_stable_since = self.final_open_hold_since
            self.global_settle = True
            self.enter("ALL_THREE_BOTTLES_SHELVED_AND_HANDS_CLEAR")
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="last_bottle_actual_shelf_only_open_hand_hold_without_withdrawal",
                    bottles=bottles,
                )
            )
        elif elapsed > 9.0:
            self.failure = "Final open-hand hold did not verify all three bottles independently upright on shelf"
        return (self.stage, self.place_phase_goals.copy(), 65.7)
    if (
        self.place_active
        and self.stage == "SHELF_OPEN_LEFT"
        and (float(self.data.time) - self.stage_start >= 2.0)
    ):
        (l, r) = self.dual_measure()
        if (
            self.shelf_stable(l)
            and self.hands.controllers["left"].finished
            and (l["hand_vertical_force_N"] < 0.2 * 0.981)
        ):
            errors = self.errors(self.place_phase_goals)
            self.place_phase_goals = self.held_goals()
            self.native.follow_current(self.scratch)
            self.policy.follow_current(self.scratch)
            self.enable_carry()
            self.carry_start = float(self.data.time) - 2.0
            self.multi_events.append(
                dict(
                    time_s=float(self.data.time),
                    event="empty_hand_withdrawal_targets_begin_at_measured_wrists",
                    previous_goal_errors=errors,
                    physical_state_changed=False,
                    rolling_histories_reset=False,
                )
            )
            self.enter("SHELF_WITHDRAW_LEFT")
    if self.place_active and self.stage == "SHELF_WITHDRAW_LEFT":
        now = float(self.data.time)
        elapsed = now - self.stage_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_RETREAT"
        self.bottle_commands = [0, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        self.post_pose[:] = self.walk_pose
        key = (self.active_bottle, self.stage_start)
        if getattr(self, "left_withdraw_key", None) != key:
            self.left_withdraw_key = key
            self.left_withdraw_clock = 0.0
            away = (
                self.scratch.xpos[self.model.body("left_hand_roll_link").id, :2]
                - self.scratch.xpos[self.cargo_body, :2]
            )
            self.left_withdraw_vector = np.r_[
                0.15 * away / max(np.linalg.norm(away), 1e-09) + np.array([0.0, -0.04]),
                0.02,
            ]
        goals = self.place_phase_goals.copy()
        goals[0, :3, 3] += (
            geometry.smoothstep(self.left_withdraw_clock / 6.0)
            * self.left_withdraw_vector
        )
        e = self.errors(goals)[0]
        if e[0] < 0.004 and e[1] < 1.5:
            self.left_withdraw_clock += 0.02
        if self.left_withdraw_clock >= 6.3:
            self.place_phase_goals = goals.copy()
            self.enter("SHELF_SETTLE")
        elif elapsed > 22.0:
            self.failure = (
                "Measured radial empty-hand withdrawal did not track before timeout"
            )
        self.place_goals = goals.copy()
        return (self.stage, goals, 65.7)
    if (
        self.use_air_trim
        and self.active_bottle > 0
        and self.place_active
        and (self.stage == "SHELF_UPRIGHT_AND_ALIGN")
    ):
        return self.upright_place_targets()
    if (
        (self.shelf_regrip_active or self.active_bottle > 0)
        and self.stage == "LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL"
        and (float(self.data.time) - self.stage_start >= 9.0)
    ):
        staging_goals = self.phase_goals.copy()
        staging_goals[0] = self.normal_left_goal
        e = self.errors(staging_goals)[0]
        if e[0] < 0.012 and e[1] < 6.0:
            self.advance("LEFT_REGRASP_PREAPPROACH", self.held_goals())
    if self.shelf_regrip_active and self.stage in [
        "LEFT_REGRASP_PREAPPROACH",
        "LEFT_REGRASP_FINAL_APPROACH",
    ]:
        return self.station_grasp_targets()
    if self.shelf_regrip_active and self.stage == "LEFT_REGRASP_RIGHT_WITHDRAW":
        return self.shelf_withdraw_targets()
    if (
        self.active_bottle > 0
        and self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD"
        and (not self.shelf_regrip_active)
    ):
        now = float(self.data.time)
        if self.both_ready():
            self.dual_ready_since = (
                now if self.dual_ready_since is None else self.dual_ready_since
            )
        else:
            self.dual_ready_since = None
        if self.dual_ready_since is not None and now - self.dual_ready_since >= 0.4:
            self.begin_dual()
            return (self.stage, self.dual_goals, 65.7)
        if self.dual_pending:
            self.bottle_commands = [1, 1]
            self.native_clock = 65.7
            self.native_start = now - 65.7
            self.source_phase = "PICK_LIFT_OUT"
            self.post_twist[:] = 0.0
            self.walk_twist[:] = 0.0
            if self.dual_ready_since is None and (
                not hasattr(self, "pending_wait_start")
            ):
                self.pending_wait_start = now
            if (
                hasattr(self, "pending_wait_start")
                and now - self.pending_wait_start > 6.0
            ):
                self.failure = (
                    "Initialized dual grasp did not show measured shared load"
                )
            return (self.stage, self.held_goals(), 65.7)
    if self.active_bottle > 0 and self.stage == "HANDOFF_LEFT_OPEN":
        (l, r) = self.dual_measure()
        elapsed = float(self.data.time) - self.stage_start
        if (
            elapsed >= 3.5
            and self.hands.controllers["left"].finished
            and (abs(l["hand_vertical_force_N"]) < 0.1 * 0.981)
            and effective_hand_control(r, 0.981)
            and (r["hand_vertical_force_N"] > 0.8 * 0.981)
        ):
            self.advance("HANDOFF_LEFT_WITHDRAW", self.held_goals())
    result = sequence.repeat_loaded_targets(self)
    if self.shelf_regrip_active:
        self.post_pose[:] = [0.82, 0.12]
        self.walk_pose[:] = self.post_pose
    return result


def carry_start_next_pick(self):
    sequence.repeat_start_next_pick(self)
    if not self.empty_return:
        self.repeat_right_park = self.held_goals()[1]
        self.right_park_program_bottle = -1


def carry_rotate_goal(self, goals, clock):
    old = self.repeat_grasp_depth
    if self.active_bottle == 2:
        self.repeat_grasp_depth = 0.002
    try:
        result = sequence.repeat_rotate_goal(self, goals, clock)
    finally:
        self.repeat_grasp_depth = old
    if (
        result is not None
        and self.active_bottle > 0
        and (self.stage == "BOTTLE_PROGRAM")
    ):
        torso = np.eye(4)
        bid = self.model.body("waist_pitch_link").id
        torso[:3, :3] = self.scratch.xmat[bid].reshape(3, 3)
        torso[:3, 3] = self.scratch.xpos[bid]
        if self.right_park_program_bottle != self.active_bottle:
            self.right_park_program_bottle = self.active_bottle
            self.right_park_relative = np.linalg.inv(torso) @ self.held_goals()[1]
            self.right_park_joints = self.scratch.qpos[self.upper_pos[9:]].copy()
        result[1] = torso @ self.right_park_relative
        if hasattr(self, "repeat_seed_positions"):
            self.repeat_seed_positions[:, 9:] = self.right_park_joints
    return result


def carry_station_object_frame(self):
    obj = self.object_pose().copy()
    target = obj @ self.station_grasp
    heading = np.arctan2(target[1, 0], target[0, 0])
    desired = self.base_yaw() + np.deg2rad(-60.0)
    obj[:3, :3] = obj[:3, :3] @ geometry.rotation_z(
        geometry.wrap_angle(desired - heading)
    )
    return obj


def carry_station_grasp_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [0, 1]
    self.post_twist[:] = 0.0
    self.walk_twist[:] = 0.0
    self.post_pose[:] = [0.82, 0.12]
    self.walk_pose[:] = self.post_pose
    pre = self.stage == "LEFT_REGRASP_PREAPPROACH"
    key = "pre" if pre else "final"
    times = self.station_path[key + "_time_s"]
    path = self.station_path[key + "_T_object_wrist"]
    bridge = 2.0 if pre else 0.0
    clock = np.clip((elapsed - bridge) * 0.7, 0.0, times[-1])
    i = min(int(np.searchsorted(times, clock, side="right")), len(times) - 1)
    j = max(i - 1, 0)
    f = float((clock - times[j]) / max(times[i] - times[j], 1e-09)) if i != j else 0.0
    relative = geometry.mix_transform(path[j], path[i], f)
    target = self.station_object_frame() @ relative
    if not pre and self.active_bottle == 0:
        target[:3, 3] += (
            0.01 * geometry.smoothstep(clock / max(times[-1], 1e-09)) * target[:3, 0]
        )
    goals = self.phase_goals.copy()
    goals[0] = (
        geometry.mix_transform(
            self.phase_goals[0], target, geometry.smoothstep(elapsed / bridge)
        )
        if bridge and elapsed < bridge
        else target
    )
    duration = bridge + times[-1] / 0.7
    err = self.errors(goals)[0]
    good = elapsed >= duration + 0.4 and err[0] < 0.0015 and (err[1] < 1.0)
    self.move_good = (
        (now if self.move_good is None else self.move_good) if good else None
    )
    if self.move_good is not None and now - self.move_good >= 0.4:
        self.advance(
            "LEFT_REGRASP_FINAL_APPROACH" if pre else "LEFT_REGRASP_CLOSE_AND_LOAD",
            goals,
        )
    elif elapsed > duration + 8.0:
        self.failure = "Object-relative known-good wrist path did not align"
    self.post_goals = goals.copy()
    return (self.stage, goals, 65.7)


def carry_upright_place_targets(self):
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
    obj[:3, :3] = geometry.rotation_z(yaw + np.deg2rad(self.object_yaw_offset))
    obj[:3, 3] = [*self.target_xy, self.middle_height + 0.105]
    target = obj @ self.object_wrist
    goals = self.place_phase_goals.copy()
    goals[0] = geometry.mix_transform(
        self.place_phase_goals[0], target, geometry.smoothstep(elapsed / self.move_s)
    )
    if elapsed >= self.move_s:
        if getattr(self, "air_trim_bottle", -1) != self.active_bottle:
            self.air_trim_bottle = self.active_bottle
            self.air_trim_goals = goals.copy()
            self.air_trim_since = None
        goals = self.air_trim_goals.copy()
        axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
        vector = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        angle = min(float(np.arctan2(np.linalg.norm(vector), axis[2])), 0.1 * 0.02)
        if np.linalg.norm(vector) > 1e-08:
            (x, y, z) = vector / np.linalg.norm(vector)
            K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            pivot = self.scratch.xpos[self.cargo_body].copy()
            goals[0, :3, 3] = pivot + rotation @ (goals[0, :3, 3] - pivot)
            goals[0, :3, :3] = rotation @ goals[0, :3, :3]
        self.air_trim_goals = goals.copy()
        (left, _) = self.dual_measure()
        err = self.errors(goals)[0]
        good = (
            left["object_tilt_deg"] < 3.0
            and left["object_angular_speed_radps"] < 0.15
            and (left["object_linear_speed_mps"] < 0.03)
            and effective_hand_control(left, 0.981)
            and (err[0] < 0.008)
            and (err[1] < 4.0)
        )
        self.air_trim_since = (
            (now if self.air_trim_since is None else self.air_trim_since)
            if good
            else None
        )
        if self.air_trim_since is not None and now - self.air_trim_since >= 0.3:
            self.place_phase_goals = goals.copy()
            self.place_lower_goal = goals.copy()
            self.place_lower_start = now
            self.enter("SHELF_LOWER_TO_ACTUAL_CONTACT")
        elif elapsed > self.move_s + 8.0:
            self.failure = (
                "Actual free bottle axis did not become upright before lowering"
            )
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def carry_start_shelf_regrip(self):
    self.dual_mode = False
    self.dual_pending = False
    self.shelf_regrip_active = True
    self.post_goals = self.held_goals()
    self.upper_pose = self.scratch.qpos[self.upper_pos].copy()
    regrasp_start_second(self)
    self.left_offset = np.array([0.16, -0.055, -0.01])
    target_grasp = self.station_object_frame() @ self.station_grasp
    self.left_normal_R = target_grasp[:3, :3].copy()
    self.normal_left_goal[:3, :3] = self.left_normal_R
    target = self.object_pose()
    target[:3, 3] = self.second_base + self.second_R @ np.array(
        [
            (
                getattr(self, "first_station_forward", 0.48)
                if self.active_bottle == 0
                else 0.52
            ),
            (
                getattr(self, "first_station_lateral", 0.1)
                if self.active_bottle == 0
                else 0.1
            ),
            1.3 - self.second_base[2],
        ]
    )
    self.second_right_goal = target @ self.right_relation
    self.left_load_since = None
    self.left_solo_since = None
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="right_supported_shelf_side_left_regrip_begin",
            active=self.active_bottle + 1,
            bottles=self.all_bottles(),
        )
    )


def carry_shelf_withdraw_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [1, 0]
    self.post_twist[:] = 0.0
    self.walk_twist[:] = 0.0
    self.post_pose[:] = [0.82, 0.12]
    goals = self.phase_goals.copy()
    key = (self.active_bottle, self.stage_start)
    if getattr(self, "shelf_withdraw_key", None) != key:
        self.shelf_withdraw_key = key
        away = (
            self.scratch.xpos[self.model.body("right_hand_roll_link").id, :2]
            - self.scratch.xpos[self.cargo_body, :2]
        )
        self.shelf_withdraw_vector = np.r_[
            0.1 * away / max(np.linalg.norm(away), 1e-09) + np.array([0.0, -0.06]), 0.0
        ]
    goals[1, :3, 3] += geometry.smoothstep(elapsed / 4.0) * self.shelf_withdraw_vector
    (l, r) = self.dual_measure()
    clear = geometry.bottle_bottom_clearance(
        self.model, self.scratch, "left", self.model.body(self.cargo_body).name
    )
    good = (
        effective_hand_control(l, 0.981)
        and (not l["delivery_table_contact"])
        and (r["active_hand_object_contact_count"] == 0)
        and (clear["hand_above_bottle_bottom_m"] >= 0.015)
        and (l["object_tilt_deg"] < 30.0)
    )
    self.left_solo_since = (
        (now if self.left_solo_since is None else self.left_solo_since)
        if good
        else None
    )
    self.post_goals = goals.copy()
    if (
        elapsed >= 5.0
        and self.left_solo_since is not None
        and (now - self.left_solo_since >= 1.0)
    ):
        self.left_normal_success = True
        self.multi_events.append(
            dict(
                time_s=now,
                event="shelf_side_left_only_bottom_exposed_verified",
                active=self.active_bottle + 1,
                clearance=clear,
                bottles=self.all_bottles(),
            )
        )
        self.begin_walk()
        return (self.stage, self.place_goals, 65.7)
    elif elapsed > 10.0:
        self.failure = "Shelf-side left normal grip did not qualify"
    return (self.stage, goals, 65.7)


def carry_begin_walk(self):
    if getattr(self, "shelf_regrip_active", False):
        self.shelf_regrip_active = False
        self.dual_mode = False
        self.done = False
        self.walk_success = True
        self.side_route_active = False
        self.extra_active = True
        self.route_start = float(self.data.time)
        self.route_arm_hold = self.executed[14:].copy()
        self.route_hold_rebased = False
        self.carry_tilt_x = 0.0
        self.carry_tilt_y = 0.0
        sequence.repeat_begin_place(self)
    else:
        sequence.repeat_begin_walk(self)
        if self.active_bottle == 0:
            before = {
                (label, n): value.copy()
                for (label, p) in [
                    ("loco", self.loco),
                    ("reach", self.policy),
                    ("native", self.native),
                ]
                for (n, value) in p.histories.items()
            }
            actions = [
                p.last_action.copy() for p in [self.loco, self.policy, self.native]
            ]
            physical = np.r_[self.data.qpos, self.data.qvel].copy()
            self.native.follow_current(self.scratch)
            self.policy.follow_current(self.scratch)
            for label, p in [
                ("loco", self.loco),
                ("reach", self.policy),
                ("native", self.native),
            ]:
                for n, value in p.histories.items():
                    assert np.array_equal(before[label, n], value)
                p.q_des = (
                    self.executed[:14].copy()
                    if label == "loco"
                    else self.executed.copy()
                )
            assert all(
                (
                    np.array_equal(a, p.last_action)
                    for (a, p) in zip(actions, [self.loco, self.policy, self.native])
                )
            )
            assert np.array_equal(physical, np.r_[self.data.qpos, self.data.qvel])
            self.multi_events.append(
                dict(
                    time_s=float(self.data.time),
                    event="carry_wrist_reference_initialized_to_measured_pose",
                    physical_state_changed=False,
                    real_histories_unchanged=True,
                    last_actions_unchanged=True,
                )
            )


def regrasp_start_second(self):
    self.done = False
    self.second_start = float(self.data.time)
    self.second_start_goals = self.post_goals.copy()
    self.second_R = geometry.rotation_z(self.base_yaw())
    self.second_base = self.scratch.xpos[self.base].copy()
    obj = self.object_pose()
    self.right_relation = np.linalg.inv(obj) @ self.second_start_goals[1]
    target = obj.copy()
    target[:3, 3] = self.second_base + self.second_R @ np.array(
        [0.62, 0.0, 1.3 - self.second_base[2]]
    )
    self.second_right_goal = target @ self.right_relation
    self.clear_left_goal = self.second_start_goals[0].copy()
    self.clear_left_goal[:3, 3] = self.second_base + self.second_R @ np.array(
        [0.38, 0.3, 1.4 - self.second_base[2]]
    )
    self.normal_left_goal = self.clear_left_goal.copy()
    self.normal_left_goal[:3, :3] = self.second_R @ geometry.rotation_z(
        np.deg2rad(-60.0)
    )
    self.left_offset = np.array([0.175, -0.07, self.left_zoffset])
    self.left_normal_R = self.normal_left_goal[:3, :3].copy()
    self.second_initial_upper = self.scratch.qpos[self.upper_pos].copy()
    self.second_normal_seed = self.second_initial_upper.copy()
    self.second_normal_seed[2:9] = self.second_initial_upper[9:16] * np.array(
        [1, -1, -1, 1, -1, 1, -1]
    )
    self.second_IK_values = self.second_initial_upper.copy()
    self.second_upper_pose = self.upper_pose.copy()
    self.second_plan_time = None
    self.second_boundary = dict(
        time_s=self.second_start,
        qpos=self.data.qpos.tolist(),
        qvel=self.data.qvel.tolist(),
    )
    self.handoff_owner = "right"
    self.advance("LEFT_REGRASP_CLEAR_AND_RIGHT_RAISE", self.second_start_goals)
    self.second_events.append(
        dict(
            time_s=self.second_start,
            event="start_right_to_left_normal_regrasp",
            left_grip_offset_wrist_m=self.left_offset.tolist(),
            physical_state_continuous=True,
        )
    )


def regrasp_ready_move(self, goals, elapsed, min_duration, side):
    errors = self.errors(goals)
    good = errors[side][0] < 0.004 and errors[side][1] < 3.0
    now = float(self.data.time)
    self.move_good = (
        (now if self.move_good is None else self.move_good)
        if good and elapsed >= min_duration
        else None
    )
    return self.move_good is not None and now - self.move_good >= 0.4


def regrasp_loaded_targets(self):
    if self.second_start is None:
        result = handoff_loaded_targets(self)
        if self.done and self.handoff_success:
            self.start_second()
            return (self.stage, self.second_start_goals, 65.7)
        return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_start = now - 65.7
    self.native_clock = 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.post_twist[:] = 0.0
    self.post_pose[:] = [0.82, 0.35]
    self.bottle_commands = [0, 1]
    goals = self.phase_goals.copy()
    if self.stage == "LEFT_REGRASP_CLEAR_AND_RIGHT_RAISE":
        goals[0] = geometry.mix_transform(
            self.second_start_goals[0],
            self.clear_left_goal,
            geometry.smoothstep(elapsed / 7.0),
        )
        goals[1] = geometry.mix_transform(
            self.second_start_goals[1],
            self.second_right_goal,
            geometry.smoothstep(elapsed / 7.0),
        )
        if self.ready_move(goals, elapsed, 8.0, 1):
            self.advance("LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL", goals)
        elif elapsed > 16.0:
            self.failure = "Right held bottle did not reach regrasp station"
    elif self.stage == "LEFT_REGRASP_ROTATE_EMPTY_TO_NORMAL":
        goals[0] = geometry.mix_transform(
            self.phase_goals[0],
            self.normal_left_goal,
            geometry.smoothstep(elapsed / 8.0),
        )
        if self.ready_move(goals, elapsed, 9.0, 0):
            self.advance("LEFT_REGRASP_PREAPPROACH", goals)
        elif elapsed > 19.0:
            self.failure = "Empty left arm did not reach normal grasp preparation"
    elif self.stage in ("LEFT_REGRASP_PREAPPROACH", "LEFT_REGRASP_FINAL_APPROACH"):
        target = np.eye(4)
        target[:3, :3] = self.left_normal_R
        target[:3, 3] = (
            self.object_pose()[:3, 3] - self.left_normal_R @ self.left_offset
        )
        pre = self.stage == "LEFT_REGRASP_PREAPPROACH"
        if pre:
            target[:3, 3] += self.second_R @ np.array([0.0, 0.065, 0.025])
        duration = 6.0 if pre else 4.0
        goals[0] = geometry.mix_transform(
            self.phase_goals[0], target, geometry.smoothstep(elapsed / duration)
        )
        if self.ready_move(goals, elapsed, duration + 1.0, 0):
            self.advance(
                "LEFT_REGRASP_FINAL_APPROACH" if pre else "LEFT_REGRASP_CLOSE_AND_LOAD",
                goals,
            )
        elif elapsed > duration + 10.0:
            self.failure = "Left normal grasp did not align with actual bottle"
    elif self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD":
        self.bottle_commands = [1, 1]
        goals[0, 2, 3] += 0.007 * geometry.smoothstep((elapsed - 3.0) / 2.0)
        (left, right) = self.dual_measure()
        loaded = (
            bool(effective_hand_control(left, 0.981))
            and (not left["delivery_table_contact"])
            and (left["active_hand_object_contact_count"] >= 2)
        )
        self.left_load_since = (
            (now if self.left_load_since is None else self.left_load_since)
            if loaded
            else None
        )
        if (
            elapsed >= 3.0
            and self.left_load_since is not None
            and (now - self.left_load_since >= 0.4)
        ):
            self.second_events.append(
                dict(
                    time_s=now,
                    event="left_actual_load_before_right_open",
                    left_contacts=left["active_hand_object_contact_count"],
                    left_vertical_force_N=left["hand_vertical_force_N"],
                    right_contacts=right["active_hand_object_contact_count"],
                )
            )
            self.handoff_owner = "left"
            self.advance("LEFT_REGRASP_RIGHT_OPEN", goals)
        elif elapsed > 10.0:
            self.failure = (
                "Left normal grasp did not bear actual load; right kept closed"
            )
    elif self.stage == "LEFT_REGRASP_RIGHT_OPEN":
        self.bottle_commands = [1, 0]
        if elapsed > 3.5:
            (left, right) = self.dual_measure()
            if right["active_hand_object_contact_count"] == 0 and bool(
                effective_hand_control(left, 0.981)
            ):
                self.advance("LEFT_REGRASP_RIGHT_WITHDRAW", goals)
            elif elapsed > 8.0:
                self.failure = (
                    "Right did not clear while left independently held bottle"
                )
    elif self.stage == "LEFT_REGRASP_RIGHT_WITHDRAW":
        self.bottle_commands = [1, 0]
        goals[1, :3, 3] += geometry.smoothstep(elapsed / 4.0) * (
            self.second_R @ np.array([-0.05, -0.1, 0.05])
        )
        (left, right) = self.dual_measure()
        clear = self.bottle_clearance(self.model, self.scratch, "left")
        ok = (
            bool(effective_hand_control(left, 0.981))
            and (not left["delivery_table_contact"])
            and (right["active_hand_object_contact_count"] == 0)
            and (clear["hand_above_bottle_bottom_m"] >= 0.015)
            and (left["object_tilt_deg"] < 30.0)
        )
        self.left_solo_since = (
            (now if self.left_solo_since is None else self.left_solo_since)
            if ok
            else None
        )
        if (
            elapsed >= 5.0
            and self.left_solo_since is not None
            and (now - self.left_solo_since >= 1.0)
        ):
            self.left_normal_success = True
            self.done = True
            self.enter("LEFT_NORMAL_GRASP_BOTTOM_EXPOSED_COMPLETE")
        elif elapsed > 10.0:
            self.failure = "Left held but exposed-bottom normal grasp did not qualify"
    self.post_goals = goals.copy()
    if int(round((now - self.second_start) / 0.02)) % 100 == 0:
        (left, right) = self.dual_measure()
        print(
            json.dumps(
                dict(
                    second_t=now - self.second_start,
                    stage=self.stage,
                    errors=self.errors(goals),
                    left_contacts=left["active_hand_object_contact_count"],
                    right_contacts=right["active_hand_object_contact_count"],
                    left_load_N=left["hand_vertical_force_N"],
                    right_load_N=right["hand_vertical_force_N"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def handoff_object_pose(self):
    T = np.eye(4)
    T[:3, :3] = self.scratch.xmat[self.cargo_body].reshape(3, 3)
    T[:3, 3] = self.scratch.xpos[self.cargo_body]
    return T


def handoff_start_post(self):
    carry_body_start_post(self)
    self.held_walk_started = False
    self.start_goals = self.held_goals()
    self.object_start = self.object_pose()
    self.object_to_left = np.linalg.inv(self.object_start) @ self.start_goals[0]
    self.object_goal = self.object_start.copy()
    self.object_goal[:3, 3] = self.post_base + self.post_R @ np.array(
        [self.bottle_x, self.bottle_y, 1.26 - self.post_base[2]]
    )
    self.left_grasp_goal = self.object_goal @ self.object_to_left
    self.right_R = self.post_R @ geometry.rotation_z(np.deg2rad(self.right_grasp_yaw))
    self.right_offset = np.array([0.152, 0.028, self.handoff_zoffset])
    self.right_approach_start = None
    self.right_final_start = None
    self.right_close_start = None
    self.left_open_start = None
    self.enter("HANDOFF_MOVE_LEFT_CLEAR_OF_BOX")


def handoff_errors(self, goals):
    out = []
    for side, T in zip(("left", "right"), goals):
        (p, q) = self.native.endpoint_pose(self.scratch, side)
        tq = np.empty(4)
        mujoco.mju_mat2Quat(tq, T[:3, :3].ravel())
        out.append(
            [
                float(np.linalg.norm(p - T[:3, 3])),
                float(np.degrees(np.linalg.norm(geometry.rotation_error(tq, q)))),
            ]
        )
    return out


def handoff_advance(self, stage, goals):
    self.phase_goals = goals.copy()
    self.move_good = None
    self.enter(stage)


def handoff_loaded_targets(self):
    if self.post_start is None:
        result = grasp.grasp_frame_loaded_targets(self)
        if self.done and self.extraction_success:
            self.start_post()
            return (self.stage, self.start_goals, 65.7)
        return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_start = now - 65.7
    self.native_clock = 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.post_twist[:] = 0
    self.post_pose[:] = [0.82, 0.35]
    self.bottle_commands = [1, 0]
    goals = self.post_goals.copy()
    if self.stage == "HANDOFF_MOVE_LEFT_CLEAR_OF_BOX":
        goals = self.start_goals.copy()
        goals[0] = geometry.mix_transform(
            self.start_goals[0],
            self.left_grasp_goal,
            geometry.smoothstep(elapsed / 8.0),
        )
        if elapsed >= 9.0:
            err = self.errors(goals)
            good = err[0][0] < 0.005 and err[0][1] < 3.0
            self.move_good = (
                (now if self.move_good is None else self.move_good) if good else None
            )
            if self.move_good is not None and now - self.move_good >= 0.5:
                self.advance("HANDOFF_RIGHT_PREAPPROACH", goals)
            elif elapsed > 18.0:
                self.failure = "Left held bottle did not reach safe handoff workpoint"
    elif self.stage in ("HANDOFF_RIGHT_PREAPPROACH", "HANDOFF_RIGHT_FINAL_APPROACH"):
        obj = self.object_pose()
        target = np.eye(4)
        target[:3, :3] = self.right_R
        target[:3, 3] = obj[:3, 3] - self.right_R @ self.right_offset
        if self.stage == "HANDOFF_RIGHT_PREAPPROACH":
            target[:3, 3] += self.post_R @ np.array([0.0, -0.065, 0.03])
        goals = self.phase_goals.copy()
        duration = 6.0 if self.stage == "HANDOFF_RIGHT_PREAPPROACH" else 4.0
        goals[1] = geometry.mix_transform(
            self.phase_goals[1], target, geometry.smoothstep(elapsed / duration)
        )
        if elapsed >= duration + 1.0:
            err = self.errors(goals)
            good = err[1][0] < 0.004 and err[1][1] < 3.0
            self.move_good = (
                (now if self.move_good is None else self.move_good) if good else None
            )
            if self.move_good is not None and now - self.move_good >= 0.4:
                self.advance(
                    (
                        "HANDOFF_RIGHT_FINAL_APPROACH"
                        if self.stage == "HANDOFF_RIGHT_PREAPPROACH"
                        else "HANDOFF_RIGHT_CLOSE_AND_LOAD"
                    ),
                    goals,
                )
            elif elapsed > duration + 10.0:
                self.failure = (
                    "Right wrist did not align with actual bottle before closing"
                )
    elif self.stage == "HANDOFF_RIGHT_CLOSE_AND_LOAD":
        self.bottle_commands = [1, 1]
        goals = self.phase_goals.copy()
        goals[1, 2, 3] += 0.007 * geometry.smoothstep((elapsed - 3.0) / 2.0)
        (left, right) = self.dual_measure()
        loaded = (
            right["active_hand_object_contact_count"] >= 2
            and right["hand_vertical_force_N"] > 0.2 * 0.981
            and (
                not any(
                    (
                        right[k]
                        for k in (
                            "crate_object_contact",
                            "table_contact",
                            "delivery_table_contact",
                            "floor_contact",
                        )
                    )
                )
            )
        )
        self.right_loaded_since = (
            (now if self.right_loaded_since is None else self.right_loaded_since)
            if loaded
            else None
        )
        if (
            elapsed >= 3.0
            and self.right_loaded_since is not None
            and (now - self.right_loaded_since >= 0.4)
        ):
            self.transfer_events.append(
                dict(
                    time_s=now,
                    event="right_actual_load_gate_before_left_open",
                    right_contacts=right["active_hand_object_contact_count"],
                    right_upward_force_N=right["hand_vertical_force_N"],
                    left_contacts=left["active_hand_object_contact_count"],
                )
            )
            self.handoff_owner = "right"
            self.advance("HANDOFF_LEFT_OPEN", goals)
        elif elapsed > 10.0:
            self.failure = "Right did not establish real airborne load before left release; left kept closed"
    elif self.stage == "HANDOFF_LEFT_OPEN":
        self.bottle_commands = [0, 1]
        goals = self.phase_goals.copy()
        if elapsed > 3.5:
            (left, right) = self.dual_measure()
            if left["active_hand_object_contact_count"] == 0 and bool(
                effective_hand_control(right, 0.981)
            ):
                self.advance("HANDOFF_LEFT_WITHDRAW", goals)
            elif elapsed > 8.0:
                self.failure = (
                    "Left did not release while right retained actual bottle control"
                )
    elif self.stage == "HANDOFF_LEFT_WITHDRAW":
        self.bottle_commands = [0, 1]
        goals = self.phase_goals.copy()
        goals[0, :3, 3] += geometry.smoothstep(elapsed / 4.0) * (
            self.post_R @ np.array([-0.04, 0.08, 0.08])
        )
        (left, right) = self.dual_measure()
        ok = (
            left["active_hand_object_contact_count"] == 0
            and bool(effective_hand_control(right, 0.981))
            and (not right["delivery_table_contact"])
        )
        self.right_solo_since = (
            (now if self.right_solo_since is None else self.right_solo_since)
            if ok
            else None
        )
        if (
            elapsed >= 5.0
            and self.right_solo_since is not None
            and (now - self.right_solo_since >= 1.0)
        ):
            self.handoff_success = True
            self.done = True
            self.enter("RIGHT_NORMAL_GRASP_TRANSFER_COMPLETE")
        elif elapsed > 9.0:
            self.failure = "Right hand did not independently hold after left withdrawal"
    self.post_goals = goals.copy()
    if int(round((now - self.post_start) / 0.02)) % 100 == 0:
        (left, right) = self.dual_measure()
        print(
            json.dumps(
                dict(
                    post_t=now - self.post_start,
                    stage=self.stage,
                    errors=self.errors(goals),
                    left_contacts=left["active_hand_object_contact_count"],
                    right_contacts=right["active_hand_object_contact_count"],
                    left_load_N=left["hand_vertical_force_N"],
                    right_load_N=right["hand_vertical_force_N"],
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 65.7)


def carry_body_start_post(self):
    self.post_start = float(self.data.time)
    self.done = False
    self.post_goals = self.held_goals()
    self.post_arms = self.executed[14:].copy()
    self.post_twist = np.zeros(3)
    self.post_pose = np.array([0.82, self.pick_pitch])
    self.post_base = self.scratch.xpos[self.base].copy()
    self.post_yaw = self.base_yaw()
    self.post_R = geometry.rotation_z(self.post_yaw)
    self.post_terminal = dict(
        time_s=self.post_start,
        qpos=self.data.qpos.tolist(),
        qvel=self.data.qvel.tolist(),
    )
    self.enter("BOTTLE_LOCO_HELD_START")
    self.held_walk_started = True
    self.transitions.append(
        dict(
            time_s=self.post_start,
            continuation=self.post_mode,
            body_controller="Loco14 already active; start velocity commands",
            arms="Held actual previously executed PD targets",
            physical_state_continuous=True,
            rolling_history_reset=False,
        )
    )
