"""Empty-hand clearance and the two physical return routes."""

from __future__ import annotations
import numpy as np
from . import geometry, placement, sequence


def departure_begin_open_back(self):
    open_retreat_begin_open_back(self)
    self.open_back_active = False
    self.departure_clear_active = True
    self.departure_clear_since = None
    self.departure_clear_goals = self.held_goals()
    away = self.departure_clear_goals[0, :2, 3] - self.scratch.xpos[self.cargo_body, :2]
    self.departure_clear_vector = np.r_[
        0.13 * away / max(np.linalg.norm(away), 1e-09), 0.015
    ]
    self.enter("THIRD_OPEN_HAND_RADIAL_CLEAR")


def departure_loaded_targets(self):
    if not self.departure_clear_active:
        return placement.support_feedback_loaded_targets(self)
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
    goals = self.departure_clear_goals.copy()
    goals[0, :3, 3] += geometry.smoothstep(elapsed / 4.0) * self.departure_clear_vector
    bottles = self.all_bottles()
    (l, r) = self.dual_measure()
    good = all(
        (
            b["upward_N"]["shelf"] > 0.8 * 0.981
            and b["tilt_deg"] < 8.0
            and (b["linear_speed_mps"] < 0.02)
            and (b["angular_speed_radps"] < 0.15)
            and b["contacts"]
            and all((n == "rack_level_1" for n in b["contacts"]))
            for b in bottles
        )
    )
    distance = np.linalg.norm(
        self.held_goals()[0, :3, 3] - self.scratch.xpos[self.cargo_body]
    )
    good = (
        good
        and distance > 0.22
        and (l["root_speed_mps"] < 0.08)
        and (l["support_feet_count"] == 2)
    )
    self.departure_clear_since = (
        (now if self.departure_clear_since is None else self.departure_clear_since)
        if good and elapsed > 4.0
        else None
    )
    if (
        self.departure_clear_since is not None
        and now - self.departure_clear_since > 0.6
    ):
        self.departure_clear_active = False
        self.open_back_active = True
        self.open_back_goals = self.held_goals()
        goals = self.open_back_goals.copy()
        self.open_back_arms = self.executed[14:].copy()
        self.multi_events.append(
            dict(
                time_s=now,
                event="third_open_hand_radially_clear_before_body_rise",
                actual_wrist_bottle_distance_m=float(distance),
                bottles=bottles,
                physical_state_changed=False,
            )
        )
        self.enter("FINAL_OPEN_HAND_LOCO_PREP")
    elif elapsed > 14.0:
        self.failure = (
            "Open hand did not clear while all three remained independently upright"
        )
    self.place_goals = goals.copy()
    return (self.stage, goals, 65.7)


def return_offset_start_return(self):
    placement.return_route_start_return(self)
    if self.active_bottle == 2:
        self.empty_target += self.source_third_return_delta
        self.empty_brake_target += self.source_third_return_delta
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_real_loco_return_workspace_goal",
                extra_goal_delta_world_xy_m=self.source_third_return_delta.tolist(),
                target_xy_m=self.empty_target.tolist(),
                physical_state_changed=False,
            )
        )


def supported_retreat_loaded_targets(self):
    result = placement.shelf_entry_loaded_targets(self)
    if (
        self.active_bottle == 2
        and self.stage == "SHELF_OPEN_LEFT"
        and (not self.open_back_started)
    ):
        now = float(self.data.time)
        bottles = self.all_bottles()
        (l, r) = self.dual_measure()
        good = now - self.stage_start >= 2.5 and all(
            (self.hands.controllers[k].finished for k in ["left", "right"])
        )
        good = good and all(
            (
                b["upward_N"]["shelf"] > 0.8 * 0.981
                and b["tilt_deg"] < 8.0
                and (b["linear_speed_mps"] < 0.02)
                and (b["angular_speed_radps"] < 0.15)
                and (b["upward_N"]["left"] < 0.2 * 0.981)
                and (b["upward_N"]["right"] < 0.2 * 0.981)
                and (b["upward_N"]["floor"] == 0.0)
                and (b["upward_N"]["crate"] == 0.0)
                for b in bottles
            )
        )
        good = good and l["root_speed_mps"] < 0.08 and (l["support_feet_count"] == 2)
        self.supported_release_since = (
            (
                now
                if self.supported_release_since is None
                else self.supported_release_since
            )
            if good
            else None
        )
        if (
            self.supported_release_since is not None
            and now - self.supported_release_since >= 0.5
        ):
            self.supported_retreat_used = True
            self.begin_open_back()
            self.multi_events[-1][
                "event"
            ] = "three_actual_shelf_supported_open_hand_retreat_begin_with_brush_allowed"
            self.multi_events[-1].update(
                actual_left_contact_count=l["active_hand_object_contact_count"],
                actual_left_Fz_N=l["hand_vertical_force_N"],
                final_hand_free_settle_still_required=True,
            )
            return (self.stage, self.open_back_goals.copy(), 65.7)
    return result


def supported_retreat_step(self):
    placement.third_return_step(self)
    if (
        self.open_back_completed
        and (not self.failure)
        and self.supported_retreat_used
        and (3 not in self.completed_bottles)
    ):
        self.completed_bottles = sorted(self.completed_bottles + [3])
        self.place_success = True
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="bottle_actual_shelf_support_and_withdrawal_complete",
                active=3,
                bottles=self.all_bottles(),
                release_rule="True shelf load, open fingers and controlled actual Loco withdrawal; final hands-clear settled support verified",
            )
        )


def open_retreat_begin_open_back(self):
    self.open_back_started = True
    self.open_back_active = True
    self.done = False
    self.global_settle = False
    self.open_back_origin = self.scratch.xpos[self.base].copy()
    self.open_back_yaw = self.base_yaw()
    self.open_back_direction = -np.array(
        [np.cos(self.open_back_yaw), np.sin(self.open_back_yaw)]
    )
    self.open_back_arms = self.executed[14:].copy()
    self.open_back_goals = self.held_goals()
    self.open_back_bottle_origins = np.array(
        [b["position_m"] for b in self.all_bottles()]
    )
    self.open_back_tilts = np.zeros(3)
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="all_three_verified_then_open_hand_loco_back_begin",
            base_xyz_m=self.open_back_origin.tolist(),
            bottles=self.all_bottles(),
        )
    )
    self.enter("FINAL_OPEN_HAND_LOCO_PREP")


def open_retreat_step(self):
    sequence.inventory_step(self)
    if (
        self.done
        and (not self.failure)
        and (self.stage == "ALL_THREE_BOTTLES_SHELVED_AND_HANDS_CLEAR")
        and (not self.open_back_started)
    ):
        self.begin_open_back()


def open_retreat_loaded_targets(self):
    if not self.open_back_active:
        result = placement.third_place_loaded_targets(self)
        if self.stage == "THIRD_FRONT_ALIGNMENT":
            now = float(self.data.time)
            elapsed = now - self.stage_start
            if self.front_step_origin is None:
                self.front_step_origin = self.scratch.xpos[self.base].copy()
            if (
                elapsed > 5.5
                and self.scratch.xpos[self.cargo_body, 1] < 2.137
                and (self.scratch.xpos[self.base, 1] - self.front_step_origin[1] < 0.08)
            ):
                world = np.array([0.0, 0.06])
                local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ world
                cmd = np.r_[local, 0.0]
                self.walk_twist[:] = cmd
                self.post_twist[:] = cmd
        return result
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_RETREAT"
    self.bottle_commands = [0, 0]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    delta = self.scratch.xpos[self.base] - self.open_back_origin
    back = float(delta[:2] @ self.open_back_direction)
    bottles = self.all_bottles()
    self.open_back_tilts = np.maximum(
        self.open_back_tilts, [b["tilt_deg"] for b in bottles]
    )
    if any((b["tilt_deg"] > 35.0 for b in bottles)):
        self.failure = "Bottle tipped during open-hand Loco withdrawal"
    if self.stage == "FINAL_OPEN_HAND_LOCO_PREP":
        f = geometry.smoothstep(elapsed / 3.0)
        self.walk_pose[:] = [
            (1 - f) * self.placement_height + f * 0.82,
            (1 - f) * self.placement_pitch + f * 0.12,
        ]
        self.post_pose[:] = self.walk_pose
        if elapsed > 3.5:
            self.enter("FINAL_OPEN_HAND_LOCO_BACK")
    elif self.stage == "FINAL_OPEN_HAND_LOCO_BACK":
        cmd = np.array(
            [
                -0.22 * geometry.smoothstep(elapsed / 1.2),
                0.0,
                np.clip(
                    0.8 * geometry.wrap_angle(self.open_back_yaw - self.base_yaw()),
                    -0.12,
                    0.12,
                ),
            ]
        )
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        if back >= 0.15:
            self.open_back_brake = cmd.copy()
            self.enter("FINAL_OPEN_HAND_LOCO_BRAKE")
        elif elapsed > 12.0:
            self.failure = "Open-hand Loco did not reach actual backward distance"
    elif self.stage == "FINAL_OPEN_HAND_LOCO_BRAKE":
        cmd = self.open_back_brake * (1 - geometry.smoothstep(elapsed / 1.2))
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        if elapsed > 2.0:
            self.enter("FINAL_OPEN_HAND_LOCO_SETTLE")
    else:
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
            and back > 0.12
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.open_back_since = (
            (now if self.open_back_since is None else self.open_back_since)
            if good
            else None
        )
        if self.open_back_since is not None and now - self.open_back_since >= 1.5:
            self.open_back_active = False
            self.open_back_completed = True
            self.done = True
            self.global_settle = True
            self.global_stable_since = self.open_back_since
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="actual_open_hand_loco_back_and_three_bottle_shelf_settle_verified",
                    actual_base_delta_xyz_m=delta.tolist(),
                    actual_backward_projection_m=back,
                    maximum_bottle_tilts_deg=self.open_back_tilts.tolist(),
                    bottles=bottles,
                )
            )
            self.enter("ALL_THREE_BOTTLES_SHELVED_AND_HANDS_CLEAR")
        elif elapsed > 10.0:
            self.failure = "Three bottles did not settle independently after open-hand Loco retreat"
    return (self.stage, self.open_back_goals.copy(), 65.7)


def contact_withdrawal_loaded_targets(self):
    if (
        self.active_bottle == 1
        and self.place_active
        and (self.stage == "SHELF_WITHDRAW_LEFT")
    ):
        now = float(self.data.time)
        key = (self.active_bottle, self.stage_start)
        if self.second_clear_key != key:
            self.second_clear_key = key
            self.second_clear_start = now
            self.second_clear_goals = self.held_goals()
            self.second_clear_origin = self.second_clear_goals[0, :3, 3].copy()
            self.second_clear_since = None
            self.enable_carry()
            self.carry_start = now - 2.0
        elapsed = now - self.second_clear_start
        self.native_clock = 65.7
        self.native_start = now - 65.7
        self.source_phase = "DELIVERY_RETREAT"
        self.bottle_commands = [0, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [self.placement_height, self.placement_pitch]
        self.post_pose[:] = self.walk_pose
        goals = self.second_clear_goals.copy()
        goals[0, :3, 3] += geometry.smoothstep(elapsed / 4.0) * np.array(
            [-0.035, -0.1, 0.025]
        )
        (l, r) = self.dual_measure()
        actual = self.held_goals()[0, :3, 3]
        delta = actual - self.second_clear_origin
        good = (
            self.shelf_stable(l)
            and l["active_hand_object_contact_count"] == 0
            and (r["active_hand_object_contact_count"] == 0)
            and (np.linalg.norm(delta) > 0.04)
            and (delta[1] < -0.03)
            and (l["root_speed_mps"] < 0.08)
            and (l["support_feet_count"] == 2)
        )
        self.second_clear_since = (
            (now if self.second_clear_since is None else self.second_clear_since)
            if good
            else None
        )
        if (
            elapsed >= 4.5
            and self.second_clear_since is not None
            and (now - self.second_clear_since >= 0.5)
        ):
            self.place_phase_goals = self.held_goals()
            self.multi_events.append(
                dict(
                    time_s=now,
                    event="second_empty_hand_actual_clearance_verified",
                    actual_wrist_delta_xyz_m=delta.tolist(),
                    wrist_error_diagnostic=self.errors(goals),
                    hands_clear=True,
                    shelf_N=l["middle_shelf_upward_N"],
                )
            )
            self.enter("SHELF_SETTLE")
        elif elapsed > 10.0:
            self.failure = "Second actual empty-hand clearance did not occur"
        self.place_goals = goals.copy()
        return (self.stage, goals, 65.7)
    return placement.front_shelf_loaded_targets(self)


def third_approach_start_return(self):
    placement.shelf_lowering_start_return(self)
    if self.active_bottle == 2:
        self.empty_target += self.third_return_offset
        self.empty_brake_target += self.third_return_offset
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="third_real_loco_return_goal_offset",
                offset_xy=self.third_return_offset.tolist(),
                target=self.empty_target.tolist(),
                physical_state_changed=False,
            )
        )


def return_heading_start_stance_arc(self):
    return False


def return_heading_empty_targets(self):
    former = self.stage
    value = sequence.repeat_empty_targets(self)
    if former == "EMPTY_LOCO_RETURN_TO_BOX":
        self.walk_twist[2] = np.clip(
            2.0 * geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()), -0.4, 0.4
        )
        self.post_twist[:] = self.walk_twist
        self.loco.command[:] = self.walk_twist
        if self.stage == "EMPTY_RETURN_BRAKE":
            self.empty_brake_cmd = self.walk_twist.copy()
    return value
