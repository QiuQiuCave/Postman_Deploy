"""Box pickup, loaded walking, placement, and release."""

from __future__ import annotations
import numpy as np
from common.r2v2_grasp_recording import body_transform as body_transform
from common.r2v2_loco_cola import interpolate_pose as interpolate_pose
import mujoco as mujoco
from . import geometry


def crate_route_diagnostic_hold(self):
    row = self.measure()
    self.loaded_guard(row)
    self.loco.command[:] = 0.0
    if self.failure:
        return (self.stage, None, self.paused_clock)
    if self.data.time - self.stage_start >= self.profile.get("hold_s", 4.0):
        self.diagnostic_finished = True
        self.done = True
        self.enter("DIAGNOSTIC_COMPLETE")
    return (self.stage, self.carry_targets(), self.paused_clock)


def crate_route_loaded_targets(self):
    if self.stage in (
        "UPRIGHT_STAND_CONTROL",
        "BACK_DIAGNOSTIC_HOLD",
        "TURN_DIAGNOSTIC_HOLD",
    ):
        cargo = self.bottle_metrics()
        now = float(self.data.time)
        lost = (
            not cargo["center_contained"]
            or cargo["ground_contact"]
            or cargo["table_contact"]
        )
        self.payload_loss_since = (
            (now if self.payload_loss_since is None else self.payload_loss_since)
            if lost
            else None
        )
        if self.payload_loss_since is not None and now - self.payload_loss_since >= 0.1:
            self.failure = "Payload escaped in diagnostic hold"
            return (self.stage, None, self.paused_clock)
        return self.diagnostic_hold()
    former = self.stage
    result = crate_retention_loaded_targets(self)
    if self.failure or self.done:
        return result
    mode = self.profile.get("mode", "full")
    if former == "LOCO_LOADED_HOLD" and self.stage == "WALK_BACK" and (mode == "stand"):
        self.enter("UPRIGHT_STAND_CONTROL")
        self.loco.command[:] = 0.0
        return (self.stage, self.carry_targets(), self.paused_clock)
    if former == "WALK_BACK" and self.stage == "WALK_BACK":
        speed = self.profile.get("back_speed", 0.2)
        acceleration = self.profile.get("acceleration", 0.0)
        if acceleration > 0.0:
            remaining = max(
                0.0,
                self.test_distance
                - (self.walk_start_position[0] - self.scratch.xpos[self.base, 0]),
            )
            target = min(
                speed, float(np.sqrt(2.0 * acceleration * max(0.0, remaining - 0.015)))
            )
            desired = np.clip(
                (target - self.command_speed) / 0.35, -acceleration, acceleration
            )
            jerk = self.profile.get("jerk", 0.2)
            self.command_accel += float(
                np.clip(desired - self.command_accel, -jerk * 0.02, jerk * 0.02)
            )
            self.command_speed = float(
                np.clip(self.command_speed + self.command_accel * 0.02, 0.0, speed)
            )
        else:
            self.command_speed = speed
        self.loco.command[:2] *= self.command_speed / 0.2
    if (
        former == "STOP_BEFORE_TURN"
        and self.stage == "TURN_LOADED"
        and (mode == "back")
    ):
        self.loco.command[:] = 0.0
        self.enter("BACK_DIAGNOSTIC_HOLD")
        return (self.stage, self.carry_targets(), self.paused_clock)
    if (
        former == "STOP_AFTER_TURN"
        and self.stage == "STOP_OVER_TARGET"
        and (mode == "turn")
    ):
        self.loco.command[:] = 0.0
        self.enter("TURN_DIAGNOSTIC_HOLD")
        return (self.stage, self.carry_targets(), self.paused_clock)
    return result


def crate_transport_placement_good(self, row, require_clear_hands=False):
    box = (
        crate_destination_placement_good(self, row, require_clear_hands)
        and row["destination_vertical_N"] >= 0.8 * self.payload_mass * 9.81
    )
    cargo = row.get("payload_bottle")
    return (
        box
        and cargo is not None
        and cargo["fully_inside_cavity"]
        and (not cargo["ground_contact"])
        and (not cargo["table_contact"])
        and (cargo["crate_floor_vertical_N"] >= 0.8 * 0.1 * 9.81)
    )


def crate_support_loaded_guard(self, row):
    if self.stage != "LOCO_LOWER":
        return crate_destination_loaded_guard(self, row)
    hands = list(row["hands"].values())
    valid = (
        not row["crate_floor_contact"]
        and (not row["source_table_contact"])
        and (row["crate_tilt_deg"] < 8.0)
        and all((h["opposed"] and h["vertical_N"] > 0.2 for h in hands))
        and (sum((h["vertical_N"] for h in hands)) >= 0.8 * self.payload_mass * 9.81)
        and all(
            (
                h["grip_slip_m"] is not None and h["grip_slip_m"] < self.slip_limit_m
                for h in hands
            )
        )
    )
    time = float(self.data.time)
    self.loaded_loss_since = (
        (time if self.loaded_loss_since is None else self.loaded_loss_since)
        if not valid
        else None
    )
    if self.loaded_loss_since is not None and time - self.loaded_loss_since >= 0.3:
        self.failure = (
            "LOCO_LOWER: bilateral load/8deg tilt/25mm slip gate failed for 0.3s"
        )


def crate_retention_retained(self, row):
    return self.physically_supported(row) and all(
        (
            h["grip_slip_m"] is not None and h["grip_slip_m"] < self.slip_limit_m
            for h in row["hands"].values()
        )
    )


def crate_retention_loaded_targets(self):
    if self.stage == "LOCO_LOWER":
        row = self.measure()
        if not row["destination_table_contact"]:
            self.loaded_guard(row)
            if self.failure:
                self.failure = self.failure.replace(
                    "unchanged support/8deg tilt/15mm slip",
                    "support/8deg tilt/25mm slip (local experiment)",
                )
                return (self.stage, None, self.paused_clock)
    result = crate_delivery_loaded_targets(self)
    if self.failure:
        self.failure = self.failure.replace(
            "unchanged support/8deg tilt/15mm slip",
            "support/8deg tilt/25mm slip (local experiment)",
        )
    return result


def crate_delivery_switch(self, owner):
    previous = self.executed.copy()
    crate_switch(self, owner)
    if owner == "loco" and self.arm_hold_mix > 0.0:
        self.arm_hold = (
            1.0 - self.arm_hold_mix
        ) * self.arm_hold + self.arm_hold_mix * previous[14:]


def crate_delivery_pre_release_good(self, row):
    return (
        crate_destination_pre_release_good(self, row)
        and row["destination_vertical_N"] >= 0.2 * self.payload_mass * 9.81
    )


def crate_delivery_loaded_targets(self):
    if self.bottle_enabled and self.cargo_body is not None:
        cargo = self.bottle_metrics()
        lost = (
            not cargo["center_contained"]
            or cargo["ground_contact"]
            or cargo["table_contact"]
        )
        t = float(self.data.time)
        self.payload_loss_since = (
            (t if self.payload_loss_since is None else self.payload_loss_since)
            if lost
            else None
        )
        if self.payload_loss_since is not None and t - self.payload_loss_since >= 0.1:
            self.failure = "Payload bottle escaped the open crate or touched external support for .1s"
            return (self.stage, None, self.paused_clock)
    if (
        self.fast_seat > 0.0
        and self.layout_mode == "arc_front"
        and (self.stage == "STOP_AFTER_TURN")
    ):
        row = self.measure()
        self.loaded_guard(row)
        self.loco.command[:] = 0.0
        if self.failure:
            return (self.stage, None, self.paused_clock)
        elapsed = float(self.data.time) - self.stage_start
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
            and (
                abs(
                    geometry.loaded_loco_tests_wrap_angle(
                        self.turn_target_yaw - self.base_yaw()
                    )
                )
                < np.deg2rad(5.0)
            )
        )
        if self.stable(good and elapsed >= 1.0, 0.5):
            position = np.asarray(row["crate_position_m"])
            local = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
                position - self.scratch.xpos[self.destination_body]
            )
            if np.any(
                abs(local[:2]) > self.model.geom_size[self.destination_geom, :2] - 0.05
            ):
                self.failure = "Natural arc exit missed fixed target support margin"
                return (self.stage, None, self.paused_clock)
            self.destination[:2] = position[:2]
            self.lower_start_height = float(self.loco.pose_command[0])
            self.lower_depth = float(position[2] - 0.98 + 0.002)
            self.event("retained_station_verified_before_early_lower", row)
            self.enter("LOCO_LOWER")
        elif elapsed > 10.0:
            self.failure = "Natural arc exit did not settle before measured lowering"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.fast_seat > 0.0 and self.stage == "LOCO_LOWER":
        row = self.measure()
        elapsed = float(self.data.time) - self.stage_start
        if not row["destination_table_contact"]:
            self.loaded_guard(row)
            if self.failure:
                return (self.stage, None, self.paused_clock)
        fraction = float(np.clip(elapsed / self.fast_seat, 0.0, 1.0))
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        extra = min(0.04, max(0.0, elapsed - self.fast_seat) * 0.004)
        self.loco.command[:] = 0.0
        self.loco.pose_command[0] = max(
            0.35, self.lower_start_height - self.lower_depth * fraction - extra
        )
        targets = self.carry_targets()
        if self.stable(self.pre_release_good(row), 0.5):
            self.event("real_table_contact_before_opening", row)
            self.release_anchor = targets.copy()
            self.enter("OPEN")
        elif elapsed > self.fast_seat + 14.0:
            self.failure = "Measured lowering did not establish stable table support"
        return (self.stage, targets, self.paused_clock)
    if self.stage == "CARGO_SETTLE":
        row = self.measure()
        self.loaded_guard(row)
        self.loco.command[:] = 0.0
        if self.failure:
            return (self.stage, None, self.paused_clock)
        cargo = row["payload_bottle"]
        elapsed = float(self.data.time) - self.stage_start
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
            and (cargo["tilt_deg"] < 8.0)
            and (cargo["relative_speed_mps"] < 0.01)
            and (cargo["angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 1.0, 0.5):
            self.segment_back_anchor = float(self.scratch.xpos[self.base, 0])
            self.segment_turn_anchor = float(self.turn_unwrapped or 0.0)
            self.event("bottle_upright_motion_resumed", row)
            self.enter(self.resume_after_settle)
        elif elapsed > 10.0:
            self.failure = (
                "Segmented cargo motion did not settle with an upright bottle"
            )
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.layout_mode == "front_big" and self.stage == "APPROACH_TARGET":
        row = self.measure()
        self.loaded_guard(row)
        if self.failure:
            return (self.stage, None, self.paused_clock)
        elapsed = float(self.data.time) - self.stage_start
        angle = np.deg2rad(self.target_layout["yaw"])
        forward = np.array([np.cos(angle), np.sin(angle)])
        left = np.array([-np.sin(angle), np.cos(angle)])
        error = self.destination[:2] - np.array(row["crate_position_m"])[:2]
        yaw = self.base_yaw()
        angular = np.clip(
            1.5 * geometry.loaded_loco_tests_wrap_angle(angle - yaw) - 0.08, -0.2, 0.2
        )
        lateral = np.clip(0.6 * float(error @ left), -0.1, 0.1)
        self.loco.command[:] = [0.15, lateral, angular]
        if float(error @ forward) < 0.02:
            self.loco.command[:] = 0.0
            self.event("frontal_destination_approach_stopped", row)
            self.enter("STOP_OVER_TARGET")
        elif elapsed > 18.0:
            self.failure = "Frontal approach did not reach the fixed landing station"
        return (self.stage, self.carry_targets(), self.paused_clock)
    former = self.stage
    result = crate_destination_loaded_targets(self)
    if self.segment_transport and self.stage == "WALK_BACK":
        if self.segment_back_anchor is None:
            self.segment_back_anchor = float(self.scratch.xpos[self.base, 0])
        if self.segment_back_anchor - self.scratch.xpos[self.base, 0] >= 0.3:
            self.resume_after_settle = "WALK_BACK"
            self.loco.command[:] = 0.0
            self.event("backward_segment_stopped", self.measure())
            self.enter("CARGO_SETTLE")
            return (self.stage, self.carry_targets(), self.paused_clock)
    if (
        self.segment_transport
        and self.stage == "TURN_LOADED"
        and (
            abs((self.turn_unwrapped or 0.0) - self.segment_turn_anchor)
            >= np.deg2rad(20.0)
        )
    ):
        self.resume_after_settle = "TURN_LOADED"
        self.loco.command[:] = 0.0
        self.event("arc_segment_stopped", self.measure())
        self.enter("CARGO_SETTLE")
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "WALK_BACK" and self.walk_ramp > 0.0:
        fraction = float(
            np.clip((self.data.time - self.stage_start) / self.walk_ramp, 0.0, 1.0)
        )
        self.loco.command[:2] *= fraction * fraction * (3.0 - 2.0 * fraction)
    if (
        former == "STOP_AFTER_TURN"
        and self.stage == "APPROACH_TARGET"
        and (self.layout_mode == "arc_front")
        and (not self.failure)
    ):
        self.event("frontal_table_already_at_natural_arc_exit", self.measure())
        self.loco.command[:] = 0.0
        self.enter("STOP_OVER_TARGET")
        return (self.stage, self.carry_targets(), self.paused_clock)
    if (
        former == "TURN_LOADED"
        and self.stage == "STOP_AFTER_TURN"
        and self.continuous_arc
        and (self.layout_mode == "front_big")
        and (not self.failure)
    ):
        self.event("arc_continues_without_stop", self.measure())
        self.approach_start = self.scratch.xpos[self.base].copy()
        self.enter("APPROACH_TARGET")
        self.loco.command[:] = [0.15, 0.0, 0.05]
        return (self.stage, self.carry_targets(), self.paused_clock)
    return result


def crate_destination_placement_good(self, row, require_clear_hands=False):
    if self.placement_mode == "reach":
        return crate_motion_placement_good(self, row, require_clear_hands)
    rel = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
        np.asarray(row["crate_position_m"]) - self.scratch.xpos[self.destination_body]
    )
    inside = bool(
        np.all(abs(rel[:2]) <= self.model.geom_size[self.destination_geom, :2] - 0.05)
    )
    return (
        inside
        and row["destination_table_contact"]
        and (row["destination_vertical_N"] >= 0.8 * 0.4 * 9.81)
        and (not row["source_table_contact"])
        and (not row["crate_floor_contact"])
        and (row["crate_tilt_deg"] < 8.0)
        and (row["crate_speed_mps"] < 0.02)
        and (row["crate_angular_speed_radps"] < 0.15)
        and (
            not require_clear_hands
            or all((h["active_contact_count"] == 0 for h in row["hands"].values()))
        )
    )


def crate_destination_pre_release_good(self, row):
    rel = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
        np.asarray(row["crate_position_m"]) - self.scratch.xpos[self.destination_body]
    )
    return (
        np.all(abs(rel[:2]) <= self.model.geom_size[self.destination_geom, :2] - 0.05)
        and row["destination_table_contact"]
        and (row["destination_vertical_N"] >= 0.2 * 0.4 * 9.81)
        and (not row["source_table_contact"])
        and (not row["crate_floor_contact"])
        and (row["crate_tilt_deg"] < 5.0)
        and (row["crate_speed_mps"] < 0.01)
        and (row["crate_angular_speed_radps"] < 0.1)
    )


def crate_destination_loaded_guard(self, row):
    time = float(self.data.time)
    valid = self.retained(row)
    self.loaded_loss_since = (
        (time if self.loaded_loss_since is None else self.loaded_loss_since)
        if not valid
        else None
    )
    if self.loaded_loss_since is not None and time - self.loaded_loss_since >= 0.3:
        self.failure = (
            f"{self.stage}: unchanged support/8deg tilt/15mm slip gate failed for 0.3s"
        )


def crate_destination_loaded_targets(self):
    if self.stage in (
        "PICKUP",
        "LOCO_LOADED_HOLD",
        "WALK_BACK",
        "STOP_BEFORE_TURN",
        "TURN_LOADED",
    ):
        return loaded_motion_loaded_targets(self)
    row = self.measure()
    time = float(self.data.time)
    elapsed = time - self.stage_start
    if self.stage in (
        "STOP_AFTER_TURN",
        "APPROACH_TARGET",
        "STOP_OVER_TARGET",
        "REACH_AT_TARGET",
    ):
        self.loaded_guard(row)
        if self.failure:
            return (self.stage, None, self.paused_clock)
    if self.stage == "STOP_AFTER_TURN":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
            and (
                abs(
                    geometry.loaded_loco_tests_wrap_angle(
                        self.turn_target_yaw - self.base_yaw()
                    )
                )
                < np.deg2rad(5.0)
            )
        )
        if self.stable(good and elapsed >= 2.0, 0.5):
            self.event("turn_stopped", row)
            self.approach_start = np.array(row["base_position_world_m"])
            self.enter("APPROACH_TARGET")
        elif elapsed > 10.0:
            self.failure = "Loaded arc did not settle"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "APPROACH_TARGET":
        np.array(row["base_position_world_m"])[:2]
        box = np.array(row["crate_position_m"])[:2]
        error = self.destination[:2] - box
        bearing = float(np.arctan2(error[1], error[0]))
        yaw = self.base_yaw()
        self.loco.command[:] = [
            0.15,
            0.0,
            np.clip(
                0.8 * geometry.loaded_loco_tests_wrap_angle(bearing - yaw), -0.2, 0.2
            ),
        ]
        if box[1] >= self.destination[1] - 0.02:
            self.loco.command[:] = 0.0
            self.event("destination_approach_stopped", row)
            self.enter("STOP_OVER_TARGET")
        elif elapsed > 15.0:
            self.failure = (
                "Frozen Loco did not bring the retained box over the fixed far target"
            )
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "STOP_OVER_TARGET":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 2.0, 0.5):
            self.event("destination_station_settled", row)
            position = np.array(row["crate_position_m"])
            rel = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
                position - self.scratch.xpos[self.destination_body]
            )
            half = self.model.geom_size[self.destination_geom]
            if np.any(abs(rel[:2]) > half[:2] - 0.04):
                self.failure = (
                    "Stopped box work point outside fixed target tabletop margin"
                )
                return (self.stage, None, self.paused_clock)
            self.destination[:2] = position[:2]
            self.return_goals = self.held_goals()
            if self.placement_mode == "reach":
                self.switch("reach")
                self.enter("REACH_AT_TARGET")
            else:
                self.lower_start_height = float(self.loco.pose_command[0])
                self.lower_depth = float(position[2] - 0.98 + 0.002)
                self.enter("LOCO_LOWER")
            return (self.stage, self.return_goals, self.paused_clock)
        elif elapsed > 10.0:
            self.failure = "Box did not settle over the fixed target"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "REACH_AT_TARGET":
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 3.0, 0.3):
            self.event("reach_return_settled", row)
            self.lower_anchor = self.commanded_goals()
            self.lower_depth = float(row["crate_position_m"][2] - 0.98 + 0.002)
            self.enter("LOWER_REVERSE")
        elif elapsed > 10.0:
            self.failure = "Loaded Reach return at far table did not settle"
        return (self.stage, self.return_goals, self.paused_clock)
    if self.stage in ("LOWER_REVERSE", "LOCO_LOWER"):
        (_, high) = self.path.sample(30.9)
        (_, seat) = self.path.sample(21.9)
        (_, reverse) = self.path.sample(15.4 + max(6.5, 15.5 - elapsed))
        progress = float(
            np.clip(
                (high[:, 2, 3].mean() - reverse[:, 2, 3].mean())
                / (high[:, 2, 3].mean() - seat[:, 2, 3].mean()),
                0,
                1,
            )
        )
        if self.stage == "LOWER_REVERSE":
            targets = self.lower_anchor.copy()
            targets[:, 2, 3] -= self.lower_depth * progress
        else:
            self.loco.command[:] = 0.0
            extra = min(0.04, max(0.0, elapsed - 9.0) * 0.004)
            self.loco.pose_command[0] = max(
                0.35, self.lower_start_height - self.lower_depth * progress - extra
            )
            targets = self.carry_targets()
        if self.stable(
            (
                self.pre_release_good(row)
                if self.stage == "LOCO_LOWER"
                else self.placement_good(row)
            )
            and elapsed >= 2.0,
            0.5,
        ):
            self.event("real_table_contact_before_opening", row)
            self.release_anchor = targets.copy()
            self.enter("OPEN")
        elif elapsed > (24.0 if self.stage == "LOCO_LOWER" else 15.0):
            self.failure = "Reverse lift descent did not transfer real box weight to the target table"
        return (self.stage, targets, self.paused_clock)
    if self.stage == "OPEN":
        self.finger_closed = False
        self.loco.command[:] = 0.0
        if self.stable(self.placement_good(row, True) and elapsed >= 3.0, 1.0):
            self.release_verified = True
            self.event("actual_box_release_verified", row)
            self.retreat_anchor = self.commanded_goals()
            self.retreat_base = self.scratch.xpos[self.base].copy()
            cr = self.scratch.xmat[self.crate_body].reshape(3, 3)
            yaw = float(np.arctan2(cr[1, 0], cr[0, 0]))
            self.retreat_rotation = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.enter("RETRACT")
        elif self.placement_mode == "loco" and self.stable(
            self.placement_good(row)
            and elapsed >= 4.0
            and (not any((h["opposed"] for h in row["hands"].values()))),
            0.0,
        ):
            self.opening_started_supported = True
            self.retreat_base = self.scratch.xpos[self.base].copy()
            self.enter("RETRACT")
        elif elapsed > 10.0:
            self.failure = "Opening did not leave the real box table-supported and clear of both hands"
        return (self.stage, self.release_anchor, self.paused_clock)
    if self.stage == "RETRACT":
        if self.placement_mode == "reach":
            (_, outside) = self.path.sample(15.4 + max(0.0, 3.5 - elapsed))
            (_, seat) = self.path.sample(18.9)
            targets = self.retreat_anchor.copy()
            targets[:, :3, 3] += (
                self.retreat_rotation @ (outside[:, :3, 3] - seat[:, :3, 3]).T
            ).T
        else:
            self.loco.command[:] = [-0.2, 0.0, 0.0]
            targets = self.carry_targets()
            if (
                np.linalg.norm(self.scratch.xpos[self.base, :2] - self.retreat_base[:2])
                > 0.3
            ):
                self.loco.command[:] = 0.0
                self.enter("FINAL_SETTLE")
        if self.placement_mode == "reach" and self.stable(
            self.placement_good(row, True) and elapsed >= 4.0, 1.0
        ):
            self.physical_placement_verified = True
            self.done = True
            self.enter("COMPLETE")
        elif elapsed > 12.0:
            self.failure = "Retraction did not leave a stable released box"
        return (self.stage, targets, self.paused_clock)
    if self.stage == "FINAL_SETTLE":
        self.loco.command[:] = 0.0
        if self.stable(self.placement_good(row, True) and elapsed >= 2.0, 1.0):
            self.release_verified = True
            self.physical_placement_verified = True
            self.done = True
            self.enter("COMPLETE")
        elif elapsed > 10.0:
            self.failure = "Released box did not remain stable after leaving table"
        return (self.stage, self.carry_targets(), self.paused_clock)
    return (self.stage, self.commanded_goals(), self.paused_clock)


def loaded_motion_base_yaw(self):
    r = self.scratch.xmat[self.base].reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def loaded_motion_event(self, name, row):
    self.segment_events.append(
        dict(event=name, time_s=float(self.data.time), metrics=row)
    )


def loaded_motion_loaded_targets(self):
    if self.stage in ("PICKUP", "LOCO_LOADED_HOLD"):
        result = crate_motion_loaded_targets(self)
        if self.stage == "WALK_BACK" and self.walk_origin is None:
            row = self.measure()
            self.walk_origin = np.array(row["base_position_world_m"])
            self.event("walking_started", row)
        return result
    row = self.measure()
    time = float(self.data.time)
    elapsed = time - self.stage_start
    valid = self.retained(row)
    self.loaded_loss_since = (
        (time if self.loaded_loss_since is None else self.loaded_loss_since)
        if not valid
        else None
    )
    if self.loaded_loss_since is not None and time - self.loaded_loss_since >= 0.3:
        self.failure = (
            f"{self.stage}: unchanged support/8deg tilt/15mm slip gate failed for 0.3s"
        )
        return (self.stage, None, self.paused_clock)
    if self.stage == "WALK_BACK":
        yaw = self.base_yaw()
        error = geometry.loaded_loco_tests_wrap_angle(self.yaw_anchor - yaw)
        world = (
            np.array([-0.2, 0.0]) if self.move_axis == "x" else np.array([0.0, 0.15])
        )
        body = (
            np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) @ world
        )
        self.loco.command[:] = [
            np.clip(body[0], -0.3, 0.6),
            np.clip(body[1], -0.15, 0.15),
            np.clip(error, -0.2, 0.2),
        ]
        distance = (
            self.walk_start_position[0] - self.scratch.xpos[self.base, 0]
            if self.move_axis == "x"
            else self.scratch.xpos[self.base, 1] - self.walk_start_position[1]
        )
        if distance >= self.test_distance - 0.01:
            self.loco.command[:] = 0.0
            self.event("walking_target_reached", row)
            self.enter("STOP_BEFORE_TURN")
        elif elapsed > 20.0:
            self.failure = (
                "Loaded translation did not reach the planned test distance within 20s"
            )
    elif self.stage == "STOP_BEFORE_TURN":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 2.0, 0.5):
            self.event("walking_stopped", row)
            if self.test_turn_deg == 0:
                self.segment_completed = True
                self.done = True
                self.enter("DIAGNOSTIC_COMPLETE")
            else:
                self.turn_origin = np.array(row["base_position_world_m"])
                self.turn_origin_yaw = self.base_yaw()
                self.turn_target_yaw = self.turn_origin_yaw + np.deg2rad(
                    self.test_turn_deg
                )
                self.turn_last_yaw = self.turn_origin_yaw
                self.turn_unwrapped = 0.0
                self.enter("TURN_LOADED")
        elif elapsed > 10.0:
            self.failure = "Loaded stop before turning did not settle within 10s"
    elif self.stage == "TURN_LOADED":
        yaw = self.base_yaw()
        self.turn_unwrapped += geometry.loaded_loco_tests_wrap_angle(
            yaw - self.turn_last_yaw
        )
        self.turn_last_yaw = yaw
        error = self.turn_target_yaw - (self.turn_origin_yaw + self.turn_unwrapped)
        self.loco.command[:] = [
            self.turn_vx * min(1.0, elapsed / 0.6),
            0.0,
            geometry.turn_command(error, elapsed, self.max_yaw_rate, self.yaw_floor),
        ]
        if abs(error) < np.deg2rad(3.0):
            self.loco.command[:] = 0.0
            self.saved_turn_progress = self.turn_unwrapped
            self.event("turn_heading_reached", self.measure())
            self.enter("STOP_AFTER_TURN")
        elif elapsed > (25.0 if abs(self.test_turn_deg) <= 90 else 40.0):
            self.failure = (
                "Loaded turn did not reach target heading within bounded duration"
            )
    elif self.stage == "STOP_AFTER_TURN":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
            and (
                abs(
                    geometry.loaded_loco_tests_wrap_angle(
                        self.turn_target_yaw - self.base_yaw()
                    )
                )
                < np.deg2rad(5.0)
            )
        )
        if self.stable(good and elapsed >= 2.0, 0.5):
            self.event("turn_stopped", row)
            self.segment_completed = True
            self.done = True
            self.enter("DIAGNOSTIC_COMPLETE")
        elif elapsed > 10.0:
            self.failure = (
                "Loaded turn reached heading but did not retain a settled heading/grip"
            )
    return (self.stage, self.carry_targets(), self.paused_clock)


def crate_motion_enter(self, stage):
    (self.stage, self.stage_start, self.stable_since) = (
        stage,
        float(self.data.time),
        None,
    )


def crate_motion_stable(self, condition, duration):
    self.stable_since = (
        (float(self.data.time) if self.stable_since is None else self.stable_since)
        if condition
        else None
    )
    return (
        self.stable_since is not None and self.data.time - self.stable_since >= duration
    )


def crate_motion_held_goals(self):
    return np.array(
        [
            body_transform(self.scratch, self.model.body(s + "_hand_roll_link").id)
            for s in ("left", "right")
        ]
    )


def crate_motion_commanded_goals(self):
    out = np.tile(np.eye(4), (2, 1, 1))
    for i, side in enumerate(("left", "right")):
        ref = self.policy.references[side]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, ref.goal_quaternion)
        (out[i, :3, :3], out[i, :3, 3]) = (rot.reshape(3, 3), ref.goal_position)
    return out


def crate_motion_capture_carry(self):
    self.carry_anchor = (
        self.commanded_goals() if self.arm_mode == "reach" else self.held_goals()
    )
    self.carry_base = self.scratch.xpos[self.base].copy()
    rot = self.scratch.xmat[self.base].reshape(3, 3)
    self.yaw_anchor = float(np.arctan2(rot[1, 0], rot[0, 0]))


def crate_motion_carry_targets(self):
    targets = self.carry_anchor.copy()
    delta = self.scratch.xpos[self.base] - self.carry_base
    targets[:, :2, 3] += delta[:2]
    return targets


def crate_motion_configure_inverse_frame(self):
    rotation = self.scratch.xmat[self.base].reshape(3, 3)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0])) - self.nominal_base_yaw
    (c, s) = (np.cos(yaw), np.sin(yaw))
    self.inverse_frame[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    self.inverse_frame[:2, 3] = (
        self.scratch.xpos[self.base, :2]
        - self.inverse_frame[:2, :2] @ self.nominal_base_xy
    )
    target = self.inverse_frame @ np.array([0.33, 0, 0.98, 1.0])
    relative = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
        target[:3] - self.scratch.xpos[self.destination_body]
    )
    half = self.model.geom_size[self.destination_geom]
    if np.any(np.abs(relative[:2]) > half[:2] - 0.025):
        self.failure = "Arrival-relative inverse pickup work point is outside the fixed destination tabletop"
        return
    self.destination = target[:3].copy()
    self.transitions.append(
        dict(
            time_s=float(self.data.time),
            planned_inverse_workpoint=self.destination.tolist(),
            reference_frame=self.inverse_frame.tolist(),
            planned_before_lowering=True,
        )
    )


def crate_motion_inverse_targets(self, source_time):
    (_, original) = self.path.sample(15.4 + source_time)
    return np.array([self.inverse_frame @ goal for goal in original])


def crate_motion_placement_good(self, row, require_clear_hands=False):
    return (
        row["destination_table_contact"]
        and row["destination_vertical_N"] >= 0.8 * 0.4 * 9.81
        and (not row["source_table_contact"])
        and (not row["crate_floor_contact"])
        and (row["destination_error_xy_m"] < 0.03)
        and (row["crate_tilt_deg"] < 8.0)
        and (row["crate_speed_mps"] < 0.02)
        and (row["crate_angular_speed_radps"] < 0.15)
        and (
            not require_clear_hands
            or all((h["active_contact_count"] == 0 for h in row["hands"].values()))
        )
    )


def crate_motion_loaded_targets(self):
    time = float(self.data.time)
    if self.stage == "PICKUP":
        if time < 30.9 - 1e-09:
            (phase, targets) = self.path.sample(time)
            self.finger_closed = 18.9 - 1e-09 <= time
            return (phase, targets, time)
        row = self.measure()
        if not self.physical_pickup_verified or not self.physically_supported(row):
            self.failure = "Pickup not physically supported before locomotion"
            return (self.stage, None, self.paused_clock)
        self.switch("loco")
        self.capture_carry()
        self.enter("LOCO_LOADED_HOLD")
    row = self.measure()
    elapsed = time - self.stage_start
    needs_grip = self.stage in (
        "LOCO_LOADED_HOLD",
        "WALK_SIDE",
        "WALK_BACK",
        "STOP_AT_DESTINATION",
        "PREPARE_REACH_RETURN",
        "REACH_RETURN",
        "PLACE_ALIGN",
    )
    if needs_grip:
        valid = self.retained(row)
        self.loaded_loss_since = (
            (time if self.loaded_loss_since is None else self.loaded_loss_since)
            if not valid
            else None
        )
        if self.loaded_loss_since is not None and time - self.loaded_loss_since >= 0.3:
            self.failure = f"{self.stage}: unchanged support/8deg tilt/15mm slip gate failed for 0.3s"
            return (self.stage, None, self.paused_clock)
    if self.stage == "LOCO_LOADED_HOLD":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 4.0, 0.5):
            self.walk_start_position = self.scratch.xpos[self.base].copy()
            self.enter("WALK_BACK" if self.route == "back" else "WALK_SIDE")
        elif elapsed > 10.0:
            self.failure = "Initial loaded Loco did not settle"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "WALK_BACK":
        rotation = self.scratch.xmat[self.base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        yaw_error = np.arctan2(
            np.sin(yaw - self.yaw_anchor), np.cos(yaw - self.yaw_anchor)
        )
        self.loco.command[:] = [
            -0.2 * np.cos(yaw),
            0.2 * np.sin(yaw),
            np.clip(-yaw_error, -0.2, 0.2),
        ]
        distance = self.walk_start_position[0] - self.scratch.xpos[self.base, 0]
        if distance >= self.walk_distance - 0.01:
            self.loco.command[:] = 0.0
            self.enter("STOP_AT_DESTINATION")
        elif elapsed > 15.0:
            self.failure = "Backward walk did not reach the short station"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "WALK_SIDE":
        rotation = self.scratch.xmat[self.base].reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        error = self.destination[:2] - np.asarray(row["crate_position_m"])[:2]
        world_velocity = np.array(
            [
                np.clip(0.6 * error[0], -0.12, 0.12),
                np.clip(0.8 * error[1], -0.0, self.speed),
            ]
        )
        body_velocity = (
            np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            @ world_velocity
        )
        yaw_error = np.arctan2(
            np.sin(yaw - self.yaw_anchor), np.cos(yaw - self.yaw_anchor)
        )
        self.loco.command[:] = [
            np.clip(body_velocity[0], -0.3, 0.6),
            np.clip(body_velocity[1], -0.15, 0.15),
            np.clip(-yaw_error, -0.2, 0.2),
        ]
        if abs(error[1]) < 0.025:
            self.loco.command[:] = 0.0
            self.enter("STOP_AT_DESTINATION")
        elif elapsed > 20.0:
            self.failure = "Lateral carry did not reach the destination station"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "STOP_AT_DESTINATION":
        self.loco.command[:] = 0.0
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 2.0, 0.5):
            if self.relative_inverse:
                self.pre_return_pitch = float(self.loco.pose_command[1])
                self.enter("PREPARE_REACH_RETURN")
            else:
                self.switch("reach")
                self.return_goals = self.held_goals()
                self.enter("REACH_RETURN")
                return (self.stage, self.return_goals, self.paused_clock)
        elif elapsed > 10.0:
            self.failure = "Loaded stop did not settle"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "PREPARE_REACH_RETURN":
        self.loco.command[:] = 0.0
        fraction = float(np.clip(elapsed / 3.0, 0, 1))
        fraction = fraction * fraction * (3 - 2 * fraction)
        self.loco.pose_command[1] = (
            1 - fraction
        ) * self.pre_return_pitch + fraction * np.deg2rad(15.0)
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 5.0, 0.5):
            self.configure_inverse_frame()
            if self.failure:
                return (self.stage, None, self.paused_clock)
            self.switch("reach")
            self.return_goals = self.inverse_targets(15.5)
            self.enter("REACH_RETURN")
            return (self.stage, self.return_goals, self.paused_clock)
        elif elapsed > 12.0:
            self.failure = (
                "Original return torso pitch did not settle with a real retained box"
            )
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "REACH_RETURN":
        good = (
            self.retained(row)
            and row["crate_speed_mps"] < 0.02
            and (row["crate_angular_speed_radps"] < 0.15)
        )
        if self.stable(good and elapsed >= 3.0, 0.3):
            if self.relative_inverse:
                self.enter("LOWER_REVERSE")
            else:
                self.begin_place()
        elif elapsed > 10.0:
            self.failure = "Loaded Reach return did not settle"
        return (self.stage, self.return_goals, self.paused_clock)
    if self.stage == "PLACE_ALIGN":
        middle = self.place_crate_anchor.copy()
        middle[0, 3] = self.place_crate_hover[0, 3]
        middle[2, 3] = self.place_crate_hover[2, 3]
        fraction = float(
            np.clip((elapsed if elapsed < 8.0 else elapsed - 8.0) / 8.0, 0, 1)
        )
        fraction = fraction * fraction * (3 - 2 * fraction)
        crate_ref = (
            geometry.mix_transform(self.place_crate_anchor, middle, fraction)
            if elapsed < 8.0
            else geometry.mix_transform(middle, self.place_crate_hover, fraction)
        )
        targets = np.array([crate_ref @ relation for relation in self.place_relations])
        good = (
            self.retained(row)
            and row["destination_error_xy_m"] < 0.025
            and (row["crate_speed_mps"] < 0.02)
        )
        if self.stable(good and elapsed >= 17.0, 0.5):
            self.lower_anchor = targets.copy()
            self.lower_depth = float(
                np.asarray(row["crate_position_m"])[2] - 0.98 + 0.002
            )
            self.enter("LOWER_REVERSE")
        elif elapsed > 25.0:
            self.failure = (
                "Box/wrist placement alignment did not settle over destination"
            )
        return (self.stage, targets, self.paused_clock)
    if self.stage == "LOWER_REVERSE":
        if self.relative_inverse:
            targets = self.inverse_targets(max(6.5, 15.5 - elapsed))
            if self.stable(self.placement_good(row) and elapsed >= 2.0, 0.5):
                self.release_anchor = targets.copy()
                self.enter("OPEN")
            elif elapsed > 14.0:
                self.failure = "Exact inverse pickup reference did not seat the box on the destination"
            return (self.stage, targets, self.paused_clock)
        (_, high) = self.path.sample(15.4 + 15.5)
        (_, seat) = self.path.sample(15.4 + 6.5)
        (_, reverse) = self.path.sample(15.4 + max(6.5, 15.5 - elapsed))
        progress = float(
            np.clip(
                (high[:, 2, 3].mean() - reverse[:, 2, 3].mean())
                / (high[:, 2, 3].mean() - seat[:, 2, 3].mean()),
                0,
                1,
            )
        )
        targets = self.lower_anchor.copy()
        targets[:, 2, 3] -= self.lower_depth * progress
        if self.stable(self.placement_good(row) and elapsed >= 2.0, 0.5):
            self.release_anchor = targets.copy()
            self.enter("OPEN")
        elif elapsed > 14.0:
            self.failure = (
                "Reverse lowering did not transfer the real box weight to the table"
            )
        return (self.stage, targets, self.paused_clock)
    if self.stage == "OPEN":
        self.finger_closed = False
        if self.stable(self.placement_good(row, True) and elapsed >= 3.0, 1.0):
            self.release_verified = True
            self.retreat_anchor = self.held_goals()
            self.enter("RETRACT")
        elif elapsed > 10.0:
            self.failure = "Real release from both hands not verified"
        return (self.stage, self.release_anchor, self.paused_clock)
    if self.stage == "RETRACT":
        if self.relative_inverse:
            targets = self.inverse_targets(max(0.0, 3.5 - elapsed))
            if self.stable(self.placement_good(row, True) and elapsed >= 4.0, 1.0):
                self.physical_placement_verified = True
                self.done = True
                self.enter("COMPLETE")
            elif elapsed > 10.0:
                self.failure = (
                    "Inverse approach did not leave the seated box clear of both hands"
                )
            return (self.stage, targets, self.paused_clock)
        targets = self.retreat_anchor.copy()
        fraction = float(np.clip(elapsed / 3.0, 0, 1))
        fraction = fraction * fraction * (3 - 2 * fraction)
        (_, outside) = self.path.sample(15.4)
        (_, seated) = self.path.sample(15.4 + 3.5)
        targets[:, :3, 3] += (outside[:, :3, 3] - seated[:, :3, 3]) * fraction
        if self.stable(self.placement_good(row, True) and elapsed >= 4.0, 1.0):
            self.physical_placement_verified = True
            self.done = True
            self.enter("COMPLETE")
        elif elapsed > 10.0:
            self.failure = "Hands did not clear the supported box after retraction"
        return (self.stage, targets, self.paused_clock)
    return (self.stage, self.commanded_goals(), self.paused_clock)


def crate_switch(self, owner):
    before = (self.data.qpos.copy(), self.data.qvel.copy())
    (self.owner, self.switch_time) = (owner, float(self.data.time))
    self.blend_from = self.executed.copy()
    if owner == "loco":
        self.arm_hold = self.scratch.qpos[self.mapping.qpos[14:]].copy()
    else:
        self.policy.follow_current(self.scratch)
        self.entry_poses = [
            self.policy.endpoint_pose(self.scratch, side) for side in ("left", "right")
        ]
    if not np.array_equal(before[0], self.data.qpos) or not np.array_equal(
        before[1], self.data.qvel
    ):
        raise RuntimeError("A controller switch changed the physical state")
    self.transitions.append(
        dict(
            time_s=float(self.data.time),
            controller=owner,
            physical_state_continuous=True,
            histories_reset=False,
        )
    )


def crate_sync(self):
    self.scratch.time = self.data.time
    self.scratch.qpos[:] = self.data.qpos
    self.scratch.qvel[:] = self.data.qvel
    self.scratch.ctrl[:] = self.data.ctrl
    mujoco.mj_forward(self.model, self.scratch)
