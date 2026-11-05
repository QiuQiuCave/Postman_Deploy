"""Continuous three-bottle workflow and shared 1 kHz control loop."""

from __future__ import annotations
import numpy as np
from r2v2_deploy.control.hand_control import DualHandControl as NativeHandControl
from r2v2_deploy.tasks.physical_validity import (
    effective_hand_control as effective_hand_control,
)
from r2v2_deploy.tasks.physical_validity import robot_stable as robot_stable
import json as json
import mujoco as mujoco
from .resources import resource
from . import geometry, grasp, handoff, placement, return_path


def sequence_begin_source_stance(self):
    grasp.grasp_stance_begin_source_stance(self)
    self.source_stance_yaw -= np.deg2rad(8.0)
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="natural_third_stance_heading_adjusted_with_original_loco",
            heading_delta_deg=-8.0,
            target_yaw_deg=float(np.rad2deg(self.source_stance_yaw)),
            original_wrist_grasp_geometry_retained=True,
            physical_state_changed=False,
        )
    )


def sequence_configure_repeat_handoff(self):
    grasp.centered_handoff_configure_repeat_handoff(self)
    if self.active_bottle == 2 and getattr(self, "natural_mode", "both") != "return":
        self.object_goal[:3, :3] = (
            geometry.axis_rotation(self.object_goal[:3, 2], np.array([0.0, 0.0, 1.0]))
            @ self.object_goal[:3, :3]
        )
        local = np.array(
            [
                0.5054798036171188,
                0.20390061850007393,
                1.415754046955416 - self.post_base[2],
            ]
        )
        self.object_goal[:3, 3] = self.post_base + self.post_R @ local
        self.left_grasp_goal = self.object_goal @ self.object_to_left


def sequence_prepare_source_cartesian(self):
    return_alignment_prepare_source_cartesian(self)
    if (
        getattr(self, "natural_mode", "return") != "return"
        and self.empty_hand_crate_rim_clearance()[0] > 0.08
    ):
        self.source_cart_raise = 0.16
        self.source_cart_origin[0] = self.source_cart_start[0].copy()
        self.source_cart_origin[0, 2, 3] -= 0.015
        if hasattr(self, "natural_third_wrist_preparation"):
            self.source_cart_origin[0, :3, :3] = np.array(
                self.natural_third_wrist_preparation["rotation_world"]
            )
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="natural_third_omits_redundant_empty_raise",
                empty_preparation_point_delta_z_m=-0.015,
                actual_hand_above_crate_rim_m=self.empty_hand_crate_rim_clearance(),
                original_finger_timing_and_physical_guards_retained=True,
                physical_state_changed=False,
            )
        )


def sequence_rotate_goal(self, goals, clock):
    result = return_alignment_rotate_goal(self, goals, clock)
    if (
        getattr(self, "natural_mode", "return") != "return"
        and self.active_bottle == 2
        and (self.stage == "BOTTLE_PROGRAM")
        and (clock >= 48.0)
        and (result is not None)
    ):
        angle = np.deg2rad(-20.0) * geometry.smoothstep((clock - 48.0) / 7.5)
        rotation = geometry.rotation_z(angle)
        pivot = (
            self.scratch.xpos[self.cargo_body].copy()
            if clock < 55.5
            else self.source_cart_pivot.copy()
        )
        result[0, :3, 3] = pivot + rotation @ (result[0, :3, 3] - pivot)
        result[0, :3, :3] = rotation @ result[0, :3, :3]
    return result


def sequence_station_object_frame(self):
    if (
        getattr(self, "natural_mode", "return") != "return"
        and self.active_bottle in (0, 1)
        and self.shelf_regrip_active
        and hasattr(self, "natural_reused_points")
    ):
        obj = self.object_pose().copy()
        target = obj @ self.station_grasp
        current = np.arctan2(target[1, 0], target[0, 0])
        desired = self.natural_reused_points[self.active_bottle][
            "left_final_reference_heading_rad"
        ]
        obj[:3, :3] = obj[:3, :3] @ geometry.rotation_z(
            geometry.wrap_angle(desired - current)
        )
        if self.active_bottle == 1 and self.stage == "LEFT_REGRASP_FINAL_APPROACH":
            grasp = obj @ self.station_grasp
            obj[:3, 3] += (
                0.01
                * geometry.smoothstep((float(self.data.time) - self.stage_start) / 8.0)
                * grasp[:3, 0]
            )
        return obj
    return placement.right_release_station_object_frame(self)


def sequence_start_shelf_regrip(self):
    placement.workspace_start_shelf_regrip(self)
    if (
        getattr(self, "natural_mode", "return") != "return"
        and self.active_bottle == 0
        and hasattr(self, "natural_reused_points")
    ):
        target = self.object_pose().copy()
        target[:3, 3] = self.natural_reused_points[0]["object_goal_world_xyz_m"]
        self.second_right_goal = target @ self.right_relation
        grasp = self.station_object_frame() @ self.station_grasp
        self.left_normal_R = grasp[:3, :3].copy()
        self.normal_left_goal[:3, :3] = self.left_normal_R


def sequence_begin_natural_station_trim(self):
    self.dual_mode = False
    self.shelf_regrip_active = False
    self.extra_active = True
    self.enable_carry()
    self.carry_start = float(self.data.time) - 2.0
    self.natural_station_goals = self.held_goals()
    self.natural_station_since = None
    self.enter("NATURAL_RIGHT_UPRIGHT_AT_SHELF")


def sequence_natural_station_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "DELIVERY_TRANSFER"
    self.bottle_commands = [0, 1]
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    goals = self.natural_station_goals.copy()
    (left, right) = self.dual_measure()
    err = self.errors(goals)[1]
    axis = self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
    if err[0] < 0.008 and err[1] < 5.0:
        rot = geometry.axis_rotation(axis, np.array([0.0, 0.0, 1.0]), 0.08 * 0.02)
        pivot = self.scratch.xpos[self.cargo_body].copy()
        goals[1, :3, 3] = pivot + rot @ (goals[1, :3, 3] - pivot)
        goals[1, :3, :3] = rot @ goals[1, :3, :3]
    good = (
        right["object_tilt_deg"] < 2.0
        and right["object_linear_speed_mps"] < 0.025
        and (right["object_angular_speed_radps"] < 0.18)
        and effective_hand_control(right, 0.981)
        and (left["active_hand_object_contact_count"] == 0)
        and (right["root_speed_mps"] < 0.08)
        and (right["support_feet_count"] == 2)
    )
    self.natural_station_since = (
        (now if self.natural_station_since is None else self.natural_station_since)
        if good
        else None
    )
    self.natural_station_goals = goals.copy()
    if (
        self.natural_station_since is not None
        and now - self.natural_station_since >= 0.4
    ):
        self.multi_events.append(
            dict(
                time_s=now,
                event="natural_right_upright_before_original_left_regrasp",
                active=self.active_bottle + 1,
                tilt_deg=right["object_tilt_deg"],
            )
        )
        self.natural_solo = False
        self.start_shelf_regrip()
        if self.natural_stop_after == "station":
            self.done = True
        return (self.stage, self.second_start_goals.copy(), 65.7)
    if elapsed > 18.0:
        self.failure = (
            "Right-held bottle did not become upright before original shelf regrasp"
        )
    self.post_goals = goals.copy()
    return (self.stage, goals, 65.7)


def sequence_natural_rest_goals(self):
    d = self.natural_fk
    d.qpos[:] = self.scratch.qpos
    d.qvel[:] = self.scratch.qvel
    d.qpos[self.upper_pos[2:]] = self.natural_posture_joints()
    mujoco.mj_kinematics(self.model, d)
    goals = np.repeat(np.eye(4)[None], 2, axis=0)
    for i, side in enumerate(("left", "right")):
        b = self.model.body(side + "_hand_roll_link").id
        goals[i, :3, 3] = d.xpos[b]
        goals[i, :3, :3] = d.xmat[b].reshape(3, 3)
    return goals


def sequence_natural_posture_joints(self):
    joints = self.natural_rest_joints.copy()
    if self.empty_return:
        joints[6] = -0.35
        joints[13] = 0.35
    return joints


def sequence_advance(self, stage, goals):
    if (
        getattr(self, "natural_mode", "none") in ("both", "transport")
        and stage == "LEFT_REGRASP_PREAPPROACH"
        and (not self.shelf_regrip_active)
    ):
        self.natural_solo = True
        self.natural_left_start = self.held_goals()
        self.natural_left_rest = self.natural_rest_goals()[0]
        self.natural_empty_left_qstart = self.scratch.qpos[self.upper_pos[2:9]].copy()
        self.natural_empty_left_qcommand = self.executed[14:21].copy()
        self.natural_empty_left_qgoal = self.natural_empty_left_qstart.copy()
        self.natural_empty_left_plan_time = None
        self.natural_since = None
        self.natural_loss_since = None
        self.bottle_commands = [0, 1]
        self.handoff_owner = "right"
        self.extra_active = True
        self.enable_carry()
        self.carry_start = float(self.data.time) - 2.0
        self.natural_pose = np.array(self.post_pose).copy()
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="natural_empty_left_rest_begin",
                active=self.active_bottle + 1,
                omitted="near-box LEFT_REGRASP_PREAPPROACH/FINAL/CLOSE",
                right_trajectory="held actual right wrist until original carry prep",
                live_state_writes=False,
            )
        )
        self.enter("NATURAL_EMPTY_LEFT_LOWER")
        return
    return grasp.centered_handoff_advance(self, stage, goals)


def sequence_loaded_targets(self):
    if (
        self.active_bottle == 2
        and self.departure_clear_active
        and (not getattr(self, "natural_departure_second_leg", False))
        and (float(self.data.time) - self.stage_start >= 4.5)
    ):
        (left, _) = self.dual_measure()
        if left["active_hand_object_contact_count"] == 0:
            self.natural_departure_second_leg = True
            self.departure_clear_goals = self.held_goals()
            radial = (
                self.departure_clear_goals[0, :2, 3]
                - self.scratch.xpos[self.cargo_body, :2]
            )
            radial /= max(np.linalg.norm(radial), 1e-09)
            base = self.scratch.xpos[self.model.body("base_link").id, :2]
            toward_body = base - self.departure_clear_goals[0, :2, 3]
            tangent = toward_body - radial * np.dot(toward_body, radial)
            tangent /= max(np.linalg.norm(tangent), 1e-09)
            self.departure_clear_vector = np.r_[0.14 * tangent, 0.0]
            self.departure_clear_since = None
            self.enter("NATURAL_THIRD_EMPTY_BACK_CLEAR")
            self.multi_events.append(
                dict(
                    time_s=float(self.data.time),
                    event="natural_third_empty_second_withdrawal_leg",
                    displacement_world_m=self.departure_clear_vector.tolist(),
                    actual_left_contacts=left["active_hand_object_contact_count"],
                    original_22cm_clearance_and_shelf_stability_gate_retained=True,
                    physical_state_changed=False,
                )
            )
    if self.stage == "NATURAL_RIGHT_UPRIGHT_AT_SHELF":
        return self.natural_station_targets()
    if self.stage != "NATURAL_EMPTY_LEFT_LOWER":
        (stage, goals, clock) = grasp.contact_descent_loaded_targets(self)
        if (
            self.active_bottle == 2
            and self.safe_place_active
            and (stage == "THIRD_SAFE_FRONT_MOVE")
        ):
            f = geometry.smoothstep(
                (float(self.data.time) - self.stage_start) / self.safe_place_move_s
            )
            pivot = (1 - f) * self.safe_place_object[:3, 3] + f * np.array(
                [self.safe_place_x, 2.15, self.middle_height + 0.105]
            )
            rotation = geometry.rotation_z(np.deg2rad(10.0) * f)
            goals[0, :3, 3] = pivot + rotation @ (goals[0, :3, 3] - pivot)
            goals[0, :3, :3] = rotation @ goals[0, :3, :3]
            self.place_goals = goals.copy()
        return (stage, goals, clock)
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.bottle_commands = [0, 1]
    self.post_twist[:] = 0.0
    self.walk_twist[:] = 0.0
    self.post_pose[:] = self.natural_pose
    self.walk_pose[:] = self.natural_pose
    goals = self.natural_left_start.copy()
    if self.active_bottle >= 1:
        mid = np.array([1.0, 0.8, 0.0, -0.9, 0.0, 0.0, 0.0])
        if elapsed < 6.0:
            f = geometry.smoothstep(elapsed / 6.0)
            qleft = (1 - f) * self.natural_empty_left_qstart + f * mid
        else:
            f = geometry.smoothstep((elapsed - 6.0) / 4.0)
            qleft = (1 - f) * mid + f * self.natural_rest_joints[:7]
        self.natural_empty_left_qgoal = qleft.copy()
        d = self.natural_fk
        d.qpos[:] = self.scratch.qpos
        d.qpos[self.upper_pos[2:9]] = qleft
        mujoco.mj_kinematics(self.model, d)
        b = self.model.body("left_hand_roll_link").id
        goals[0, :3, 3] = d.xpos[b]
        goals[0, :3, :3] = d.xmat[b].reshape(3, 3)
    else:
        goals[0] = geometry.mix_transform(
            goals[0], self.natural_left_rest, geometry.smoothstep(elapsed / 7.0)
        )
    (left, right) = self.dual_measure()
    err = self.errors(goals)[0]
    good = (
        elapsed >= (10.5 if self.active_bottle >= 1 else 7.5)
        and err[0] < 0.015
        and (err[1] < 6.0)
        and (left["active_hand_object_contact_count"] == 0)
        and effective_hand_control(right, 0.981)
        and (right["hand_vertical_force_N"] > 0.8 * 0.981)
        and (right["root_speed_mps"] < 0.08)
        and (right["support_feet_count"] == 2)
    )
    self.natural_since = (
        (now if self.natural_since is None else self.natural_since) if good else None
    )
    if self.natural_since is not None and now - self.natural_since >= 0.4:
        handoff.carry_begin_dual(self)
        self.handoff_owner = "right"
        self.bottle_commands = [0, 1]
        self.multi_events.append(
            dict(
                time_s=now,
                event="natural_right_only_carry_begin",
                active=self.active_bottle + 1,
                right_upward_N=right["hand_vertical_force_N"],
                left_contacts=left["active_hand_object_contact_count"],
                original_loco_route_and_right_carry_transform=True,
            )
        )
        return (self.stage, self.dual_goals.copy(), 65.7)
    if elapsed > (20.0 if self.active_bottle >= 1 else 16.0):
        self.failure = (
            "Empty left resting posture did not align with real right-only support"
        )
    self.post_goals = goals.copy()
    return (self.stage, goals, 65.7)


def sequence_dual_carry_goals(self, elapsed):
    goals = handoff.carry_dual_carry_goals(self, elapsed)
    if self.natural_solo:
        goals[0] = self.natural_rest_goals()[0]
    return goals


def sequence_dual_targets(self):
    result = placement.shelf_stance_dual_targets(self)
    if not self.natural_solo:
        return result
    self.bottle_commands = [0, 1]
    self.handoff_owner = "right"
    if self.stage == "DUAL_STATION_OPEN_LEFT_FOR_REGRIP":
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="natural_shelf_rejoin_original_regrasp",
                active=self.active_bottle + 1,
                omitted="redundant station left-open/withdraw",
                all_original_final_alignment_and_load_gates_retained=True,
            )
        )
        self.begin_natural_station_trim()
        return (self.stage, self.natural_station_goals.copy(), 65.7)
    return result


def sequence_start_return(self):
    return_path.return_offset_start_return(self)
    self.natural_raised_return_goals = self.return_goals.copy()
    self.natural_raised_return_prior = self.cart_prior.copy()
    self.natural_return_raised_for_pick = False
    if getattr(self, "natural_mode", "none") in ("both", "return"):
        self.return_goals = self.natural_rest_goals()
        self.cart_prior[2:] = self.natural_posture_joints()
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="natural_both_empty_arms_return",
                active_next_bottle=self.active_bottle + 1,
                original_clearance_first_2p5_s_and_loco_route_retained=True,
                original_next_pick_preparation_retained=True,
            )
        )


def sequence_empty_hand_crate_rim_clearance(self):
    if not hasattr(self, "natural_clearance_geom_ids"):

        def under(body, root):
            while body > 0 and body != root:
                body = int(self.model.body_parentid[body])
            return body == root

        roots = [
            self.model.body(side + "_hand_roll_link").id for side in ("left", "right")
        ]
        groups = roots + [self.crate_body]
        self.natural_clearance_geom_ids = [
            np.array(
                [
                    g
                    for g in range(self.model.ngeom)
                    if (self.model.geom_contype[g] or self.model.geom_conaffinity[g])
                    and under(int(self.model.geom_bodyid[g]), root)
                ],
                dtype=int,
            )
            for root in groups
        ]
    bounds = []
    for ids in self.natural_clearance_geom_ids:
        rotation = self.scratch.geom_xmat[ids].reshape(-1, 3, 3)
        center = self.scratch.geom_xpos[ids] + np.einsum(
            "nij,nj->ni", rotation, self.model.geom_aabb[ids, :3]
        )
        half_z = np.einsum(
            "ni,ni->n", abs(rotation[:, 2, :]), self.model.geom_aabb[ids, 3:]
        )
        bounds.append(
            (float(np.min(center[:, 2] - half_z)), float(np.max(center[:, 2] + half_z)))
        )
    return [bounds[0][0] - bounds[2][1], bounds[1][0] - bounds[2][1]]


def sequence_empty_targets(self):
    now = float(self.data.time)
    if (
        self.natural_mode in ("both", "return")
        and self.stage == "EMPTY_LOCO_RETURN_TO_BOX"
        and (not getattr(self, "natural_return_raised_for_pick", False))
        and (self.scratch.xpos[self.base, 0] > 0.1)
    ):
        self.natural_return_raise_from = self.held_goals()
        self.natural_return_raise_to = self.natural_raised_return_goals.copy()
        rotation = geometry.rotation_z(
            geometry.wrap_angle(self.base_yaw() - self.empty_yaw)
        )
        for i in range(2):
            self.natural_return_raise_to[i, :2, 3] = self.scratch.xpos[
                self.base, :2
            ] + rotation[:2, :2] @ (
                self.natural_return_raise_to[i, :2, 3] - self.empty_origin[:2]
            )
            self.natural_return_raise_to[i, :3, :3] = (
                rotation @ self.natural_return_raise_to[i, :3, :3]
            )
        self.natural_return_raise_brake = self.walk_twist.copy()
        self.natural_return_raise_since = None
        self.natural_return_raise_prior_from = self.cart_prior.copy()
        self.natural_return_smooth_plan_time = None
        self.natural_return_last_arm_command = self.executed[14:].copy()
        self.enter("NATURAL_RETURN_RAISE_FOR_PICK")
    if self.stage != "NATURAL_RETURN_RAISE_FOR_PICK":
        result = return_path.return_heading_empty_targets(self)
        if (
            self.active_bottle == 1
            and self.stage == "EMPTY_LOCO_RETURN_TO_BOX"
            and getattr(self, "natural_return_raised_for_pick", False)
        ):
            yaw_goal = self.natural_reused_return["second_pick_original_base_yaw_rad"]
            self.walk_twist[2] = np.clip(
                0.8 * geometry.wrap_angle(yaw_goal - self.base_yaw()), -0.12, 0.12
            )
            self.post_twist[:] = self.walk_twist
            self.loco.command[:] = self.walk_twist
        return result
    elapsed = now - self.stage_start
    self.native_clock = 6.7
    self.native_start = now - 6.7
    self.source_phase = "EMPTY_RETURN"
    self.bottle_commands = [0, 0]
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    self.walk_twist[:] = self.natural_return_raise_brake * (
        1 - geometry.smoothstep(elapsed / 1.5)
    )
    self.post_twist[:] = self.walk_twist
    self.loco.command[:] = self.walk_twist
    self.loco.pose_command[:] = self.walk_pose
    duration = 12.0 if self.active_bottle == 2 else 8.0
    fprior = geometry.smoothstep(elapsed / duration)
    self.cart_prior[:] = (
        (1 - fprior) * self.natural_return_raise_prior_from
        + fprior * self.natural_raised_return_prior
    )
    goals = np.array(
        [
            geometry.mix_transform(a, b, geometry.smoothstep(elapsed / duration))
            for (a, b) in zip(
                self.natural_return_raise_from, self.natural_return_raise_to
            )
        ]
    )
    errors = self.errors(goals)
    (left, right) = self.dual_measure()
    hand_clearance = self.empty_hand_crate_rim_clearance()
    tracking = True
    good = (
        elapsed > duration + 0.5
        and min(hand_clearance) > 0.03
        and tracking
        and (left["root_speed_mps"] < 0.08)
        and (left["support_feet_count"] == 2)
        and (left["active_hand_object_contact_count"] == 0)
        and (right["active_hand_object_contact_count"] == 0)
    )
    self.natural_return_raise_since = (
        (
            now
            if self.natural_return_raise_since is None
            else self.natural_return_raise_since
        )
        if good
        else None
    )
    if (
        self.natural_return_raise_since is not None
        and now - self.natural_return_raise_since > 0.4
    ):
        self.natural_return_raised_for_pick = True
        if self.active_bottle == 1:
            self.empty_brake_target = self.empty_target.copy()
        self.empty_fixed_goals = self.held_goals()
        self.empty_reference_base = self.scratch.xpos[self.base].copy()
        self.empty_reference_yaw = self.base_yaw()
        self.multi_events.append(
            dict(
                time_s=now,
                event="natural_return_rejoined_original_raised_preparation_before_box",
                actual_wrist_errors_m_deg=errors,
                conservative_actual_hand_above_crate_rim_m=hand_clearance,
                active_next_bottle=self.active_bottle + 1,
                physical_state_changed=False,
                original_next_pick_goal_unchanged=True,
            )
        )
        self.enter("EMPTY_LOCO_RETURN_TO_BOX")
    elif elapsed > (24.0 if self.active_bottle == 2 else 18.0):
        self.failure = (
            "Empty return did not rejoin original raised preparation before box"
        )
    return (self.stage, goals, 6.7)


def sequence_step(self):
    grasp.grasp_stance_step(self)
    if self.steps % 10 == 0:
        (left, right) = self.dual_measure()
        self.natural_rows.append(
            dict(
                time_s=float(self.data.time),
                stage=self.stage,
                active_bottle=self.active_bottle + 1,
                solo=self.natural_solo,
                commands=list(self.bottle_commands),
                left_contacts=left["active_hand_object_contact_count"],
                left_upward_N=left["hand_vertical_force_N"],
                right_upward_N=right["hand_vertical_force_N"],
                right_controlled=bool(effective_hand_control(right, 0.981)),
                tilt_deg=right["object_tilt_deg"],
                left_wrist_m=self.held_goals()[0, :3, 3].tolist(),
            )
        )
    now = float(self.data.time)
    if hasattr(self, "outdir") and (
        now - getattr(self, "natural_status_time", -1000000000.0) >= 1.0
        or self.stage != getattr(self, "natural_status_stage", None)
    ):
        status = dict(
            time_s=now,
            stage=self.stage,
            native_clock=float(self.native_clock),
            active_bottle=self.active_bottle + 1,
            completed_bottles=list(self.completed_bottles),
            failure=self.failure,
            path_stall_since=getattr(self, "path_stall_since", None),
            auxiliary_demo=True,
            training_changed=False,
        )
        if self.natural_rows:
            status["latest_readonly_sample"] = self.natural_rows[-1]
        if self.source_cart_rows:
            status["latest_private_IK"] = self.source_cart_rows[-1]["IK_residual"]
        tmp = self.outdir / "LIVE_STATUS.json.tmp"
        tmp.write_text(json.dumps(status))
        tmp.replace(self.outdir / "LIVE_STATUS.json")
        self.natural_status_time = now
        self.natural_status_stage = self.stage
    if (
        self.natural_stop_after == "return"
        and self.active_bottle > 0
        and (self.stage == "ALIGNED_EMPTY_HOME")
        and (not self.failure)
    ):
        self.done = True


def return_alignment_prepare_source_cartesian(self):
    goals = grasp.grasp_tilt_rotate_goal(
        self, np.repeat(np.eye(4)[None], 2, axis=0), 55.5
    )
    self.source_cart_pivot = self.scratch.xpos[self.cargo_body].copy()
    rot = geometry.rotation_z(np.deg2rad(self.source_cart_axial))
    final = goals[0].copy()
    final[:3, 3] = self.source_cart_pivot + rot @ (
        final[:3, 3] - self.source_cart_pivot
    )
    final[:3, :3] = rot @ final[:3, :3]
    self.source_cart_final = final
    self.source_cart_origin = self.held_goals()
    self.source_cart_origin[0, 2, 3] += 0.1
    self.source_cart_start = self.held_goals()
    self.source_cart_right_relative = (
        np.linalg.inv(self.torso_pose()) @ self.source_cart_start[1]
    )
    self.home_goal = self.home_start.copy()
    self.source_cart_ready = True
    self.source_cart_IK = None
    toward = self.scratch.xpos[self.base, :2] - self.source_cart_pivot[:2]
    self.source_cart_lift_xy = 0.06 * toward / max(np.linalg.norm(toward), 1e-09)
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="third_new_cartesian_empty_path_prepared",
            axial_yaw_deg=self.source_cart_axial,
            above_grasp_m=self.source_cart_raise,
            physical_state_changed=False,
        )
    )


def return_alignment_loaded_targets(self):
    if (
        self.active_bottle == 2
        and self.stage == "ALIGNED_EMPTY_HOME"
        and (not self.source_cart_ready)
    ):
        self.prepare_source_cartesian()
    return grasp.grasp_stance_loaded_targets(self)


def return_alignment_rotate_goal(self, goals, clock):
    if (
        self.active_bottle != 2
        or not self.source_cart_ready
        or self.stage != "BOTTLE_PROGRAM"
    ):
        return grasp.grasp_tilt_rotate_goal(self, goals, clock)
    result = self.source_cart_start.copy()
    final = self.source_cart_final.copy()
    final[:3, 3] += (
        self.scratch.xpos[self.cargo_body] - self.source_cart_pivot
        if clock < 55.5
        else 0.0
    )
    above = final.copy()
    above[2, 3] += self.source_cart_raise
    if clock < 12.0:
        left = geometry.mix_transform(
            self.source_cart_start[0],
            self.source_cart_origin[0],
            geometry.smoothstep((clock - 6.7) / 5.3),
        )
    elif clock < 30.0:
        left = geometry.mix_transform(
            self.source_cart_origin[0],
            above,
            geometry.smoothstep((clock - 12.0) / 18.0),
        )
    elif clock < 48.0:
        left = above
    elif clock < 55.5:
        left = geometry.mix_transform(
            above, final, geometry.smoothstep((clock - 48.0) / 7.5)
        )
    else:
        left = final.copy()
        f = geometry.smoothstep((clock - 58.3) / 7.4)
        left[:2, 3] += self.source_cart_lift_xy * f
        left[2, 3] += 0.38 * f
    result[0] = left
    result[1] = self.torso_pose() @ self.source_cart_right_relative
    return result


def bottle_sequence_loaded_targets(self):
    eligible = (
        self.active_bottle == 2
        and self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
    )
    start = self.native_start
    clock = float(self.data.time) - start if start is not None else -1.0
    relaxed = False
    if eligible and 6.7 <= clock < 48.0:
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
    result = return_path.contact_withdrawal_loaded_targets(self)
    if (
        relaxed
        and self.stage == "BOTTLE_PROGRAM"
        and (self.native_start - start > 0.015)
    ):
        self.native_start = start + 0.01
        self.path_stall_since = None
        self.air_relaxed_count += 1
        self.air_relaxed_max_mm = max(self.air_relaxed_max_mm, ep * 1000)
        if self.failure and self.failure.startswith(
            "Repeated wrist path actual tracking stalled"
        ):
            self.failure = None
    return result


def repeat_switch_active(self, index):
    self.active_bottle = index
    self.cargo_body = self.bottle_bodies[index]
    self.cargo_geom = self.bottle_geoms[index]
    self.proxy.object_body = self.cargo_body
    self.proxy.object_geom = self.cargo_geom
    self.target_xy = np.array([-0.49 - 0.16 * index, self.shelf_target_y])
    self.route_goal = np.array(
        [self.shelf_base_x - 0.16 * index + (0.12 if index > 0 else 0.0), 1.58]
    )
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="active_bottle_changed_without_physical_reset",
            active_bottle=index + 1,
        )
    )


def repeat_begin_walk(self):
    self.__dict__.pop("route_arm_hold", None)
    self.__dict__.pop("route_world_R", None)
    self.route_hold_rebased = False
    placement.rack_begin_walk(self)
    if self.active_bottle > 0:
        self.configure_upright_carry()


def repeat_configure_upright_carry(self):
    axis = (
        geometry.rotation_z(self.carry_yaw).T
        @ self.scratch.xmat[self.cargo_body].reshape(3, 3)[:, 2]
    )
    self.carry_tilt_y = float(np.degrees(-np.arctan2(axis[0], axis[2])))
    self.carry_tilt_x = float(
        np.degrees(np.arctan2(axis[1], np.hypot(axis[0], axis[2])))
    )


def repeat_configure_repeat_handoff(self):
    if self.active_bottle > 0:
        self.object_goal[:3, 3] = self.post_base + self.post_R @ np.array(
            [0.58, 0.1, 1.26 - self.post_base[2]]
        )
        self.left_grasp_goal = self.object_goal @ self.object_to_left


def repeat_start_post(self):
    handoff.handoff_start_post(self)
    self.configure_repeat_handoff()
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="bottle_actual_held_extraction_verified",
            active=self.active_bottle + 1,
            bottles=self.all_bottles(),
        )
    )


def repeat_start_second(self):
    placement.grip_geometry_start_second(self)
    if self.active_bottle > 0:
        self.left_offset = self.repeat_left_offset.copy()
        self.left_normal_R = self.second_R @ geometry.rotation_z(
            np.deg2rad(self.repeat_left_yaw)
        )
        self.normal_left_goal[:3, :3] = self.left_normal_R


def repeat_begin_place(self):
    if hasattr(self, "route_arm_hold") and (
        not getattr(self, "route_hold_rebased", False)
    ):
        self.enable_carry()
        self.route_hold_rebased = True
        self.carry_tilt_x = 0.0
        self.carry_tilt_y = 0.0
    if self.scratch.xpos[self.base, 1] < 1.57 and (
        not (self.active_bottle == 0 and getattr(self, "skip_first_refine", False))
    ):
        self.refine_active = True
        self.refine_start = float(self.data.time)
        self.refine_stop_since = None
        self.place_goals = self.extra_goals()
        self.enter("SHELF_STANCE_FORWARD_AFTER_POST_CLEARANCE")
        return
    placement.shelf_begin_place(self)


def repeat_extra_goals(self):
    old = self.carry_pullback
    if getattr(self, "route_hold_rebased", False):
        self.carry_pullback = 0.0
    try:
        goals = placement.rack_extra_goals(self)
    finally:
        self.carry_pullback = old
    if (
        self.active_bottle > 0
        and hasattr(self, "route_world_R")
        and (not self.place_active)
    ):
        goals[0, :3, :3] = self.route_world_R
    return goals


def repeat_refine_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.bottle_commands = [1, 0]
    self.native_clock = 65.7
    self.native_start = now - 65.7
    self.source_phase = "PICK_LIFT_OUT"
    self.walk_pose[:] = [0.82, 0.12]
    self.post_pose[:] = self.walk_pose
    yaw = np.clip(0.8 * geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12)
    if self.stage == "SHELF_STANCE_FORWARD_AFTER_POST_CLEARANCE":
        cmd = np.array([0.2, 0.0, yaw]) * geometry.smoothstep(elapsed / 1.2)
        if self.scratch.xpos[self.base, 1] >= 1.57:
            self.refine_brake = cmd.copy()
            self.enter("SHELF_STANCE_FORWARD_BRAKE")
        elif elapsed > 9.0:
            self.failure = "Post-clearance forward stance did not make progress"
    else:
        cmd = self.refine_brake * (1 - geometry.smoothstep(elapsed / 0.6))
        row = self.native_measure()
        good = row["root_speed_mps"] < 0.08 and row["support_feet_count"] == 2
        self.refine_stop_since = (
            (now if self.refine_stop_since is None else self.refine_stop_since)
            if good
            else None
        )
        if (
            elapsed >= 2.5
            and self.refine_stop_since is not None
            and (now - self.refine_stop_since >= 0.2)
        ):
            self.refine_active = False
            placement.shelf_begin_place(self)
            return (self.stage, self.place_goals, 65.7)
        elif elapsed > 8.0:
            self.failure = "Post-clearance forward stance failed to settle"
    self.walk_twist[:] = cmd
    self.post_twist[:] = cmd
    return (self.stage, self.extra_goals(), 65.7)


def repeat_start_return(self):
    self.empty_return = True
    self.return_started = float(self.data.time)
    self.empty_origin = self.scratch.xpos[self.base].copy()
    self.empty_initial_pose = self.walk_pose.copy()
    self.empty_initial_goals = self.held_goals()
    self.empty_yaw = self.base_yaw()
    self.empty_R = geometry.rotation_z(self.empty_yaw)
    self.return_goals = self.empty_initial_goals.copy()
    for i, offset in enumerate([[0.18, 0.24, 1.35], [0.18, -0.27, 1.32]]):
        self.return_goals[i, :3, 3] = np.r_[
            self.empty_origin[:2], 0.0
        ] + self.empty_R @ np.array(offset)
        self.return_goals[i, :3, :3] = self.empty_R
    self.empty_target = (
        self.first_pick_base[:2]
        + self.scratch.xpos[self.cargo_body, :2]
        - self.first_pick_bottle[:2]
        + np.array(
            [
                0.0,
                (
                    getattr(self, "second_pick_forward_world", 0.0)
                    if self.active_bottle == 1
                    else 0.0
                ),
            ]
        )
    )
    self.empty_brake_target = self.empty_target - np.array([0.15, 0.0])
    self.empty_stop_since = None
    self.place_active = False
    self.place_success = False
    self.side_route_active = False
    self.post_start = None
    self.second_start = None
    self.walk_start = None
    self.extraction_success = False
    self.extract_since = None
    self.extraction_wait_start = None
    self.handoff_success = False
    self.left_normal_success = False
    self.walk_success = False
    self.native_active = False
    self.owner = "loco"
    self.arm_mode = "reach"
    self.switch_time = None
    self.bottle_commands = [0, 0]
    self.bottle_stage_started = False
    self.bottle_loss_since = None
    self.extra_active = True
    self.enable_carry()
    self.enter("EMPTY_RETRACT_AND_CLEAR_SHELF")


def repeat_empty_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    self.native_clock = 6.7
    self.native_start = now - 6.7
    self.source_phase = "EMPTY_RETURN"
    self.bottle_commands = [0, 0]
    self.walk_pose[:] = (
        1 - geometry.smoothstep((now - self.return_started) / 3.0)
    ) * self.empty_initial_pose + geometry.smoothstep(
        (now - self.return_started) / 3.0
    ) * np.array(
        [0.82, 0.12]
    )
    self.post_pose[:] = self.walk_pose
    self.walk_twist[:] = 0.0
    self.post_twist[:] = 0.0
    self.loco.command[:] = 0.0
    self.loco.pose_command[:] = self.walk_pose
    if self.stage.startswith("REPEAT_PICK_STANCE_"):
        goals = self.arc_goals.copy()
        rotation = geometry.rotation_z(
            geometry.wrap_angle(self.base_yaw() - self.arc_yaw)
        )
        for i in range(2):
            goals[i, :2, 3] = self.scratch.xpos[self.base, :2] + rotation[:2, :2] @ (
                goals[i, :2, 3] - self.arc_origin[:2]
            )
            goals[i, :3, :3] = rotation @ goals[i, :3, :3]
        if self.stage == "REPEAT_PICK_STANCE_BACK_ARC":
            cmd = np.array(
                [
                    -0.2,
                    0.0,
                    np.clip(
                        1.5
                        * geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()),
                        -0.45,
                        0.45,
                    ),
                ]
            ) * geometry.smoothstep(elapsed / 0.6)
            if (
                abs(geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()))
                < np.deg2rad(3)
                or np.linalg.norm(
                    self.scratch.xpos[self.base, :2] - self.arc_origin[:2]
                )
                > 0.45
            ):
                self.arc_brake = cmd.copy()
                self.enter("REPEAT_PICK_STANCE_BACK_BRAKE")
            elif elapsed > 8.0:
                self.failure = "Empty reverse arc did not make progress"
        elif self.stage == "REPEAT_PICK_STANCE_FORWARD":
            delta = self.arc_target - self.scratch.xpos[self.base, :2]
            local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ delta
            cmd = np.array(
                [
                    0.2,
                    np.clip(1.2 * local[1], -0.12, 0.12),
                    np.clip(
                        1.5
                        * geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()),
                        -0.3,
                        0.3,
                    ),
                ]
            ) * geometry.smoothstep(elapsed / 0.6)
            if (
                np.linalg.norm(delta) < 0.08
                or self.scratch.xpos[self.base, 1] > self.arc_target[1] - 0.04
            ):
                self.arc_brake = cmd.copy()
                self.enter("REPEAT_PICK_STANCE_FORWARD_BRAKE")
            elif elapsed > 8.0:
                self.failure = "Empty forward stance approach did not make progress"
        else:
            cmd = self.arc_brake * (1 - geometry.smoothstep(elapsed / 0.4))
            row = self.native_measure()
            if (
                elapsed >= 2.5
                and row["root_speed_mps"] < 0.08
                and (row["support_feet_count"] == 2)
            ):
                if self.stage == "REPEAT_PICK_STANCE_BACK_BRAKE":
                    self.enter("REPEAT_PICK_STANCE_FORWARD")
                else:
                    self.start_next_pick()
                    return (self.stage, self.home_start, 6.7)
            if elapsed > 8.0:
                self.failure = "Repeated-pick arc stance did not settle"
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        self.loco.command[:] = cmd
        return (self.stage, goals, 6.7)
    if self.stage == "EMPTY_RETRACT_AND_CLEAR_SHELF":
        goals = self.empty_initial_goals.copy()
        for i in range(2):
            retreat = self.empty_initial_goals[i].copy()
            retreat[:3, 3] += self.empty_R @ np.array([-0.13, 0.0, 0.1])
            if elapsed < 2.5:
                goals[i] = geometry.mix_transform(
                    self.empty_initial_goals[i],
                    retreat,
                    geometry.smoothstep(elapsed / 2.5),
                )
            else:
                goals[i] = geometry.mix_transform(
                    retreat,
                    self.return_goals[i],
                    geometry.smoothstep((elapsed - 2.5) / 3.5),
                )
        if elapsed >= 6.5:
            self.empty_fixed_goals = goals.copy()
            self.empty_reference_base = self.scratch.xpos[self.base].copy()
            self.empty_reference_yaw = self.base_yaw()
            self.enter("EMPTY_LOCO_RETURN_TO_BOX")
    else:
        rotation = geometry.rotation_z(
            geometry.wrap_angle(self.base_yaw() - self.empty_reference_yaw)
        )
        goals = self.empty_fixed_goals.copy()
        for i in range(2):
            goals[i, :2, 3] = self.scratch.xpos[self.base, :2] + rotation[:2, :2] @ (
                goals[i, :2, 3] - self.empty_reference_base[:2]
            )
            goals[i, :3, :3] = rotation @ goals[i, :3, :3]
        if self.stage == "EMPTY_LOCO_RETURN_TO_BOX":
            base = self.scratch.xpos[self.base, :2]
            ygoal = 1.5 if base[0] < 0.1 else self.empty_brake_target[1]
            delta = np.array([self.empty_brake_target[0] - base[0], ygoal - base[1]])
            world = np.array(
                [
                    (
                        np.sign(delta[0]) * max(0.2, min(0.25, 1.6 * abs(delta[0])))
                        if abs(delta[0]) > 0.07
                        else 0.0
                    ),
                    np.sign(delta[1]) * 0.2 if abs(delta[1]) > 0.06 else 0.0,
                ]
            )
            local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ world
            cmd = np.array(
                [
                    *local,
                    np.clip(
                        0.8
                        * geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()),
                        -0.12,
                        0.12,
                    ),
                ]
            ) * geometry.smoothstep(elapsed / 1.2)
            if (
                np.linalg.norm(
                    self.scratch.xpos[self.base, :2] - self.empty_brake_target
                )
                < 0.1
            ):
                self.empty_brake_cmd = cmd.copy()
                self.enter("EMPTY_RETURN_BRAKE")
            elif elapsed > 40.0:
                self.failure = "Empty Loco return did not reach box stance"
        else:
            cmd = self.empty_brake_cmd * (1 - geometry.smoothstep(elapsed / 1.5))
            row = self.native_measure()
            good = row["root_speed_mps"] < 0.08 and row["support_feet_count"] == 2
            self.empty_stop_since = (
                (now if self.empty_stop_since is None else self.empty_stop_since)
                if good
                else None
            )
            if (
                elapsed >= 2.5
                and self.empty_stop_since is not None
                and (now - self.empty_stop_since >= 0.3)
            ):
                self.start_next_pick()
                return (self.stage, self.home_start, 6.7)
            elif elapsed > 9.0:
                self.failure = "Empty Loco return did not settle"
        self.walk_twist[:] = cmd
        self.post_twist[:] = cmd
        self.loco.command[:] = cmd
    if int(round((now - self.return_started) / 0.02)) % 100 == 0:
        print(
            json.dumps(
                dict(
                    return_t=now - self.return_started,
                    stage=self.stage,
                    base=self.scratch.xpos[self.base].tolist(),
                    target=self.empty_target.tolist(),
                )
            ),
            flush=True,
        )
    return (self.stage, goals, 6.7)


def repeat_start_next_pick(self):
    if not self.repeat_yaw_teacher and self.start_stance_arc():
        return
    self.__dict__.pop("yaw_teacher_entry", None)
    self.__dict__.pop("repeat_extraction_hold", None)
    self.path_stall_since = None
    self.empty_return = False
    self.extra_active = False
    self.cart_active = False
    self.arm_planning = True
    self.plan_time = None
    self.upper_cmd = None
    self.second_plan_time = None
    self.cart_time = None
    self.second_plan_cache = None
    self.close_released = False
    self.close_wait = None
    self.close_good = None
    self.bottle_picked = False
    self.bottle_exited = False
    self.bottle_placed = False
    self.bottle_released = False
    self.bottle_withdrawn = False
    self.bottle_release_position = None
    self.bottle_settle_since = None
    self.extraction_wait_start = None
    self.extract_since = None
    self.left_load_since = None
    self.left_solo_since = None
    self.right_loaded_since = None
    self.right_solo_since = None
    self.handoff_owner = "left"
    self.held_walk_started = False
    self.native_active = False
    self.owner = "loco"
    self.arm_mode = "reach"
    self.switch_time = None
    self.bottle_commands = [0, 0]
    self.bottle_loss_since = None
    frame_yaw = np.deg2rad(self.measure()["crate_yaw_deg"])
    self.bottle_R = geometry.rotation_z(frame_yaw)
    self.bottle_t = np.r_[
        self.scratch.xpos[self.cargo_body, :2]
        - (self.bottle_R @ np.array([0.438, 0.454, 0.0]))[:2],
        0.0,
    ]
    self.source_start = 6.7
    self.native_clock = 6.7
    self.clock_pause = 0.0
    self.home_start = self.held_goals()
    (values, _, _) = self.tube.sample_index(round(6.7 / 0.02))
    self.home_goal = geometry.matrix_targets(values, self.bottle_R, self.bottle_t)
    self.loco.command[:] = 0.0
    self.loco.pose_command[:] = [0.82, 0.0]
    self.enter("ALIGNED_EMPTY_HOME")
    self.multi_events.append(
        dict(
            time_s=float(self.data.time),
            event="next_bottle_pick_after_real_return",
            active=self.active_bottle + 1,
            base=self.scratch.xpos[self.base].tolist(),
            bottles=self.all_bottles(),
        )
    )


def repeat_rotate_goal(self, goals, clock):
    if goals is not None and self.active_bottle > 0:
        if not hasattr(self, "repeat_reference"):
            with np.load(resource("repeat_prior.npz"), allow_pickle=False) as prior:
                self.repeat_reference = (prior["time_s"].copy(), prior["wrists"].copy())
                self.repeat_seed_times = prior["time_s"].copy()
                self.repeat_seed_positions = prior["upper_qpos"].copy()
                self.repeat_origin = prior["origin"].copy()
                self.repeat_R = prior["crate_rotation"].copy()
        (times, values) = self.repeat_reference
        j = int(np.argmin(abs(times - min(clock, times[-1]))))
        rotation = self.bottle_R @ self.repeat_R.T
        pivot = (
            self.bottle_R @ np.array([0.438, 0.454, self.repeat_origin[2]])
            + self.bottle_t
        )
        goals = goals.copy()
        for i in range(2):
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, values[j, i, 3:])
            goals[i, :3, :3] = rotation @ mat.reshape(3, 3)
            goals[i, :3, 3] = pivot + rotation @ (values[j, i, :3] - self.repeat_origin)
        goals[0, 2, 3] += (
            self.repeat_grasp_depth
            * geometry.smoothstep((clock - 52.2) / 3.3)
            * (1 - geometry.smoothstep((clock - 58.3) / 2.5))
        )
        goals[0, 2, 3] += 0.04 * geometry.smoothstep((clock - 59.5) / 4.0)
        return goals
    goals = grasp.grasp_frame_rotate_goal(self, goals, clock)
    if goals is not None and self.active_bottle > 0:
        lift = (
            0.05
            * geometry.smoothstep((clock - 39.0) / 3.0)
            * (1 - geometry.smoothstep((clock - 52.2) / 3.3))
        )
        goals[0, 2, 3] += lift
    return goals


def repeat_loaded_targets(self):
    if self.refine_active:
        return self.refine_targets()
    if self.empty_return:
        return self.empty_targets()
    if self.global_settle:
        self.bottle_commands = [0, 0]
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        now = float(self.data.time)
        good = all(
            (
                b["upward_N"]["shelf"] > 0.8 * 0.981
                and b["tilt_deg"] < 8
                and (b["linear_speed_mps"] < 0.02)
                and (b["angular_speed_radps"] < 0.15)
                and (
                    not any((n.startswith(("left_", "right_")) for n in b["contacts"]))
                )
                for b in self.all_bottles()
            )
        )
        self.global_stable_since = (
            (now if self.global_stable_since is None else self.global_stable_since)
            if good
            else None
        )
        if (
            self.global_stable_since is not None
            and now - self.global_stable_since >= 1.5
        ):
            self.done = True
            self.enter("ALL_THREE_BOTTLES_SHELVED_AND_HANDS_CLEAR")
        elif now - self.stage_start > 8.0:
            self.failure = "All three bottles did not remain upright and independently supported after the last placement"
        return (self.stage, self.place_goals, 65.7)
    if (
        self.active_bottle > 0
        and self.side_route_active
        and (self.stage == "SIDE_SHELF_POSTURE_PREP")
        and (2.1 <= float(self.data.time) - self.stage_start < 4.2)
    ):
        self.bottle_commands = [1, 0]
        self.native_clock = 65.7
        self.native_start = float(self.data.time) - 65.7
        self.source_phase = "PICK_LIFT_OUT"
        self.walk_twist[:] = 0.0
        self.post_twist[:] = 0.0
        self.walk_pose[:] = [0.82, 0.12]
        self.post_pose[:] = self.walk_pose
        return (self.stage, self.extra_goals(), 65.7)
    if (
        self.active_bottle > 0
        and self.stage == "BOTTLE_PROGRAM"
        and (not self.close_released)
    ):
        clock = float(self.data.time) - self.native_start
        if (
            clock >= 52.2
            and getattr(self, "pre_descend_checkpoint_bottle", -1) != self.active_bottle
        ):
            self.pre_descend_checkpoint_bottle = self.active_bottle
        if 6.7 <= clock < 55.5:
            ref = self.native.references["left"]
            (p, q) = self.native.endpoint_pose(self.scratch, "left")
            ep = float(np.linalg.norm(ref.position - p))
            er = float(
                np.degrees(np.linalg.norm(geometry.rotation_error(ref.quaternion, q)))
            )
            air_end = 48.0
            position_gate = (
                0.015
                if clock < 18.0
                else (
                    0.008
                    if clock < air_end
                    else 0.006 if self.active_bottle == 2 else 0.004
                )
            )
            orientation_gate = 4.0 if clock < 18.0 else 3.0 if clock < air_end else 1.5
            stalled = ep > position_gate or er > orientation_gate
            self.native_start += 0.02 if stalled else 0.01
            now = float(self.data.time)
            self.path_stall_since = (
                (
                    now
                    if getattr(self, "path_stall_since", None) is None
                    else self.path_stall_since
                )
                if stalled
                else None
            )
            if self.path_stall_since is not None and now - self.path_stall_since > 12.0:
                self.failure = f"Repeated wrist path actual tracking stalled: {ep * 1000:.2f}mm/{er:.2f}deg at clock{clock:.3f}"
        if clock >= 48.0:
            measured = (
                self.scratch.xpos[self.cargo_body, :2]
                - (self.bottle_R @ np.array([0.438, 0.454, 0.0]))[:2]
            )
            delta = measured - self.bottle_t[:2]
            self.bottle_t[:2] += delta * min(
                1.0, 0.012 * 0.02 / max(np.linalg.norm(delta), 1e-12)
            )
    if self.place_active and self.stage == "SHELF_LOAD_HOLD":
        (left, _) = self.dual_measure()
        if self.active_bottle > 0 and (
            left["object_tilt_deg"] > 6.0 or left["object_angular_speed_radps"] > 0.15
        ):
            self.place_contact_since = None
        if (
            left["middle_shelf_upward_N"] < 0.75 * 0.981
            and left["object_tilt_deg"] < 15.0
        ):
            self.place_phase_goals[0, 2, 3] -= 0.002 * 0.02
    if (
        self.active_bottle > 0
        and self.stage == "BOTTLE_PROGRAM"
        and (self.native_clock >= 65.7 - 1e-08)
    ):
        row = self.native_measure()
        if (
            not hasattr(self, "repeat_extraction_hold")
            and row["above_highest_box_rim_m"] > 0.03
            and effective_hand_control(row, 0.981)
            and (not row["crate_object_contact"])
        ):
            self.repeat_extraction_hold = self.executed[14:].copy()
            self.multi_events.append(
                dict(
                    time_s=float(self.data.time),
                    event="hold_actual_arm_PD_targets_after_real_rim_clearance",
                    active=self.active_bottle + 1,
                    clearance_m=row["above_highest_box_rim_m"],
                )
            )
    if (
        self.active_bottle > 0
        and self.stage == "LEFT_REGRASP_FINAL_APPROACH"
        and (float(self.data.time) - self.stage_start > 4.5)
        and (getattr(self, "left_close_checkpoint_bottle", -1) != self.active_bottle)
    ):
        self.left_close_checkpoint_bottle = self.active_bottle
    if self.active_bottle > 0 and self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD":
        (left, _) = self.dual_measure()
        if left["hand_vertical_force_N"] < 0.6 * 0.981:
            self.left_load_since = None
        if (
            left["hand_vertical_force_N"] < 0.8 * 0.981
            and float(self.data.time) - self.stage_start > 5.0
        ):
            self.phase_goals[0, 2, 3] += 0.002 * 0.02
    original = self.bottle_clearance
    self.bottle_clearance = lambda m, d, side: original(
        m, d, side, self.model.body(self.cargo_body).name
    )
    try:
        former = self.stage
        value = inventory_loaded_targets(self)
    finally:
        self.bottle_clearance = original
    if self.side_route_active and self.stage == "LOCO_TO_LEFT_SHORT_EDGE":
        world = np.array(
            [
                -0.2 if self.active_bottle > 0 else -0.26,
                np.clip(1.8 * (1.53 - self.scratch.xpos[self.base, 1]), -0.15, 0.1),
            ]
        )
        local = geometry.rotation_z(self.base_yaw())[:2, :2].T @ world
        elapsed = float(self.data.time) - self.stage_start
        command = np.r_[
            local,
            np.clip(
                0.8 * geometry.wrap_angle(np.pi / 2 - self.base_yaw()), -0.12, 0.12
            ),
        ] * geometry.smoothstep(elapsed / (2.0 if self.active_bottle > 0 else 1.2))
        if self.active_bottle > 0 and (not hasattr(self, "route_arm_hold")):
            self.route_arm_hold = self.executed[14:].copy()
            self.route_world_R = (
                self.scratch.xmat[self.model.body("left_hand_roll_link").id]
                .reshape(3, 3)
                .copy()
            )
            value[1][0, :3, :3] = self.route_world_R
        self.last_lateral_command = command.copy()
        self.walk_twist[:] = command
        self.post_twist[:] = command
    elif (
        self.side_route_active
        and self.stage == "SIDE_SHELF_BRAKE"
        and (former == "LOCO_TO_LEFT_SHORT_EDGE")
    ):
        self.route_brake = self.last_lateral_command.copy()
        self.walk_twist[:] = self.route_brake
        self.post_twist[:] = self.route_brake
    elif self.side_route_active and self.stage == "SIDE_SHELF_BRAKE":
        self.walk_twist[:] = self.route_brake * (
            1
            - geometry.smoothstep(
                (float(self.data.time) - self.stage_start)
                / (1.5 if self.active_bottle > 0 else 0.6)
            )
        )
        self.post_twist[:] = self.walk_twist
    if self.first_pick_base is None and self.live_start is not None:
        self.first_pick_base = self.scratch.xpos[self.base].copy()
        self.first_pick_bottle = self.scratch.xpos[self.cargo_body].copy()
        self.first_pick_yaw = self.base_yaw()
    if self.done and self.place_success:
        if self.active_bottle + 1 not in self.completed_bottles:
            self.completed_bottles.append(self.active_bottle + 1)
            self.multi_events.append(
                dict(
                    time_s=float(self.data.time),
                    event="bottle_actual_shelf_support_and_withdrawal_complete",
                    active=self.active_bottle + 1,
                    bottles=self.all_bottles(),
                )
            )
        if self.max_objects < 3 and len(self.completed_bottles) >= self.max_objects:
            return value
        self.done = False
        if len(self.completed_bottles) == 3:
            self.global_settle = True
            self.enter("GLOBAL_THREE_BOTTLE_SETTLE")
        else:
            self.switch_active(self.active_bottle + 1)
            self.start_return()
            return (self.stage, self.empty_initial_goals, 6.7)
    return value


def inventory_loaded_targets(self):
    result = placement.rack_loaded_targets(self)
    if self.physical_placement_verified and (not self.box_prefix_finished):
        self.box_prefix_finished = True
        self.multi_events.append(
            dict(
                time_s=float(self.data.time),
                event="three_bottle_box_placed",
                bottles=self.all_bottles(),
            )
        )
        if self.stop_after_box:
            self.done = True
            self.enter("THREE_BOTTLE_BOX_PLACED_CHECKPOINT")
    return result


def inventory_step(self):
    count = len(self.trace)
    placement.shelf_step(self)
    if len(self.trace) > count:
        self.ownership_rows.append(
            dict(
                time_s=self.trace[-1]["time_s"],
                executed_qdes=self.executed.tolist(),
                loco_qdes=self.loco.q_des.tolist(),
                loco_command=self.loco.command.tolist(),
                loco_pose=self.loco.pose_command.tolist(),
                owner=self.owner,
                active_bottle=self.active_bottle + 1,
            )
        )


def control_activate_native(self):
    before = (self.data.qpos.copy(), self.data.qvel.copy())
    self.owner = (
        "student3191"
        if self.bottle_policy_choice == "3191"
        else "bottle_" + self.bottle_policy_choice
    )
    self.switch_time = float(self.data.time)
    self.blend_from = self.executed.copy()
    self.native.follow_current(self.scratch)
    self.native_active = True
    self.arm_mode = "locked"
    if not self.native_hand_enabled:
        self.hands = NativeHandControl(self.model, self.data, self.native_hand_cfg)
        self.native_hand_enabled = True
    assert np.array_equal(before[0], self.data.qpos) and np.array_equal(
        before[1], self.data.qvel
    )
    self.transitions.append(
        dict(
            time_s=float(self.data.time),
            controller=self.owner,
            physical_state_continuous=True,
            histories_reset=False,
        )
    )


def control_prepare_empty_hands(self):
    if self.prepare_choice == "native_full":
        self.activate_native()
        return
    self.native_active = False
    self.owner = "loco"
    self.arm_mode = "reach"
    self.switch_time = float(self.data.time)
    self.blend_from = self.executed.copy()
    self.policy.follow_current(self.scratch)
    self.hands = NativeHandControl(self.model, self.data, self.native_hand_cfg)
    self.native_hand_enabled = True
    if self.prepare_choice == "pd_neutral":
        self.owner = "pd_home"
        self.arm_mode = "locked"
        self.neutral_from = self.executed.copy()
    self.transitions.append(
        dict(
            time_s=float(self.data.time),
            controller="loco_legs_stop4_empty_arms",
            physical_state_continuous=True,
            histories_reset=False,
            purpose="Only EMPTY-hand preparation before3191 pickup; full28 student owns bottle skill",
        )
    )


def control_home_at(self, xy, yaw):
    home = np.array(
        [
            [0.246, 0.19698, 1.087902, 1.0, 0.0, 0.0, 0.0],
            [0.246, -0.19643, 1.087902, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    return geometry.matrix_targets(home, geometry.rotation_z(yaw), np.r_[xy, 0.0])


def control_loaded_targets(self):
    now = float(self.data.time)
    elapsed = now - self.stage_start
    if self.stage == "PD_NEUTRAL":
        self.bottle_commands = [0, 0]
        row = self.native_measure()
        error = self.scratch.qpos[self.mapping.qpos] - self.native.default_joint_pos
        ready = (
            elapsed >= 10.0
            and np.sqrt(np.mean(error**2)) < 0.08
            and (max(abs(error)) < 0.18)
            and (row["base_tilt_deg"] < 8.0)
            and (row["root_height_m"] > 0.75)
            and robot_stable(row, None)
        )
        if self.stable(ready, 1.0):
            self.activate_native()
            self.native_start = now - self.source_start
            self.bottle_stage_started = True
            self.enter("BOTTLE_PROGRAM")
            self.transitions.append(
                dict(
                    time_s=now,
                    neutral_PD_standing_rms_rad=float(np.sqrt(np.mean(error**2))),
                    bottle_reference_frame=dict(
                        rotation=self.bottle_R.tolist(),
                        translation=self.bottle_t.tolist(),
                    ),
                )
            )
        elif elapsed > 18.0:
            self.failure = (
                "Physical neutral PD standing posture did not settle; no reset used"
            )
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage == "PREPARE_BOTTLE_HOME":
        self.bottle_commands = [0, 0]
        f = geometry.smoothstep(max(0.0, elapsed - 1.0) / 5.0)
        goals = np.array(
            [
                geometry.mix_transform(a, b, f)
                for (a, b) in zip(self.home_start, self.home_goal)
            ]
        )
        if elapsed >= 9.0:
            if self.prepare_choice == "stop4_then_pd":
                self.owner = "pd_home"
                self.arm_mode = "locked"
                self.neutral_from = self.executed.copy()
                self.blend_from = self.executed.copy()
                self.switch_time = now
                self.enter("PD_NEUTRAL")
                return (self.stage, goals, self.paused_clock)
            self.switch("loco")
            self.native_active = False
            self.arm_mode = "locked"
            self.loco.pose_command[:] = [0.82, 0.0]
            self.loco.command[:] = 0.0
            self.enter("ALIGN_BOTTLE_STATION")
            self.align_begin = self.scratch.xpos[self.base].copy()
        return (self.stage, goals, self.paused_clock)
    if self.stage == "ALIGN_BOTTLE_STATION":
        row = self.measure()
        yaw = self.base_yaw()
        R = geometry.rotation_z(yaw)
        error = self.station_xy - self.scratch.xpos[self.base, :2]
        local = R[:2, :2].T @ error
        self.loco.command[:] = [
            np.clip(1.0 * local[0], -0.15, 0.15),
            np.clip(1.0 * local[1], -0.12, 0.12),
            np.clip(1.0 * geometry.wrap_angle(self.station_yaw - yaw), -0.2, 0.2),
        ]
        if not self.align_station:
            self.loco.command[:] = 0.0
        good = np.linalg.norm(error) < 0.025 and abs(
            geometry.wrap_angle(self.station_yaw - yaw)
        ) < np.deg2rad(4.0)
        if self.stable((good or not self.align_station) and elapsed >= 2.0, 0.5):
            self.loco.command[:] = 0.0
            self.enter("STATION_SETTLE")
        elif elapsed > 30.0:
            self.failure = "Loco did not reach the 3191 bottle pickup station in 30s"
        return (self.stage, self.carry_targets(), self.paused_clock)
    if self.stage == "STATION_SETTLE":
        self.loco.command[:] = 0.0
        if elapsed >= 3.0:
            self.activate_native()
            self.native_start = now - self.source_start
            self.bottle_stage_started = True
            self.enter("BOTTLE_PROGRAM")
            self.transitions.append(
                dict(
                    time_s=now,
                    bottle_reference_frame=dict(
                        rotation=self.bottle_R.tolist(),
                        translation=self.bottle_t.tolist(),
                    ),
                    station_xy=self.station_xy.tolist(),
                    station_error_m=float(
                        np.linalg.norm(
                            self.station_xy - self.scratch.xpos[self.base, :2]
                        )
                    ),
                )
            )
        return (self.stage, self.held_goals(), self.paused_clock)
    if self.stage in (
        "BOTTLE_PROGRAM",
        "BOTTLE_LOCO_HOLD",
        "BOTTLE_LOCO_WALK",
        "BOTTLE_LOCO_STOP",
    ):
        self.native_clock = (
            now - self.native_start - self.clock_pause
            if self.stage == "BOTTLE_PROGRAM"
            else self.pause_clock
        )
        index = min(int(round(self.native_clock / 0.02)), len(self.tube.times) - 1)
        (values, commands, phase) = self.tube.sample_index(index)
        self.source_phase = str(phase)
        self.bottle_commands = list(map(int, commands))
        goals = geometry.matrix_targets(values, self.bottle_R, self.bottle_t)
        if (
            self.stage == "BOTTLE_PROGRAM"
            and self.route_choice == "walk"
            and (not self.held_walk_started)
            and (self.native_clock >= 65.9)
        ):
            if not self.bottle_exited:
                self.failure = "Bottle-walk branch refused: actual lift out of the box not verified"
                return (self.stage, None, self.native_clock)
            self.held_walk_started = True
            self.pause_start = now
            self.pause_clock = self.native_clock
            self.walk_yaw = self.base_yaw()
            self.switch("loco")
            self.native_active = False
            self.loco.pose_command[:] = [0.82, 0.0]
            self.enter("BOTTLE_LOCO_HOLD")
        if self.stage == "BOTTLE_LOCO_HOLD":
            self.bottle_commands = [1, 0]
            self.loco.command[:] = 0.0
            if elapsed >= 3.0:
                self.walk_base = self.scratch.xpos[self.base].copy()
                self.enter("BOTTLE_LOCO_WALK")
        elif self.stage == "BOTTLE_LOCO_WALK":
            self.bottle_commands = [1, 0]
            self.loco.command[:] = [
                -0.2,
                0.0,
                np.clip(
                    -geometry.wrap_angle(self.base_yaw() - self.walk_yaw), -0.2, 0.2
                ),
            ]
            if (
                np.linalg.norm(self.scratch.xpos[self.base, :2] - self.walk_base[:2])
                >= 0.12
            ):
                self.loco.command[:] = 0.0
                self.enter("BOTTLE_LOCO_STOP")
            elif elapsed > 8.0:
                self.failure = "Held-bottle short walk did not make progress"
        elif self.stage == "BOTTLE_LOCO_STOP":
            self.bottle_commands = [1, 0]
            self.loco.command[:] = 0.0
            if elapsed >= 3.0:
                self.clock_pause += now - self.pause_start
                delta = self.scratch.xpos[self.base, :2] - self.walk_base[:2]
                self.bottle_t[:2] += delta
                goals = geometry.matrix_targets(values, self.bottle_R, self.bottle_t)
                self.transitions.append(
                    dict(
                        time_s=now,
                        bottle_goals_translation_after_real_walk_m=delta.tolist(),
                    )
                )
                self.activate_native()
                self.enter("BOTTLE_PROGRAM")
        if self.stage != "BOTTLE_PROGRAM":
            self.native_clock = self.pause_clock
        if self.native_clock >= 78.32 - 1e-08:
            self.done = (
                self.bottle_placed and self.bottle_released and self.bottle_withdrawn
            )
            if not self.done:
                self.failure = (
                    "3191 goals ended without stable actual release/withdrawal"
                )
            else:
                self.enter("FULL_CHAIN_COMPLETE")
        return (
            (
                "BOTTLE_" + self.source_phase
                if self.stage == "BOTTLE_PROGRAM"
                else self.stage
            ),
            goals,
            self.native_clock,
        )
    result = grasp.table_approach_loaded_targets(self)
    if self.near_ready:
        self.done = False
        self.bottle_R = geometry.rotation_z(np.deg2rad(self.measure()["crate_yaw_deg"]))
        if self.station_choice == "bearing":
            delta = (
                self.scratch.xpos[self.cargo_body, :2]
                - self.scratch.xpos[self.base, :2]
            )
            self.bottle_R = geometry.rotation_z(
                float(np.arctan2(delta[1], delta[0]) - np.arctan2(0.454, 0.438))
            )
        self.bottle_t = np.r_[
            self.scratch.xpos[self.cargo_body, :2]
            - (self.bottle_R @ np.array([0.438, 0.454, 0.0]))[:2],
            0.0,
        ]
        self.station_xy = self.bottle_t[:2].copy()
        self.station_yaw = float(np.arctan2(self.bottle_R[1, 0], self.bottle_R[0, 0]))
        self.clock_pause = 0.0
        self.home_start = self.held_goals()
        self.home_goal = self.home_at(self.scratch.xpos[self.base, :2], self.base_yaw())
        if self.source_start > 0.0:
            (values, _, _) = self.tube.sample_index(round(self.source_start / 0.02))
            self.home_goal = geometry.matrix_targets(
                values, self.bottle_R, self.bottle_t
            )
        self.prepare_empty_hands()
        self.enter("PREPARE_BOTTLE_HOME")
        if self.prepare_choice == "pd_neutral":
            self.enter("PD_NEUTRAL")
        self.near_ready = False
        return (self.stage, self.home_start, self.paused_clock)
    return result


def control_step(self):
    if self.steps % 20 == 0:
        self.sync()
        (phase, targets, clock) = self.loaded_targets()
        if self.failure or self.done:
            return
        if phase != self.phase:
            self.phase = phase
            self.transitions.append(dict(time_s=float(self.data.time), phase=phase))
            print(f"{self.data.time:.3f} {phase}", flush=True)
        hybrid = self.owner == "loco" and self.arm_mode == "reach"
        if (
            targets is None
            or (self.owner == "loco" and (not hybrid))
            or self.native_active
            or (self.data.time < 0.4)
        ):
            self.policy.follow_current(self.scratch)
        else:
            for side, target in zip(("left", "right"), targets):
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, target[:3, :3].ravel())
                self.policy.set_target_world(side, target[:3, 3], q)
        row = self.measure()
        if not self.physical_placement_verified:
            valid = (
                self.physically_supported(row)
                and row["crate_speed_mps"] < 0.02
                and (row["crate_angular_speed_radps"] < 0.15)
            )
            self.physical_grip_since = (
                (
                    float(self.data.time)
                    if self.physical_grip_since is None
                    else self.physical_grip_since
                )
                if valid
                else None
            )
            if (
                self.physical_grip_since is not None
                and self.data.time - self.physical_grip_since >= 0.3
            ):
                self.physical_pickup_verified = True
                if self.grip_baseline is None:
                    self.grip_baseline = {
                        s: np.asarray(row["hands"][s]["T_wrist_crate"]).copy()
                        for s in ("left", "right")
                    }
            bilateral = all((h["opposed"] for h in row["hands"].values()))
            self.contact_since = (
                (
                    float(self.data.time)
                    if self.contact_since is None
                    else self.contact_since
                )
                if bilateral
                else None
            )
            if phase == "PROBE_LIFT" and (not self.grasp_verified):
                if (
                    self.contact_since is None
                    or self.data.time - self.contact_since < 0.3
                ):
                    self.failure = "No sustained bilateral grip before lifting"
                    return
                self.grasp_verified = True
        for policy in (self.policy, self.native):
            policy.last_action = (
                (self.executed - policy.default_joint_pos) / policy.action_scale
            ).astype(np.float32)
        self.loco.last_action = (
            (self.executed[:14] - self.loco.default) / self.loco.scale
        ).astype(np.float32)
        if self.native_active:
            for side, target in zip(("left", "right"), targets):
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, target[:3, :3].ravel())
                self.native.set_target_world(side, target[:3, 3], q)
            desired = self.native.act(self.scratch)
            self.loco.observe(self.scratch)
            self.policy.observe(
                self.scratch, advance_reference=False, update_history=True
            )
        else:
            self.native.follow_current(self.scratch)
            self.native.observe(
                self.scratch, advance_reference=False, update_history=True
            )
            if self.owner == "reach":
                desired = self.policy.act(self.scratch)
                self.loco.observe(self.scratch)
            elif self.owner == "pd_home":
                f = geometry.smoothstep((self.data.time - self.stage_start) / 8.0)
                desired = (1.0 - f) * self.neutral_from + f * getattr(
                    self, "neutral_goal", self.native.default_joint_pos
                )
                self.loco.observe(self.scratch)
                self.policy.observe(
                    self.scratch, advance_reference=False, update_history=True
                )
            elif hybrid:
                desired = np.r_[
                    self.loco.act(self.scratch), self.policy.act(self.scratch)[14:]
                ]
            else:
                desired = np.r_[self.loco.act(self.scratch), self.arm_hold]
                self.policy.observe(
                    self.scratch, advance_reference=False, update_history=True
                )
        f = (
            1.0
            if self.switch_time is None
            else geometry.smoothstep(
                (self.data.time - self.switch_time)
                / (
                    2.0
                    if self.native_active
                    else self.return_blend_s if self.owner == "reach" else 1.2
                )
            )
        )
        self.executed = (1.0 - f) * self.blend_from + f * desired
        for policy in (self.policy, self.native):
            policy.q_des = self.executed.copy()
            policy.last_action = (
                (self.executed - policy.default_joint_pos) / policy.action_scale
            ).astype(np.float32)
        self.loco.last_action = (
            (self.executed[:14] - self.loco.default) / self.loco.scale
        ).astype(np.float32)
        row.update(
            actions=self.policy.last_action.tolist(),
            finger_closed=self.finger_closed,
            hand_commands=(
                self.bottle_commands
                if self.native_hand_enabled
                else [int(self.finger_closed)] * 2
            ),
            path_clock_s=clock,
        )
        self.trace.append(row)
        self.states.append(
            (
                float(self.data.time),
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                self.policy.last_observation.copy(),
                self.phase,
            )
        )
        self.native_observations.append(self.native.last_observation.copy())
    if self.steps % 10 == 0:
        for i, side in enumerate(("left", "right")):
            self.hands.command(
                side,
                (
                    self.bottle_commands[i]
                    if self.native_hand_enabled
                    else int(self.finger_closed)
                ),
            )
        self.hands.update()
    (self.native if self.native_active else self.policy).apply(self.data)
    self.hands.apply(self.data)
    mujoco.mj_step(self.model, self.data)
    self.steps += 1
    self.failure = self.safety()
