"""Explicit adapters composing frozen body policies and Cartesian arm targets.

The adapters preserve policy histories across controller changes. Their factory
order follows the workflow initialization; closures capture the previous callable
once, so no module globals or class hierarchy are changed.
"""

import numpy as np
from . import geometry


def configure_grasp_gate_live_act(self):
    self.original_native_act = self.native.act

    def grasp_gate_live_act(data):
        grasp_gate_target = self.original_native_act(data)
        if self.live_start is not None:
            grasp_gate_f = geometry.smoothstep(
                (float(data.time) - self.live_start) / max(self.live_blend_s, 1e-06)
            )
            if grasp_gate_f < 1.0:
                for grasp_gate_side in ("left", "right"):
                    grasp_gate_ref = self.native.references[grasp_gate_side]
                    self.policy.set_target_world(
                        grasp_gate_side,
                        grasp_gate_ref.goal_position,
                        grasp_gate_ref.goal_quaternion,
                    )
                grasp_gate_source = np.r_[
                    self.loco.act(data), self.policy.act(data)[14:]
                ]
                self.owner = "LIVE_HANDOFF_TO_3191"
                return (
                    1.0 - grasp_gate_f
                ) * grasp_gate_source + grasp_gate_f * grasp_gate_target
            self.owner = "student3191"
        return grasp_gate_target

    self.native.act = grasp_gate_live_act


def configure_arm_planner_body(self):
    arm_planner_raw = self.original_native_act

    def arm_planner_body(data):
        arm_planner_q = arm_planner_raw(data)
        self.raw_body_command = arm_planner_q.copy()
        return arm_planner_q

    self.original_native_act = arm_planner_body


def configure_arm_planner_prepared(self):
    arm_planner_old = self.policy.act

    def arm_planner_prepared(data):
        arm_planner_q = arm_planner_old(data)
        self.arm_nn_prior = arm_planner_q[14:].copy()
        if self.native_active and self.raw_body_command is not None:
            arm_planner_f = geometry.smoothstep(
                (float(data.time) - self.live_start) / max(self.live_blend_s, 1e-06)
            )
            self.arm_nn_prior = (
                1.0 - arm_planner_f
            ) * self.arm_nn_prior + arm_planner_f * self.raw_body_command[14:]
        if self.arm_planning:
            arm_planner_q[14:] = self.plan(
                data, self.native if self.native_active else self.policy
            )
        return arm_planner_q

    self.policy.act = arm_planner_prepared


def configure_arm_planner_mixed(self):
    arm_planner_native = self.native.act

    def arm_planner_mixed(data):
        arm_planner_q = arm_planner_native(data)
        if (
            self.live_start is not None
            and data.time - self.live_start >= self.live_blend_s
        ):
            self.arm_nn_prior = self.raw_body_command[14:].copy()
        if self.arm_planning:
            arm_planner_q[14:] = self.plan(data, self.native)
        self.owner = (
            "3191_LEGS_WAIST__PLANNED_ARMS"
            if self.live_start is not None
            and data.time - self.live_start >= self.live_blend_s
            else "LOCO_TO_3191_LEGS__PLANNED_ARMS"
        )
        return arm_planner_q

    self.native.act = arm_planner_mixed


def configure_posture_ik_mixed(self):
    posture_ik_old = self.native.act

    def posture_ik_mixed(data):
        posture_ik_q = posture_ik_old(data)
        self.loco.command[:] = 0.0
        self.loco.pose_command[:] = [0.82, self.pick_pitch]
        posture_ik_lower = self.loco.act(data)
        posture_ik_q[:12] = posture_ik_lower[:12]
        if self.upper_cmd is not None:
            posture_ik_q[12:14] = self.upper_cmd[:2]
        self.owner = "LOCO_LEGS12__SEEDED_WAIST_ARMS_TEACHER16"
        self.lower_rows.append(
            dict(
                time_s=float(data.time),
                loco_qdes_rad=posture_ik_lower.tolist(),
                executed_leg_dim=12,
            )
        )
        return posture_ik_q

    self.native.act = posture_ik_mixed


def configure_arms_only_mixed(self):
    arms_only_old = self.native.act

    def arms_only_mixed(data):
        arms_only_q = arms_only_old(data)
        self.loco.command[:] = 0.0
        self.loco.pose_command[:] = [0.82, self.pick_pitch]
        arms_only_lower = self.loco.act(data)
        arms_only_q[:14] = arms_only_lower
        self.owner = "LOCO14__NATIVE_BRANCH_ARM_IK14"
        self.lower_rows.append(
            dict(
                time_s=float(data.time),
                loco_qdes_rad=arms_only_lower.tolist(),
                executed_leg_dim=14,
            )
        )
        return arms_only_q

    self.native.act = arms_only_mixed


def configure_carry_body_commanded(self):
    carry_body_old = self.loco.act

    def carry_body_commanded(data):
        if self.post_start is not None:
            self.loco.command[:] = self.post_twist
            self.loco.pose_command[:] = self.post_pose
        return carry_body_old(data)

    self.loco.act = carry_body_commanded


def configure_loaded_route_commanded(self):
    loaded_route_old_loco = self.loco.act

    def loaded_route_commanded(data):
        if self.walk_start is not None:
            self.loco.command[:] = self.walk_twist
            self.loco.pose_command[:] = self.walk_pose
        return loaded_route_old_loco(data)

    self.loco.act = loaded_route_commanded


def configure_repeat_compensate_yaw(self):
    repeat_old_loco = self.loco.act

    def repeat_compensate_yaw(data):
        repeat_q = repeat_old_loco(data)
        if (
            self.repeat_yaw_teacher
            and self.active_bottle > 0
            and (self.stage in ["ALIGNED_EMPTY_HOME", "BOTTLE_PROGRAM"])
        ):
            if not hasattr(self, "yaw_teacher_entry"):
                self.yaw_teacher_entry = float(data.time)
                self.yaw_teacher_from = float(repeat_q[12])
            repeat_wanted = np.clip(
                geometry.wrap_angle(self.first_pick_yaw - self.base_yaw()), -0.5, 0.5
            )
            repeat_wanted += float(
                data.qfrc_bias[self.mapping.dofs[12]] / self.native.kp[12]
            )
            repeat_f = geometry.smoothstep(
                (float(data.time) - self.yaw_teacher_entry) / 3.0
            )
            repeat_q[12] = (
                1 - repeat_f
            ) * self.yaw_teacher_from + repeat_f * repeat_wanted
            self.yaw_teacher_last = float(repeat_q[12])
            self.yaw_teacher_release = float(data.time)
        elif self.repeat_yaw_teacher and hasattr(self, "yaw_teacher_last"):
            repeat_f = geometry.smoothstep(
                (float(data.time) - self.yaw_teacher_release) / 4.0
            )
            repeat_q[12] = (1 - repeat_f) * self.yaw_teacher_last + repeat_f * repeat_q[
                12
            ]
        return repeat_q

    self.loco.act = repeat_compensate_yaw


def configure_repeat_repeat_waist(self):
    repeat_old_native = self.native.act

    def repeat_repeat_waist(data):
        repeat_q = repeat_old_native(data)
        if (
            self.repeat_use_waist
            and self.active_bottle > 0
            and (self.stage == "BOTTLE_PROGRAM")
            and (self.upper_cmd is not None)
        ):
            repeat_q[12:14] = self.upper_cmd[:2]
            self.repeat_waist_hold = self.upper_cmd[:2].copy()
            self.repeat_waist_release = float(data.time)
            self.owner = "LOCO_LEGS12__EXTERNAL_WAIST2_ARM_IK14"
        elif self.repeat_use_waist and hasattr(self, "repeat_waist_hold"):
            repeat_f = geometry.smoothstep(
                (float(data.time) - self.repeat_waist_release) / 4.0
            )
            repeat_q[12:14] = (
                1 - repeat_f
            ) * self.repeat_waist_hold + repeat_f * repeat_q[12:14]
            if repeat_f < 1.0:
                self.owner = "LOCO_LEGS12__WAIST_TEACHER_RETURN_BLEND__ARM_IK14"
        if self.active_bottle > 0:
            self.waist_teacher_rows.append(
                dict(
                    time_s=float(data.time),
                    stage=self.stage,
                    owner=self.owner,
                    desired_qdes=repeat_q.tolist(),
                    loco_qdes=self.loco.q_des.tolist(),
                )
            )
        return repeat_q

    self.native.act = repeat_repeat_waist


def configure_grasp_stance_source_command(self):
    grasp_stance_old_loco = self.loco.act

    def grasp_stance_source_command(data):
        if self.source_stance_active:
            self.loco.command[:] = self.walk_twist
            self.loco.pose_command[:] = self.walk_pose
        return grasp_stance_old_loco(data)

    self.loco.act = grasp_stance_source_command
