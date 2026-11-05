"""Physical contact measurements, inventory, and safety checks."""

from __future__ import annotations
import numpy as np
from common.r2v2_crate_lift_metrics import (
    _contact_force_on_geom as _contact_force_on_geom,
)
from common.r2v2_crate_lift_metrics import _descendant as _descendant
from common.r2v2_crate_lift_metrics import measure_crate_lift as measure_crate_lift
from common.r2v2_grasp_recording import body_transform as body_transform
from r2v2_deploy.simulation.measurements import (
    ContactMeasurements as ContactMeasurements,
)
from r2v2_deploy.simulation.robot import urdf_hand_joints as urdf_hand_joints
from r2v2_deploy.tasks.contact_reach import safety_conditions as safety_conditions
from r2v2_deploy.tasks.physical_validity import (
    effective_hand_control as effective_hand_control,
)
from r2v2_deploy.tasks.physical_validity import robot_stable as robot_stable
from r2v2_deploy.tasks.physical_validity import (
    stable_table_contact as stable_table_contact,
)
import mujoco as mujoco
import types as types
from . import geometry


def sequence_safety(self):
    failure = carry_safety(self)
    if failure or not self.natural_solo or self.steps % 10:
        return failure
    now = float(self.data.time)
    (left, right) = self.dual_measure()
    controlled = effective_hand_control(right, 0.981)
    self.natural_loss_since = (
        (now if self.natural_loss_since is None else self.natural_loss_since)
        if not controlled
        else None
    )
    if self.natural_loss_since is not None and now - self.natural_loss_since >= 0.2:
        return "Right-only transport lost real right-hand control for 0.2s"
    if left["active_hand_object_contact_count"]:
        return "Resting empty left hand contacted the transported bottle"
    return None


def carry_safety(self):
    if not (self.dual_mode or self.dual_pending):
        return repeat_safety(self)
    result = handoff_safety(self)
    result = result or self.inventory_guard()
    if self.steps % 10 == 0:
        (l, r) = self.dual_measure()
        self.dual_rows.append(
            dict(
                time_s=float(self.data.time),
                stage=self.stage,
                left=l,
                right=r,
                bottles=self.all_bottles(),
            )
        )
        if l["shelf_bottle_contact"]:
            return (
                result
                or "Unexpected active bottle/shelf contact during dual-hand transport"
            )
    return result


def repeat_safety(self):
    if not self.empty_return:
        return inventory_safety(self)
    if self.steps % 10:
        return None
    (left, right) = self.dual_measure()
    row = left.copy()
    row["parked_hand_contact"] = False
    row["hand_object_penetration_m"] = max(
        left["hand_object_penetration_m"], right["hand_object_penetration_m"]
    )
    for name, bad in safety_conditions(row, 0.0):
        if bool(bad):
            return name
    return self.inventory_guard()


def inventory_all_bottles(self):
    result = []
    for i, (body, geom) in enumerate(zip(self.bottle_bodies, self.bottle_geoms)):
        forces = {"crate": 0.0, "shelf": 0.0, "left": 0.0, "right": 0.0, "floor": 0.0}
        contacts = []
        for k, c in enumerate(self.scratch.contact):
            (a, b) = (int(c.geom1), int(c.geom2))
            if c.dist > 0 or geom not in (a, b):
                continue
            other = b if a == geom else a
            f = np.empty(6)
            mujoco.mj_contactForce(self.model, self.scratch, k, f)
            if f[0] < 0.001:
                continue
            world = c.frame.reshape(3, 3).T @ f[:3] * (1.0 if b == geom else -1.0)
            kind = (
                "crate"
                if other in self.crate_geoms
                else (
                    "shelf"
                    if other == self.middle_geom
                    else (
                        "floor"
                        if other == self.floor
                        else self.proxy.geom_sides.get(other)
                    )
                )
            )
            if kind in forces:
                forces[kind] += float(world[2])
            contacts.append(self.model.geom(other).name)
        joint = int(self.model.body_jntadr[body])
        assert self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
        qa = int(self.model.jnt_qposadr[joint])
        position = self.data.qpos[qa : qa + 3]
        quaternion = self.data.qpos[qa + 3 : qa + 7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        matrix = rotation.reshape(3, 3)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.scratch, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
        )
        result.append(
            dict(
                id=i + 1,
                position_m=position.tolist(),
                quaternion_wxyz=quaternion.tolist(),
                pose_source="Current physical free-joint qpos; raw contact/velocity sample convention unchanged",
                tilt_deg=float(np.degrees(np.arccos(np.clip(matrix[2, 2], -1, 1)))),
                linear_speed_mps=float(np.linalg.norm(velocity[3:])),
                angular_speed_radps=float(np.linalg.norm(velocity[:3])),
                upward_N=forces,
                contacts=contacts,
            )
        )
    return result


def inventory_safety(self):
    result = shelf_pose_safety(self)
    extra = self.inventory_guard()
    return result or extra


def inventory_inventory_guard(self):
    if hasattr(self, "bottle_bodies") and self.steps % 10 == 0:
        bottles = self.all_bottles()
        physical = self.native_measure()
        self.multi_rows.append(
            dict(
                time_s=float(self.data.time),
                active_bottle=self.active_bottle + 1,
                stage=self.stage,
                bottles=bottles,
                physical={
                    k: physical[k]
                    for k in [
                        "body_joint_violation_rad",
                        "base_tilt_deg",
                        "root_height_m",
                        "robot_self_penetration_m",
                        "robot_prop_penetration_m",
                        "robot_prop_force_N",
                        "hand_joint_violation_rad",
                        "mimic_error_rad",
                        "support_feet_count",
                        "root_speed_mps",
                    ]
                },
            )
        )
        if any((b["upward_N"]["floor"] > 0.001 for b in bottles)):
            return "A three-bottle payload touched the floor"
        force = 0.0
        penetration = 0.0
        for k, c in enumerate(self.scratch.contact):
            (a, b) = (int(c.geom1), int(c.geom2))
            idle = set(self.bottle_geoms) - {self.cargo_geom}
            if c.dist < 0 and (
                a in idle and b in self.robot or (b in idle and a in self.robot)
            ):
                value = np.empty(6)
                mujoco.mj_contactForce(self.model, self.scratch, k, value)
                force += float(np.linalg.norm(value[:3]))
                penetration = max(penetration, -float(c.dist))
        if force > 150 or penetration > 0.003:
            return "Non-target bottle robot contact exceeded original 3mm/150N physical limits"
    return None


def shelf_pose_safety(self):
    result = shelf_safety(self)
    if self.side_route_active and self.steps % 10 == 0:
        (l, rr) = self.dual_measure()
        row = l.copy()
        row.update(
            time_s=float(self.data.time),
            route_elapsed_s=float(self.data.time) - self.route_start,
            left_contact_count=l["active_hand_object_contact_count"],
            right_contact_count=rr["active_hand_object_contact_count"],
            route_command=self.walk_twist.tolist(),
            base=self.scratch.xpos[self.base].tolist(),
        )
        self.route_rows.append(row)
    return result


def shelf_dual_measure(self):
    (left, right) = loaded_route_dual_measure(self)
    if not hasattr(self, "middle_geom"):
        return (left, right)
    force = 0.0
    count = 0
    other_rack = False
    for i, c in enumerate(self.scratch.contact):
        if c.dist > 0:
            continue
        (a, b) = (int(c.geom1), int(c.geom2))
        if self.cargo_geom not in (a, b):
            continue
        other = b if a == self.cargo_geom else a
        if other not in self.rack_geoms:
            continue
        f = np.empty(6)
        mujoco.mj_contactForce(self.model, self.scratch, i, f)
        if f[0] < 0.001:
            continue
        if other == self.middle_geom:
            count += 1
            world = (
                c.frame.reshape(3, 3).T
                @ f[:3]
                * (1.0 if b == self.cargo_geom else -1.0)
            )
            force += float(world[2])
        else:
            other_rack = True
    pos = self.scratch.xpos[self.cargo_body]
    local = self.scratch.xmat[self.rack_body].reshape(3, 3).T @ (
        pos - self.scratch.xpos[self.rack_body]
    )
    margin = float(np.min(self.model.geom_size[self.middle_geom, :2] - abs(local[:2])))
    for row in (left, right):
        row.update(
            middle_shelf_contact_count=count,
            middle_shelf_upward_N=force,
            middle_shelf_xy_margin_m=margin,
            other_shelf_bottle_contact=other_rack,
        )
    return (left, right)


def shelf_shelf_support(self, row):
    return (
        row["middle_shelf_contact_count"] > 0
        and row["middle_shelf_upward_N"] > 0.01
        and (row["middle_shelf_xy_margin_m"] > 0.025)
        and (row["object_tilt_deg"] < 30.0)
        and (
            not any(
                (
                    row[k]
                    for k in (
                        "crate_object_contact",
                        "floor_contact",
                        "table_contact",
                        "delivery_table_contact",
                        "other_shelf_bottle_contact",
                    )
                )
            )
        )
    )


def shelf_shelf_stable(self, row):
    return (
        self.shelf_support(row)
        and row["middle_shelf_upward_N"] > 0.8 * 0.981
        and (row["object_tilt_deg"] < 8.0)
        and (row["object_linear_speed_mps"] < 0.02)
        and (row["object_angular_speed_radps"] < 0.15)
    )


def shelf_safety(self):
    if not self.place_active:
        return loaded_route_safety(self)
    if self.steps % 10:
        return None
    (left, right) = self.dual_measure()
    row = left.copy()
    grip = bool(effective_hand_control(left, 0.981)) or bool(
        effective_hand_control(right, 0.981)
    )
    supported = self.shelf_support(left)
    now = float(self.data.time)
    self.bottle_loss_since = (
        (now if self.bottle_loss_since is None else self.bottle_loss_since)
        if not (grip or supported)
        else None
    )
    row["raw_single_hand_parked_contact"] = row["parked_hand_contact"]
    row["parked_hand_contact"] = False
    row["hand_object_penetration_m"] = max(
        left["hand_object_penetration_m"], right["hand_object_penetration_m"]
    )
    row.update(
        time_s=now,
        left_object_contact_count=left["active_hand_object_contact_count"],
        right_object_contact_count=right["active_hand_object_contact_count"],
        left_vertical_force_N=left["hand_vertical_force_N"],
        right_vertical_force_N=right["hand_vertical_force_N"],
        stage=self.stage,
        placement_elapsed_s=now - self.place_start,
        independent_shelf_support=supported,
        actual_hand_control=grip,
        hand_commands=self.bottle_commands.copy(),
    )
    for name, bad in safety_conditions(
        row, 0.0 if self.bottle_loss_since is None else now - self.bottle_loss_since
    ):
        if bool(bad):
            row["safety_failure"] = name
            self.place_rows.append(row)
            return name
    if row["other_shelf_bottle_contact"]:
        self.place_rows.append(row)
        return "Bottle contacted non-target shelf structure"
    self.place_rows.append(row)
    return None


def loaded_route_dual_measure(self):
    (left, right) = handoff_dual_measure(self)
    return (self.rack_measure(left), self.rack_measure(right))


def loaded_route_native_measure(self):
    return self.rack_measure(control_native_measure(self))


def loaded_route_rack_measure(self, row):
    if not hasattr(self, "rack_geoms"):
        return row
    penetration = 0.0
    force = 0.0
    touches = 0
    object_contact = False
    for i, c in enumerate(self.scratch.contact):
        if c.dist > 0:
            continue
        (a, b) = (int(c.geom1), int(c.geom2))
        if a not in self.rack_geoms and b not in self.rack_geoms:
            continue
        other = b if a in self.rack_geoms else a
        f = np.empty(6)
        mujoco.mj_contactForce(self.model, self.scratch, i, f)
        if other in self.robot:
            penetration = max(penetration, -float(c.dist))
            force += float(np.linalg.norm(f[:3]))
            touches += int(f[0] > 0.001)
        if other == self.cargo_geom and f[0] > 0.001:
            object_contact = True
    row["robot_prop_penetration_m"] = max(row["robot_prop_penetration_m"], penetration)
    row["robot_prop_force_N"] += force
    row.update(
        shelf_robot_penetration_m=penetration,
        shelf_robot_contact_force_N=force,
        shelf_robot_contact_count=touches,
        shelf_bottle_contact=object_contact,
    )
    return row


def loaded_route_safety(self):
    result = regrasp_safety(self)
    if self.walk_start is not None and self.steps % 10 == 0:
        row = self.post_rows[-1].copy()
        row.update(
            walk_elapsed_s=float(self.data.time) - self.walk_start,
            walk_twist=self.walk_twist.tolist(),
            walk_pose=self.walk_pose.tolist(),
        )
        self.walk_rows.append(row)
        if row.get("shelf_bottle_contact", False):
            return "Unexpected bottle/shelf contact during held locomotion"
    return result


def regrasp_safety(self):
    result = handoff_safety(self)
    if self.second_start is not None and self.steps % 10 == 0:
        self.second_rows.append(self.post_rows[-1].copy())
    return result


def handoff_dual_measure(self):
    old = self.proxy.side
    self.proxy.side = "left"
    left = control_native_measure(self)
    self.proxy.side = "right"
    right = control_native_measure(self)
    self.proxy.side = old
    return (left, right)


def handoff_safety(self):
    if self.post_start is None:
        return carry_body_safety(self)
    if self.steps % 10:
        return None
    (left, right) = self.dual_measure()
    self.right_contact_seen |= right["active_hand_object_contact_count"] > 0
    row = (left if self.handoff_owner == "left" else right).copy()
    both_hold = bool(effective_hand_control(left, 0.981)) or bool(
        effective_hand_control(right, 0.981)
    )
    now = float(self.data.time)
    self.bottle_loss_since = (
        (now if self.bottle_loss_since is None else self.bottle_loss_since)
        if not both_hold
        else None
    )
    row["raw_single_hand_parked_contact"] = row["parked_hand_contact"]
    row["parked_hand_contact"] = False
    row["hand_object_penetration_m"] = max(
        left["hand_object_penetration_m"], right["hand_object_penetration_m"]
    )
    row.update(
        hand_owner=self.handoff_owner,
        both_hands_designated_grasping=True,
        left_object_contact_count=left["active_hand_object_contact_count"],
        right_object_contact_count=right["active_hand_object_contact_count"],
        left_vertical_force_N=left["hand_vertical_force_N"],
        right_vertical_force_N=right["hand_vertical_force_N"],
        post_elapsed_s=now - self.post_start,
        twist=self.post_twist.tolist(),
    )
    for name, bad in safety_conditions(
        row, 0.0 if self.bottle_loss_since is None else now - self.bottle_loss_since
    ):
        if bool(bad):
            row["safety_failure"] = name
            self.bottle_rows.append(row)
            self.post_rows.append(row)
            return name
    self.bottle_rows.append(row)
    self.post_rows.append(row)
    return None


def carry_body_safety(self):
    result = grasp_frame_safety(self)
    if self.post_start is not None and self.steps % 10 == 0:
        row = self.bottle_rows[-1].copy()
        row.update(
            hand_owner="left",
            post_elapsed_s=float(self.data.time) - self.post_start,
            twist=self.post_twist.tolist(),
        )
        self.post_rows.append(row)
    return result


def grasp_frame_safety(self):
    result = control_safety(self)
    for name in (
        "bottle_picked",
        "bottle_exited",
        "bottle_placed",
        "bottle_released",
        "bottle_withdrawn",
    ):
        if hasattr(self, name):
            setattr(self, name, bool(getattr(self, name)))
    return result


def control_make_measurement_proxy(self):
    p = types.SimpleNamespace(
        model=self.model,
        data=self.data,
        scratch=self.scratch,
        object_body=self.cargo_body,
        object_geom=self.cargo_geom,
        wrist_bodies={
            s: self.model.body(s + "_hand_roll_link").id for s in ("left", "right")
        },
        side="left",
        profile={"height_m": 0.12, "radius_m": 0.02},
        table_height=0.98,
        table_geom=self.model.geom("lift_table_geom").id,
        floor_geom=self.floor,
        baseline_relation=None,
        alignment_relation=None,
        release_commanded=False,
        grasp_verified=False,
        active_goal=np.eye(4),
        geom_sides={},
        mimics=[],
    )
    p.geom_sides = {
        g: s
        for g in self.hand_geoms
        for s in ("left", "right")
        if _descendant(self.model, int(self.model.geom_bodyid[g]), p.wrist_bodies[s])
    }
    joints = urdf_hand_joints()
    p.hand_joints = np.array([self.model.joint(n).id for n in joints])
    for name, j in joints.items():
        mm = j.find("mimic")
        if mm is not None:
            p.mimics.append(
                (
                    self.model.joint(name).qposadr[0],
                    self.model.joint(mm.get("joint")).qposadr[0],
                    float(mm.get("multiplier", "1")),
                    float(mm.get("offset", "0")),
                )
            )
    return p


def control_native_measure(self):
    p = self.proxy
    p.active_goal = self.held_goals()[0]
    p.release_commanded = self.bottle_release_position is not None
    row = ContactMeasurements._object_sync(p)
    q = self.scratch.qpos[self.mapping.qpos]
    limits = self.model.jnt_range[self.mapping.joints]
    row["body_joint_violation_rad"] = float(
        max(0.0, np.max(limits[:, 0] - q), np.max(q - limits[:, 1]))
    )
    R = self.scratch.xmat[self.base].reshape(3, 3)
    row.update(
        base_tilt_deg=float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))),
        root_height_m=float(self.scratch.xpos[self.base, 2]),
        robot_self_penetration_m=0.0,
        robot_prop_penetration_m=0.0,
        robot_prop_force_N=0.0,
        nonfoot_ground_contact=False,
        parked_hand_contact=row["parked_hand_contact_count"] > 0,
        nonhand_object_contact=False,
        crate_object_contact=False,
        delivery_table_contact=False,
        delivery_table_vertical_N=0.0,
        delivery_table_xy_margin_m=0.0,
    )
    (jp, jr) = (np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv)))
    mujoco.mj_jacBody(self.model, self.scratch, jp, jr, self.base)
    row["root_speed_mps"] = float(np.linalg.norm(jp @ self.scratch.qvel))
    row["root_angular_speed_radps"] = float(np.linalg.norm(jr @ self.scratch.qvel))
    feet_speed = []
    supported = set()
    for side in ("left", "right"):
        mujoco.mj_jacBody(
            self.model,
            self.scratch,
            jp,
            jr,
            self.model.body(side + "_ankle_roll_link").id,
        )
        feet_speed.append(float(np.linalg.norm(jp @ self.scratch.qvel)))
    for i, c in enumerate(self.scratch.contact):
        if c.dist > 0.0:
            continue
        (a, b) = (int(c.geom1), int(c.geom2))
        if a in self.robot and b in self.robot:
            row["robot_self_penetration_m"] = max(
                row["robot_self_penetration_m"], -float(c.dist)
            )
        if self.floor in (a, b):
            other = b if a == self.floor else a
            row["nonfoot_ground_contact"] |= (
                other in self.robot and other not in self.foot
            )
            if other in self.foot:
                for side in ("left", "right"):
                    if _descendant(
                        self.model,
                        int(self.model.geom_bodyid[other]),
                        self.model.body(side + "_ankle_roll_link").id,
                    ):
                        supported.add(side)
        props = self.crate_geoms | {
            self.destination_geom,
            self.model.geom("lift_table_geom").id,
        }
        if a in self.robot and b in props or (b in self.robot and a in props):
            f = np.empty(6)
            mujoco.mj_contactForce(self.model, self.scratch, i, f)
            row["robot_prop_penetration_m"] = max(
                row["robot_prop_penetration_m"], -float(c.dist)
            )
            row["robot_prop_force_N"] += float(np.linalg.norm(f[:3]))
        if self.cargo_geom in (a, b):
            other = b if a == self.cargo_geom else a
            f = np.empty(6)
            mujoco.mj_contactForce(self.model, self.scratch, i, f)
            if f[0] <= 0.001:
                continue
            row["nonhand_object_contact"] |= (
                other in self.robot and other not in self.hand_geoms
            )
            row["crate_object_contact"] |= other in self.crate_geoms
            if other == self.destination_geom:
                row["delivery_table_contact"] = True
                world = (
                    c.frame.reshape(3, 3).T
                    @ f[:3]
                    * (1.0 if b == self.cargo_geom else -1.0)
                )
                row["delivery_table_vertical_N"] += float(world[2])
    local = self.scratch.xmat[self.destination_body].reshape(3, 3).T @ (
        np.asarray(row["object_position_m"]) - self.scratch.xpos[self.destination_body]
    )
    row["delivery_table_xy_margin_m"] = float(
        np.min(self.model.geom_size[self.destination_geom, :2] - abs(local[:2]))
    )
    crate = self.scratch.xpos[self.crate_body]
    cr = self.scratch.xmat[self.crate_body].reshape(3, 3)
    row["above_highest_box_rim_m"] = row["bottom_height_m"] - (
        crate[2] + 0.16 * cr[2, 2] + abs(cr[2, 0]) * 0.12 + abs(cr[2, 1]) * 0.18
    )
    row.update(
        foot_linear_speed_mps=max(feet_speed),
        support_feet_count=len(supported),
        foot_drift_m=self.peaks["foot_drift_m"],
    )
    row.update(source_phase=getattr(self, "source_phase", "PREPARE"), stage=self.stage)
    for name in ("finite_state", "floor_contact"):
        row[name] = np.bool_(row[name])
    return row


def control_measure(self):
    row = crate_delivery_measure(self)
    if getattr(self, "native", None) is not None:
        row["student3191_active"] = (
            self.native_active and self.bottle_policy_choice == "3191"
        )
        row["bottle_policy_active"] = self.native_active
        row["bottle_reference_clock_s"] = self.native_clock
        if self.native_active:
            for side in ("left", "right"):
                ref = self.native.references[side]
                (p, q) = self.native.endpoint_pose(self.scratch, side)
                row["wrists"][side].update(
                    goal_m=ref.goal_position.tolist(),
                    position_error_m=float(np.linalg.norm(p - ref.goal_position)),
                    orientation_error_deg=float(
                        np.degrees(
                            2
                            * np.arccos(np.clip(abs(q @ ref.goal_quaternion), 0.0, 1.0))
                        )
                    ),
                )
    return row


def control_safety(self):
    if not getattr(self, "native_hand_enabled", False):
        return crate_delivery_safety(self)
    if self.steps % 10:
        return None
    row = self.native_measure()
    now = float(self.data.time)
    requires = (
        self.bottle_stage_started
        and self.bottle_commands[0] == 1
        and (
            getattr(self, "source_phase", "")
            in (
                "PICK_VERIFY",
                "PICK_LIFT_OUT",
                "DELIVERY_TRANSFER",
                "DELIVERY_UPRIGHT",
                "DELIVERY_LOWER",
                "DELIVERY_RELEASE",
                "DELIVERY_RETREAT",
            )
            or self.stage.startswith("BOTTLE_LOCO")
        )
    )
    held = bool(effective_hand_control(row, 0.981))
    lost = requires and (not held)
    self.bottle_loss_since = (
        (now if self.bottle_loss_since is None else self.bottle_loss_since)
        if lost
        else None
    )
    for name, bad in safety_conditions(
        row, 0.0 if self.bottle_loss_since is None else now - self.bottle_loss_since
    ):
        if bool(bad):
            self.bottle_rows.append(dict(row, safety_failure=name))
            return name
    self.bottle_picked |= requires and held and (not row["crate_object_contact"])
    self.bottle_exited |= (
        self.bottle_picked and held and (row["above_highest_box_rim_m"] >= 0.01)
    )
    placed = bool(stable_table_contact(row))
    if (
        self.bottle_stage_started
        and self.bottle_exited
        and (self.bottle_commands[0] == 0)
        and (self.bottle_release_position is None)
    ):
        self.bottle_release_position = np.asarray(row["T_world_wrist"])[:3, 3].copy()
    self.bottle_released |= (
        self.bottle_exited
        and placed
        and (self.bottle_release_position is not None)
        and (row["active_hand_object_contact_count"] == 0)
    )
    if self.bottle_release_position is not None:
        self.bottle_withdrawn |= (
            self.bottle_released
            and placed
            and (row["active_hand_object_contact_count"] == 0)
            and (
                np.linalg.norm(
                    np.asarray(row["T_world_wrist"])[:3, 3]
                    - self.bottle_release_position
                )
                >= 0.03
            )
        )
    stable = (
        self.bottle_withdrawn
        and placed
        and (row["active_hand_object_contact_count"] == 0)
        and (getattr(self, "source_phase", "") == "DELIVERY_SETTLE")
        and robot_stable(row, None)
    )
    self.bottle_settle_since = (
        (now if self.bottle_settle_since is None else self.bottle_settle_since)
        if stable
        else None
    )
    self.bottle_placed |= (
        self.bottle_settle_since is not None and now - self.bottle_settle_since >= 1.0
    )
    self.bottle_rows.append(row)
    return None


def crate_delivery_bottle_metrics(self):
    if not self.bottle_enabled:
        return None
    (model, data) = (self.model, self.scratch)
    T = body_transform(data, self.cargo_body)
    C = body_transform(data, self.crate_body)
    relative = np.linalg.inv(C) @ T
    axis = relative[:3, 2]
    extent = 0.06 * np.abs(axis) + 0.02 * np.sqrt(np.maximum(0.0, 1.0 - axis**2))
    inside = bool(
        np.all(np.abs(relative[:2, 3]) + extent[:2] <= np.array([0.116, 0.176]) + 0.002)
        and relative[2, 3] - extent[2] >= 0.005 - 0.002
        and (relative[2, 3] + extent[2] <= 0.16 + 0.002)
    )
    center_contained = bool(
        np.all(np.abs(relative[:2, 3]) <= np.array([0.12, 0.18]) + 0.015)
        and -0.005 <= relative[2, 3] <= 0.25
    )
    mujoco.mj_jacBody(model, data, self.cargo_jp, self.cargo_jr, self.cargo_body)
    mujoco.mj_jacBody(model, data, self.crate_jp, self.crate_jr, self.crate_body)
    velocity = self.cargo_jp @ data.qvel
    angular = self.cargo_jr @ data.qvel
    relative_velocity = velocity - (
        self.crate_jp @ data.qvel
        + np.cross(self.crate_jr @ data.qvel, T[:3, 3] - C[:3, 3])
    )
    floorN = crateN = 0.0
    ground = table = False
    robot_contacts = 0
    for i, c in enumerate(data.contact):
        (a, b) = (int(c.geom1), int(c.geom2))
        if self.cargo_geom not in (a, b):
            continue
        other = b if a == self.cargo_geom else a
        (local, force) = _contact_force_on_geom(model, data, i, self.cargo_geom)
        active = bool(local[0] > 1e-08 or c.dist <= 0)
        if other in self.crate_geoms:
            crateN += float(force[2])
        if other == self.crate_floor_geom:
            floorN += float(force[2])
        ground |= active and other == self.floor
        table |= active and other in (
            self.destination_geom,
            self.model.geom("lift_table_geom").id,
        )
        robot_contacts += int(active and other in self.robot)
    return dict(
        position_m=T[:3, 3].tolist(),
        quaternion_wxyz=data.xquat[self.cargo_body].tolist(),
        relative_crate_position_m=relative[:3, 3].tolist(),
        tilt_deg=float(np.rad2deg(np.arccos(np.clip(T[2, 2], -1, 1)))),
        relative_tilt_deg=float(np.rad2deg(np.arccos(np.clip(relative[2, 2], -1, 1)))),
        speed_mps=float(np.linalg.norm(velocity)),
        angular_speed_radps=float(np.linalg.norm(angular)),
        relative_speed_mps=float(np.linalg.norm(relative_velocity)),
        fully_inside_cavity=inside,
        center_contained=center_contained,
        crate_floor_vertical_N=floorN,
        total_crate_vertical_N=crateN,
        ground_contact=ground,
        table_contact=table,
        robot_contacts=robot_contacts,
    )


def crate_delivery_measure(self):
    row = crate_destination_measure(self)
    if not hasattr(self, "cargo_jp"):
        return row
    row["payload_bottle"] = self.bottle_metrics()
    row["total_carried_mass_kg"] = self.payload_mass
    angle = np.deg2rad(row["crate_yaw_deg"] - self.target_layout["yaw"])
    row["box_table_alignment_deg"] = float(
        abs(np.rad2deg(0.5 * np.arctan2(np.sin(2 * angle), np.cos(2 * angle))))
    )
    row["robot_table_alignment_deg"] = float(
        abs(
            np.rad2deg(
                geometry.loaded_loco_tests_wrap_angle(
                    np.deg2rad(row["base_yaw_deg"] - self.target_layout["yaw"])
                )
            )
        )
    )
    return row


def crate_delivery_physically_supported(self, row):
    return (
        crate_physically_supported(self, row)
        and sum((h["vertical_N"] for h in row["hands"].values()))
        >= 0.8 * self.payload_mass * 9.81
    )


def crate_delivery_safety(self):
    failure = crate_motion_safety(self)
    if failure or not getattr(self, "cargo_geom", None):
        return failure
    for c in self.data.contact:
        (a, b) = (int(c.geom1), int(c.geom2))
        if self.cargo_geom in (a, b):
            other = b if a == self.cargo_geom else a
            if other == self.floor and c.dist <= 0:
                return "Payload bottle reached the floor"
            if other in self.robot and c.dist < -0.002:
                return "Robot/payload bottle penetration >2mm"
    return None


def crate_destination_measure(self):
    row = loaded_motion_measure(self)
    if hasattr(self, "placement_mode"):
        row["loco_pose_command"] = self.loco.pose_command.tolist()
        row["point_accuracy_3cm_passed"] = row["destination_error_xy_m"] < 0.03
    return row


def loaded_motion_measure(self):
    row = crate_motion_measure(self)
    if not hasattr(self, "crate_body"):
        return row
    yaw = self.base_yaw()
    cr = self.scratch.xmat[self.crate_body].reshape(3, 3)
    row.update(
        base_yaw_deg=float(np.rad2deg(yaw)),
        crate_yaw_deg=float(np.rad2deg(np.arctan2(cr[1, 0], cr[0, 0]))),
        base_angular_speed_radps=float(np.linalg.norm(self.scratch.qvel[3:6])),
    )
    if self.turn_target_yaw is not None:
        row["turn_error_deg"] = float(
            np.rad2deg(
                geometry.loaded_loco_tests_wrap_angle(self.turn_target_yaw - yaw)
            )
        )
        row["turn_progress_deg"] = float(np.rad2deg(self.turn_unwrapped))
    return row


def crate_motion_measure(self):
    r = crate_measure(self)
    if not hasattr(self, "destination_geom"):
        return r
    r["source_table_contact"] = r["crate_table_contact"]
    destination_force = 0.0
    destination_contact = False
    for i, c in enumerate(self.scratch.contact):
        (a, b) = (int(c.geom1), int(c.geom2))
        target = a if a in self.crate_geoms else b if b in self.crate_geoms else None
        if target is not None and self.destination_geom in (a, b):
            (local, force) = _contact_force_on_geom(self.model, self.scratch, i, target)
            destination_force += float(force[2])
            destination_contact |= bool(local[0] > 1e-08 or c.dist <= 0.0)
    r.update(
        destination_table_contact=destination_contact,
        destination_vertical_N=destination_force,
        source_vertical_N=r["table_vertical_N"],
        crate_table_contact=r["crate_table_contact"] or destination_contact,
        destination_error_xy_m=float(
            np.linalg.norm(np.asarray(r["crate_position_m"])[:2] - self.destination[:2])
        ),
        arm_mode=self.arm_mode,
    )
    r["table_vertical_N"] += destination_force
    return r


def crate_motion_safety(self):
    failure = crate_safety(self)
    if failure or not hasattr(self, "destination_body"):
        return failure
    for contact in self.data.contact:
        if contact.dist >= -0.002:
            continue
        (a, b) = (int(contact.geom1), int(contact.geom2))
        for robot, other in ((a, b), (b, a)):
            if robot not in self.robot or robot in self.hand_geoms:
                continue
            body = int(self.model.geom_bodyid[other])
            if any(
                (
                    _descendant(self.model, body, table)
                    for table in (self.source_body, self.destination_body)
                )
            ):
                return "Non-hand robot/table penetration >2mm"
    return None


def crate_physically_supported(self, row):
    return (
        row["clearance_m"] > 0.008
        and (not row["crate_table_contact"])
        and (not row["crate_floor_contact"])
        and (row["crate_tilt_deg"] < 8.0)
        and all((h["opposed"] and h["vertical_N"] > 0.2 for h in row["hands"].values()))
        and (sum((h["vertical_N"] for h in row["hands"].values())) >= 0.8 * 0.4 * 9.81)
    )


def crate_safety(self):
    if (
        not np.isfinite(self.data.qpos).all()
        or not np.isfinite(self.data.qvel).all()
        or self.data.warning.number.any()
    ):
        return "Nonfinite physical state or MuJoCo warning"
    q = self.data.qpos[self.mapping.qpos]
    limits = self.model.jnt_range[self.mapping.joints]
    excess = np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0.0)
    self.peaks["joint_limit_excess_rad"] = max(
        self.peaks["joint_limit_excess_rad"], float(excess.max())
    )
    if excess.max() > 0.01:
        return (
            "Body joint physical limit excess >0.01rad: "
            + (
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_arm_pitch_joint",
                "left_arm_yaw_joint",
                "left_hand_pitch_joint",
                "left_hand_roll_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_arm_pitch_joint",
                "right_arm_yaw_joint",
                "right_hand_pitch_joint",
                "right_hand_roll_joint",
            )[int(excess.argmax())]
        )
    for contact in self.data.contact:
        if contact.dist >= 0:
            continue
        (a, b) = (int(contact.geom1), int(contact.geom2))
        if self.floor in (a, b):
            other = b if a == self.floor else a
            if other in self.robot and other not in self.foot:
                return "Non-foot floor contact: " + self.model.geom(other).name
        if a in self.robot and b in self.robot:
            self.peaks["self_penetration_m"] = max(
                self.peaks["self_penetration_m"], -float(contact.dist)
            )
            if -contact.dist > 0.002:
                return "Robot self-penetration >2mm"
        if self.mode == "contact" and (a in self.robot) != (b in self.robot):
            other = b if a in self.robot else a
            robot = a if a in self.robot else b
            if self.model.geom(other).name.startswith("crate_"):
                self.peaks["hand_crate_penetration_m"] = max(
                    self.peaks["hand_crate_penetration_m"], -float(contact.dist)
                )
                if robot not in self.hand_geoms:
                    return "Non-hand robot/crate contact"
                if -contact.dist > 0.003:
                    return "Hand/crate penetration >3mm"
    return None


def crate_measure(self):
    self.sync()
    wrists = {}
    for side in ("left", "right"):
        (position, quaternion) = self.policy.endpoint_pose(self.scratch, side)
        ref = self.policy.references[side]
        wrists[side] = dict(
            position_m=position.tolist(),
            goal_m=ref.goal_position.tolist(),
            position_error_m=float(np.linalg.norm(position - ref.goal_position)),
            orientation_error_deg=float(
                np.rad2deg(
                    2 * np.arccos(np.clip(abs(quaternion @ ref.goal_quaternion), 0, 1))
                )
            ),
            speed_mps=float(
                np.linalg.norm(self.policy.wrist_twist(self.scratch, side)[0])
            ),
        )
    feet = np.array(
        [
            self.scratch.xpos[self.model.body(s + "_ankle_roll_link").id]
            for s in ("left", "right")
        ]
    )
    drift = float(np.linalg.norm(feet - self.foot_start, axis=1).max())
    self.peaks["foot_drift_m"] = max(self.peaks["foot_drift_m"], drift)
    result = dict(
        time_s=float(self.data.time),
        phase=self.phase,
        wrists=wrists,
        foot_drift_m=drift,
        base_height_m=float(self.scratch.xpos[self.base, 2]),
        base_position_world_m=self.scratch.xpos[self.base].tolist(),
        body_owner=self.owner,
        velocity_command_body=None if self.loco is None else self.loco.command.tolist(),
    )
    if self.mode != "stand":
        raw = measure_crate_lift(
            self.model, self.scratch, 0.98, virtual_props=self.mode == "air"
        )
        result.update(
            clearance_m=raw["clearance_m"],
            table_vertical_N=raw["table_vertical_force_N"],
            crate_position_m=raw["crate_position_m"],
            crate_tilt_deg=raw["crate_tilt_deg"],
            crate_speed_mps=raw["crate_linear_speed_m_s"],
            crate_angular_speed_radps=raw["crate_angular_speed_rad_s"],
            crate_table_contact=raw["table_contact"],
            crate_floor_contact=raw["floor_contact"],
        )
        result["hands"] = {}
        for side in ("left", "right"):
            hand = raw["hands"][side]
            contacts = [
                c
                for c in hand["contacts"]
                if c["crate_geom"] == f"crate_{side}_vertical_grip"
            ]
            thumb = sum((c["normal_force_N"] for c in contacts if c["part"] == "thumb"))
            fingers = sum(
                (
                    c["normal_force_N"]
                    for c in contacts
                    if c["part"] in ("index", "middle", "ring", "pinky")
                )
            )
            result["hands"][side] = dict(
                opposed=bool(thumb > 0.05 and fingers > 0.05),
                thumb_normal_N=thumb,
                finger_normal_N=fingers,
                vertical_N=hand["vertical_force_N"],
                active_contact_count=sum((int(c["active"]) for c in hand["contacts"])),
                T_wrist_crate=hand["T_wrist_crate"],
                grip_slip_m=(
                    None
                    if self.grip_baseline is None
                    else float(
                        np.linalg.norm(
                            np.asarray(hand["T_wrist_crate"])[:3, 3]
                            - self.grip_baseline[side][:3, 3]
                        )
                    )
                ),
            )
    return result
