"""Native CONTACT measurements; copied equations preserve the physical metric protocol."""

import copy
import mujoco
import numpy as np
from r2v2_deploy.data.grasp_recording import body_transform
from r2v2_deploy.control.geometry import (
    pose_quaternion,
    rotation_error_deg,
    contact_opposition_angle,
)
from r2v2_deploy.tasks.scene_spec import cylinder_envelope_in_crate
from r2v2_deploy.simulation.robot import SIDES


class ContactMeasurements:
    def _object_sync(self):
        for name in (
            "qpos",
            "qvel",
            "act",
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "qacc_warmstart",
            "xfrc_applied",
            "qfrc_applied",
        ):
            getattr(self.scratch, name)[:] = getattr(self.data, name)
        self.scratch.time = self.data.time
        mujoco.mj_forward(self.model, self.scratch)
        m, d = (self.model, self.scratch)
        object_T = body_transform(d, self.object_body)
        wrist_T = body_transform(d, self.wrist_bodies[self.side])
        relation = np.linalg.inv(wrist_T) @ object_T
        rotation = object_T[:3, :3]
        axis_z = float(rotation[2, 2])
        half_extent = self.profile["height_m"] / 2.0 * abs(axis_z) + self.profile[
            "radius_m"
        ] * np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        bottom = float(object_T[2, 3] - half_extent)
        jac_p, jac_r = (np.zeros((3, m.nv)), np.zeros((3, m.nv)))
        mujoco.mj_jacBody(m, d, jac_p, jac_r, self.object_body)
        velocity, angular = (jac_p @ d.qvel, jac_r @ d.qvel)
        fingers, hand_force = ({}, np.zeros(3))
        contact_records = []
        thumb_normals, finger_normals = ([], [])
        table_force = np.zeros(3)
        table = floor = False
        active_count = parked_count = table_hand = 0
        hand_penetration = self_penetration = 0.0
        for index, contact in enumerate(d.contact):
            a, b = (int(contact.geom1), int(contact.geom2))
            sa, sb = (self.geom_sides.get(a), self.geom_sides.get(b))
            if contact.dist <= 0 and sa and (sa == sb):
                self_penetration = max(self_penetration, -float(contact.dist))
            if contact.dist <= 0 and (
                a == self.table_geom and sb or (b == self.table_geom and sa)
            ):
                table_hand += 1
            if self.object_geom not in (a, b):
                continue
            other = b if a == self.object_geom else a
            side = self.geom_sides.get(other)
            if side:
                hand_penetration = max(hand_penetration, max(0.0, -float(contact.dist)))
            local = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, local)
            if contact.dist > 0 or local[0] <= 0.001:
                continue
            force = np.asarray(contact.frame).reshape(3, 3).T @ local[:3]
            force *= 1.0 if b == self.object_geom else -1.0
            if other == self.table_geom:
                table = True
                table_force += force
            elif other == self.floor_geom:
                floor = True
            elif side == self.side:
                active_count += 1
                hand_force += force
                name = m.geom(other).name
                part = next(
                    (
                        p
                        for p in ("thumb", "index", "middle", "ring", "pinky")
                        if f"_{p}_" in name
                    ),
                    "palm",
                )
                fingers[part] = fingers.get(part, 0.0) + max(0.0, float(local[0]))
                object_local = rotation.T @ (np.asarray(contact.pos) - object_T[:3, 3])
                contact_records.append(
                    dict(
                        part=part,
                        normal_force_N=float(local[0]),
                        force_on_object_world_N=force.tolist(),
                        position_world_m=np.asarray(contact.pos).tolist(),
                        position_object_m=object_local.tolist(),
                        depth_below_object_top_m=float(
                            self.profile["height_m"] / 2.0 - object_local[2]
                        ),
                    )
                )
                if local[0] > 0.05 and part != "palm":
                    normal = np.asarray(contact.frame).reshape(3, 3)[0] * (
                        1.0 if b == self.object_geom else -1.0
                    )
                    (thumb_normals if part == "thumb" else finger_normals).append(
                        normal
                    )
            elif side:
                parked_count += 1
        q = d.qpos[m.jnt_qposadr[self.hand_joints]]
        limits = m.jnt_range[self.hand_joints]
        violation = max(
            0.0, float(np.max(limits[:, 0] - q)), float(np.max(q - limits[:, 1]))
        )
        mimic_error = max(
            (abs(d.qpos[a] - r * d.qpos[b] - o) for a, b, r, o in self.mimics)
        )
        slip = angle = None
        if self.baseline_relation is not None and (not self.release_commanded):
            slip = float(
                np.linalg.norm(relation[:3, 3] - self.baseline_relation[:3, 3])
            )
            angle = rotation_error_deg(relation[:3, :3], self.baseline_relation[:3, :3])
        alignment_slip = alignment_angle = None
        if self.alignment_relation is not None:
            alignment_slip = float(
                np.linalg.norm(relation[:3, 3] - self.alignment_relation[:3, 3])
            )
            alignment_angle = rotation_error_deg(
                relation[:3, :3], self.alignment_relation[:3, :3]
            )
        opposition = contact_opposition_angle(thumb_normals, finger_normals)
        pair_contact = bool(
            fingers.get("thumb", 0.0) > 0.05
            and any(
                (
                    fingers.get(p, 0.0) > 0.05
                    for p in ("index", "middle", "ring", "pinky")
                )
            )
        )
        self.current_metrics = dict(
            time_s=float(d.time),
            object_position_m=object_T[:3, 3].tolist(),
            object_quaternion_wxyz=pose_quaternion(object_T).tolist(),
            T_world_object=object_T.tolist(),
            T_world_wrist=wrist_T.tolist(),
            T_wrist_object=relation.tolist(),
            bottom_height_m=bottom,
            clearance_m=bottom - self.table_height,
            object_tilt_deg=float(np.rad2deg(np.arccos(np.clip(axis_z, -1.0, 1.0)))),
            object_linear_speed_mps=float(np.linalg.norm(velocity)),
            object_angular_speed_radps=float(np.linalg.norm(angular)),
            table_contact=table,
            floor_contact=floor,
            table_vertical_force_N=float(table_force[2]),
            hand_vertical_force_N=float(hand_force[2]),
            fingers_normal_force_N=fingers,
            hand_object_contacts=contact_records,
            finger_pair_contact=pair_contact,
            directional_opposition_deg=opposition,
            opposed_contact=bool(
                pair_contact and opposition is not None and (opposition >= 120.0)
            ),
            active_hand_object_contact_count=active_count,
            parked_hand_contact_count=parked_count,
            hand_table_contact_count=table_hand,
            hand_object_penetration_m=hand_penetration,
            hand_self_penetration_m=self_penetration,
            hand_joint_violation_rad=violation,
            mimic_error_rad=float(mimic_error),
            actuator_torque_Nm=float(np.max(np.abs(d.actuator_force))),
            grasp_slip_m=slip,
            grasp_rotation_slip_deg=angle,
            grasp_verified=self.grasp_verified,
            alignment_slip_m=alignment_slip,
            alignment_rotation_slip_deg=alignment_angle,
            wrist_position_error_m=float(
                np.linalg.norm(wrist_T[:3, 3] - self.active_goal[:3, 3])
            ),
            wrist_orientation_error_deg=rotation_error_deg(
                wrist_T[:3, :3], self.active_goal[:3, :3]
            ),
            finite_state=bool(
                np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
            ),
        )
        return self.current_metrics

    def _robot_sync(self):
        self._object_sync()
        m, d, row = (self.model, self.scratch, self.current_metrics)
        errors = {}
        for side, body in self.wrist_bodies.items():
            actual, goal = (body_transform(d, body), self.goal_wrist_transforms[side])
            v, w = self.policy.wrist_twist(d, side)
            errors[side] = dict(
                position_m=float(np.linalg.norm(actual[:3, 3] - goal[:3, 3])),
                orientation_deg=rotation_error_deg(actual[:3, :3], goal[:3, :3]),
                linear_speed_mps=float(np.linalg.norm(v)),
                angular_speed_radps=float(np.linalg.norm(w)),
                T_world_wrist=actual.tolist(),
                T_world_wrist_goal=goal.tolist(),
            )
        crate = body_transform(d, self.crate_body)
        dims = self.path.scene["crate_dimensions_depth_width_height_m"]
        envelope = cylinder_envelope_in_crate(
            np.asarray(row["T_world_object"]),
            crate,
            0.02,
            0.06,
            dims,
            self.path.scene["crate_wall_thickness_m"],
            self.layout["crate_opening_bounds_local_m"],
        )
        jp, jr = (np.zeros((3, m.nv)), np.zeros((3, m.nv)))
        mujoco.mj_jacBody(m, d, jp, jr, self.base)
        root_v, root_w = (np.linalg.norm(jp @ d.qvel), np.linalg.norm(jr @ d.qvel))
        mujoco.mj_jacBody(m, d, jp, jr, self.crate_body)
        crate_v, crate_w = (np.linalg.norm(jp @ d.qvel), np.linalg.norm(jr @ d.qvel))
        crate_force, crate_contact, robot_prop, nonhand_object, self_depth = (
            0.0,
            False,
            [],
            0,
            0.0,
        )
        crate_table = False
        pickup_tray_contact, pickup_tray_force = (False, 0.0)
        for index, c in enumerate(d.contact):
            a, b = map(int, c.geom)
            if c.dist > 0.0:
                continue
            if a in self.robot_geoms and b in self.robot_geoms:
                self_depth = max(self_depth, -float(c.dist))
            if (
                a in self.crate_geoms
                and b in self.table_geoms
                or (b in self.crate_geoms and a in self.table_geoms)
            ):
                crate_table = True
            if (
                a in self.robot_geoms
                and b in self.table_geoms | self.crate_geoms
                or (b in self.robot_geoms and a in self.table_geoms | self.crate_geoms)
            ):
                robot_prop.append(
                    dict(
                        geoms=[m.geom(a).name, m.geom(b).name],
                        penetration_m=-float(c.dist),
                    )
                )
            if self.object_geom in (a, b):
                other = b if a == self.object_geom else a
                if other in self.pickup_tray_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, index, force)
                    if force[0] > 0.001:
                        pickup_tray_contact = True
                        world_force = c.frame.reshape(3, 3).T @ force[:3]
                        pickup_tray_force += float(world_force[2]) * (
                            1.0 if b == self.object_geom else -1.0
                        )
                if other in self.robot_geoms and other not in self.geom_sides:
                    nonhand_object += 1
                if other in self.crate_geoms:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, index, force)
                    if force[0] > 0.001:
                        crate_contact = True
                        world_force = c.frame.reshape(3, 3).T @ force[:3]
                        crate_force += float(world_force[2]) * (
                            1.0 if b == self.object_geom else -1.0
                        )
        if self.mode == "air":
            diagnostic = self.diagnostic_data
            diagnostic.qpos[:] = d.qpos
            diagnostic.qvel[:] = d.qvel
            diagnostic.ctrl[:] = d.ctrl
            mujoco.mj_forward(self.diagnostic_model, diagnostic)
            for c in diagnostic.contact:
                a, b = map(int, c.geom)
                if c.dist < 0.0 and (
                    a in self.robot_geoms
                    and b in self.table_geoms | self.crate_geoms
                    or (
                        b in self.robot_geoms
                        and a in self.table_geoms | self.crate_geoms
                    )
                ):
                    robot_prop.append(
                        dict(
                            geoms=[m.geom(a).name, m.geom(b).name],
                            penetration_m=-float(c.dist),
                        )
                    )
        hand_bottom = float("inf")
        for g, side in self.geom_sides.items():
            if side != "left" or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            center, half = (m.geom_aabb[g, :3], m.geom_aabb[g, 3:])
            rotation = d.geom_xmat[g].reshape(3, 3)
            hand_bottom = min(
                hand_bottom,
                float(
                    (d.geom_xpos[g] + rotation @ center)[2] - np.abs(rotation[2]) @ half
                ),
            )
        rim = float(
            crate[2, 3]
            + crate[2, 2] * dims[2]
            + abs(crate[2, 0]) * dims[0] / 2.0
            + abs(crate[2, 1]) * dims[1] / 2.0
        )
        low, high = (
            np.asarray(envelope["local_min_m"]),
            np.asarray(envelope["local_max_m"]),
        )
        row.update(
            wrist_errors=errors,
            **envelope,
            root_speed_mps=float(root_v),
            root_angular_speed_radps=float(root_w),
            pickup_tray_contact=pickup_tray_contact,
            pickup_tray_vertical_force_N=pickup_tray_force,
            pickup_tray_clearance_m=row["bottom_height_m"]
            - self.table_height
            - self.layout["delivery_environment"]["pickup_tray"]["height_m"]
            if self.config["pickup_tray_height_m"]
            else None,
            root_height_m=float(d.xpos[self.base, 2]),
            base_tilt_deg=float(
                np.degrees(
                    np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1.0, 1.0))
                )
            ),
            foot_drift_m=float(
                np.max(np.linalg.norm(self._feet(d) - self.foot_anchor, axis=1))
            ),
            crate_object_contact=crate_contact,
            crate_table_contact=crate_table,
            crate_vertical_force_N=crate_force,
            T_world_crate=crate.tolist(),
            crate_tilt_deg=float(
                np.degrees(np.arccos(np.clip(crate[2, 2], -1.0, 1.0)))
            ),
            crate_speed_mps=float(crate_v),
            crate_angular_speed_radps=float(crate_w),
            object_inside_crate=bool(
                envelope["interior_xy_margin_m"] >= 0.0
                and low[2] >= self.path.scene["crate_bottom_thickness_m"] - 0.003
                and (high[2] <= dims[2] + 0.003)
            ),
            hand_above_rim_m=hand_bottom - rim,
            robot_prop_contacts=robot_prop,
            nonhand_object_contact_count=nonhand_object,
            robot_self_penetration_m=self_depth,
            mode=self.mode,
            grasp_verified=self.grasp_verified,
            drop_verified=getattr(self, "drop_verified", False),
        )
        if pickup_tray_contact:
            row["table_contact"] = True
            row["table_vertical_force_N"] += pickup_tray_force
        if robot_prop:
            self.stage_collision = True
        return row

    def _pickup_sync(self):
        row = self._robot_sync()
        supported = set()
        for contact in self.scratch.contact:
            if contact.dist > 0.0:
                continue
            a, b = map(int, contact.geom)
            if self.floor_geom not in (a, b):
                continue
            other = b if a == self.floor_geom else a
            if other in self.foot_geoms:
                name = self.model.body(self.model.geom_bodyid[other]).name
                supported.update((s for s in SIDES if name.startswith(s + "_")))
        foot_speeds = []
        for side in SIDES:
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.scratch,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.body(side + "_ankle_roll_link").id,
                velocity,
                0,
            )
            foot_speeds.append(float(np.linalg.norm(velocity[3:])))
        row.update(
            support_feet_count=len(supported),
            foot_linear_speed_mps=max(foot_speeds),
            waist_pitch_rad=float(
                self.scratch.qpos[self.model.joint("waist_pitch_joint").qposadr[0]]
            ),
            waist_yaw_rad=float(
                self.scratch.qpos[self.model.joint("waist_yaw_joint").qposadr[0]]
            ),
        )
        if self.mode == "contact":
            crate = np.array(row["T_world_crate"])
            floor = (
                crate[2, 3] + crate[2, 2] * self.path.scene["crate_bottom_thickness_m"]
            )
            row["clearance_m"] = row["bottom_height_m"] - floor
            if row["crate_object_contact"]:
                row["clearance_m"] = min(row["clearance_m"], 0.0)
            depth, width, height = self.path.scene[
                "crate_dimensions_depth_width_height_m"
            ]
            rim = (
                crate[2, 3]
                + crate[2, 2] * height
                + abs(crate[2, 0]) * depth / 2
                + abs(crate[2, 1]) * width / 2
            )
            row["above_highest_box_rim_m"] = row["bottom_height_m"] - rim
        return row

    def _delivery_sync(self):
        row = self._pickup_sync()
        if not hasattr(self, "delivery_geoms"):
            return row
        force, touching = (0.0, False)
        for i, contact in enumerate(self.scratch.contact):
            a, b = map(int, contact.geom)
            if contact.dist > 0 or self.object_geom not in (a, b):
                continue
            other = b if a == self.object_geom else a
            if other == self.model.geom("dropoff_table_top").id:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.scratch, i, f)
                if f[0] > 0.001:
                    touching = True
                    force += float((contact.frame.reshape(3, 3).T @ f[:3])[2]) * (
                        1 if b == self.object_geom else -1
                    )
        row.update(
            delivery_table_contact=touching,
            delivery_table_vertical_force_N=force,
            delivery_table_xy_margin_m=float(
                np.min(
                    np.array([0.1, 0.12])
                    - np.abs(
                        np.asarray(row["object_position_m"])[:2] - self.delivery_xy
                    )
                    - 0.02
                )
            ),
        )
        return row

    def sync(self):
        row = self._delivery_sync()
        if hasattr(self, "delivery_geoms"):
            table = self.program.document["scene"]["delivery_dropoff_table"]
            row["delivery_table_xy_margin_m"] = float(
                np.min(
                    np.asarray(table["half_size_m"][:2])
                    - np.abs(
                        np.asarray(row["object_position_m"][:2])
                        - table["center_world_m"][:2]
                    )
                    - 0.02
                )
            )
        return row

    @property
    def done(self):
        return self.phase in ("COMPLETE", "FAILED")

    def fail(self, reason):
        if self.done:
            return
        self.failure, self.failure_phase = (str(reason), self.phase)
        self.phase = "FAILED"
        self.transitions.append(
            dict(time_s=float(self.data.time), state="FAILED", reason=str(reason))
        )

    def _stable(self, valid, duration):
        if not valid:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = float(self.data.time)
        return self.data.time - self.stable_since >= duration - 1e-08

    def record(self):
        if not self.keep_trace:
            return
        mapping = self.hands.maps[self.side]
        reference = self.hands.controllers[self.side].reference
        self.samples.append(
            dict(
                time_s=float(self.data.time),
                phase=self.phase,
                command=int(self.hands.controllers[self.side].command),
                metrics=copy.deepcopy(self.current_metrics),
                qpos=self.data.qpos.tolist(),
                qvel=self.data.qvel.tolist(),
                reference_q_rad=reference.position.tolist(),
                reference_qd_rad_s=reference.velocity.tolist(),
                reference_qdd_rad_s2=reference.acceleration.tolist(),
                active_hand_q_rad=self.data.qpos[mapping.qpos].tolist(),
                active_hand_torque_Nm=self.data.actuator_force[
                    mapping.actuators
                ].tolist(),
            )
        )
