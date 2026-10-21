"""Single-policy native CONTACT simulation, without teacher or IK construction."""

import copy
import mujoco
import numpy as np
from r2v2_deploy.control.exit_placement_policy import (
    ExitPlacementReachPolicy,
    TASK_ID,
    TASK_SCOPE,
)
from r2v2_deploy.control.hand_control import DualHandControl
from r2v2_deploy.control.geometry import pose_quaternion
from r2v2_deploy.data.grasp_recording import body_transform
from r2v2_deploy.simulation.measurements import ContactMeasurements
from r2v2_deploy.simulation.front_drop_scene import (
    build_front_drop_model,
    build_front_drop_collision_diagnostic,
)
from r2v2_deploy.simulation.reach_sim import (
    initialize_robot,
    foot_collision_ids,
    load_reach_config,
    resolve_asset,
    require_parity,
    sha256,
)
from r2v2_deploy.simulation.robot import BODY_JOINTS, SIDES, JointMap, urdf_hand_joints
from r2v2_deploy.simulation.tabletop_scene import _descendant
from r2v2_deploy.tasks.scene_spec import FrontDropPath, load_config, canonical_digest
from r2v2_deploy.simulation.cylinder import load_cylinder_profile


class NativeContactScene(ContactMeasurements):
    POLICY_CLASS = ExitPlacementReachPolicy

    def __init__(
        self, config, mode="contact", *, program, initial_seed=0, run_bindings=None
    ):
        if mode != "contact":
            raise ValueError("Native Reach scene supports real CONTACT only")
        self.program, self.initial_seed = program, int(initial_seed)
        self.run_bindings = dict(run_bindings or {})
        self.delivery_xy = np.asarray(
            program.document["scene"]["delivery_dropoff_table"]["center_world_m"][:2]
        )
        self.delivery_relation = np.eye(4)
        self.delivery_duration = 0.0
        self.segmented = False
        self.delivery_started = self.placement_completed = False
        self.box_exit_snapshot = None
        self.defer_air_check = self.contact_air_verified = self.can_exit_verified = (
            False
        )
        self.arm_assist = self.paired_assist = self.waist_assist = (
            self.virtual_upper
        ) = False
        self.balance_gain = 0.0
        self.top_active = self.tracking_verified = False
        self.plan = None
        self._initialize_physics(
            config,
            mode="contact",
            keep_trace=False,
            scene_override=program.document["scene"],
        )
        d = program.document
        self.home = np.asarray(d["home"]).copy()
        self.pick_stage_count = d["pick_stage_count"]
        self.pick_exit_target = np.asarray(d["contact_pick_exit_wrist_goal"])
        self.carry_paths = copy.deepcopy(d["carry_wrist_goals"])
        self.hand_cfg = copy.deepcopy(d["hand_config"])
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.plan = copy.deepcopy(d["plan"])
        self.delivery_geoms = {
            self.model.geom("dropoff_table_top").id,
            *[self.model.geom("dropoff_table_leg_" + str(i)).id for i in range(4)],
        }
        self.table_geoms |= self.delivery_geoms
        self.sync()
        self.policy.closure_provider = lambda: 0.0
        self.training_only = True

    def _policy_contract(self):
        return dict(
            task_id=TASK_ID,
            task_scope=TASK_SCOPE,
            scene_path_sha256=self.path.sha256,
            delivery_program_sha256=self.program.sha256,
        )

    def sync(self):
        previous = self.stage_collision
        row = super().sync()
        total = 0.0
        for i, c in enumerate(self.data.contact):
            a, b = map(int, c.geom)
            if c.dist <= 0 and (
                (a in self.robot_geoms and b in self.table_geoms | self.crate_geoms)
                or (b in self.robot_geoms and a in self.table_geoms | self.crate_geoms)
            ):
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += float(np.linalg.norm(force[:3]))
        row["robot_prop_force_N"] = total
        depth = max([c["penetration_m"] for c in row["robot_prop_contacts"]] or [0.0])
        row["dangerous_robot_prop_contact"] = bool(depth > 0.003 or total > 150.0)
        self.stage_collision = previous or row["dangerous_robot_prop_contact"]
        return row

    def _bindings(self):
        return dict(
            program_sha256=self.program.sha256,
            scene_sha256=self.path.sha256,
            onnx_sha256=self.policy_sha,
            checkpoint_sha256=self.parity["checkpoint_sha256"],
            hand_config_sha256=canonical_digest(self.hand_cfg),
            policy_dt=0.02,
            physics_dt=0.001,
            hand_dt=0.01,
        )

    def _initial_safety(self):
        if (
            self.model.nmocap
            or np.any(self.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
            or np.any(self.data.xfrc_applied)
            or np.any(self.data.qfrc_applied)
        ):
            raise ValueError("Native scene forbids fixtures and applied forces")
        from r2v2_deploy.evaluation.contact_reach import scalar_measurements
        from r2v2_deploy.tasks.contact_reach import safety_conditions

        for reason, bad in safety_conditions(scalar_measurements(self), 0.0):
            if bool(bad):
                self.fail(reason)

    def _safety(self):
        raise RuntimeError(
            "A validated Reach reference program must be installed before stepping"
        )

    def _gate(self):
        raise RuntimeError(
            "A validated Reach reference program must be installed before stepping"
        )

    def _stream(self):
        raise RuntimeError(
            "A validated Reach reference program must be installed before stepping"
        )

    def report(self):
        return dict(
            schema="native_contact_reach_result_v1",
            mode="contact",
            success=bool(
                self.completed and self.placement_completed and not self.failure
            ),
            failure=self.failure,
            failure_phase=self.failure_phase,
            duration_s=float(self.data.time),
            phase=self.phase,
            grasp_verified=self.grasp_verified,
            can_exit_verified=self.can_exit_verified,
            placement_completed=self.placement_completed,
            release_commanded=self.release_commanded,
            actor_input_dim=1461,
            action_dim=28,
            single_student_body_policy=True,
            teacher_constructed=False,
            arm_ik_overlay=False,
            ankle_balance_residual_gain=0.0,
            fixed_base=False,
            runtime_state_restore=False,
            program_sha256=self.program.sha256,
            config=self.config,
            bindings=self.bindings,
            reach_configuration=self.cfg,
            final_metrics=copy.deepcopy(self.current_metrics),
            initial_metrics=copy.deepcopy(self.initial_metrics),
            transitions=self.transitions,
            physics_fingerprint=__import__(
                "r2v2_deploy.data.delivery_state", fromlist=["physics_fingerprint"]
            ).physics_fingerprint(self.model),
        )

    def _initialize_physics(
        self,
        config=None,
        mode="air",
        keep_trace=True,
        *,
        crate_entry_height_m=None,
        scene_override=None,
    ):
        if mode not in ("air", "contact"):
            raise ValueError("Mode must be air or contact")
        self.config, self.mode = (load_config(config), mode)
        self.crate_entry_height_m = crate_entry_height_m
        self.cfg = load_reach_config(self.config["reach_config"])
        self.parity = require_parity(self.config["parity_report"], self.cfg)
        self.path = FrontDropPath(self.config["path_manifest"])
        self.scene_override = (
            None if scene_override is None else copy.deepcopy(scene_override)
        )
        if self.scene_override is not None:
            self.path.scene = copy.deepcopy(self.scene_override)
        self.policy_sha = sha256(resolve_asset(self.cfg["policy_path"]))
        if self.parity.get("task") != self._policy_contract()["task_id"]:
            raise ValueError(
                "Numerical observation parity must belong to the front task"
            )
        scene_options = dict(
            delivery_background=self.config["delivery_background"],
            pickup_tray_height_m=self.config["pickup_tray_height_m"],
            crate_entry_height_m=crate_entry_height_m,
        )
        self.model, self.hand_cfg, self.layout = build_front_drop_model(
            self.cfg, self.path.scene, mode, **scene_options
        )
        self.data, self.scratch = (mujoco.MjData(self.model), mujoco.MjData(self.model))
        initialize_robot(self.model, self.data, self.hand_cfg)
        self.body_map = JointMap.create(self.model, BODY_JOINTS)
        limits = self.model.jnt_range[self.body_map.joints]
        middle, half = (limits.mean(1), 0.45 * (limits[:, 1] - limits[:, 0]))
        self.data.qpos[self.body_map.qpos] = np.clip(
            self.data.qpos[self.body_map.qpos], middle - half, middle + half
        )
        self._initialize_task_posture()
        mujoco.mj_forward(self.model, self.data)
        self.policy = self.POLICY_CLASS(
            self.model,
            self.data,
            resolve_asset(self.cfg["policy_path"]),
            expected_endpoint_contract="wrist_world_v2",
        )
        for key, expected in self._policy_contract().items():
            if self.policy.metadata.get(key) != expected:
                raise ValueError("Policy/path metadata mismatch: " + key)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        assert not set(self.body_map.actuators).intersection(
            np.concatenate([x.actuators for x in self.hands.maps.values()])
        )
        self.side, self.profile = ("left", load_cylinder_profile("baseline_40mm_100g"))
        self.object_body, self.object_geom = (
            self.model.body("test_cylinder").id,
            self.model.geom("cylinder_geom").id,
        )
        self.table_geom, self.floor_geom = (
            self.model.geom("tabletop_geom").id,
            self.model.geom("floor").id,
        )
        self.base = self.model.body("base_link").id
        self.crate_body = self.model.body("cargo_crate").id
        self.wrist_bodies = {
            s: self.model.body(s + "_hand_roll_link").id for s in SIDES
        }
        self.robot_geoms = {
            g
            for g in range(self.model.ngeom)
            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.base)
        }
        self.crate_geoms = {
            g
            for g in range(self.model.ngeom)
            if _descendant(self.model, int(self.model.geom_bodyid[g]), self.crate_body)
        }
        self.table_geoms = {
            g
            for g in range(self.model.ngeom)
            if _descendant(
                self.model,
                int(self.model.geom_bodyid[g]),
                self.model.body("tabletop").id,
            )
        }
        self.pickup_tray_geoms = {
            g
            for g in self.table_geoms
            if self.model.geom(g).name.startswith("pickup_tray_")
        }
        self.diagnostic_model = (
            build_front_drop_collision_diagnostic(
                self.cfg, self.path.scene, **scene_options
            )
            if mode == "air"
            else None
        )
        self.diagnostic_data = (
            mujoco.MjData(self.diagnostic_model)
            if self.diagnostic_model is not None
            else None
        )
        self.foot_geoms = set(foot_collision_ids(self.model))
        self.geom_sides = {
            g: s
            for g in self.robot_geoms
            for s, b in self.wrist_bodies.items()
            if _descendant(self.model, int(self.model.geom_bodyid[g]), b)
        }
        self.hand_joints = np.array(
            [self.model.joint(n).id for n in urdf_hand_joints()]
        )
        self.mimics = []
        for name, joint in urdf_hand_joints().items():
            mimic = joint.find("mimic")
            if mimic is not None:
                self.mimics.append(
                    (
                        self.model.joint(name).qposadr[0],
                        self.model.joint(mimic.get("joint")).qposadr[0],
                        float(mimic.get("multiplier", "1")),
                        float(mimic.get("offset", "0")),
                    )
                )
        self.weight_N, self.table_height = (
            self.profile["mass_kg"] * 9.81,
            self.path.scene["tabletop_height_m"],
        )
        self.phase, self.phase_start, self.steps = ("STAND", 0.0, 0)
        self.failure = self.failure_phase = None
        self.grasp_verified = self.release_commanded = self.completed = False
        self.baseline_relation = self.alignment_relation = None
        self.upright_attempted, self.max_verified_hold_s = (False, 0.0)
        self.stable_since = self.bad_since = None
        self.keep_trace, self.samples, self.targets = (bool(keep_trace), [], [])
        self.transitions = [dict(time_s=0.0, state="STAND")]
        self.stage_results, self.safety_events, self.contact_events = ([], [], [])
        self.current_metrics = {}
        self.goal_wrist_transforms = {
            s: body_transform(self.data, b) for s, b in self.wrist_bodies.items()
        }
        self.active_goal = self.goal_wrist_transforms["left"].copy()
        self.initial_object_position = self.data.xpos[self.object_body].copy()
        self.foot_anchor = self._feet(self.data)
        self.stage_collision = False
        self.peaks = dict.fromkeys(
            (
                "foot_drift_m",
                "base_tilt_deg",
                "hand_object_penetration_m",
                "hand_self_penetration_m",
                "hand_joint_violation_rad",
                "mimic_error_rad",
                "grasp_slip_m",
                "grasp_rotation_slip_deg",
                "clearance_m",
                "body_joint_violation_rad",
                "robot_self_penetration_m",
            ),
            0.0,
        )
        self.bindings = self._bindings()
        if mode == "contact":
            self.contact_air_verified = False
        scene = self.path.scene
        dims = np.asarray(scene["crate_dimensions_depth_width_height_m"])
        self.render_wireframes = [
            dict(
                kind="box",
                position=scene["table_center_world_m"],
                size=scene["table_half_size_m"],
                color=(1.0, 0.73, 0.32, 1.0),
            ),
            dict(
                kind="box",
                position=np.asarray(scene["crate_base_world_m"])
                + [0.0, 0.0, dims[2] / 2.0],
                size=dims / 2.0,
                color=(0.25, 0.85, 1.0, 1.0),
            ),
            dict(
                kind="cylinder",
                position=scene["can_initial_world_m"],
                size=[0.02, 0.06],
                color=(1.0, 0.2, 0.15, 1.0),
            ),
        ]
        self.sync()
        self.initial_metrics = copy.deepcopy(self.current_metrics)
        self._initial_safety()
        self.record()

    def _feet(self, data):
        return np.array(
            [data.xpos[self.model.body(s + "_ankle_roll_link").id] for s in SIDES]
        )

    def _set_goals(self, targets):
        for side, target in zip(SIDES, targets):
            self.goal_wrist_transforms[side] = target.copy()
            self.policy.set_target_world(side, target[:3, 3], pose_quaternion(target))
        self.active_goal = self.goal_wrist_transforms["left"].copy()

    def step(self):
        if self.done:
            return
        if self.steps % 20 == 0:
            self.sync()
            self._safety()
            if self.done:
                self.record()
                return
            self._gate()
            if self.done:
                self.record()
                return
            self._stream()
            self.policy.act(self.scratch)
        if self.steps % 10 == 0:
            self.hands.update()
        self.policy.apply(self.data)
        self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or np.any(self.data.warning.number)
        ):
            self.fail("Nonfinite state or MuJoCo warning")
        if self.steps % 10 == 0 or self.done:
            self.sync()
            self._safety()
            self.record()

    def _initialize_task_posture(self):
        # Legal root-only seed perturbation before time0; no mid-episode resets.
        if self.initial_seed:
            import mujoco

            delta = np.random.default_rng(self.initial_seed).uniform(-1, 1, 3) * [
                0.002,
                0.002,
                np.deg2rad(0.3),
            ]
            j = int(self.model.joint("floating_base_joint").qposadr[0])
            self.data.qpos[j : j + 2] += delta[:2]
            q = np.empty(4)
            mujoco.mju_mulQuat(
                q,
                np.array([np.cos(delta[2] / 2), 0, 0, np.sin(delta[2] / 2)]),
                self.data.qpos[j + 3 : j + 7],
            )
            self.data.qpos[j + 3 : j + 7] = q
