"""Construct the fixed three-bottle scene and controller state."""

from __future__ import annotations
from . import body_control
import numpy as np
from common.r2v2_crate_lift_metrics import _descendant as _descendant
from common.r2v2_hand_control import DualHandControl as BoxHandControl
from common.r2v2_loco_policy import LocoPolicy as LocoPolicy
from common.r2v2_reach_policy import ReachPolicy as BoxReachPolicy
from common.r2v2_reach_sim import foot_collision_ids as foot_collision_ids
from common.r2v2_reach_sim import initialize_robot as initialize_robot
from r2v2_deploy.tasks.contact_reach import GoalTube as GoalTube
from r2v2_deploy.tasks.contact_reach_program import (
    reference_closure as reference_closure,
)
from r2v2_description.model import JointMap as JointMap
import hashlib as hashlib
import json as json
import mujoco as mujoco
from .resources import resource
from . import geometry


def initialize(self):
    self.contact_descent_speed = 0.04
    self.contact_descent_key = None
    self.contact_descent_hold = None
    self.departure_clear_active = False
    self.departure_clear_since = None
    self.safe_feedback_live = None
    self.safe_feedback_nominal = None
    self.safe_feedback_stage = None
    self.safe_feedback_rows = []
    self.safe_place_initialized = False
    self.safe_place_active = False
    self.safe_grip_lower = 0.0
    self.safe_place_height = 0.75
    self.safe_place_move_s = 7.0
    self.safe_place_x = -1.35
    self.task_ik_hold_pose = False
    self.task_ik_values = None
    self.task_ik_pose = None
    self.task_ik_rows = []
    self.task_ik_time = None
    self.body_workpoint = np.array([0.375, 0.095])
    self.body_place_x = -1.48
    self.source_third_station_x = -1.1
    self.source_center_right_yaw = 60.0
    self.source_center_right_R = None
    self.source_center_waypoint_since = None
    self.source_center_empty_since = None
    self.source_cart_ready = False
    self.source_cart_axial = -35.0
    self.source_cart_raise = 0.2
    self.source_cart_rows = []
    self.source_cart_IK = None
    self.source_stance_active = False
    self.source_stance_done = False
    self.source_stance_speed = 0.2
    self.source_stance_yaw_cap = 0.35
    self.source_left_open_vector = None
    self.gradual_right_load_since = None
    self.source_third_right_x = 0.152
    self.source_third_right_z = 0.072
    self.source_right_pre_gate_m = 0.02
    self.second_early_share_since = None
    self.second_early_release_used = False
    self.third_soft_right_pre_count = 0
    self.final_second_trim_done = False
    self.final_second_trim_since = None
    self.third_right_trim_done = False
    self.third_right_trim_since = None
    self.source_third_return_delta = np.array([0.04, -0.12])
    self.source_second_place_yaw = 75.0
    self.second_radial_tilt_deg = 6.0
    self.second_tilt_axis = None
    self.second_axial_yaw_deg = -8.0
    self.source_layout = "original"
    self.supported_release_since = None
    self.supported_retreat_used = False
    self.station_inward = 0.005
    self.station_path_adjusted = False
    self.third_world_grasp_yaw = 15.0
    self.third_tilt = 6.0
    self.tilt_axis = None
    self.no_push_origin = None
    self.no_push_max_m = 0.0
    self.stop_after_third = False
    self.third_shallow = 0.0
    self.shortcut_enabled = False
    self.efficient_pick_rate = 1.2
    self.third_clearance_raise = 0.08
    self.third_stance_delta = np.array([-0.04, 0.0])
    self.fast_shared_since = None
    self.shortcut_count = 0
    self.accelerated_empty_steps = 0
    self.third_task_active = False
    self.third_task_target = np.array([-1.45, 2.15])
    self.third_return_offset = np.array([0.06, 0.1])
    self.row_width = 1.4
    self.right_park_relative_place = None
    self.second_stand_y = 1.51
    self.second_stand_x = -0.69
    self.second_stance_done = False
    self.axis_trim_done = False
    self.axis_trim_active = False
    self.near_y = 2.17
    self.second_yaw = 0.0
    self.second_x = -0.74
    self.practical_active = False
    self.flip_height = 0.03
    self.flip_back = 0.0
    self.flip_yaw = -10.0
    self.first_station_forward = 0.42
    self.first_station_lateral = -0.12
    self.skip_first_refine = True
    self.second_pick_forward_world = 0.1
    self.stop_after_box = False
    self.box_prefix_finished = False
    self.multi_rows = []
    self.multi_events = []
    self.ownership_rows = []
    self.active_bottle = 0
    self.completed_bottles = []
    self.empty_return = False
    self.target_xy = np.array([-0.49, 2.1700333121935995])
    self.lower_right = False
    self.right_lower_origin = None
    self.adaptive_xy = False
    self.upright_fraction = 1.0
    self.completion_root_limit = 0.08
    self.place_active = False
    self.place_rows = []
    self.place_events = []
    self.place_success = False
    self.place_contact_since = None
    self.place_hold_since = None
    self.place_edge = -0.26
    self.middle_height = 1.05
    self.place_target_y = 2.19
    self.place_pitch = 0.12
    self.move_s = 2.5
    self.place_body = "loco"
    self.back_distance = 0.45
    self.object_yaw_offset = 0.0
    shelf_layout = dict(
        center=[-0.26 - 0.2, 2.485, 0.0],
        yaw_deg=-90.0,
        width=0.8,
        depth=0.4,
        levels=[0.55, 1.05, 1.8],
        stance=[0.8, 1.0],
        heading_deg=115.0,
    )
    self.extra_active = False
    self.extra_route = "perpendicular"
    self.carry_kind = "cartesian"
    self.extra_speed = 0.14
    self.extra_yaw = 0.12
    self.turn_vx = 0.0
    self.seed_folder = None
    self.checkpoint_file = None
    self.checkpoint_only = False
    self.branch_mode = None is not None or None is not None
    self.extra_controls = []
    self.extra_rows = []
    self.cart_active = False
    self.checkpoint_written = False
    self.walk_start = None
    self.walk_rows = []
    self.walk_events = []
    self.walk_success = False
    self.route_kind = "back"
    self.route_speed = 0.2
    self.walk_pitch = 0.0
    self.walk_height = 0.82
    self.leg_index = 0
    self.brake_start = None
    self.walk_settle = None
    self.shelf_layout = {
        "perpendicular": {
            "center": [3.3, 1.2, 0.0],
            "yaw_deg": -90.0,
            "width": 1.2,
            "depth": 0.5,
            "levels": [0.55, 1.1, 1.65],
            "stance": [2.5, 1.15],
            "heading_deg": 0.0,
        },
        "parallel": {
            "center": [2.4, 3.6, 0.0],
            "yaw_deg": 0.0,
            "width": 1.2,
            "depth": 0.5,
            "levels": [0.55, 1.1, 1.65],
            "stance": [2.4, 2.8],
            "heading_deg": 90.0,
        },
    }["perpendicular"].copy()
    self.grip_offset = [0.165, -0.06, -0.01]
    self.second_start = None
    self.second_rows = []
    self.second_events = []
    self.left_normal_success = False
    self.left_load_since = None
    self.left_solo_since = None
    self.clearance_rows = []
    self.left_zoffset = -0.01
    self.second_plan_cache = None
    self.handoff_zoffset = 0.072
    self.right_grasp_yaw = 60.0
    self.bottle_x = 0.62
    self.bottle_y = 0.1
    self.handoff_owner = "left"
    self.right_loaded_since = None
    self.right_solo_since = None
    self.right_contact_seen = False
    self.transfer_events = []
    self.move_good = None
    self.post_start = None
    self.post_rows = []
    self.post_mode = "handoff"
    self.walk_speed = 0.2
    self.walk_target = 0.4
    self.walk_success = False
    self.handoff_success = False
    self.post_goals = None
    self.walk_stop = None
    self.stop_since = None
    self.walk_pitch = 0.1
    self.walk_height = 0.75
    self.arm_planning = False
    self.plan_time = None
    self.arm_rows = []
    self.gate_rows = []
    self.close_released = False
    self.close_wait = None
    self.close_good = None
    self.cart_kp = 6
    self.cart_kr = 4
    self.arm_speed = 0.8
    self.planner_kind = "integral"
    self.arm_nn_prior = None
    self.nn_offset = np.zeros(14)
    self.closer = 0.06
    self.live_blend_s = 6.0
    self.brake_lead = 0.06
    self.prep_s = 11.0
    self.empty_pitch = 0.0
    self.live_start = None
    self.extraction_success = False
    self.extract_since = None
    self.extraction_wait_start = None
    self.forward_enabled = True
    self.forward_done = False
    self.forward_lead = 0.05
    self.foot_gap = None
    self.forward_vy = 0.0
    self.goal_yaw_offset = 0.0
    self.native_active = False
    self.native = None
    self.route_choice = "direct"
    self.align_station = False
    self.prepare_choice = "stop4_arms"
    self.station_choice = "crate"
    self.source_start = 0.0
    self.bottle_policy_choice = "3191"
    self.native_start = None
    self.native_clock = 0.0
    self.bottle_commands = [0, 0]
    self.bottle_picked = False
    self.bottle_exited = False
    self.bottle_placed = False
    self.bottle_released = False
    self.bottle_withdrawn = False
    self.bottle_release_position = None
    self.bottle_loss_since = None
    self.bottle_settle_since = None
    self.bottle_rows = []
    self.native_observations = []
    self.native_hand_enabled = False
    self.bottle_stage_started = False
    self.held_walk_started = False
    self.variant = "rise"
    self.near_ready = False
    self.release_base = None
    self.profile = {
        "mode": "full",
        "back_speed": 0.2,
        "back_distance": 0.55,
        "turn_vx": 0.1,
        "turn_wz": 0.2,
        "table_center": [0.39, 2.485, 0.96],
    }
    self.command_speed = 0.0
    self.command_accel = 0.0
    self.diagnostic_finished = False
    self.slip_limit_m = 0.025
    self.bottle_enabled = True
    self.payload_mass = 0.4 + 0.1
    self.layout_mode = "arc_front"
    self.requested_pitch = 11.0
    self.continuous_arc = True
    self.walk_ramp = 0.0
    self.commanded_arm_hold = False
    self.selected_reach = "dual_wrist_vertical_grip_stop4"
    self.arm_hold_mix = 0.0
    self.fast_seat = 0.0
    self.segment_transport = False
    self.segment_back_anchor = None
    self.segment_turn_anchor = 0.0
    self.resume_after_settle = None
    crate_layout = json.loads(resource("crate_layout.json").read_text())
    self.target_layout = {
        key: np.asarray(value) if isinstance(value, list) else value
        for (key, value) in crate_layout.items()
    }
    self.cargo_body = self.cargo_geom = None
    self.payload_loss_since = None
    self.side_distance = 2.485
    self.arm_mode = "locked"
    self.speed = 0.1
    self.route = "back"
    self.destination_x = 0.39
    self.destination_yaw_deg = 90
    self.walk_distance = 1.0
    self.relative_inverse = False
    self.return_blend_s = 0.8
    self.inverse_frame = np.eye(4)
    if (
        not 0 < 0.1 <= 0.15
        or not np.isfinite(2.485)
        or ("back" == "side" and (not 0.4 <= 2.485 <= 1.0))
        or (not 0.05 <= 1.0 <= 1.0)
    ):
        raise ValueError(
            "Lateral command must be in the saved training range and tables must not overlap"
        )
    self.path = geometry.FrozenPath()
    (self.mode, self.model_name) = ("contact", "dual_wrist_vertical_grip_stop4")
    self.entry_loco = False
    self.loaded_handoff = True
    self.hold_current_on_return = True
    self.carry_distance_m = 0.0
    self.carry_speed_mps = 0.15
    if not np.isfinite(0.0) or not 0 <= 0.0 <= 5.0:
        raise ValueError("Carry command ramp must be between zero and five seconds")
    self.carry_ramp_s = 0.0
    if not np.isfinite(0.0) or not 0 <= 0.0 <= 0.15 or (0.0 and (not True)):
        raise ValueError(
            "Carry-pose adaptation requires loaded handoff and at most 15cm retraction"
        )
    self.carry_retract_m = 0.0
    self.carry_pose_anchor = None
    if not np.isfinite(11.0) or not -17.0 <= 11.0 <= 17.0:
        raise ValueError("Pitch command outside the bounded local diagnostic range")
    self.loco_pitch_deg = 11.0
    self.carry_start_position = None
    self.carry_stop_position = None
    (self.startup_blend_s, self.approach_speed_mps) = (0.0, 0.15)
    crate_manifest = json.loads(resource("@skills/manifest.json").read_text())
    self.spec = crate_manifest["models"]["dual_wrist_vertical_grip_stop4"]
    self.onnx = resource("@skills") / self.spec["onnx"]
    self.checkpoint = resource("@skills") / self.spec["checkpoint"]
    for crate_path in (self.onnx, self.checkpoint):
        if (
            geometry.digest(crate_path)
            != crate_manifest["files"][
                str(crate_path.relative_to(resource("@skills")))
            ]["sha256"]
        ):
            raise ValueError("Frozen model checksum differs")
    self.model = mujoco.MjModel.from_xml_path(str(resource("scene.xml")))
    self.hand_cfg = json.loads(resource("hands.json").read_text())
    (self.data, self.scratch) = (mujoco.MjData(self.model), mujoco.MjData(self.model))
    initialize_robot(self.model, self.data, self.hand_cfg)
    self.initial_base_x = -0.3
    if self.initial_base_x:
        self.data.qpos[
            self.model.joint("floating_base_joint").qposadr[0]
        ] += self.initial_base_x
        mujoco.mj_forward(self.model, self.data)
        self.path.home[:, 0] += self.initial_base_x
    self.mapping = JointMap.create(
        self.model,
        (
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
        ),
    )
    self.policy = BoxReachPolicy(
        self.model, self.data, self.onnx, expected_endpoint_contract="wrist_world_v2"
    )
    self.hands = BoxHandControl(self.model, self.data, self.hand_cfg)
    self.loco = LocoPolicy(
        self.model, self.data, resource("@skills/loco_9999/policy.onnx")
    )
    if self.loco is not None:
        self.loco.pose_command[1] = np.deg2rad(11.0)
        if not np.array_equal(self.loco.kp, self.policy.kp[:14]) or not np.array_equal(
            self.loco.kd, self.policy.kd[:14]
        ):
            raise ValueError("Loco/Reach leg-waist PD mismatch")
    (self.owner, self.executed) = ("reach", self.policy.default_joint_pos.copy())
    self.switch_time = None
    self.blend_from = self.executed.copy()
    self.arm_hold = self.data.qpos[self.mapping.qpos[14:]].copy()
    (self.entry_stage, self.entry_stage_start) = ("PREPARE", 0.0)
    self.path_start = None
    self.ready_since = None
    self.entry_poses = None
    (self.loaded_stage, self.loaded_start) = ("PATH", 0.0)
    (self.paused_clock, self.clock_offset) = (30.9 - 1e-05, 0.0)
    self.loaded_loss_since = None
    self.loaded_stable_since = None
    self.physical_grip_since = None
    self.physical_pickup_verified = False
    self.physical_placement_verified = False
    self.placement_since = None
    self.grip_baseline = None
    self.base = self.model.body("base_link").id
    self.floor = self.model.geom("floor").id
    self.foot = set(foot_collision_ids(self.model))
    self.robot = {
        crate_g
        for crate_g in range(self.model.ngeom)
        if _descendant(self.model, int(self.model.geom_bodyid[crate_g]), self.base)
    }
    self.hand_geoms = {
        crate_g
        for crate_g in self.robot
        if any(
            (
                _descendant(
                    self.model,
                    int(self.model.geom_bodyid[crate_g]),
                    self.model.body(crate_side + "_hand_roll_link").id,
                )
                for crate_side in ("left", "right")
            )
        )
    }
    self.foot_start = np.array(
        [
            self.data.xpos[self.model.body(crate_s + "_ankle_roll_link").id]
            for crate_s in ("left", "right")
        ]
    )
    (self.phase, self.failure) = ("STAND", None)
    (self.transitions, self.trace, self.states) = ([], [], [])
    self.steps = 0
    self.contact_since = None
    self.grasp_verified = False
    self.peaks = dict(
        foot_drift_m=0.0,
        joint_limit_excess_rad=0.0,
        self_penetration_m=0.0,
        hand_crate_penetration_m=0.0,
    )
    (self.stage, self.stage_start, self.stable_since) = ("PICKUP", 0.0, None)
    self.done = False
    self.finger_closed = False
    self.carry_anchor = self.carry_base = None
    self.yaw_anchor = 0.0
    self.place_start = self.place_end = self.place_crate_anchor = None
    self.place_relations = None
    self.release_verified = False
    self.destination = np.array([*[0.33, 2.485], 0.98])
    self.walk_start_position = None
    self.destination_geom = self.model.geom("destination_table_geom").id
    self.destination_body = self.model.body("destination_table").id
    self.source_body = self.model.body("tabletop").id
    self.crate_body = self.model.body("cargo_crate").id
    self.crate_geoms = {
        crate_motion_g
        for crate_motion_g in range(self.model.ngeom)
        if _descendant(
            self.model, int(self.model.geom_bodyid[crate_motion_g]), self.crate_body
        )
    }
    self.post_failure_time = None
    (self.test_distance, self.test_turn_deg, self.layout_id) = (1.75, 90.0, "C")
    self.max_yaw_rate = 0.2
    (self.turn_vx, self.move_axis, self.yaw_floor) = (0.15, "x", 0.0)
    self.segment_completed = False
    self.walk_origin = self.turn_origin = None
    self.turn_origin_yaw = self.turn_target_yaw = None
    self.turn_last_yaw = self.turn_unwrapped = None
    self.saved_turn_progress = 0.0
    self.segment_events = []
    self.destination = np.array([[0.39, 2.485, 0.96][0], [0.39, 2.485, 0.96][1], 0.98])
    self.return_blend_s = 2.0
    self.placement_mode = "loco"
    self.approach_start = None
    self.goal_yaw = np.pi / 2
    self.lower_start_height = None
    self.lower_depth = None
    self.opening_started_supported = False
    self.destination_yaw_deg = self.target_layout["yaw"]
    self.loco.pose_command[1] = np.deg2rad(11.0)
    self.loco_pitch_deg = 11.0
    self.approach_yaw_start = None
    self.destination = self.target_layout["landing"].copy()
    self.initial_landing_goal = self.destination.copy()
    self.cargo_body = self.model.body("test_cylinder").id
    self.cargo_geom = self.model.geom("cylinder_geom").id
    self.crate_floor_geom = self.model.geom("crate_bottom").id
    self.cargo_jp = np.zeros((3, self.model.nv))
    self.cargo_jr = np.zeros((3, self.model.nv))
    self.crate_jp = np.zeros((3, self.model.nv))
    self.crate_jr = np.zeros((3, self.model.nv))
    self.test_distance = {
        "mode": "full",
        "back_speed": 0.2,
        "back_distance": 0.55,
        "turn_vx": 0.1,
        "turn_wz": 0.2,
        "table_center": [0.39, 2.485, 0.96],
    }.get("back_distance", 0.55)
    self.turn_vx = {
        "mode": "full",
        "back_speed": 0.2,
        "back_distance": 0.55,
        "turn_vx": 0.1,
        "turn_wz": 0.2,
        "table_center": [0.39, 2.485, 0.96],
    }.get("turn_vx", 0.1)
    self.max_yaw_rate = {
        "mode": "full",
        "back_speed": 0.2,
        "back_distance": 0.55,
        "turn_vx": 0.1,
        "turn_wz": 0.2,
        "table_center": [0.39, 2.485, 0.96],
    }.get("turn_wz", 0.2)
    control_Student = geometry.WristReferencePolicy
    self.bottle_policy_file = resource("models/wrist_reference.onnx")
    self.native = control_Student(
        self.model,
        self.data,
        self.bottle_policy_file,
        expected_endpoint_contract="wrist_world_v2",
    )
    for control_key in ("kp", "kd", "default_joint_pos", "action_scale"):
        np.testing.assert_array_equal(
            getattr(self.native, control_key), getattr(self.policy, control_key)
        )
    self.native.closure_provider = lambda: reference_closure(
        self.hands.controllers["left"]
    )
    self.native_hand_cfg = json.loads(resource("reference/program.json").read_text())[
        "hand_config"
    ]
    control_program_sha = hashlib.sha256(
        resource("reference/program.json").read_bytes()
    ).hexdigest()
    self.tube = GoalTube(resource("reference/reference.json"), control_program_sha)
    self.proxy = self.make_measurement_proxy()
    self.mode = "zigzag"
    self.wz = -0.2
    self.after_action = "pick"
    self.frame_mode = "crate"
    self.dx = 0.0
    self.arrived = False
    self.stance_events = []
    self.zig_sign = 1.0
    self.post_turn_vx = None
    self.turned = False
    self.prehold = 0.0
    self.fixed_frame_yaw = None
    stance_measured = json.loads(resource("stance.json").read_text())
    self.native_base = np.array(
        stance_measured["native"][1]["bodies"]["base_link"]["native_position"]
    )
    self.native_yaw = np.deg2rad(
        stance_measured["native"][1]["bodies"]["base_link"]["native_yaw_deg"]
    )
    body_control.configure_grasp_gate_live_act(self)
    self.arm_joints = np.asarray(self.mapping.joints[14:])
    self.arm_pos = np.asarray(self.mapping.qpos[14:])
    self.arm_dofs = np.asarray(self.mapping.dofs[14:])
    self.arm_limits = self.model.jnt_range[self.arm_joints]
    self.jp = np.zeros((3, self.model.nv))
    self.jr = np.zeros((3, self.model.nv))
    self.plan_command = None
    self.plan_seed = None
    self.raw_body_command = None
    body_control.configure_arm_planner_body(self)
    body_control.configure_arm_planner_prepared(self)
    body_control.configure_arm_planner_mixed(self)
    self.closer = 0.06
    self.pick_pitch = 0.35
    self.bias_scale = 1.0
    self.upper_cmd = None
    self.upper_rows = []
    self.lower_rows = []
    self.upper_pos = np.asarray(self.mapping.qpos[12:])
    self.upper_dofs = np.asarray(self.mapping.dofs[12:])
    self.upper_joints = np.asarray(self.mapping.joints[12:])
    self.upper_limits = self.model.jnt_range[self.upper_joints]
    self.ik = mujoco.MjData(self.model)
    posture_ik_record = resource("posture_prior.npz")
    posture_ik_z = np.load(posture_ik_record)
    self.prior_times = posture_ik_z["time_s"]
    self.prior_positions = posture_ik_z["upper_qpos"]
    self.prior_sha = hashlib.sha256(posture_ik_record.read_bytes()).hexdigest()
    body_control.configure_posture_ik_mixed(self)
    self.fixed_waist = True
    self.lower_rows = []
    self.original_upper_limits = self.upper_limits.copy()
    body_control.configure_arms_only_mixed(self)
    self.grasp_yaw = -15.0
    body_control.configure_carry_body_commanded(self)
    self.rack_geoms = {
        loaded_route_g
        for loaded_route_g in range(self.model.ngeom)
        if self.model.geom(loaded_route_g).name.startswith("rack_")
    }
    assert len(self.rack_geoms) == 7
    body_control.configure_loaded_route_commanded(self)
    self.branch_initial_time = 0.0
    self.shelf_layout = shelf_layout
    self.open_front = False
    self.middle_geom = self.model.geom("rack_level_1").id
    self.rack_body = self.model.body("storage_rack").id
    self.branch_mode = None is not None
    self.branch_initial_time = 0.0
    self.placement_pitch = 0.25
    self.slow_near = True
    self.placement_height = 0.75
    self.carry_pullback = 0.16
    self.route_lateral = 0.3
    self.carry_tilt_x = 0.0
    self.carry_tilt_y = 0.0
    self.route_goal = np.array([-0.15, 1.6])
    self.route_pitch = 0.12
    self.route_speed = 0.2
    self.side_route_active = False
    self.carry_mode = "cartesian"
    self.route_stop_since = None
    self.route_rows = []
    self.route_event = []
    self.route_arrived = False
    self.branch_mode = None is not None
    self.branch_initial_time = 0.0
    self.shelf_layout = dict(
        center=self.model.body_pos[self.rack_body].tolist(),
        yaw_deg=0.0,
        width=float(2 * self.model.geom_size[self.middle_geom, 0]),
        depth=0.4,
        levels=[0.55, 1.05, 1.8],
        stance=[-0.15, 1.6],
        heading_deg=90.0,
    )
    self.branch_mode = None is not None
    self.branch_initial_time = 0.0
    self.bottle_bodies = [
        self.model.body(inventory_n).id
        for inventory_n in ["test_cylinder", "test_cylinder_2", "test_cylinder_3"]
    ]
    self.bottle_geoms = [
        self.model.geom(inventory_n).id
        for inventory_n in ["cylinder_geom", "cylinder_geom_2", "cylinder_geom_3"]
    ]
    self.payload_mass = 0.7
    self.initial_three = self.all_bottles()
    self.max_objects = 3
    self.global_settle = False
    self.global_stable_since = None
    self.first_pick_base = None
    self.first_pick_bottle = None
    self.first_pick_yaw = None
    self.refine_active = False
    self.load_place_initialization(None)
    self.shelf_base_x = -0.4
    self.route_goal = np.array(
        [
            -0.4
            - 0.16 * self.active_bottle
            + (0.12 if self.active_bottle > 0 else 0.0),
            1.58,
        ]
    )
    self.carry_pullback = 0.22
    self.placement_height = 0.75
    self.placement_pitch = 0.25
    self.shelf_target_y = 2.1700333121935995
    self.object_yaw_offset = 45.0
    self.target_xy = np.array([-0.49 - 0.16 * self.active_bottle, 2.1700333121935995])
    self.last_lateral_command = np.zeros(3)
    self.repeat_use_waist = False
    self.repeat_yaw_teacher = False
    self.repeat_grasp_depth = -0.006
    self.repeat_left_offset = np.array([0.16, -0.055, -0.01])
    self.repeat_left_yaw = -60.0
    self.waist_teacher_rows = []
    body_control.configure_repeat_compensate_yaw(self)
    body_control.configure_repeat_repeat_waist(self)
    self.dual_mode = getattr(self, "dual_mode", False)
    self.dual_pending = (
        self.active_bottle > 0 and self.stage == "LEFT_REGRASP_CLOSE_AND_LOAD"
    )
    self.dual_ready_since = None
    self.dual_rows = []
    self.dual_pending_arms = self.executed[14:].copy()
    self.shelf_regrip_active = getattr(self, "shelf_regrip_active", False)
    self.station_grasp = np.load(resource("station_grasp.npz"))[
        "T_object_left_wrist_preclose"
    ]
    self.station_path = dict(np.load(resource("station_path.npz")))
    self.use_air_trim = False
    self.right_park_program_bottle = -1
    if not hasattr(self, "walk_twist"):
        self.walk_twist = np.zeros(3)
    if not hasattr(self, "walk_pose"):
        self.walk_pose = np.array(
            (
                getattr(self, "post_pose", [0.82, 0.12])
                if getattr(self, "post_pose", None) is not None
                else [0.82, 0.12]
            ),
            dtype=float,
        )
    if not hasattr(self, "walk_arms"):
        self.walk_arms = self.executed[14:].copy()
    self.max_objects = 3
    self.first_shared_since = None
    self.second_pick_forward_world = 0.1
    self.right_place_active = getattr(self, "right_place_active", False)
    self.target_xy[1] = 2.29 if self.active_bottle == 1 else 2.1700333121935995
    self.left_after_stance_pending = (
        self.active_bottle == 0
        and self.right_place_active
        and (self.stage == "RIGHT_SHELF_ALIGN_AIR")
    )
    self.first_station_forward = 0.42
    self.first_station_lateral = -0.12
    self.skip_first_refine = True
    self.first_trim_active = False
    self.first_trim_done = False
    self.first_contact_release_since = None
    self.flip_height = 0.03
    self.flip_back = 0.0
    self.flip_yaw = -10.0
    self.extra_left_distance = 0.15
    self.third_withdraw_enabled = True
    self.third_extra_done = False
    self.third_place_yaw = 75.0
    self.near_y = 2.17
    self.second_yaw = 0.0
    self.second_x = -0.74
    self.max_objects = 3
    self.extra_left_distance = 0.3
    self.second_stand_y = 1.51
    self.target_xy[:] = [-0.49 - 0.25 * self.active_bottle, 2.17]
    self.second_x = -0.9
    self.near_y = 2.15
    self.second_yaw = 0.0
    self.target_xy[:] = [[-0.49, 2.17], [-0.9, 2.15], [-1.25, 2.15]][self.active_bottle]
    self.extra_left_distance = 0.55
    self.third_return_offset = np.array([0.06, 0.1])
    self.second_clear_key = None
    self.air_relaxed_count = 0
    self.air_relaxed_max_mm = 0.0
    self.compact_logs = True
    self.multi_rows = geometry.Inventory50Hz()
    for bottle_sequence_k in [
        "bottle_rows",
        "post_rows",
        "second_rows",
        "dual_rows",
        "place_rows",
        "extra_controls",
        "upper_rows",
    ]:
        if hasattr(self, bottle_sequence_k):
            setattr(self, bottle_sequence_k, geometry.SampledLog())
    self.soft_staging_key = None
    self.third_clear_key = None
    self.third_right_park_key = None
    self.third_task_target = np.array([-1.45, 2.15])
    self.third_withdraw_enabled = False
    self.front_step_origin = None
    self.open_back_started = False
    self.open_back_active = False
    self.open_back_completed = False
    self.open_back_since = None
    self.max_objects = 3
    self.shortcut_enabled = False
    self.efficient_pick_rate = 1.2
    self.third_clearance_raise = 0.08
    self.third_stance_delta = np.array([-0.04, 0.0])
    self.fast_shared_since = None
    self.stop_after_third = False
    self.third_shallow = 0.0
    self.third_tilt = 6.0
    self.third_world_grasp_yaw = 15.0
    self.third_task_target = np.array([-1.23, 2.15])
    self.station_inward = 0.005
    self.station_path_adjusted = False
    self.stop_after_box = False
    self.source_speed_scale = 1.0
    self.profile = dict(self.profile)
    self.profile.update(acceleration=0.08, jerk=0.08)
    self.extended_turn_timeout_steps = 0
    assert self.branch_initial_time == 0.0 and (not self.branch_mode)
    self.second_high_relaxed_steps = 0
    self.second_high_relaxed_max_mm = 0.0
    self.stop_after_second_extraction = False
    self.second_axial_yaw_deg = -8.0
    self.second_radial_tilt_deg = 6.0
    self.second_tilt_axis = None
    self.source_second_place_yaw = 75.0
    self.second_yaw = 75.0
    self.stop_after_second_shelf = False
    self.source_third_return_delta = np.array([0.04, -0.12])
    self.stop_after_second_shelf = False
    self.source_third_right_x = 0.152
    self.source_third_right_z = 0.072
    self.source_right_pre_gate_m = 0.02
    self.stop_after_right_handoff = False
    self.source_stance_speed = 0.2
    self.source_stance_yaw_cap = 0.35
    body_control.configure_grasp_stance_source_command(self)
    self.source_cart_axial = -35.0
    self.source_cart_raise = 0.2
    self.source_center_right_yaw = 60.0
    self.right_grasp_yaw = 60.0
    self.source_third_right_x = 0.16
    self.source_third_right_z = 0.072
    self.source_third_station_x = -1.1
    self.body_place_x = -1.48
    self.extra_left_distance = 0.38
    self.safe_place_height = 0.75
    self.safe_place_move_s = 7.0
    self.safe_place_x = -1.35
    self.body_place_x = -1.35
    self.extra_left_distance = 0.55
    self.natural_stop_after = "none"
    self.natural_mode = "both"
    self.natural_solo = False
    self.natural_rows = []
    self.natural_empty_joint_rows = []
    self.natural_since = None
    self.natural_loss_since = None
    self.natural_start_time = float(self.data.time)
    self.natural_initial_completed = list(self.completed_bottles)
    self.natural_initial_qpos = self.data.qpos.copy()
    self.natural_initial_qvel = self.data.qvel.copy()
    self.natural_fk = mujoco.MjData(self.model)
    self.natural_rest_joints = np.array(
        [0.5, 0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5, -0.2, 0.0, 0.25, 0.0, 0.0, 0.0]
    )
    self.natural_reused_points = json.loads(resource("station_points.json").read_text())
    assert all(
        (
            "left_final_reference_heading_rad" in sequence_p
            for sequence_p in self.natural_reused_points
        )
    )
    self.natural_reused_return = json.loads(resource("return_points.json").read_text())
    self.natural_third_wrist_preparation = json.loads(
        resource("third_wrist.json").read_text()
    )
    self.safe_grip_lower = 0.0
    self.contact_descent_speed = 0.01
