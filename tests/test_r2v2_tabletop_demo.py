"""FSM behavior is checked independently of an ONNX file or successful rollout."""

from common.path_config import PROJECT_ROOT

import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import common.r2v2_tabletop_demo as demo
from common.r2v2_grasp_recording import CONTACT_PARTS
from common.r2v2_reach_sim import ReachCompatibilityExperiment, build_reach_model, initialize_robot
from r2v2_description.model import SIDES


class HandStub:
    def __init__(self):
        self.controllers = {side: SimpleNamespace(command=int(side == "left"), finished=True,
                                                  status=lambda *args: "CLOSED") for side in SIDES}
        self.commands = []

    def command(self, side, command):
        self.commands.append((side, command))
        self.controllers[side].command = command


class PolicyStub:
    def __init__(self, *args):
        self.references = {side: SimpleNamespace(position=np.zeros(3), goal_position=np.zeros(3),
                            quaternion=np.array([1., 0, 0, 0]), goal_quaternion=np.array([1., 0, 0, 0]),
                            linear_velocity=np.zeros(3), angular_velocity=np.zeros(3)) for side in SIDES}
        self.commands = []
        self.last_action = self.q_des = self.last_torque = np.zeros(28)
        self.histories = {"sample": np.zeros((10, 3))}
        self.last_observation = np.zeros(1460)

    def set_target_world(self, side, position, quaternion):
        self.commands.append((side, np.array(position).copy(), np.array(quaternion).copy()))
        self.references[side].goal_position = np.array(position).copy()
        self.references[side].goal_quaternion = np.array(quaternion).copy()


def gate(phase="CLOSE"):
    exp = demo.TabletopDemoExperiment.__new__(demo.TabletopDemoExperiment)
    exp.demo_cfg = demo.load_demo_config()
    exp.cfg = exp.demo_cfg["reach"]
    exp.demo_cfg["contact_stable_s"] = exp.demo_cfg["motion_settle_s"] = 0.3
    exp.data = SimpleNamespace(time=0.0)
    exp.scratch = SimpleNamespace(time=0.0, xpos=np.array([[0, 0, 0.82]]), xmat=np.eye(3).reshape(1, 9))
    exp.base = 0
    exp.phase, exp.phase_start = phase, 0.0
    exp.stable_time = 0.0
    exp.stable_since = None
    exp.condition_since = {}
    exp.precision_events, exp.transitions, exp.targets = [], [], []
    exp.failure = exp.failure_phase = None
    exp.hands, exp.policy = HandStub(), PolicyStub()
    exp.grasp_confirmed = exp.released = False
    exp.grasp_relation = None
    exp.current_slip_m = exp.current_slip_deg = 0.0
    exp.opposed_hold_samples = exp.hold_samples = 0
    exp.approach_index = exp.lower_index = exp.touch_index = 0
    exp.table_height = 1.0
    exp.peaks = {key: 0.0 for key in ("base_tilt_deg", "inactive_wrist_position_error_m",
                                    "inactive_orientation_error_deg", "object_clearance_m",
                                    "hand_object_penetration_m", "slip_m", "slip_deg")}
    exp.fake_wrist = np.eye(4)
    exp.fake_wrist[:3, 3] = [0.255, 0.235, 1.06]
    exp.wrist = lambda: exp.fake_wrist.copy()
    transform = np.eye(4)
    transform[:3, 3] = [0.40, 0.20, 1.06]
    exp.object_state = {"T_world_cylinder": transform, "position_m": transform[:3, 3].copy(),
                        "bottom_height_m": 1.0, "linear_velocity_mps": np.zeros(3), "tilt_deg": 0.0}
    exp.initial_object_transform = transform.copy()
    exp.destination_transform = transform.copy()
    exp.destination_transform[1, 3] -= 0.1
    exp.closing_relation = exp.relative_object().copy()
    exp.contact_state = {"opposed": False, "table_contact": True, "floor_contact": False,
                         "max_hand_object_penetration_m": 0.0,
                         "fingers_normal_force_N": dict.fromkeys(CONTACT_PARTS, 0.0)}
    errors = {side: {"position_m": 0.0, "orientation_deg": 0.0, "linear_speed_mps": 0.0,
                     "wrist_position_m": 0.0, "wrist_linear_speed_mps": 0.0} for side in SIDES}
    exp.errors = lambda side: errors[side].copy()
    exp.record = lambda: None
    return exp, errors


def tick(exp, time):
    exp.data.time = exp.scratch.time = time
    exp.update_gate()


def test_final_hold_rejects_residual_thumb_contact():
    exp, _ = gate("FINAL_HOLD")
    exp.object_state["position_m"] = exp.destination_transform[:3, 3].copy()
    exp.contact_state["fingers_normal_force_N"]["thumb"] = 0.2
    for time in (0, 0.5, 1.0, 2.0):
        tick(exp, time)
        assert exp.phase == "FINAL_HOLD"
    exp.contact_state["fingers_normal_force_N"]["thumb"] = 0.0
    tick(exp, 2.1)
    tick(exp, 3.1)
    assert exp.phase == "COMPLETE"


def enable_unsupported_grip(exp):
    exp.contact_state["opposed"] = True
    exp.contact_state["table_contact"] = False
    exp.object_state["bottom_height_m"] = exp.table_height + 0.02


def test_settled_motion_explicitly_bypasses_strict_reach_error():
    exp, errors = gate("OUTWARD_ALIGN")
    errors["left"].update(wrist_position_m=0.08, orientation_deg=25.0)
    exp.pregrasp_wrist_target = exp.wrist()
    for time in (0.0, 0.1, 0.28):
        tick(exp, time)
        assert exp.phase == "OUTWARD_ALIGN"
    tick(exp, 0.30)
    assert exp.phase == "SIDE_PREGRASP"
    assert exp.precision_events[-1]["mode"] == "diagnostic_only"
    assert exp.precision_events[-1]["strict_reach_passed"] is False
    assert exp.precision_events[-1]["errors"]["wrist_position_m"] == pytest.approx(0.08)


@pytest.mark.parametrize("reason", ["reference_moving", "physical_wrist_moving"])
def test_motion_settled_still_requires_finished_reference_and_low_speed(reason):
    exp, errors = gate("OUTWARD_ALIGN")
    exp.pregrasp_wrist_target = exp.wrist()
    if reason == "reference_moving":
        exp.policy.references["left"].goal_position[0] = 0.1
    else:
        errors["left"]["wrist_linear_speed_mps"] = exp.demo_cfg["motion_speed_mps"] + 0.01
    for time in (0.0, 0.3, 1.0):
        tick(exp, time)
    assert exp.phase == "OUTWARD_ALIGN"
    assert not exp.precision_events


def test_closed_finished_hand_without_opposition_cannot_enter_trial_lift():
    exp, _ = gate()
    assert exp.hands.controllers["left"].finished
    assert exp.hands.controllers["left"].status() == "CLOSED"
    for time in (2.5, 2.8, 3.0, 5.0):
        tick(exp, time)
        assert exp.phase == "CLOSE"
        assert not exp.grasp_confirmed
    assert not exp.hands.commands


def test_opposing_contacts_are_required_for_full_duration_before_trial_lift():
    exp, _ = gate()
    exp.contact_state["opposed"] = True
    tick(exp, 2.5)
    tick(exp, 2.78)
    assert exp.phase == "CLOSE"
    tick(exp, 2.8)
    assert exp.phase == "TRIAL_LIFT"
    assert not exp.grasp_confirmed
    target = np.asarray(exp.targets[-1]["wrist_transform_world"])
    assert target[2, 3]-exp.wrist()[2, 3] == pytest.approx(exp.demo_cfg["trial_lift_m"])


def test_opposition_without_minimum_closure_duration_does_not_lift():
    exp, _ = gate()
    exp.contact_state["opposed"] = True
    tick(exp, 0.0)
    tick(exp, 0.3)
    assert exp.phase == "CLOSE"
    tick(exp, exp.demo_cfg["close_minimum_s"])
    assert exp.phase == "TRIAL_LIFT"


def test_table_supported_opposition_is_not_confirmed_grasp():
    exp, _ = gate("TRIAL_LIFT")
    exp.contact_state["opposed"] = True
    exp.object_state["bottom_height_m"] = exp.table_height + 0.02
    for time in (0.0, 0.3, 0.6, 1.0):
        tick(exp, time)
        assert exp.phase == "TRIAL_LIFT"
        assert not exp.grasp_confirmed


def test_unsupported_opposition_and_relative_stability_need_elapsed_point_three_seconds():
    exp, _ = gate("TRIAL_LIFT")
    enable_unsupported_grip(exp)
    for index in range(15):
        tick(exp, index*0.02)
        assert exp.phase == "TRIAL_LIFT"  # Fifteen samples span only .28 s.
        assert not exp.grasp_confirmed
    tick(exp, 0.30)
    assert exp.phase == "LIFT"
    assert exp.grasp_confirmed
    np.testing.assert_allclose(exp.grasp_relation, exp.relative_object())
    assert not exp.hands.commands


def test_lost_contact_resets_trial_confirmation_timer():
    exp, _ = gate("TRIAL_LIFT")
    enable_unsupported_grip(exp)
    tick(exp, 0.0)
    tick(exp, 0.18)
    exp.contact_state["opposed"] = False
    tick(exp, 0.20)
    exp.contact_state["opposed"] = True
    for time in (0.22, 0.4, 0.5):
        tick(exp, time)
        assert not exp.grasp_confirmed
    tick(exp, 0.52)
    assert exp.grasp_confirmed


@pytest.mark.parametrize("failure", ["translation_slip", "rotation_slip", "insufficient_clearance"])
def test_unstable_relative_pose_or_insufficient_lift_cannot_confirm(failure):
    exp, _ = gate("TRIAL_LIFT")
    enable_unsupported_grip(exp)
    if failure == "translation_slip":
        exp.object_state["T_world_cylinder"][0, 3] += exp.demo_cfg["max_slip_m"] + 0.005
    elif failure == "rotation_slip":
        angle = np.deg2rad(exp.demo_cfg["max_slip_deg"] + 5)
        exp.object_state["T_world_cylinder"][:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    else:
        exp.object_state["bottom_height_m"] = exp.table_height + exp.demo_cfg["trial_clearance_m"] / 2
    for time in (0.0, 0.3, 0.6):
        tick(exp, time)
        assert exp.phase == "TRIAL_LIFT"
        assert not exp.grasp_confirmed


def test_no_support_never_commands_release():
    exp, _ = gate("TOUCH_0")
    enable_unsupported_grip(exp)
    exp.motion_settled = lambda: False
    for time in (0.0, 0.3, 1.0):
        tick(exp, time)
        assert exp.phase == "TOUCH_0"
        assert not exp.released
        assert exp.hands.controllers["left"].command == 1
    assert not exp.hands.commands


def test_stable_table_contact_precedes_release_command():
    exp, _ = gate("TOUCH_0")
    exp.motion_settled = lambda: False
    for time in (0.0, 0.1, 0.28):
        tick(exp, time)
        assert not exp.released
        assert exp.hands.controllers["left"].command == 1
    tick(exp, 0.3)
    assert exp.phase == "RELEASE"
    assert exp.released
    assert exp.hands.commands == [("left", 0)]


@pytest.mark.parametrize("reason", ["linear_motion", "tilt", "interrupted_support"])
def test_unstable_support_does_not_release(reason):
    exp, _ = gate("TOUCH_0")
    exp.motion_settled = lambda: False
    if reason == "linear_motion":
        exp.object_state["linear_velocity_mps"] = np.array([0.021, 0, 0])
    elif reason == "tilt":
        exp.object_state["tilt_deg"] = exp.demo_cfg["final_tilt_deg"] + 1
    tick(exp, 0.0)
    if reason == "interrupted_support":
        exp.contact_state["table_contact"] = False
        tick(exp, 0.2)
        exp.contact_state["table_contact"] = True
        tick(exp, 0.21)
    tick(exp, 0.3)
    assert exp.phase == "TOUCH_0"
    assert not exp.released
    assert not exp.hands.commands


def test_failure_preserves_current_hand_command_and_cannot_keep_stepping():
    exp, _ = gate("TRANSFER")
    exp.fail("Injected transport failure")
    assert exp.phase == "FAILED"
    assert exp.failure_phase == "TRANSFER"
    assert exp.hands.controllers["left"].command == 1
    assert exp.hands.controllers["right"].command == 0
    assert not exp.hands.commands
    with pytest.raises(RuntimeError, match="already stopped"):
        exp.step()


def test_empty_grasp_times_out_without_automatically_opening():
    exp, _ = gate("CLOSE")
    tick(exp, exp.demo_cfg["phase_timeout_s"])
    assert exp.phase == "FAILED"
    assert exp.failure_phase == "CLOSE"
    assert not exp.grasp_confirmed
    assert exp.hands.controllers["left"].command == 1
    assert not exp.hands.commands


def test_sustained_grip_loss_timer_survives_motion_phase_boundary():
    exp, _ = gate("LIFT")
    enable_unsupported_grip(exp)
    exp.grasp_confirmed = True
    exp.grasp_relation = exp.relative_object().copy()
    exp.contact_state["opposed"] = False
    exp.motion_settled = lambda: False
    duration = exp.demo_cfg["slip_violation_s"]
    tick(exp, 0.0)
    tick(exp, duration/2)
    exp.transition("TRANSFER")
    tick(exp, duration-0.01)
    assert not exp.done
    tick(exp, duration)
    assert exp.phase == "FAILED"
    assert exp.failure_phase == "TRANSFER"
    assert exp.hands.controllers["left"].command == 1
    assert not exp.hands.commands


@pytest.mark.parametrize("phase,should_fail", [("LIFT", True), ("TRANSFER", True),
                                               ("LOWER_1", False), ("TOUCH_0", False)])
def test_early_table_contact_counts_as_lost_grip_only_before_placement(phase, should_fail):
    exp, _ = gate(phase)
    exp.grasp_confirmed = True
    exp.grasp_relation = exp.relative_object().copy()
    exp.contact_state["opposed"] = True
    exp.motion_settled = lambda: False
    tick(exp, 0.0)
    tick(exp, exp.demo_cfg["slip_violation_s"])
    assert exp.done is should_fail
    if should_fail:
        assert exp.phase == "FAILED"
        assert "Grasp lost" in exp.failure
    else:
        assert exp.phase == phase
    assert exp.hands.controllers["left"].command == 1


@pytest.mark.parametrize("field,limit", [("wrist_position_m", "inactive_position_m"),
                                        ("orientation_deg", "inactive_orientation_deg")])
def test_inactive_world_wrist_violation_timer_survives_phase_boundary(field, limit):
    exp, errors = gate("CLOSE")
    errors["right"][field] = exp.cfg["gates"][limit] + 0.01
    duration = exp.cfg["gates"]["inactive_violation_s"]
    tick(exp, 0.0)
    tick(exp, duration/2)
    exp.transition("TRIAL_LIFT")
    tick(exp, duration-0.01)
    assert not exp.done
    tick(exp, duration)
    assert exp.phase == "FAILED"
    assert "Inactive right wrist" in exp.failure
    assert not exp.hands.commands


def test_contact_seeking_descent_advances_one_configured_step():
    exp, _ = gate("TOUCH_0")
    enable_unsupported_grip(exp)
    exp.grasp_confirmed = True
    exp.grasp_relation = exp.relative_object().copy()
    exp.motion_settled = lambda: True
    tick(exp, 0.0)
    assert exp.touch_index == 1
    assert exp.phase == "TOUCH_1"
    wrist = np.array(exp.targets[-1]["wrist_transform_world"])
    object_target = wrist @ exp.grasp_relation
    assert exp.destination_transform[2, 3]-object_target[2, 3] == pytest.approx(exp.demo_cfg["touch_step_m"])
    assert exp.hands.controllers["left"].command == 1


def test_approach_uses_hand_local_positive_y_then_returns_laterally():
    exp, _ = gate("READY")
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    exp.object_state["T_world_cylinder"][:3, :3] = rotation
    exp.prepare_approach()
    assert exp.phase == "OUTWARD_ALIGN"
    delta = exp.pregrasp_wrist_target[:3, 3] - exp.grasp_wrist_target[:3, 3]
    np.testing.assert_allclose(delta, rotation[:, 1]*exp.demo_cfg["pregrasp_retreat_m"], atol=1e-12)
    assert np.dot(delta, rotation[:, 0]) == pytest.approx(0)
    align = np.array(exp.targets[-1]["wrist_transform_world"])
    # Alignment changes only the coordinate along the hand's local +Y axis.
    movement = align[:3, 3]-exp.wrist()[:3, 3]
    np.testing.assert_allclose(np.cross(movement, rotation[:, 1]), np.zeros(3), atol=1e-12)
    exp.next_approach()
    step = np.array(exp.targets[-1]["wrist_transform_world"])
    remaining = max(0., exp.demo_cfg["pregrasp_retreat_m"]-exp.demo_cfg["approach_step_m"])
    np.testing.assert_allclose(step[:3, 3]-exp.grasp_wrist_target[:3, 3], rotation[:, 1]*remaining, atol=1e-12)
    assert all(side == "left" for side, _, _ in exp.policy.commands)


def test_final_phase_records_demo_completion_without_claiming_strict_acceptance(monkeypatch):
    exp, _ = gate("COMPLETE")
    exp.scene_cfg, exp.warmup_duration_s = {}, 2.0
    exp.model = SimpleNamespace(opt=SimpleNamespace(enableflags=int(mujoco.mjtEnableBit.mjENBL_MULTICCD)))
    monkeypatch.setattr(ReachCompatibilityExperiment, "report", lambda self: {"phase": self.phase})
    report = exp.report()
    assert report["passed"] and report["demo_completed"]
    assert report["reach_precision_gate_bypassed"] is True
    assert report["strict_task_acceptance"] is False
    assert report["object_runtime_resets"] == report["object_welds"] == 0
    assert report["multiccd_enabled"]


def test_release_retreat_includes_backoff_for_thumb_clearance():
    exp, _ = gate("RELEASE")
    exp.released = True
    tick(exp, 0.0)
    tick(exp, 0.3)
    assert exp.phase == "RETREAT"
    target = np.array(exp.targets[-1]["wrist_transform_world"])
    delta = target[:3, 3] - exp.wrist()[:3, 3]
    np.testing.assert_allclose(delta, [-exp.demo_cfg["retreat_backoff_m"],
                                      exp.demo_cfg["retreat_distance_m"], 0.0])


def test_named_initial_state_copy_preserves_extra_free_object_even_when_reordered():
    robot = '''<body name="base_link" pos="0 0 .8"><freejoint name="floating_base_joint"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body><joint name="left_hip_pitch_joint"/><geom type="sphere" size=".03" mass=".1"/>
        <body pos="0 0 .1"><joint name="left_thumb_metacarpal_joint"/>
          <geom type="sphere" size=".01" mass=".01"/></body></body></body>'''
    object_body = '''<body name="test_cylinder" pos="2 3 4"><freejoint name="cylinder_free"/>
                    <geom type="cylinder" size=".02 .06" mass=".1"/></body>'''
    motor = '<motor name="{0}" joint="{1}"/>'
    actuators = [motor.format("hip_motor", "left_hip_pitch_joint"),
                 motor.format("finger_motor", "left_thumb_metacarpal_joint")]
    old = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot+'</worldbody><actuator>'
                                       + ''.join(actuators) + '</actuator></mujoco>')
    new = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+object_body+robot+'</worldbody><actuator>'
                                       + ''.join(reversed(actuators)) + '</actuator></mujoco>')
    source, target = mujoco.MjData(old), mujoco.MjData(new)
    source.qpos[-2:] = [0.23, 0.45]
    source.qvel[:] = np.arange(old.nv)/10
    source.qacc_warmstart[:] = np.arange(old.nv)+20
    source.ctrl[:] = [1.2, -0.5]
    target.qvel[:6] = np.arange(6)+40
    target.qacc_warmstart[:6] = np.arange(6)+50
    object_q, object_v = target.qpos[:7].copy(), target.qvel[:6].copy()
    demo.copy_robot_initial_state(old, source, new, target)
    np.testing.assert_array_equal(target.qpos[:7], object_q)
    np.testing.assert_array_equal(target.qvel[:6], object_v)
    for joint in range(old.njnt):
        name = old.joint(joint).name
        nq, nv = (7, 6) if old.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
        oq, nqadr = old.joint(name).qposadr[0], new.joint(name).qposadr[0]
        ov, nvadr = old.joint(name).dofadr[0], new.joint(name).dofadr[0]
        np.testing.assert_array_equal(target.qpos[nqadr:nqadr+nq], source.qpos[oq:oq+nq])
        np.testing.assert_array_equal(target.qvel[nvadr:nvadr+nv], source.qvel[ov:ov+nv])
    assert target.ctrl[new.actuator("hip_motor").id] == 1.2
    assert target.ctrl[new.actuator("finger_motor").id] == -0.5


def test_constructor_copies_input_configuration_and_policy_history(monkeypatch):
    cfg = demo.load_demo_config()
    model, hands = build_reach_model(cfg["reach"])
    data = mujoco.MjData(model)
    initialize_robot(model, data, hands)
    data.time = cfg["warmup_standing_s"]
    warm = SimpleNamespace(model=model, data=data, scratch=data, phase="STANDING_CHECK", phase_start=0.,
                           sync=lambda: None, policy=PolicyStub(), hands=HandStub(), right_lock=None,
                           initial_base_height=0.82, initial_base_xy=np.zeros(2))
    warm.policy.histories["sample"][:] = 7.0
    monkeypatch.setattr(demo, "ReachCompatibilityExperiment", lambda ignored: warm)
    monkeypatch.setattr(demo, "ReachPolicy", PolicyStub)
    monkeypatch.setattr(demo.TabletopDemoExperiment, "record", lambda self: None)
    monkeypatch.setattr(demo.TabletopDemoExperiment, "_contacts", lambda self: [])

    def sync(exp):
        transform = np.eye(4)
        transform[:3, 3] = exp.scene_cfg["cylinder_position_xyz"]
        exp.object_state = {"T_world_cylinder": transform}
        exp.contact_state = {"robot_table_contacts": []}

    monkeypatch.setattr(demo.TabletopDemoExperiment, "sync", sync)
    exp = demo.TabletopDemoExperiment(cfg)
    original_half_size = cfg["table_half_size"].copy()
    original_gate = cfg["reach"]["gates"]["position_m"]
    cfg["table_half_size"][0] += 0.5
    cfg["reach"]["gates"]["position_m"] = 123
    warm.policy.histories["sample"][0, 0] = 100
    assert exp.demo_cfg["table_half_size"] == original_half_size
    assert exp.scene_cfg["table_half_size"] == original_half_size
    assert exp.cfg["gates"]["position_m"] == original_gate
    assert exp.policy.histories["sample"][0, 0] == 7
