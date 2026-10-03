"""Front-drop acceptance and command-bus regressions; not rollout evidence."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from common.r2v2_front_drop import (
    FrontDropExperiment, FrontDropPath, PHASES, SCOPE, canonical_digest,
    cylinder_envelope_in_crate, deposited, load_config, measured_pickup, pose, valid_drop_location,
)
import common.r2v2_front_drop as front_drop


def document():
    left = [[.246, .19698, 1.0879], [.246, .19698, 1.0879],
            [.255, .215, 1.0879], [.255, .215, 1.041], [.255, .165, 1.041],
            [.255, .165, 1.061], [.255, .165, 1.235], [.305, .475, 1.235],
            [.305, .475, 1.235], [.255, .165, 1.235], [.246, .19698, 1.0879]]
    result = dict(schema_version=1, endpoint_contract="wrist_world_v2",
        path_contract="front_manipulation_path_v1", task_scope=SCOPE,
        geometry_validation=dict(passed=True, scope="synthetic unit-test geometry, no physics claim"),
        phases=[dict(name=name, duration_s=3., motion_s=0. if name in ("STAND", "HOME", "DROP_RELEASE") else 2.,
                     loaded=name in ("PROBE", "LIFT_CLEAR", "TRANSFER_ABOVE"),
                     position_m=[p, [.246, -.19643, 1.0879]],
                     quaternion_wxyz=[[1., 0., 0., 0.], [1., 0., 0., 0.]])
                for name, p in zip(PHASES, left)],
        provenance=dict(scene=dict(can_initial_world_m=[.4, .13, 1.041],
            can_initial_quaternion_wxyz=[1., 0., 0., 0.],
            grasp_can_in_wrist_position_m=[.145, -.035, 0.],
            grasp_can_in_wrist_quaternion_wxyz=[1., 0., 0., 0.])))
    result["content_sha256"] = canonical_digest(result)
    return result


def write_path(tmp_path, value=None, rehash=True):
    value = copy.deepcopy(document() if value is None else value)
    if rehash:
        value["content_sha256"] = canonical_digest({k: v for k, v in value.items() if k != "content_sha256"})
    filename = tmp_path/"path.json"
    filename.write_text(json.dumps(value))
    return filename


def good_metrics():
    return dict(finite_state=True, opposed_contact=True, clearance_m=.02,
        table_contact=False, floor_contact=False, hand_table_contact_count=0,
        parked_hand_contact_count=0, hand_vertical_force_N=.981, object_tilt_deg=16.,
        object_linear_speed_mps=.001, object_angular_speed_radps=.01,
        crate_object_contact=False, grasp_slip_m=.001, grasp_rotation_slip_deg=.2,
        crate_table_contact=True, opening_xy_margin_m=.03, bottom_above_rim_m=.04,
        hand_above_rim_m=.04, crate_tilt_deg=0., crate_speed_mps=0., crate_angular_speed_radps=0.,
        interior_xy_margin_m=.03, object_inside_crate=False, crate_vertical_force_N=0.,
        active_hand_object_contact_count=3, T_wrist_object=np.eye(4).tolist(),
        wrist_errors={side: dict(position_m=.001, orientation_deg=.2, linear_speed_mps=.001,
                                 angular_speed_radps=.01) for side in ("left", "right")},
        foot_drift_m=.001, root_speed_mps=.001, root_angular_speed_radps=.01,
        base_tilt_deg=1., root_height_m=.82)


def deposited_metrics():
    return {**good_metrics(), "object_inside_crate": True, "crate_object_contact": True,
            "crate_vertical_force_N": .981, "active_hand_object_contact_count": 0}


def stub_experiment(tmp_path, phase="GRASP", mode="contact", metrics=None):
    """Real FSM methods with no simulator construction or model/ONNX needed."""
    exp = object.__new__(FrontDropExperiment)
    exp.path = FrontDropPath(write_path(tmp_path))
    exp.mode, exp.phase, exp.phase_start = mode, phase, 0.
    exp.data = SimpleNamespace(time=0.)
    exp.current_metrics = good_metrics() if metrics is None else copy.deepcopy(metrics)
    exp.hands = SimpleNamespace(command=Mock(), controllers={})
    exp.transitions, exp.stage_results, exp.safety_events = [], [], []
    exp.stable_since = exp.bad_since = None
    exp.stage_collision = False
    exp.grasp_verified = exp.release_commanded = exp.completed = False
    exp.failure = exp.failure_phase = None
    exp.baseline_relation = None
    exp.weight_N = .981
    exp.config = dict(stage_timeout_s=10.)
    exp.sync = Mock()
    return exp


def test_archived_path_uses_wrist_calibration_not_can_center(tmp_path):
    path = FrontDropPath(write_path(tmp_path))
    actual = path.target("GRASP")[0]
    np.testing.assert_allclose(actual[:3, 3], [.255, .165, 1.041], atol=1e-12)
    np.testing.assert_allclose(actual @ pose([.145, -.035, 0.], [1., 0., 0., 0.]),
                               pose([.4, .13, 1.041], [1., 0., 0., 0.]), atol=1e-12)
    assert not np.allclose(actual[:3, 3], path.scene["can_initial_world_m"])


def test_phase_cubic_endpoints_and_smoothstep(tmp_path):
    path = FrontDropPath(write_path(tmp_path))
    start, end = path.target("HOME"), path.target("SAFE_OUT")
    np.testing.assert_allclose(path.sample_phase("SAFE_OUT", 0.), start)
    np.testing.assert_allclose(path.sample_phase("SAFE_OUT", 2.), end)
    np.testing.assert_allclose(path.sample_phase("SAFE_OUT", 10.), end)
    np.testing.assert_allclose(path.sample_phase("SAFE_OUT", .5)[:, :3, 3],
                               (.84375*start+.15625*end)[:, :3, 3])
    np.testing.assert_allclose(path.sample_phase("HOME", 0.), path.target("HOME"))


def test_quaternion_slerp_uses_shortest_arc_and_preserves_right(tmp_path):
    value = document()
    index = PHASES.index("SAFE_OUT")
    angle = np.deg2rad(20.)
    value["phases"][index]["quaternion_wxyz"][0] = (-np.array([np.cos(angle/2), 0., 0., np.sin(angle/2)])).tolist()
    path = FrontDropPath(write_path(tmp_path, value))
    middle = path.sample_phase("SAFE_OUT", 1.)
    expected = pose([0., 0., 0.], [np.cos(angle/4), 0., 0., np.sin(angle/4)])
    np.testing.assert_allclose(middle[0, :3, :3], expected[:3, :3], atol=1e-12)
    np.testing.assert_allclose(middle[1], path.target("HOME")[1])


@pytest.mark.parametrize("elapsed", [-.001, float("nan"), float("inf")])
def test_bad_sample_time_rejected(tmp_path, elapsed):
    with pytest.raises(ValueError):
        FrontDropPath(write_path(tmp_path)).sample_phase("GRASP", elapsed)


@pytest.mark.parametrize("change", ["hash", "contract", "sequence", "right_position", "right_orientation", "calibration", "can_center"])
def test_invalid_archived_path_rejected(tmp_path, change):
    value = document()
    if change == "hash":
        value["phases"][2]["position_m"][0][0] += .01
    elif change == "contract":
        value["endpoint_contract"] = "legacy_tcp_v1"
    elif change == "sequence":
        value["phases"][2]["name"] = "UNTRAINED"
    elif change == "right_position":
        value["phases"][2]["position_m"][1][0] += .01
    elif change == "right_orientation":
        value["phases"][2]["quaternion_wxyz"][1] = [np.cos(.01), 0., 0., np.sin(.01)]
    elif change == "calibration":
        value["provenance"]["scene"]["grasp_can_in_wrist_position_m"][0] += .01
    else:
        value["phases"][PHASES.index("GRASP")]["position_m"][0] = [.4, .13, 1.041]
    with pytest.raises(ValueError):
        FrontDropPath(write_path(tmp_path, value, rehash=change != "hash"))


def test_tilted_front_grip_can_be_valid_but_closed_pose_alone_is_not():
    assert measured_pickup(good_metrics(), .981)
    assert not measured_pickup({**good_metrics(), "opposed_contact": False}, .981)


@pytest.mark.parametrize("key,value", [
    ("finite_state", False), ("opposed_contact", False), ("clearance_m", .0079),
    ("table_contact", True), ("floor_contact", True), ("hand_table_contact_count", 1),
    ("parked_hand_contact_count", 1), ("hand_vertical_force_N", .1),
    ("object_linear_speed_mps", .021), ("object_angular_speed_radps", .16),
    ("object_tilt_deg", 31.), ("crate_object_contact", True),
    ("grasp_slip_m", None), ("grasp_slip_m", .015), ("grasp_slip_m", float("nan")),
    ("grasp_rotation_slip_deg", 5.),
])
def test_pickup_requires_real_opposed_support_and_relative_stability(key, value):
    assert not measured_pickup({**good_metrics(), key: value}, .981)


@pytest.mark.parametrize("key,value", [
    ("finite_state", False), ("crate_table_contact", False), ("opening_xy_margin_m", .0049),
    ("bottom_above_rim_m", .0099), ("hand_above_rim_m", .0099), ("crate_tilt_deg", 5.),
    ("crate_speed_mps", .02), ("crate_angular_speed_radps", .15),
])
def test_drop_location_requires_complete_object_and_hand_clearance(key, value):
    assert valid_drop_location(good_metrics())
    assert not valid_drop_location({**good_metrics(), key: value})


def test_cylinder_envelope_includes_tilt_and_full_radius():
    crate = pose([.45, .44, .98], [1., 0., 0., 0.])
    can = pose([.45, .44, 1.235], [1., 0., 0., 0.])
    report = cylinder_envelope_in_crate(can, crate, .02, .06, [.24, .36, .16], .004)
    assert report["bottom_above_rim_m"] == pytest.approx(.035)
    assert report["interior_xy_margin_m"] == pytest.approx(.096)
    angle = np.deg2rad(30.)
    tilted = pose([.45, .44, 1.235], [np.cos(angle/2), 0., np.sin(angle/2), 0.])
    result = cylinder_envelope_in_crate(tilted, crate, .02, .06, [.24, .36, .16], .004)
    assert result["interior_xy_margin_m"] < report["interior_xy_margin_m"]
    assert result["bottom_above_rim_m"] == pytest.approx(.255-.06*np.cos(angle)-.02*np.sin(angle)-.16)


@pytest.mark.parametrize("key,value", [
    ("finite_state", False), ("crate_table_contact", False), ("crate_tilt_deg", 5.),
    ("crate_speed_mps", .02), ("crate_angular_speed_radps", .15),
    ("interior_xy_margin_m", -.00001), ("object_inside_crate", False),
    ("crate_object_contact", False), ("crate_vertical_force_N", .1),
    ("floor_contact", True), ("table_contact", True), ("active_hand_object_contact_count", 1),
    ("parked_hand_contact_count", 1), ("object_linear_speed_mps", .02),
    ("object_angular_speed_radps", .15),
])
def test_deposit_needs_actual_crate_support_containment_and_stillness(key, value):
    assert deposited(deposited_metrics(), .981)
    assert not deposited({**deposited_metrics(), key: value}, .981)


@pytest.mark.parametrize("mode,verified,change", [
    ("air", True, {}), ("contact", False, {}),
    ("contact", True, {"opening_xy_margin_m": -.001}),
    ("contact", True, {"bottom_above_rim_m": .001}),
    ("contact", True, {"hand_above_rim_m": .001}),
])
def test_open_command_is_impossible_before_verified_safe_release(tmp_path, mode, verified, change):
    exp = stub_experiment(tmp_path, mode=mode, metrics={**good_metrics(), **change})
    exp.grasp_verified = verified
    exp.enter("OPEN")
    assert exp.phase == "FAILED" and exp.release_commanded is False
    exp.hands.command.assert_not_called()


def test_open_is_authorized_after_actual_grasp_and_clearance(tmp_path):
    exp = stub_experiment(tmp_path)
    exp.grasp_verified = True
    exp.enter("OPEN")
    exp.hands.command.assert_called_once_with("left", 0)
    assert exp.release_commanded and exp.phase == "OPEN"
    assert exp.release_snapshot == exp.current_metrics


def test_air_can_never_close_fingers(tmp_path):
    exp = stub_experiment(tmp_path, mode="air")
    with pytest.raises(RuntimeError, match="AIR"):
        exp.enter("CLOSE")
    exp.hands.command.assert_not_called()


def test_contact_grasp_reaches_close_only_after_settled_pose(tmp_path):
    exp = stub_experiment(tmp_path)
    exp.data.time = 2.9
    exp._gate()
    exp.hands.command.assert_not_called()
    exp.data.time = 3.3
    exp._gate()
    assert exp.phase == "CLOSE"
    exp.hands.command.assert_called_once_with("left", 1)


def test_air_grasp_phase_advances_without_closing(tmp_path):
    exp = stub_experiment(tmp_path, mode="air")
    exp.data.time = 2.9; exp._gate()
    exp.data.time = 3.3; exp._gate()
    assert exp.phase == "PROBE"
    exp.hands.command.assert_not_called()


@pytest.mark.parametrize("phase", ["GRASP", "CLOSE", "VERIFY_GRASP", "LIFT_CLEAR", "TRANSFER_ABOVE", "DROP_RELEASE", "WAIT_DROP"])
def test_deadline_failure_never_automatically_opens_hand(tmp_path, phase):
    exp = stub_experiment(tmp_path, phase=phase, metrics={**good_metrics(), "opposed_contact": False})
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .10
    exp.data.time = 10.
    exp._gate()
    assert exp.phase == "FAILED"
    assert exp.failure_phase == phase
    assert exp.release_commanded is False
    exp.hands.command.assert_not_called()


def test_drop_not_credited_until_one_second_of_actual_deposit(tmp_path):
    exp = stub_experiment(tmp_path, phase="WAIT_DROP", metrics=deposited_metrics())
    exp.data.time = .1; exp._gate()
    exp.data.time = 1.05; exp._gate()
    assert exp.phase == "WAIT_DROP" and not getattr(exp, "drop_verified", False)
    exp.data.time = 1.11; exp._gate()
    assert exp.phase == "RETREAT_HIGH" and exp.drop_verified is True


def test_verified_drop_snapshot_refreshes_the_success_flag_before_copy(tmp_path):
    exp = stub_experiment(tmp_path, phase="WAIT_DROP",
        metrics={**deposited_metrics(), "drop_verified": False})
    exp.drop_verified = False

    def refresh_metrics():
        # Mirror sync(): measured metrics acquire the current task flag.
        exp.current_metrics["drop_verified"] = exp.drop_verified

    exp.sync.side_effect = refresh_metrics
    exp.data.time = .1
    exp._gate()
    exp.sync.assert_not_called()
    assert exp.drop_verified is False
    exp.data.time = 1.11
    exp._gate()
    exp.sync.assert_called_once_with()
    assert exp.phase == "RETREAT_HIGH"
    assert exp.drop_verified is True
    assert exp.drop_snapshot["drop_verified"] is True
    # The archived snapshot must remain independent of later measurement rows.
    exp.current_metrics["drop_verified"] = False
    assert exp.drop_snapshot["drop_verified"] is True


def test_runtime_control_does_not_replay_qpos_or_attach_object():
    tree = ast.parse(inspect.getsource(FrontDropExperiment))
    for method in tree.body[0].body:
        if not isinstance(method, ast.FunctionDef) or method.name == "__init__":
            continue
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("mj_resetData", "mj_setState")
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    source = ast.unparse(target)
                    assert not source.startswith(("self.data.qpos", "self.data.qvel", "self.data.mocap_", "self.data.xfrc_applied"))


def safety_stub(tmp_path, mode="air", phase="STAND"):
    exp = stub_experiment(tmp_path, phase=phase, mode=mode)
    exp.model = SimpleNamespace(nmocap=0, eq_type=np.array([], dtype=int),
        jnt_range=np.array([[-1., 1.]]))
    exp.data = SimpleNamespace(time=.14, warning=SimpleNamespace(number=np.zeros(8)),
        xfrc_applied=np.zeros((1, 6)), qfrc_applied=np.zeros(1), qpos=np.zeros(1), contact=[])
    exp.body_map = SimpleNamespace(qpos=np.array([0]), joints=np.array([0]))
    exp.peaks = {}
    exp.floor_geom, exp.robot_geoms, exp.foot_geoms = 0, set(), set()
    exp.current_metrics.update(robot_self_penetration_m=0., hand_joint_violation_rad=0.,
        mimic_error_rad=0., robot_prop_contacts=[], nonhand_object_contact_count=0,
        hand_object_penetration_m=0.)
    return exp


@pytest.mark.parametrize("phase", ["STAND", "HOME", "SAFE_OUT"])
def test_air_predicted_prop_contact_is_not_erased_on_phase_change(tmp_path, phase):
    exp = safety_stub(tmp_path, phase=phase)
    exp.current_metrics["robot_prop_contacts"] = [dict(geoms=["left_pinky", "tabletop"], penetration_m=.0001)]
    exp._safety()
    assert exp.phase == "FAILED" and exp.failure_phase == phase
    assert "AIR collision diagnostic" in exp.failure
    exp.hands.command.assert_not_called()


def test_air_runtime_guard_rejects_finger_closure(tmp_path):
    exp = safety_stub(tmp_path)
    exp.hands.controllers = {"left": SimpleNamespace(command=1)}
    exp._safety()
    assert exp.phase == "FAILED" and "finger closure" in exp.failure


def test_verified_grasp_loss_stops_without_automatic_release(tmp_path):
    exp = safety_stub(tmp_path, mode="contact", phase="TRANSFER_ABOVE")
    exp.grasp_verified = True
    exp.current_metrics["grasp_slip_m"] = .016
    exp._safety()
    assert exp.phase == "FAILED" and not exp.release_commanded
    assert "not automatically opened" in exp.failure
    exp.hands.command.assert_not_called()


def test_hold_home_startup_only_sets_world_goals(tmp_path):
    exp = stub_experiment(tmp_path, phase="STAND")
    exp.config["startup_mode"] = "hold_home"
    exp.policy = SimpleNamespace(set_target_world=Mock(), follow_current=Mock())
    exp.goal_wrist_transforms = {s: np.eye(4) for s in ("left", "right")}
    exp.targets = []
    exp._stream()
    exp.policy.follow_current.assert_not_called()
    assert exp.policy.set_target_world.call_count == 2
    for index, side in enumerate(("left", "right")):
        call = exp.policy.set_target_world.call_args_list[index]
        assert call.args[0] == side
        np.testing.assert_allclose(call.args[1], exp.path.target("HOME")[index, :3, 3])


@pytest.mark.parametrize("startup", ["", "scripted_pose", True, 1, None])
def test_unknown_startup_mode_is_rejected(startup):
    with pytest.raises(ValueError, match="startup"):
        load_config({"startup_mode": startup})


def test_air_evidence_binding_changes_with_startup_but_not_report_filename(tmp_path, monkeypatch):
    source = tmp_path/"source"
    source.mkdir()
    (source/"asset.xml").write_text("unit test source")
    monkeypatch.setattr(front_drop, "SOURCE", source)
    monkeypatch.setattr(front_drop, "sha256", lambda path: "test-digest:"+str(path))
    exp = stub_experiment(tmp_path)
    exp.config.update(startup_mode="follow_current", air_evidence="air-one.json")
    exp.policy_sha = "frozen-onnx"
    exp.parity = dict(checkpoint_sha256="frozen-checkpoint")
    exp.hand_cfg = dict(unit_test=True)
    original = exp._bindings()
    exp.config["air_evidence"] = "air-two.json"
    assert exp._bindings() == original
    exp.config["startup_mode"] = "hold_home"
    assert exp._bindings()["task_config_sha256"] != original["task_config_sha256"]
