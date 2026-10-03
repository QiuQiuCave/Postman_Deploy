"""CPU safety/contract regressions; synthetic gates are not rollout evidence."""

import copy
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.path_config import PROJECT_ROOT
from common.r2v2_top_grasp_fullbody import (
    AIR_STAGES, TopGraspFullbodyExperiment, at_wrist_goal, build_fullbody_model,
    fingerprint, load_config, require_air_evidence,
)
import common.r2v2_top_grasp_fullbody as fullbody
from r2v2_description.model import (
    BODY_JOINTS, JointMap, SIDES, build_model, hand_names,
)


def config(**updates):
    result = dict(
        reach_config=str(PROJECT_ROOT / "deploy_mujoco/config/r2v2_reach_wrist_v2.yaml"),
        parity_report="unused-in-scene-only-tests.json",
        table_height_m=.75,
        can_xy_m=[.4, .2],
        yaw_deg=180.,
        table_center_xy_m=[.45, .15],
        table_half_size_m=[.25, .3, .02],
        stage_timeout_s=10.,
        air_continue_on_precision_failure=True,
    )
    result.update(updates)
    return result


def errors(**updates):
    result = dict(position_m=.004, orientation_deg=2., linear_speed_mps=.01)
    result.update(updates)
    return result


def test_actual_wrist_gate_uses_all_three_required_measurements():
    assert at_wrist_goal(errors())
    assert not at_wrist_goal(errors(position_m=.0051))
    assert not at_wrist_goal(errors(orientation_deg=3.01))
    assert not at_wrist_goal(errors(linear_speed_mps=.0201))


@pytest.mark.parametrize("field", ["position_m", "orientation_deg", "linear_speed_mps"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -.001])
def test_actual_wrist_gate_rejects_invalid_error_metrics(field, value):
    assert not at_wrist_goal(errors(**{field: value}))


@pytest.mark.parametrize("field", ["position_m", "orientation_deg", "linear_speed_mps"])
def test_actual_wrist_gate_missing_measurement_cannot_pass(field):
    measurement = errors()
    measurement.pop(field)
    assert not at_wrist_goal(measurement)


@pytest.fixture(scope="module", params=["air", "contact"])
def scene(request):
    return request.param, build_fullbody_model(config(), request.param)


def test_no_fixture_no_robot_weld_and_all_body_hand_channels_disjoint(scene):
    mode, (model, hands, layout) = scene
    assert model.nmocap == 0
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert model.neq == 10  # Source finger mimics, never object/robot welds.
    assert model.joint("floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("base_link").mocapid[0] == -1
    assert model.body("tabletop").jntnum[0] == 0
    assert model.body("test_cylinder").mocapid[0] == -1
    assert model.nu == 40
    body = JointMap.create(model, BODY_JOINTS)
    hand_maps = {side: JointMap.create(model, hand_names(side)) for side in SIDES}
    np.testing.assert_array_equal(body.actuators, np.arange(28))
    channels = [set(body.actuators)] + [set(hand_maps[s].actuators) for s in SIDES]
    assert len(channels[1]) == len(channels[2]) == 6
    assert all(not channels[a] & channels[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    assert set.union(*channels) == set(range(model.nu))
    if mode == "contact":
        assert model.joint("cylinder_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert model.body("test_cylinder").jntnum[0] == 1


def test_original_robot_limits_mass_collision_and_actuators_are_preserved(scene):
    _, (model, hands, _) = scene
    original = build_model(hands)
    for key in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(model, key)[:original.nbody], getattr(original, key))
    for key in ("jnt_type", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited", "jnt_stiffness"):
        np.testing.assert_array_equal(getattr(model, key)[:original.njnt], getattr(original, key))
    for key in ("dof_damping", "dof_armature", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(model, key)[:original.nv], getattr(original, key))
    for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype",
                "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(model, key)[:original.ngeom], getattr(original, key))
    for key in ("actuator_trnid", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
                "actuator_gainprm", "actuator_biasprm", "eq_type", "eq_data", "exclude_signature"):
        np.testing.assert_array_equal(getattr(model, key), getattr(original, key))


def test_air_props_are_noncolliding_and_contact_object_is_real_physics(scene):
    mode, (model, _, _) = scene
    for name in ("tabletop_geom", "cylinder_geom"):
        geom = model.geom(name).id
        if mode == "air":
            assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0
        else:
            assert model.geom_contype[geom] or model.geom_conaffinity[geom]
    if mode == "contact":
        assert model.body("test_cylinder").mass[0] == pytest.approx(.1)
        np.testing.assert_allclose(model.geom("cylinder_geom").size[:2], [.02, .06])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()


def test_world_endpoint_sites_are_true_wrist_origins(scene):
    _, (model, _, _) = scene
    for side in SIDES:
        wrist = model.body(f"{side}_hand_roll_link").id
        site = model.site(f"{side}_wrist").id
        assert model.site_bodyid[site] == wrist
        np.testing.assert_array_equal(model.site_pos[site], [0., 0., 0.])
        np.testing.assert_array_equal(model.site_quat[site], [1., 0., 0., 0.])


def test_missing_air_evidence_never_allows_contact(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        require_air_evidence(tmp_path / "absent.json", config(), "calibration", "policy")


@pytest.mark.parametrize("report", [
    {}, {"success": True}, {"mode": "air", "air_passed": True},
    {"mode": "contact", "success": True, "air_passed": True},
    {"mode": "air", "air_passed": True, "stage_results": []},
])
def test_incomplete_or_wrong_scope_air_evidence_is_rejected(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        require_air_evidence(path, config(), "calibration", "policy")


@pytest.mark.parametrize("change", [
    {"stage_timeout_s": 10.001}, {"stage_timeout_s": 0},
    {"stage_timeout_s": float("nan")}, {"stage_timeout_s": True},
    {"table_height_m": float("inf")}, {"yaw_deg": 181.},
    {"table_half_size_m": [.2, -.2, .02]},
    {"table_half_size_m": [.2, .2, .5]},
    {"can_xy_m": [.4, float("nan")]}, {"can_xy_m": [.4]},
    {"can_xy_m": [True, .2]}, {"air_continue_on_precision_failure": 1},
    {"profile": "sleek_330ml_approx_full"}, {"unreviewed_relaxation": True},
])
def test_invalid_configuration_fails_before_loading_an_actor(change):
    with pytest.raises(ValueError):
        load_config(config(**change))


def positive_air_report():
    return dict(
        schema_version=1, mode="air", air_passed=True, success=True, failure=None,
        experiment_completed=True, phase="COMPLETE", wrist_fixture=False,
        object_pose_replay=False, object_weld=False, additional_external_forces=False,
        grasp_verified=False, release_commanded=False,
        safety_violations_seen=[], endpoint_contract="wrist_world_v2",
        stage_results=[dict(name=phase, passed=True, duration_s=20. if phase == "STAND" else .5,
                            settled_duration_s=.3, standing_final_base_tilt_deg=1.,
                            standing_final_root_speed=.01,
                            final_errors={side: errors() for side in SIDES},
                            possible_table_collision=False) for phase in AIR_STAGES],
        config_sha256=fingerprint(config()), calibration_sha256="calibration", onnx_sha256="policy",
        controller_sha256=fullbody.sha256(fullbody.__file__),
        adapter_sha256=fullbody.sha256(PROJECT_ROOT / "common/r2v2_reach_policy.py"),
        **fullbody.evidence_input_bindings(config()),
    )


def test_matching_complete_air_report_is_accepted(tmp_path):
    report = positive_air_report()
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    assert require_air_evidence(path, config(), "calibration", "policy") == report


@pytest.mark.parametrize("field", ["config_sha256", "calibration_sha256", "onnx_sha256",
                                   "controller_sha256", "adapter_sha256", "endpoint_contract",
                                   "model_source_sha256", "source_asset_manifest",
                                   "reach_config_sha256", "mujoco_version"])
def test_air_evidence_cannot_be_reused_for_other_scene_policy_or_code(tmp_path, field):
    report = positive_air_report()
    report[field] = "different"
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_air_evidence(path, config(), "calibration", "policy")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reorder", "failed", "safety"])
def test_air_evidence_requires_every_stage_once_in_order(tmp_path, mutation):
    report = positive_air_report()
    if mutation == "missing":
        report["stage_results"].pop()
    elif mutation == "duplicate":
        report["stage_results"][-1] = copy.deepcopy(report["stage_results"][-2])
    elif mutation == "reorder":
        report["stage_results"][3:5] = report["stage_results"][3:5][::-1]
    elif mutation == "failed":
        report["stage_results"][-1]["passed"] = False
    else:
        report["safety_violations_seen"].append({"reason": "non-foot ground contact"})
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_air_evidence(path, config(), "calibration", "policy")


def supervisor(phase="GRASP", mode="contact"):
    """Real supervisor methods with synthetic kinematics; never grasp evidence."""
    exp = TopGraspFullbodyExperiment.__new__(TopGraspFullbodyExperiment)
    exp.config, exp.mode = config(), mode
    exp.data = SimpleNamespace(time=0.)
    exp.phase, exp.phase_start = phase, 0.
    exp.failure = exp.failure_phase = None
    exp.completed = exp.grasp_verified = exp.release_commanded = False
    exp.stable_since = exp.bad_since = None
    exp.transitions, exp.targets, exp.stage_results = [], [], []
    exp.safety_violations_seen, exp.hand_commands, exp.policy_commands = [], [], []
    exp.feedback_target_checks = []
    exp.stage_collision = False
    exp.stage_peak = dict(position_m=0., orientation_deg=0., right_position_m=0., right_orientation_deg=0.)
    exp.active_goal = np.eye(4)
    exp.goal_wrist_transforms = {side: np.eye(4) for side in SIDES}
    exp.path_targets = {phase: np.eye(4) for phase in fullbody.PATH_NAMES}
    controllers = {side: SimpleNamespace(command=0) for side in SIDES}

    def hand_command(side, value):
        controllers[side].command = value
        exp.hand_commands.append((side, value))

    exp.hands = SimpleNamespace(controllers=controllers, command=hand_command)
    exp.policy = SimpleNamespace(set_wrist_target_world=lambda *args: exp.policy_commands.append(args))
    exp.current_metrics = dict(wrist_errors={side: errors() for side in SIDES}, opposed_contact=False)
    return exp


def tick(exp, time):
    exp.data.time = float(time)
    exp._gate()


def test_contact_cannot_close_before_actual_pose_and_speed_hold_point_three_seconds():
    exp = supervisor()
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .06
    tick(exp, 0.); tick(exp, .5)
    assert exp.phase == "GRASP" and not exp.hand_commands
    exp.current_metrics["wrist_errors"]["left"] = errors()
    tick(exp, .6); tick(exp, .88)
    assert not exp.hand_commands
    tick(exp, .9)
    assert exp.phase == "CLOSE" and exp.hand_commands == [("left", 1)]
    for time in (1., 1.5, 2.):
        tick(exp, time)
    assert exp.hand_commands == [("left", 1)]
    assert not exp.grasp_verified


@pytest.mark.parametrize("side,field,value", [
    ("left", "position_m", .0051), ("left", "orientation_deg", 3.01),
    ("left", "linear_speed_mps", .021), ("right", "position_m", .021),
    ("right", "orientation_deg", 5.01), ("right", "linear_speed_mps", .051),
])
def test_any_required_endpoint_violation_resets_closure_hold(side, field, value):
    exp = supervisor()
    tick(exp, 0.); tick(exp, .2)
    exp.current_metrics["wrist_errors"][side][field] = value
    tick(exp, .25)
    exp.current_metrics["wrist_errors"][side] = errors()
    tick(exp, .3); tick(exp, .58)
    assert exp.phase == "GRASP" and not exp.hand_commands
    tick(exp, .6)
    assert exp.hand_commands == [("left", 1)]


@pytest.mark.parametrize("phase", ["CLOSE", "OPEN"])
def test_air_cannot_execute_binary_grasp_or_release_states(phase):
    exp = supervisor(mode="air")
    exp.enter(phase)
    assert exp.phase == "FAILED"
    assert not exp.hand_commands and not exp.release_commanded


def test_air_precision_timeout_can_continue_but_is_recorded_failed():
    exp = supervisor(phase="HOVER", mode="air")
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .08
    tick(exp, 9.9)
    assert exp.phase == "HOVER"
    tick(exp, 10.)
    assert exp.phase == "APPROACH"
    assert exp.stage_results[0]["name"] == "HOVER"
    assert exp.stage_results[0]["passed"] is False
    assert not exp.hand_commands


def test_contact_precision_timeout_never_uses_air_continue_override():
    exp = supervisor()
    assert exp.config["air_continue_on_precision_failure"]
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .08
    tick(exp, 10.)
    assert exp.phase == "FAILED" and exp.failure_phase == "GRASP"
    assert not exp.hand_commands


def test_air_table_overlap_is_remembered_even_if_endpoint_later_clears_table():
    exp = supervisor(phase="HOVER", mode="air")
    exp.stage_collision = True
    for time in (0., .3, 1., 10.):
        tick(exp, time)
    result = exp.stage_results[0]
    assert result["possible_table_collision"] is True and result["passed"] is False
    assert not exp.hand_commands


@pytest.mark.parametrize("mutation", [
    "collision", "left_error", "right_error", "moving", "nan_error", "short_hold",
    "hold_exceeds_phase", "short_stand", "stand_tilt", "stand_speed", "missing_error",
    "not_completed", "fixture", "replay", "external_force", "closed_in_air",
    "stage_not_object", "report_not_object", "bool_schema",
])
def test_contradictory_air_pass_flags_do_not_override_actual_evidence(tmp_path, mutation):
    report = positive_air_report()
    stage = report["stage_results"][-1]
    stand = next(s for s in report["stage_results"] if s["name"] == "STAND")
    if mutation == "collision":
        stage["possible_table_collision"] = True
    elif mutation == "left_error":
        stage["final_errors"]["left"]["position_m"] = .006
    elif mutation == "right_error":
        stage["final_errors"]["right"]["orientation_deg"] = 5.1
    elif mutation == "moving":
        stage["final_errors"]["left"]["linear_speed_mps"] = .03
    elif mutation == "nan_error":
        stage["final_errors"]["left"]["position_m"] = float("nan")
    elif mutation == "short_hold":
        stage["settled_duration_s"] = .299
    elif mutation == "hold_exceeds_phase":
        stage["settled_duration_s"] = 1.
    elif mutation == "short_stand":
        stand["duration_s"] = 19.9
    elif mutation == "stand_tilt":
        stand["standing_final_base_tilt_deg"] = 8.01
    elif mutation == "stand_speed":
        stand["standing_final_root_speed"] = .21
    elif mutation == "missing_error":
        stage["final_errors"]["left"].pop("position_m")
    elif mutation == "not_completed":
        report["experiment_completed"] = False
    elif mutation == "fixture":
        report["wrist_fixture"] = True
    elif mutation == "replay":
        report["object_pose_replay"] = True
    elif mutation == "external_force":
        report["additional_external_forces"] = True
    elif mutation == "closed_in_air":
        report["grasp_verified"] = True
    elif mutation == "stage_not_object":
        report["stage_results"][-1] = True
    elif mutation == "report_not_object":
        report = []
    elif mutation == "bool_schema":
        report["schema_version"] = True
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_air_evidence(path, config(), "calibration", "policy")


def test_home_evidence_has_explicit_loose_gate_but_task_endpoints_do_not(tmp_path):
    report = positive_air_report()
    report["stage_results"][0]["final_errors"]["left"] = errors(position_m=.02, orientation_deg=6., linear_speed_mps=.04)
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    require_air_evidence(path, config(), "calibration", "policy")
    report["stage_results"][-1]["final_errors"]["left"] = errors(position_m=.02)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        require_air_evidence(path, config(), "calibration", "policy")


@pytest.mark.parametrize("phase", ["UPRIGHT", "PLACE"])
@pytest.mark.parametrize("deviation", ["position", "orientation", "invalid_rotation", "nan"])
def test_feedback_target_outside_air_neighborhood_stops_without_opening(phase, deviation):
    exp = supervisor(phase="VERIFY")
    exp.hands.controllers["left"].command = 1
    target = np.eye(4)
    if deviation == "position":
        target[0, 3] = .0201
    elif deviation == "orientation":
        angle = np.deg2rad(5.1)
        target[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    elif deviation == "invalid_rotation":
        target[0, 0] = 2.
    else:
        target[0, 3] = np.nan
    exp.enter(phase, target)
    assert exp.phase == "FAILED"
    assert not exp.policy_commands and not exp.hand_commands
    assert exp.hands.controllers["left"].command == 1
    assert not exp.release_commanded
    assert exp.feedback_target_checks[-1]["accepted"] is False


@pytest.mark.parametrize("phase", ["UPRIGHT", "PLACE"])
def test_small_feedback_target_is_recorded_but_not_claimed_as_a_safety_certificate(phase):
    exp = supervisor(phase="VERIFY")
    target = np.eye(4); target[0, 3] = .005
    exp.enter(phase, target)
    assert exp.phase == phase and len(exp.policy_commands) == 1
    check = exp.feedback_target_checks[-1]
    assert check["accepted"] is True
    assert check["is_reachability_or_collision_certificate"] is False


@pytest.mark.parametrize("changed_input", ["asset", "reach_config"])
def test_modifying_input_bytes_at_same_path_invalidates_existing_air_report(tmp_path, monkeypatch, changed_input):
    report = positive_air_report()
    path = tmp_path / "air.json"
    path.write_text(json.dumps(report))
    original_hash = fullbody.sha256
    changed_path = (fullbody.SOURCE / "r2v2_with_hand.xml" if changed_input == "asset"
                    else PROJECT_ROOT / "deploy_mujoco/config/r2v2_reach_wrist_v2.yaml")
    monkeypatch.setattr(fullbody, "sha256", lambda p: "new-bytes-at-same-path"
                        if str(p) == str(changed_path) else original_hash(p))
    with pytest.raises(ValueError, match="source/asset evidence mismatch"):
        require_air_evidence(path, config(), "calibration", "policy")


def test_config_fingerprint_excludes_evidence_path_but_binds_scene_and_timeouts():
    expected = fingerprint(config())
    assert fingerprint(config(air_evidence="another-air-report.json")) == expected
    assert fingerprint(config(air_continue_on_precision_failure=False)) == expected
    for change in (dict(table_height_m=.8), dict(can_xy_m=[.41, .2]), dict(yaw_deg=90.),
                   dict(table_center_xy_m=[.5, .15]), dict(stage_timeout_s=8.)):
        assert fingerprint(config(**change)) != expected


def safety_supervisor(phase="PLACE", mode="contact"):
    exp = supervisor(phase=phase, mode=mode)
    exp.model = SimpleNamespace(nmocap=0, eq_type=np.zeros(0, dtype=int),
        jnt_range=np.tile([-1., 1.], (28, 1)), geom=lambda i: SimpleNamespace(name=f"geom_{i}"))
    exp.data = SimpleNamespace(time=0., qpos=np.zeros(57), qvel=np.zeros(56),
        warning=SimpleNamespace(number=np.zeros(8, dtype=int)),
        xfrc_applied=np.zeros((2, 6)), qfrc_applied=np.zeros(56),
        contact=[], xpos=np.array([[0., 0., .8]]))
    exp.body_map = SimpleNamespace(qpos=np.arange(28), joints=np.arange(28))
    exp.base, exp.floor_geom, exp.robot_geoms, exp.foot_geoms = 0, 100, {0, 1}, {1}
    exp.peaks = dict(body_joint_violation_rad=0., robot_self_penetration_m=0.)
    exp.right_locked, exp.inactive_bad_since = True, None
    exp.current_metrics.update(
        finite_state=True, base_tilt_deg=0., hand_joint_violation_rad=0., mimic_error_rad=0.,
        robot_table_contact_count=0, nonhand_object_contact_count=0, parked_hand_contact_count=0,
        active_hand_object_contact_count=0, hand_table_contact_count=0, floor_contact=False,
        hand_object_penetration_m=0., hand_self_penetration_m=0., object_tilt_deg=0.,
        table_contact=False, opposed_contact=True, grasp_slip_m=0., grasp_rotation_slip_deg=0.,
    )
    return exp


@pytest.mark.parametrize("violation", ["opposition", "tilt", "slip", "rotation", "unknown_slip"])
def test_airborne_placement_retention_failure_never_opens_hand(violation):
    exp = safety_supervisor()
    exp.grasp_verified = True
    exp.hands.controllers["left"].command = 1
    changes = dict(opposition={"opposed_contact": False}, tilt={"object_tilt_deg": 10.1},
                   slip={"grasp_slip_m": .0151}, rotation={"grasp_rotation_slip_deg": 5.1},
                   unknown_slip={"grasp_slip_m": None})
    exp.current_metrics.update(changes[violation])
    exp._safety()
    assert exp.phase == "FAILED" and "placement" in exp.failure
    assert not exp.hand_commands and exp.hands.controllers["left"].command == 1
    assert not exp.release_commanded


@pytest.mark.parametrize("field,value", [("position_m", .021), ("orientation_deg", 5.1)])
def test_inactive_world_wrist_violation_requires_continuous_point_three_seconds(field, value):
    exp = safety_supervisor()
    exp.current_metrics["wrist_errors"]["right"][field] = value
    for time in (0., .29):
        exp.data.time = time; exp._safety()
        assert not exp.done
    exp.data.time = .31; exp._safety()
    assert exp.done and "Inactive wrist" in exp.failure
    assert not exp.hand_commands


def test_inactive_wrist_recovery_resets_violation_timer():
    exp = safety_supervisor()
    exp.current_metrics["wrist_errors"]["right"]["position_m"] = .021
    exp._safety()
    exp.data.time = .2
    exp.current_metrics["wrist_errors"]["right"] = errors()
    exp._safety()
    assert exp.inactive_bad_since is None
    exp.current_metrics["wrist_errors"]["right"]["position_m"] = .021
    for time in (.25, .54):
        exp.data.time = time; exp._safety()
        assert not exp.done
    exp.data.time = .56; exp._safety()
    assert exp.done


@pytest.mark.parametrize("phase", ["RESET_SETTLE", "HOME", "STAND", "SAFE_OUT", "TURN_WRIST", "HOVER"])
def test_premature_hand_contact_cannot_be_hidden_as_successful_approach(phase):
    exp = safety_supervisor(phase)
    exp.current_metrics["active_hand_object_contact_count"] = 1
    exp._safety()
    assert exp.phase == "FAILED" and "Premature" in exp.failure
    assert not exp.hand_commands


def test_contact_during_actual_approach_is_not_premature_grasp_success():
    exp = safety_supervisor("APPROACH")
    exp.current_metrics["active_hand_object_contact_count"] = 1
    exp._safety()
    assert not exp.done and not exp.grasp_verified and not exp.hand_commands


@pytest.mark.parametrize("field", ["robot_table_contact_count", "nonhand_object_contact_count", "parked_hand_contact_count"])
def test_body_or_inactive_hand_cannot_supply_unreported_object_support(field):
    exp = safety_supervisor()
    exp.current_metrics[field] = 1
    exp._safety()
    assert exp.done and "non-grasping-part" in exp.failure


@pytest.mark.parametrize("violation", ["nonfoot_ground", "self_collision", "external_force", "body_limit", "air_closed"])
def test_unsafe_state_stops_even_in_diagnostic_air_mode(violation):
    exp = safety_supervisor("HOVER", "air")
    if violation == "nonfoot_ground":
        exp.data.contact.append(SimpleNamespace(geom=(100, 0), dist=-.001))
    elif violation == "self_collision":
        exp.data.contact.append(SimpleNamespace(geom=(0, 1), dist=-.006))
    elif violation == "external_force":
        exp.data.xfrc_applied[0, 2] = 1.
    elif violation == "body_limit":
        exp.data.qpos[0] = 1.051
    else:
        exp.hands.controllers["left"].command = 1
    exp._safety()
    assert exp.done and not exp.hand_commands


def test_step_uses_separate_torque_channels_and_never_assigns_object_pose(monkeypatch):
    exp = supervisor("CLOSE", "contact")
    exp.steps = 1
    exp.data = SimpleNamespace(qpos=np.arange(64, dtype=float), ctrl=np.zeros(40))
    exp.model = object()
    exp._safety = lambda: None
    exp.sync = lambda: None
    exp.record = lambda: None
    before = exp.data.qpos.copy()
    calls = []

    def body_apply(data):
        calls.append("body")
        data.ctrl[:28] = np.arange(28)

    def hands_apply(data):
        calls.append("hands")
        np.testing.assert_array_equal(data.ctrl[:28], np.arange(28))
        data.ctrl[28:] = 100 + np.arange(12)

    exp.policy.apply, exp.hands.apply = body_apply, hands_apply
    monkeypatch.setattr(fullbody.mujoco, "mj_step", lambda model, data: calls.append("physics"))
    exp.step()
    assert calls == ["body", "hands", "physics"]
    np.testing.assert_array_equal(exp.data.qpos, before)
    np.testing.assert_array_equal(exp.data.ctrl[:28], np.arange(28))
    np.testing.assert_array_equal(exp.data.ctrl[28:], 100 + np.arange(12))
