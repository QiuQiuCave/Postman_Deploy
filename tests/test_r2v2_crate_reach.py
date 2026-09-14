"""Whole-body crate contracts; synthetic gates do not establish grasp success."""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import yaml

import common.r2v2_crate_reach as reach
from common.r2v2_crate import load_crate_config
from common.r2v2_crate_hand_preview import _world_vertices
from common.r2v2_crate_lift import load_lift_config
from common.r2v2_crate_lift_metrics import _descendant
from common.r2v2_crate_lift_scene import build_crate_lift_model
from common.r2v2_crate_reach import (
    CrateReachExperiment, load_crate_reach_config, wrist_goal_to_tcp,
)
from common.r2v2_crate_reach_scene import build_crate_reach_model, crate_wrist_targets
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_policy import ReachPolicy, ReachReference
from common.r2v2_reach_sim import (
    TCP_OFFSETS, foot_collision_ids, initialize_robot, load_reach_config,
    quaternion_from_matrix,
)
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, build_model_xml, urdf_hand_joints


@pytest.fixture(scope="module")
def scene():
    return build_crate_reach_model(load_reach_config(), load_crate_config(), table_top_m=.88)


def test_scene_keeps_the_complete_robot_physics_and_only_original_mimics(scene):
    model, hand_cfg, layout = scene
    source = mujoco.MjModel.from_xml_string(build_model_xml(hand_cfg, fixture=False))
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (64, 62, 40, 10, 0)
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 2
    assert model.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("cargo_crate").mocapid[0] == -1
    assert layout["driver"] == "dual_arm_reach_policy_no_wrist_fixtures"
    for name in ("test_cylinder", "left_wrist_fixture", "right_wrist_fixture"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    for name in ("cylinder_free", "left_wrist_free", "right_wrist_free"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
    for index in range(source.nbody):
        actual = model.body(source.body(index).name).id
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])
    for index in range(source.njnt):
        actual = model.joint(source.joint(index).name).id
        for field in ("jnt_type", "jnt_range", "jnt_limited", "jnt_axis", "jnt_pos", "jnt_stiffness"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(model, field)[model.jnt_dofadr[actual]],
                                          getattr(source, field)[source.jnt_dofadr[index]])
    for index in range(source.ngeom):
        actual = model.geom(source.geom(index).name).id
        for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                      "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])
    for index in range(source.nu):
        actual = model.actuator(source.actuator(index).name).id
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])
    for index in range(source.neq):
        actual = model.equality(source.equality(index).name).id
        np.testing.assert_array_equal(model.eq_data[actual], source.eq_data[index])
        np.testing.assert_array_equal(model.eq_solref[actual], source.eq_solref[index])
        np.testing.assert_array_equal(model.eq_solimp[actual], source.eq_solimp[index])


def test_target_calibration_matches_the_committed_fixture_with_world_translation(scene):
    _, _, layout = scene
    p = load_crate_config()
    _, _, fixture = build_crate_lift_model(p, insertion_m=.060, table_height=.88)
    for side in SIDES:
        for phase in ("initial", "inserted"):
            target = layout[f"{phase}_wrist_transforms"][side]
            expected = fixture[f"{phase}_wrist_positions"][side]+[.43, 0., 0.]
            np.testing.assert_allclose(target[:3, 3], expected, rtol=0, atol=1e-14)
            fixture_quaternion = fixture["wrist_quaternions"][side]
            if fixture_quaternion[0] < 0:
                fixture_quaternion = -fixture_quaternion
            np.testing.assert_array_equal(quaternion_from_matrix(target[:3, :3]), fixture_quaternion)
            assert abs(target[1, 3]) > p.width/2
    shift = np.array([.05, -.07, .09])
    translated = crate_wrist_targets(p, table_top_m=.88+shift[2], crate_center_xy=np.array([.43, 0.])+shift[:2])
    for key in ("initial_wrist_transforms", "inserted_wrist_transforms"):
        for side in SIDES:
            np.testing.assert_allclose(translated[key][side][:3, 3]-layout[key][side][:3, 3], shift, atol=1e-14)
            np.testing.assert_array_equal(translated[key][side][:3, :3], layout[key][side][:3, :3])
    translated["initial_wrist_transforms"]["left"][:] = 0
    fresh = crate_wrist_targets(p, table_top_m=.88, crate_center_xy=[.43, 0.])
    np.testing.assert_array_equal(fresh["initial_wrist_transforms"]["left"], layout["initial_wrist_transforms"]["left"])


@pytest.mark.parametrize("side", SIDES)
def test_wrist_to_tcp_rotates_the_calibrated_local_offset_and_preserves_rotation(scene, side):
    wrist = scene[2]["inserted_wrist_transforms"][side].copy()
    original = wrist.copy()
    tcp = wrist_goal_to_tcp(side, wrist)
    np.testing.assert_allclose(tcp[:3, 3], wrist[:3, 3]+wrist[:3, :3]@TCP_OFFSETS[side], atol=1e-14)
    np.testing.assert_array_equal(tcp[:3, :3], wrist[:3, :3])
    np.testing.assert_array_equal(wrist, original)
    assert not np.allclose(tcp[:3, 3], wrist[:3, 3]+TCP_OFFSETS[side])
    assert not np.allclose(tcp[:3, 3], scene[2]["crate_initial_position"])


@pytest.mark.parametrize("side,wrist", [("both", np.eye(4)), ("left", np.eye(3)), ("right", np.full((4, 4), np.nan))])
def test_wrist_target_rejects_invalid_side_shape_or_nonfinite(side, wrist):
    with pytest.raises(ValueError):
        wrist_goal_to_tcp(side, wrist)


def test_body_and_hand_torque_writes_are_disjoint_without_onnx(scene):
    model, hand_cfg, _ = scene
    data = mujoco.MjData(model)
    initialize_robot(model, data, hand_cfg)
    hands = DualHandControl(model, data, hand_cfg)
    body = JointMap.create(model, BODY_JOINTS)
    hand_ids = np.concatenate([h.actuators for h in hands.maps.values()])
    assert not set(hand_ids) & set(body.actuators)
    assert set(hand_ids) | set(body.actuators) == set(range(model.nu))
    assert len(body.actuators) == 28 and len(hand_ids) == 12
    policy = ReachPolicy.__new__(ReachPolicy)
    policy._direct_torque, policy.body_map = True, body
    policy.kp, policy.kd = np.ones(28), np.zeros(28)
    policy.q_des = data.qpos[body.qpos]+.123
    policy.torque_lower, policy.torque_upper = np.full(28, -1.), np.ones(28)
    data.ctrl[hand_ids] = .271
    policy.apply(data)
    np.testing.assert_array_equal(data.ctrl[hand_ids], np.full(12, .271))
    before_body = data.ctrl[body.actuators].copy()
    hands.apply(data)
    np.testing.assert_array_equal(data.ctrl[body.actuators], before_body)
    np.testing.assert_allclose(before_body, .123, atol=1e-14)


@pytest.mark.parametrize("patch", [
    {"warmup_standing_s": 0}, {"phase_timeout_s": True}, {"phase_timeout_s": float("inf")},
    {"crate_center_xy": [0, 0, 0]}, {"table_half_size": [.22, -.35, .02]},
    {"motion_stable_s": "slow"}, {"unexpected_key": 1},
])
def test_configuration_guards_reject_invalid_numeric_shapes_and_keys(tmp_path, patch):
    cfg = load_crate_reach_config()
    cfg.update(patch)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_crate_reach_config(path)


@pytest.mark.parametrize("kwargs", [
    {"table_top_m": .02}, {"table_center_xy": [.4, 0]}, {"crate_center_xy": [.65, .3]},
    {"table_half_size": [.22, .35, 0]}, {"crate_center_xy": [np.nan, 0]},
])
def test_scene_layout_guards_reject_unsupported_or_unsafe_table_geometry(kwargs):
    settings = {"table_top_m": .88, **kwargs}
    with pytest.raises(ValueError):
        build_crate_reach_model(load_reach_config(), load_crate_config(), **settings)


COMPACT_CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_crate_reach_compact.yaml"


@pytest.fixture(scope="module")
def compact_scene():
    cfg = load_crate_reach_config(COMPACT_CONFIG)
    params = replace(load_crate_config(), width=cfg["crate_width_m"])
    model, hands, layout = build_crate_reach_model(
        load_reach_config(), params, table_top_m=1.+cfg["table_above_waist_m"],
        crate_center_xy=cfg["crate_center_xy"], table_center_xy=cfg["table_center_xy"],
        table_half_size=cfg["table_half_size"])
    return model, hands, layout, cfg, params


def test_compact_profile_is_isolated_and_preserves_holes_contact_parameters_and_mass(compact_scene):
    model, _, _, cfg, params = compact_scene
    baseline = load_crate_config()
    original = load_crate_reach_config()
    assert params.width == pytest.approx(.26)
    assert (params.depth, params.height, params.mass) == pytest.approx((.24, .16, .4))
    assert params.handle_opening_width == pytest.approx(.120)
    assert params.handle_opening_height == pytest.approx(.055)
    changed = {k for k, v in asdict(params).items() if v != asdict(baseline)[k]}
    assert changed == {"width"}
    assert baseline.width == pytest.approx(.36)
    assert "crate_width_m" not in original
    assert original["table_above_waist_m"] == pytest.approx(.10)
    assert original["crate_center_xy"] == [.48, 0.]
    assert cfg["table_above_waist_m"] == 0.
    assert cfg["crate_center_xy"] == [.38, 0.]
    assert cfg["table_center_xy"] == original["table_center_xy"]
    assert cfg["table_half_size"] == original["table_half_size"]
    crate = model.body("cargo_crate").id
    assert model.body_mass[crate] == pytest.approx(.4, abs=1e-12)
    np.testing.assert_allclose(model.geom("crate_bottom").size, [.12, .13, .0025], atol=1e-14)
    for side in SIDES:
        front, back = model.geom(f"crate_{side}_front_post"), model.geom(f"crate_{side}_back_post")
        opening_x = front.pos[0]-front.size[0]-(back.pos[0]+back.size[0])
        assert opening_x == pytest.approx(.120, abs=1e-12)
        assert 2*front.size[2] == pytest.approx(.055, abs=1e-12)
        assert model.site(f"crate_{side}_handle").pos[2] == pytest.approx(.1125)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (64, 62, 40, 10, 0)


def test_compact_crate_physical_bounds_fit_table_with_front_edges_coincident(compact_scene):
    model, _, layout, _, params = compact_scene
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    crate = model.body("cargo_crate").id
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == crate]
    vertices = np.concatenate([_world_vertices(model, data, g) for g in geoms])
    lower, upper = vertices.min(0), vertices.max(0)
    np.testing.assert_allclose(upper-lower, [params.depth, params.width, params.height], atol=1e-8)
    table = model.geom("lift_table_geom").id
    table_lower = data.geom_xpos[table]-model.geom_size[table]
    table_upper = data.geom_xpos[table]+model.geom_size[table]
    assert lower[0] == pytest.approx(.26, abs=1e-12)
    assert lower[0] == pytest.approx(table_lower[0], abs=1e-12)
    assert np.all(lower[:2] >= table_lower[:2]-1e-10)
    assert np.all(upper[:2] <= table_upper[:2]+1e-10)
    assert lower[2]-table_upper[2] == pytest.approx(.001, abs=1e-12)
    assert layout["table_top_m"] == pytest.approx(1.)


def test_compact_world_targets_shift_closer_lower_and_fifty_mm_inward_per_hand(compact_scene):
    _, _, layout, _, _ = compact_scene
    original = load_crate_reach_config()
    baseline = crate_wrist_targets(load_crate_config(),
        table_top_m=1.+original["table_above_waist_m"], crate_center_xy=original["crate_center_xy"],
        insertion_m=.060)
    for key in ("initial_wrist_transforms", "inserted_wrist_transforms"):
        for side, inward in (("left", -.05), ("right", .05)):
            before, after = baseline[key][side], layout[key][side]
            np.testing.assert_allclose(after[:3, 3]-before[:3, 3], [-.1, inward, -.1], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(after[:3, :3], before[:3, :3])
            np.testing.assert_allclose(wrist_goal_to_tcp(side, after)[:3, 3]-wrist_goal_to_tcp(side, before)[:3, 3],
                                       [-.1, inward, -.1], rtol=0, atol=1e-14)


def test_compact_width_override_is_applied_privately_before_any_warmup_or_rollout(monkeypatch):
    class StopBeforeWarmup(Exception):
        pass

    cfg = load_crate_reach_config(COMPACT_CONFIG)
    original_cfg = copy.deepcopy(cfg)
    exp = CrateReachExperiment.__new__(CrateReachExperiment)
    monkeypatch.setattr(reach, "require_parity", lambda *args: {})
    warmup = Mock(side_effect=StopBeforeWarmup)
    monkeypatch.setattr(reach, "ReachCompatibilityExperiment", warmup)
    with pytest.raises(StopBeforeWarmup):
        exp.__init__(cfg)
    warmup.assert_called_once()
    assert exp.crate_params.width == pytest.approx(.26)
    assert load_crate_config().width == pytest.approx(.36)
    assert cfg == original_cfg
    assert exp.fullbody_cfg is not cfg


def test_zero_waist_offset_and_positive_optional_width_are_accepted(tmp_path):
    cfg = load_crate_reach_config()
    cfg.update(table_above_waist_m=0., crate_width_m=.26)
    path = tmp_path / "valid_compact.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert load_crate_reach_config(path) == cfg


@pytest.mark.parametrize("patch", [
    {"table_above_waist_m": -.001}, {"table_above_waist_m": False},
    {"crate_width_m": -.26}, {"crate_width_m": 0.}, {"crate_width_m": True},
    {"crate_width_m": float("nan")}, {"crate_width_m": [.26]},
])
def test_compact_scalar_guard_rejects_negative_offset_or_invalid_width(tmp_path, patch):
    cfg = load_crate_reach_config()
    cfg.update(patch)
    path = tmp_path / "invalid_compact.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_crate_reach_config(path)


def synthetic_experiment(phase="READY"):
    exp = CrateReachExperiment.__new__(CrateReachExperiment)
    exp.params, exp.crate_params = load_lift_config(), load_crate_config()
    exp.fullbody_cfg = load_crate_reach_config()
    exp.layout = crate_wrist_targets(exp.crate_params, table_top_m=.88, crate_center_xy=[.43, 0.])
    exp.phase, exp.phase_start, exp.failure, exp.failure_phase = "READY", 0., None, None
    exp.data = SimpleNamespace(time=0., qpos=np.arange(4, dtype=float), ctrl=np.arange(4, dtype=float))
    exp.stable_since = exp.bad_since = None
    exp.transitions, exp.targets, exp.samples, exp.hold_samples = [], [], [], []
    exp.baseline_relations = None
    exp.trial_confirmed = exp.prealign_passed = False
    exp.hands = SimpleNamespace(command=Mock(), apply=Mock(), update=Mock())
    policy = ReachPolicy.__new__(ReachPolicy)
    policy.references = {side: ReachReference(np.zeros(3), np.array([1., 0, 0, 0]),
        np.array([.11, .22, .33]), np.array([.02, .03, .04]), np.zeros(3), np.array([1., 0, 0, 0])) for side in SIDES}
    policy.histories = {"sentinel": np.arange(30).reshape(10, 3)}
    policy.last_action = np.arange(28)
    policy.reset, policy.follow_current = Mock(), Mock()
    policy.set_target_world = Mock(wraps=policy.set_target_world)
    policy.act, policy.apply = Mock(), Mock()
    exp.policy = policy
    exp.goal_wrist_transforms = copy.deepcopy(exp.layout["initial_wrist_transforms"])
    exp.current_metrics = {
        "wrist_errors": {side: {"position_m": 0., "orientation_deg": 0., "linear_speed_mps": 0.} for side in SIDES},
        "hands": {side: {"T_wrist_crate": np.eye(4), "finger_normal_force_N": 1.,
                          "finger_handle_vertical_force_N": 1., "vertical_force_N": 1.} for side in SIDES},
        "clearance_m": .10, "table_vertical_force_N": 0., "grasp_slip_m": 0.,
        "crate_tilt_deg": 0., "crate_position_m": [.43, 0., .98],
    }
    if phase != "READY":
        exp.enter(phase)
    return exp


def finish_reference(exp):
    for reference in exp.policy.references.values():
        reference.position = reference.goal_position.copy()
        reference.quaternion = reference.goal_quaternion.copy()


def test_phase_entry_sets_each_world_target_once_with_seat_trial_and_lift_offsets():
    exp = synthetic_experiment()
    reference_ids = {s: id(r) for s, r in exp.policy.references.items()}
    history_id, history = id(exp.policy.histories), copy.deepcopy(exp.policy.histories)
    action = exp.policy.last_action.copy()
    phases = (("PREALIGN", "initial", 0.), ("INSERT", "inserted", 0.),
              ("CLOSE", "inserted", .0175), ("TRIAL_LIFT", "inserted", .0375),
              ("LIFT", "inserted", .1175))
    for number, (phase, source, z) in enumerate(phases, 1):
        exp.enter(phase)
        assert exp.policy.set_target_world.call_count == 2*number
        assert len(exp.targets) == number
        for side in SIDES:
            wrist = exp.goal_wrist_transforms[side]
            np.testing.assert_allclose(wrist[:3, 3], exp.layout[f"{source}_wrist_transforms"][side][:3, 3]+[0, 0, z])
            expected = wrist_goal_to_tcp(side, wrist)
            reference = exp.policy.references[side]
            np.testing.assert_allclose(reference.goal_position, expected[:3, 3])
            assert id(reference) == reference_ids[side]
            np.testing.assert_array_equal(reference.position, np.zeros(3))
            np.testing.assert_array_equal(reference.linear_velocity, [.11, .22, .33])
            np.testing.assert_array_equal(reference.angular_velocity, [.02, .03, .04])
    for phase in ("TRIAL_HOLD", "HOLD", "COMPLETE"):
        exp.enter(phase)
    assert exp.policy.set_target_world.call_count == 10
    assert exp.hands.command.call_args_list == [(("left", 1),), (("right", 1),)]
    assert id(exp.policy.histories) == history_id
    np.testing.assert_array_equal(exp.policy.histories["sentinel"], history["sentinel"])
    np.testing.assert_array_equal(exp.policy.last_action, action)
    exp.policy.reset.assert_not_called()
    exp.policy.follow_current.assert_not_called()


def test_prealign_gross_actual_error_times_out_without_insertion_or_grasp():
    exp = synthetic_experiment("PREALIGN")
    finish_reference(exp)
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .1
    exp.data.time = exp.phase_start+exp.fullbody_cfg["phase_timeout_s"]
    exp._gate()
    assert exp.phase == "FAILED" and exp.failure_phase == "PREALIGN"
    assert not exp.prealign_passed
    assert not any(t["state"] in ("INSERT", "CLOSE") for t in exp.transitions)
    assert exp.policy.set_target_world.call_count == 2
    exp.hands.command.assert_not_called()


def test_prealign_requires_finished_reference_and_continuous_actual_stability():
    exp = synthetic_experiment("PREALIGN")
    exp.data.time = 1.
    exp._gate()
    assert exp.phase == "PREALIGN" and exp.stable_since is None
    finish_reference(exp)
    exp._gate()
    exp.data.time = 1.29
    exp._gate()
    assert exp.phase == "PREALIGN"
    exp.current_metrics["wrist_errors"]["right"]["orientation_deg"] = 15.
    exp.data.time = 1.30
    exp._gate()
    assert exp.stable_since is None
    exp.current_metrics["wrist_errors"]["right"]["orientation_deg"] = 0.
    exp.data.time = 1.31
    exp._gate()
    exp.data.time = 1.62
    exp._gate()
    assert exp.phase == "INSERT" and exp.prealign_passed
    assert exp.policy.set_target_world.call_count == 4
    exp.hands.command.assert_not_called()


@pytest.mark.parametrize("field,bad", [("position_m", .006), ("orientation_deg", 3.1), ("linear_speed_mps", .03)])
def test_insertion_settle_withholds_grasp_for_each_actual_precision_failure(field, bad):
    exp = synthetic_experiment("INSERT")
    exp.enter("INSERT_SETTLE")
    finish_reference(exp)
    exp.current_metrics["wrist_errors"]["right"][field] = bad
    exp.data.time = exp.fullbody_cfg["phase_timeout_s"]+.01
    exp._gate()
    assert exp.phase == "FAILED" and exp.failure_phase == "INSERT_SETTLE"
    assert "grasp command withheld" in exp.failure
    exp.hands.command.assert_not_called()


def test_insertion_settle_precision_then_stability_closes_once_not_every_gate():
    exp = synthetic_experiment("INSERT")
    exp.enter("INSERT_SETTLE")
    finish_reference(exp)
    exp.data.time = .49
    exp._gate()
    assert exp.stable_since is None
    exp.data.time = .51
    exp._gate()
    exp.data.time = .82
    exp._gate()
    assert exp.phase == "CLOSE" and exp.hands.command.call_count == 2
    for time in (.83, .84, .85):
        exp.data.time = time
        exp._gate()
    assert exp.phase == "CLOSE" and exp.hands.command.call_count == 2
    assert exp.policy.set_target_world.call_count == 4


@pytest.mark.parametrize("terminal", ["FAILED", "COMPLETE"])
def test_terminal_step_never_advances_physics_targets_or_releases(monkeypatch, terminal):
    exp = synthetic_experiment("CLOSE")
    exp.sync, exp.record = Mock(), Mock()
    exp.steps = 123
    before_q, before_ctrl = exp.data.qpos.copy(), exp.data.ctrl.copy()
    calls = exp.hands.command.call_count
    if terminal == "FAILED":
        exp.fail("synthetic test stop")
        assert exp.failure_phase == "CLOSE"
    else:
        exp.enter("COMPLETE")
    step = Mock()
    monkeypatch.setattr(reach.mujoco, "mj_step", step)
    exp.step()
    assert exp.steps == 123 and exp.data.time == 0.
    np.testing.assert_array_equal(exp.data.qpos, before_q)
    np.testing.assert_array_equal(exp.data.ctrl, before_ctrl)
    assert exp.hands.command.call_count == calls
    for method in (step, exp.sync, exp.record, exp.policy.act, exp.policy.apply, exp.hands.apply, exp.hands.update):
        method.assert_not_called()


def safety_experiment(scene):
    exp = synthetic_experiment()
    exp.model, hand_cfg, _ = scene
    exp.data = mujoco.MjData(exp.model)
    initialize_robot(exp.model, exp.data, hand_cfg)
    exp.cfg = load_reach_config()
    exp.hands = DualHandControl(exp.model, exp.data, hand_cfg)
    exp.body_map = JointMap.create(exp.model, BODY_JOINTS)
    exp.base, exp.floor = exp.model.body("base_link").id, exp.model.geom("floor").id
    exp.foot_geoms = set(foot_collision_ids(exp.model))
    exp.robot_geoms = {g for g in range(exp.model.ngeom) if _descendant(exp.model, int(exp.model.geom_bodyid[g]), exp.base)}
    exp.hand_geoms = {g for g in exp.robot_geoms if any(_descendant(exp.model, int(exp.model.geom_bodyid[g]),
        exp.model.body(f"{s}_hand_roll_link").id) for s in SIDES)}
    exp.table_geoms = {g for g in range(exp.model.ngeom) if _descendant(exp.model, int(exp.model.geom_bodyid[g]), exp.model.body("tabletop").id)}
    exp.crate_geoms = {g for g in range(exp.model.ngeom) if int(exp.model.geom_bodyid[g]) == exp.model.body("cargo_crate").id}
    exp.joints = [exp.model.joint(n).id for n in urdf_hand_joints()]
    exp.mimics = []
    for name, joint in urdf_hand_joints().items():
        mimic = joint.find("mimic")
        if mimic is not None:
            exp.mimics.append((exp.model.joint(name).qposadr[0], exp.model.joint(mimic.get("joint")).qposadr[0],
                float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
    exp.peaks = dict.fromkeys(("joint_violation_rad", "body_joint_violation_rad", "mimic_error_rad",
        "robot_self_penetration_m", "hand_crate_penetration_m", "hand_self_penetration_m", "base_tilt_deg",
        "crate_tilt_deg", "clearance_m", "slip_m", "wrist_tracking_error_m", "wrist_tracking_error_deg", "actuator_torque_Nm"), 0.)
    exp.current_metrics.update({"finite_state": True, "max_hand_crate_penetration_m": 0.,
        "max_hand_self_penetration_m": 0., "base_tilt_deg": 0., "wrist_tracking_error_m": 0.,
        "wrist_tracking_error_deg": 0., "floor_contacts": []})
    exp.scratch = SimpleNamespace(contact=[], xpos=exp.data.xpos.copy())
    return exp


@pytest.mark.parametrize("kind,reason", [
    ("hand_table", "robot/table collision"), ("arm_table", "robot/table collision"),
    ("hand_floor", "non-foot ground contact"), ("arm_crate", "non-hand robot/crate collision"),
    ("self", "robot self penetration"), ("foot_floor", None), ("hand_crate", None),
])
def test_whole_body_safety_distinguishes_feet_hands_and_forbidden_contacts(scene, kind, reason):
    exp = safety_experiment(scene)
    hand = min(exp.hand_geoms)
    arm = min(exp.robot_geoms-exp.hand_geoms-exp.foot_geoms)
    table, crate, foot = min(exp.table_geoms), min(exp.crate_geoms), min(exp.foot_geoms)
    pairs = {"hand_table": (hand, table), "arm_table": (arm, table), "hand_floor": (hand, exp.floor),
             "arm_crate": (arm, crate), "self": (arm, hand), "foot_floor": (foot, exp.floor), "hand_crate": (hand, crate)}
    exp.scratch.contact = [SimpleNamespace(geom=pairs[kind], dist=-.006 if kind == "self" else -.0001)]
    exp._safety()
    if reason:
        assert exp.phase == "FAILED" and reason in exp.failure
    else:
        assert exp.phase == "READY" and exp.failure is None


def test_hand_self_collision_retains_fixture_threshold_not_looser_body_threshold(scene):
    exp = safety_experiment(scene)
    assert exp.params.max_penetration_m < exp.cfg["gates"]["max_self_penetration_m"]
    exp.current_metrics["max_hand_self_penetration_m"] = exp.params.max_penetration_m+.0001
    exp._safety()
    assert exp.phase == "FAILED" and "hand self penetration" in exp.failure


def test_inherited_report_cannot_call_a_failed_attempt_successful_and_has_no_fixture_claims(scene, monkeypatch):
    exp = synthetic_experiment("PREALIGN")
    exp.model, exp.hand_cfg, _ = scene
    exp.cfg, exp.peaks = load_reach_config(), {}
    exp.warmup_duration_s, exp.warmup_peaks = 2., {}
    exp.parity_path = PROJECT_ROOT / "artifacts/diagnostic-only-synthetic-parity.json"
    exp.parity_evidence = {"onnx_sha256": "synthetic", "checkpoint_sha256": "synthetic"}
    exp.initial_crate_position = np.array([.43, 0., .881])
    exp.data.qfrc_applied = np.zeros(1)
    exp.data.xfrc_applied = np.zeros((1, 6))
    exp.data.warning = SimpleNamespace(number=np.zeros(8, dtype=int))
    monkeypatch.setattr(reach, "sha256", lambda _: "synthetic-test-not-real-evidence")
    exp.fail("Synthetic gross wrist error")
    report = exp.report()
    assert not report["lift_passed"] and not report["experiment_completed"]
    assert not report["checks"]["trial_lift_confirmed"]
    assert report["failure_phase"] == "PREALIGN"
    assert not report["insertion_attempted"] and not report["grasp_commanded"]
    assert report["mocap_wrist_count"] == report["wrist_support_welds"] == report["crate_welds"] == 0
    assert report["model_dimensions"]["nmocap"] == 0
    assert "real new free-base robot" in report["scope"]
    assert "no wrist fixtures" in report["wrist_drives"]
