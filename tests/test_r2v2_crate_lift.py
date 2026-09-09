"""Core/scene contracts; synthetic gate evidence is not a physical lift demo.

In particular, the no-contact negative control only checks the CLOSE gate.
It does not establish that forced wrist lifting with open fingers cannot
support the handle mechanically.
"""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import common.r2v2_crate_lift as lift
from common.r2v2_crate import load_crate_config
from common.r2v2_crate_lift import CrateLiftExperiment, CrateLiftParameters, _blend, load_lift_config
from common.r2v2_crate_lift_scene import build_crate_lift_model
from r2v2_description.model import SIDES, build_model_xml, hand_names, load_config, urdf_hand_joints


@pytest.fixture
def experiment():
    return CrateLiftExperiment(CrateLiftParameters())


def gate_experiment(phase="CLOSE", grasp_enabled=True):
    """Exercise the production gate with explicit diagnostic-only evidence."""
    exp = CrateLiftExperiment.__new__(CrateLiftExperiment)
    exp.params = CrateLiftParameters(grasp_enabled=grasp_enabled)
    exp.phase, exp.phase_start, exp.failure = "READY", 0., None
    exp.data = SimpleNamespace(time=0., qfrc_applied=np.zeros(1), xfrc_applied=np.zeros((1, 6)),
                               warning=SimpleNamespace(number=np.zeros(8, dtype=int)))
    commands = {side: 0 for side in SIDES}
    exp.hands = SimpleNamespace(command=lambda side, value: commands.__setitem__(side, value))
    exp.test_commands = commands
    exp.current_metrics = {"clearance_m": .10, "table_vertical_force_N": 0.,
                           "grasp_slip_m": 0., "crate_tilt_deg": 0., "crate_position_m": [0, 0, .5],
                           "hands": {side: {"finger_normal_force_N": 1., "finger_handle_vertical_force_N": 1.,
                                             "vertical_force_N": 1.,
                                             "T_wrist_crate": np.eye(4).tolist()} for side in SIDES}}
    exp.stable_since = exp.bad_since = None
    exp.baseline_relations = None
    exp.trial_confirmed = False
    exp.hold_samples, exp.transitions = [], []
    exp.samples, exp.peaks, exp.layout, exp.hand_cfg = [], {}, {}, {}
    exp.crate_params = load_crate_config()
    exp.initial_crate_position = np.array([0., 0., .401])
    exp.enter(phase)
    return exp


def gate_at(exp, seconds):
    exp.data.time = seconds
    exp._gate()
    return exp.phase


def test_lift_yaml_matches_requested_deeper_and_tighter_defaults():
    params = CrateLiftParameters()
    assert load_lift_config() == params
    assert params.insertion_m == pytest.approx(.060)
    assert params.closed_curl_rad == pytest.approx(.80)
    assert params.closure_seating_m == pytest.approx(.0175)


def test_sixty_mm_insertion_moves_both_targets_thirty_mm_inward_without_more_torque(experiment):
    exp = experiment
    shallow_model, shallow_cfg, shallow = build_crate_lift_model(
        exp.crate_params, insertion_m=.030,
        start_palm_clearance_m=exp.params.start_palm_clearance_m,
        table_height=exp.params.table_height,
    )
    source_cfg = load_config()
    torque_limits = [.15, .30, .35, .35, .35, .35]
    assert exp.hand_cfg["servo"]["torque_limit"] == torque_limits
    assert exp.hand_cfg["servo"] == shallow_cfg["servo"] == source_cfg["servo"]
    for side, sign in (("left", 1), ("right", -1)):
        np.testing.assert_allclose(exp.initial_wrists[side], shallow["initial_wrist_positions"][side], atol=1e-12)
        np.testing.assert_allclose(exp.inserted_wrists[side]-shallow["inserted_wrist_positions"][side],
                                   [0., -sign*.030, 0.], atol=1e-12)
        np.testing.assert_array_equal(exp.layout["wrist_quaternions"][side], shallow["wrist_quaternions"][side])
        assert exp.hand_cfg["hands"][side]["closed"][2:] == [.80]*4
    for field in ("actuator_ctrlrange", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(exp.model, field), getattr(shallow_model, field))


def test_hook_profile_does_not_modify_original_or_builder_hand_configuration(monkeypatch):
    original = copy.deepcopy(load_config())
    builder = lift.build_crate_lift_model
    captured = {}
    def remember_config(*args, **kwargs):
        model, cfg, layout = builder(*args, **kwargs)
        captured["cfg"], captured["before"] = cfg, copy.deepcopy(cfg)
        return model, cfg, layout
    monkeypatch.setattr(lift, "build_crate_lift_model", remember_config)
    exp = CrateLiftExperiment(CrateLiftParameters(closed_curl_rad=.45))
    assert load_config() == original
    assert captured["cfg"] == captured["before"]
    assert exp.hand_cfg is not captured["cfg"]
    for side in SIDES:
        opened = original["hands"][side]["open"]
        assert exp.hand_cfg["hands"][side]["open"] == opened
        assert exp.hand_cfg["hands"][side]["closed"] == opened[:2]+[.45]*4
        assert captured["cfg"]["hands"][side]["closed"] == original["hands"][side]["closed"]
        measured = [exp.data.qpos[exp.model.joint(name).qposadr[0]] for name in hand_names(side)]
        np.testing.assert_allclose(measured, opened, atol=1e-12)
        assert measured[0] == pytest.approx(np.deg2rad(75))


def test_scene_preserves_real_hands_with_two_dynamic_wrists_and_only_wrist_support_welds(experiment):
    exp, model = experiment, experiment.model
    source = mujoco.MjModel.from_xml_string(build_model_xml(load_config(), fixture=True))
    assert (model.nq, model.nv, model.nu, model.nmocap, model.neq) == (43, 40, 12, 2, 12)
    assert np.count_nonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT) == 10
    weld_ids = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    assert len(weld_ids) == 2
    assert model.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("cargo_crate").mocapid[0] == -1
    assert not np.any(exp.data.qfrc_applied)
    assert not np.any(exp.data.xfrc_applied)
    for side in SIDES:
        fixture = model.body(f"{side}_wrist_fixture")
        wrist = model.body(f"{side}_hand_roll_link")
        assert fixture.mocapid[0] == exp.layout["mocap_ids"][side]
        assert fixture.mass[0] == 0
        assert wrist.parentid[0] == 0  # Never a fixed child of the mocap target.
        assert wrist.mocapid[0] == -1
        assert wrist.jntnum[0] == 1
        assert model.joint(f"{side}_wrist_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
        weld = model.equality(f"{side}_wrist_tracking_weld").id
        assert weld in weld_ids
        assert {int(model.eq_obj1id[weld]), int(model.eq_obj2id[weld])} == {wrist.id, fixture.id}
    crate = model.body("cargo_crate").id
    assert crate not in model.eq_obj1id[weld_ids]
    assert crate not in model.eq_obj2id[weld_ids]
    equality_rows = exp.scratch.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    tracking_rows = equality_rows & np.isin(exp.scratch.efc_id, weld_ids)
    assert np.count_nonzero(tracking_rows) == 12
    np.testing.assert_allclose(exp.scratch.efc_pos[tracking_rows], 0, atol=1e-12)
    for index in range(source.nbody):
        name = source.body(index).name
        if not name.startswith(("left_", "right_")) or name.endswith("_fixture"):
            continue
        actual = model.body(name).id
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(model, field)[actual], getattr(source, field)[index], atol=1e-14)
    for name in urdf_hand_joints():
        old, new = source.joint(name).id, model.joint(name).id
        for field in ("jnt_type", "jnt_axis", "jnt_range", "jnt_limited"):
            np.testing.assert_array_equal(getattr(model, field)[new], getattr(source, field)[old])
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(model, field)[model.jnt_dofadr[new]],
                                          getattr(source, field)[source.jnt_dofadr[old]])
    for index in range(source.ngeom):
        name = source.geom(index).name
        if not name.startswith(("left_", "right_")) or name.endswith("_fixture_visual"):
            continue
        actual = model.geom(name).id
        for field in ("geom_type", "geom_size", "geom_contype", "geom_conaffinity", "geom_condim",
                      "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])
    for index in range(source.nu):
        actual = model.actuator(source.actuator(index).name).id
        for field in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_array_equal(getattr(model, field)[actual], getattr(source, field)[index])


def test_motion_has_mirrored_insertion_constant_angles_and_synchronous_vertical_offsets(experiment):
    exp, p = experiment, experiment.params
    seating = p.closure_seating_m
    cases = (("READY", 0., 0., 0.), ("INSERT", p.insert_s/2, .5, 0.),
             ("INSERT_SETTLE", 0., 1., 0.), ("CLOSE", 0., 1., 0.),
             ("CLOSE", p.close_minimum_s/2, 1., seating/2),
             ("CLOSE", p.close_minimum_s, 1., seating),
             ("CLOSE", p.close_timeout_s, 1., seating),
             ("TRIAL_LIFT", p.trial_lift_s/2, 1., seating+p.trial_height_m/2),
             ("TRIAL_HOLD", 0., 1., seating+p.trial_height_m),
             ("LIFT", p.lift_s/2, 1., seating+(p.trial_height_m+p.lift_height_m)/2),
             ("HOLD", 0., 1., seating+p.lift_height_m),
             ("COMPLETE", 0., 1., seating+p.lift_height_m))
    original_qpos = exp.data.qpos.copy()
    for phase, elapsed, alpha, height in cases:
        exp.phase, exp.phase_start, exp.data.time = phase, 0., elapsed
        exp._motion()
        for side in SIDES:
            mid = exp.layout["mocap_ids"][side]
            expected = (1-alpha)*exp.initial_wrists[side]+alpha*exp.inserted_wrists[side]+[0, 0, height]
            np.testing.assert_allclose(exp.data.mocap_pos[mid], expected, atol=1e-12)
            np.testing.assert_array_equal(exp.data.mocap_quat[mid], exp.layout["wrist_quaternions"][side])
            assert exp.data.mocap_pos[mid, 2]-exp.inserted_wrists[side][2] == pytest.approx(height)
        np.testing.assert_array_equal(exp.data.qpos, original_qpos)


def test_zero_seating_reproduces_the_original_constant_height_closure_path(experiment):
    exp = experiment
    exp.params = replace(exp.params, closure_seating_m=0.)
    p = exp.params
    cases = (("CLOSE", 0., 0.), ("CLOSE", p.close_minimum_s/2, 0.),
             ("CLOSE", p.close_timeout_s, 0.), ("TRIAL_LIFT", 0., 0.),
             ("TRIAL_LIFT", p.trial_lift_s/2, p.trial_height_m/2),
             ("TRIAL_HOLD", 0., p.trial_height_m),
             ("LIFT", p.lift_s/2, (p.trial_height_m+p.lift_height_m)/2),
             ("HOLD", 0., p.lift_height_m), ("COMPLETE", 0., p.lift_height_m))
    original_qpos = exp.data.qpos.copy()
    for phase, elapsed, height in cases:
        exp.phase, exp.phase_start, exp.data.time = phase, 0., elapsed
        exp._motion()
        for side in SIDES:
            np.testing.assert_allclose(exp.data.mocap_pos[exp.layout["mocap_ids"][side]],
                                       exp.inserted_wrists[side]+[0., 0., height], atol=1e-12)
        np.testing.assert_array_equal(exp.data.qpos, original_qpos)


def test_no_grasp_closure_keeps_open_hand_at_original_insertion_height(experiment):
    exp = experiment
    exp.params = replace(exp.params, grasp_enabled=False)
    assert exp.params.closure_seating_m > 0
    exp.enter("CLOSE")
    original_qpos = exp.data.qpos.copy()
    for elapsed in (0., exp.params.close_minimum_s/2, exp.params.close_minimum_s, exp.params.close_timeout_s):
        exp.data.time = exp.phase_start+elapsed
        exp._motion()
        for side in SIDES:
            assert exp.hands.controllers[side].command == 0
            np.testing.assert_allclose(exp.data.mocap_pos[exp.layout["mocap_ids"][side]],
                                       exp.inserted_wrists[side], atol=1e-12)
        np.testing.assert_array_equal(exp.data.qpos, original_qpos)


def test_motion_phase_boundaries_are_position_continuous(experiment):
    exp, p = experiment, experiment.params
    pairs = (("READY", p.ready_s, "INSERT"), ("INSERT", p.insert_s, "INSERT_SETTLE"),
             ("INSERT_SETTLE", p.insert_settle_s, "CLOSE"), ("CLOSE", p.close_minimum_s, "TRIAL_LIFT"),
             ("TRIAL_LIFT", p.trial_lift_s, "TRIAL_HOLD"), ("TRIAL_HOLD", p.contact_stable_s, "LIFT"),
             ("LIFT", p.lift_s, "HOLD"), ("HOLD", p.hold_s, "COMPLETE"))
    for before, elapsed, after in pairs:
        exp.phase, exp.phase_start, exp.data.time = before, 0., elapsed
        exp._motion()
        end = exp.data.mocap_pos.copy()
        exp.phase, exp.phase_start, exp.data.time = after, 0., 0.
        exp._motion()
        np.testing.assert_allclose(exp.data.mocap_pos, end, atol=1e-12)
    assert _blend(-1) == 0
    assert _blend(2) == 1
    values = np.array([_blend(t) for t in np.linspace(0, 1, 101)])
    assert np.all(np.diff(values) >= 0)
    assert _blend(1e-4) < 1e-9
    assert 1-_blend(1-1e-4) < 1e-9


def test_moving_target_does_not_teleport_dynamic_wrist_and_record_distinguishes_both(experiment):
    exp = experiment
    side, offset = "left", np.array([0., 0., .002])
    target = exp.layout["mocap_ids"][side]
    original = np.array(exp.current_metrics["hands"][side]["T_world_wrist"])
    original_qpos = exp.data.qpos.copy()
    exp.data.mocap_pos[target] += offset
    metrics = exp.sync()
    actual = np.array(metrics["hands"][side]["T_world_wrist"])
    np.testing.assert_allclose(actual, original, atol=1e-12)
    np.testing.assert_array_equal(exp.data.qpos, original_qpos)
    assert metrics["wrist_tracking_error_m"] == pytest.approx(.002)
    exp.record()
    row = exp.samples[-1]["hands"][side]
    np.testing.assert_allclose(row["wrist_position_m"], original[:3, 3], atol=1e-12)
    np.testing.assert_allclose(row["wrist_target_position_m"], original[:3, 3]+offset, atol=1e-12)


@pytest.mark.parametrize("side", SIDES)
def test_dynamic_wrist_velocity_reaches_finger_contact_jacobian_and_uses_wrist_not_com(experiment, side):
    exp, model, data = experiment, experiment.model, experiment.data
    dof = model.joint(f"{side}_wrist_free").dofadr[0]
    linear, angular_local = np.array([.01, -.02, .04]), np.array([.3, -.2, .1])
    data.qvel[dof:dof+3], data.qvel[dof+3:dof+6] = linear, angular_local
    metrics = exp.sync()
    scratch = exp.scratch
    wrist = model.body(f"{side}_hand_roll_link").id
    angular_world = scratch.xmat[wrist].reshape(3, 3) @ angular_local
    np.testing.assert_allclose(metrics["wrist_tracking"][side]["linear_velocity_world_m_s"], linear, atol=1e-12)
    np.testing.assert_allclose(metrics["wrist_tracking"][side]["angular_velocity_world_rad_s"], angular_world, atol=1e-12)
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        body = model.body(f"{side}_{finger}_distal_link").id
        geom = model.geom(f"{side}_{finger}_distal_link_col_1").id
        point = scratch.geom_xpos[geom]
        jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jac(model, scratch, jacp, jacr, point, body)
        expected = linear+np.cross(angular_world, point-scratch.xpos[wrist])
        np.testing.assert_allclose(jacp @ scratch.qvel, expected, atol=1e-12)
        assert np.linalg.norm(jacp @ scratch.qvel) > .001


@pytest.mark.parametrize("fault", ["position", "orientation", "hand_floor"])
def test_safety_rejects_wrist_tracking_error_and_hand_floor_collision(experiment, fault):
    exp = experiment
    if fault == "position":
        exp.current_metrics["wrist_tracking_error_m"] = exp.params.max_wrist_tracking_m+.00001
    elif fault == "orientation":
        exp.current_metrics["wrist_tracking_error_deg"] = exp.params.max_wrist_tracking_deg+.001
    else:
        exp.current_metrics["hand_floor_contacts"] = [{"geoms": ["left_index_distal_link_col_1", "floor"]}]
    exp._safety()
    assert exp.phase == "FAILED"
    assert ("wrist " + fault + " tracking" if fault != "hand_floor" else "hand/floor collision") in exp.failure


def test_step_never_resets_or_kinematically_drives_the_free_crate(experiment, monkeypatch):
    exp = experiment
    address = exp.model.joint("crate_free").qposadr[0]
    exp.data.qpos[address:address+3] = [.03, -.02, .6]
    before = exp.data.qpos[address:address+7].copy()
    # Suppress physics only in this unit test: any pose change would now be
    # an illicit direct assignment by the controller rather than dynamics.
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: None)
    exp.step()
    np.testing.assert_array_equal(exp.data.qpos[address:address+7], before)
    assert not np.any(exp.data.qfrc_applied)
    assert not np.any(exp.data.xfrc_applied)
    assert exp.steps == 1


@pytest.mark.parametrize("grasp_enabled", [True, False])
def test_close_requires_actual_bilateral_finger_contact_not_a_binary_command(grasp_enabled):
    exp = gate_experiment(grasp_enabled=grasp_enabled)
    assert set(exp.test_commands.values()) == {int(grasp_enabled)}
    for side in SIDES:
        exp.current_metrics["hands"][side]["finger_normal_force_N"] = 0.
    assert gate_at(exp, exp.params.close_minimum_s) == "CLOSE"
    assert gate_at(exp, exp.params.close_timeout_s) == "FAILED"
    assert not exp.trial_confirmed
    assert not exp.report()["lift_passed"]


def test_close_contact_must_be_bilateral_and_continuous():
    exp, start = gate_experiment(), CrateLiftParameters().close_minimum_s
    assert gate_at(exp, start) == "CLOSE"
    exp.current_metrics["hands"]["right"]["finger_normal_force_N"] = 0
    assert gate_at(exp, start+.2) == "CLOSE"
    exp.current_metrics["hands"]["right"]["finger_normal_force_N"] = 1
    assert gate_at(exp, start+.21) == "CLOSE"
    assert gate_at(exp, start+.21+exp.params.contact_stable_s-1e-5) == "CLOSE"
    assert gate_at(exp, start+.21+exp.params.contact_stable_s) == "TRIAL_LIFT"
    assert exp.baseline_relations is not None
    exp.current_metrics["hands"]["left"]["T_wrist_crate"][0][3] = 1
    assert exp.baseline_relations["left"][0, 3] == 0  # Captured baseline is private.


@pytest.mark.parametrize("violation", ["clearance", "table", "left_load", "right_load", "slip", "tilt"])
def test_trial_hold_rejects_each_required_physical_condition(violation):
    exp = gate_experiment("TRIAL_HOLD")
    if violation == "clearance": exp.current_metrics["clearance_m"] = exp.params.trial_clearance_m-1e-6
    if violation == "table": exp.current_metrics["table_vertical_force_N"] = 1.
    if violation.endswith("_load"):
        side = violation.split("_")[0]
        exp.current_metrics["hands"][side]["finger_normal_force_N"] = 100.
        exp.current_metrics["hands"][side]["finger_handle_vertical_force_N"] = 0.
    if violation == "slip": exp.current_metrics["grasp_slip_m"] = exp.params.max_slip_m
    if violation == "tilt": exp.current_metrics["crate_tilt_deg"] = exp.params.hold_tilt_deg+1e-6
    assert gate_at(exp, 0.) == "TRIAL_HOLD"
    assert gate_at(exp, exp.params.trial_timeout_s) == "FAILED"
    assert not exp.trial_confirmed


def test_trial_needs_sustained_good_measurements_before_full_lift():
    exp = gate_experiment("TRIAL_HOLD")
    assert gate_at(exp, 0.) == "TRIAL_HOLD"
    assert gate_at(exp, exp.params.contact_stable_s-1e-5) == "TRIAL_HOLD"
    assert gate_at(exp, exp.params.contact_stable_s) == "LIFT"
    assert exp.trial_confirmed


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("phase", ["TRIAL_HOLD", "LIFT", "HOLD"])
@pytest.mark.parametrize("net_vertical_force_N", [-1., 0., .1999])
def test_internal_clamp_force_cannot_replace_each_whole_hands_net_bearing_load(
        side, phase, net_vertical_force_N):
    exp = gate_experiment(phase)
    # An upward finger/beam force opposed by a stronger downward palm force
    # is an internal clamp, not evidence that this hand bears the crate.
    hand = exp.current_metrics["hands"][side]
    hand["finger_normal_force_N"] = 100.
    hand["finger_handle_vertical_force_N"] = 5.
    hand["vertical_force_N"] = net_vertical_force_N
    assert not exp._bilateral_load()
    gate_at(exp, 0.)
    timeout = exp.params.trial_timeout_s if phase == "TRIAL_HOLD" else exp.params.violation_s+.000001
    gate_at(exp, timeout)
    assert exp.phase == "FAILED"
    assert not exp.report()["lift_passed"]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("field", ["finger_handle_vertical_force_N", "vertical_force_N"])
def test_each_finger_and_whole_hand_bearing_threshold_is_required(side, field):
    exp = gate_experiment()
    hand = exp.current_metrics["hands"][side]
    hand[field] = exp.params.min_side_vertical_force_N
    assert exp._bilateral_load()
    hand[field] -= 1e-8
    assert not exp._bilateral_load()


def test_lift_violation_timer_resets_after_contact_recovers():
    exp = gate_experiment("LIFT")
    hand = exp.current_metrics["hands"]["left"]
    hand["finger_handle_vertical_force_N"] = 0
    assert gate_at(exp, 0.) == "LIFT"
    assert gate_at(exp, .1) == "LIFT"
    hand["finger_handle_vertical_force_N"] = 1
    assert gate_at(exp, .19) == "LIFT"
    assert exp.bad_since is None
    hand["finger_handle_vertical_force_N"] = 0
    assert gate_at(exp, .2) == "LIFT"
    assert gate_at(exp, .2+exp.params.violation_s+.000001) == "FAILED"


def test_hold_requires_complete_duration_and_report_requires_confirmed_trial():
    exp = gate_experiment("HOLD")
    assert gate_at(exp, 0.) == "HOLD"
    assert gate_at(exp, exp.params.hold_s-1e-5) == "HOLD"
    assert gate_at(exp, exp.params.hold_s) == "COMPLETE"
    assert not exp.report()["lift_passed"]  # A manual state jump cannot certify a trial.
    exp.trial_confirmed = True
    assert exp.report()["lift_passed"]  # Gate-unit evidence only, not a dynamic trial.
    assert exp.report()["checks"]["hold_duration"]


def test_invalid_hold_sample_prevents_completion_even_near_end():
    exp = gate_experiment("HOLD")
    exp.trial_confirmed = True
    assert gate_at(exp, 0.) == "HOLD"
    exp.current_metrics["grasp_slip_m"] = exp.params.max_slip_m
    assert gate_at(exp, exp.params.hold_s-1e-5) == "FAILED"
    assert not exp.report()["lift_passed"]
    assert not exp.report()["checks"]["full_hold_valid"]


def test_report_rejects_incomplete_observed_hold_duration():
    exp = gate_experiment("HOLD")
    exp.trial_confirmed = True
    assert gate_at(exp, .5) == "HOLD"  # First observed valid sample is late.
    assert gate_at(exp, exp.params.hold_s) == "COMPLETE"
    report = exp.report()
    assert not report["checks"]["hold_duration"]
    assert not report["lift_passed"]


@pytest.mark.parametrize("terminal", ["FAILED", "COMPLETE"])
def test_terminal_step_does_not_move_wrists_crate_or_open_fingers(experiment, terminal, monkeypatch):
    exp = experiment
    exp.enter("CLOSE")
    exp.enter("TRIAL_LIFT")
    exp.data.time = exp.params.trial_lift_s/2
    exp._motion()
    arrays = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")
    before = {name: getattr(exp.data, name).copy() for name in arrays}
    reference = {side: exp.hands.controllers[side].reference.position.copy() for side in SIDES}
    if terminal == "FAILED":
        exp.fail("diagnostic injected fault")
    else:
        exp.enter("COMPLETE")
    time_before, steps_before = exp.data.time, exp.steps
    samples_before, transitions_before = len(exp.samples), len(exp.transitions)
    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal experiments must not step or reset")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_resetData", forbidden)
    for _ in range(3):
        exp.step()
    assert exp.data.time == time_before
    assert exp.steps == steps_before
    assert len(exp.samples) == samples_before
    assert len(exp.transitions) == transitions_before
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(exp.data, name), value)
    for side in SIDES:
        assert exp.hands.controllers[side].command == 1
        np.testing.assert_array_equal(exp.hands.controllers[side].reference.position, reference[side])


def test_sync_uses_private_scratch_and_computes_slip_from_current_wrist_relation(experiment):
    exp = experiment
    exp.enter("TRIAL_LIFT")
    index = exp.model.joint("crate_free").qposadr[0]
    exp.data.qpos[index] += .007
    live_xpos, live_xmat = exp.data.xpos.copy(), exp.data.xmat.copy()
    result = exp.sync()
    assert result["grasp_slip_m"] == pytest.approx(.007)
    assert result["side_slip_m"] == pytest.approx({"left": .007, "right": .007})
    np.testing.assert_array_equal(exp.data.xpos, live_xpos)
    np.testing.assert_array_equal(exp.data.xmat, live_xmat)


@pytest.mark.parametrize("kwargs", [
    {"grasp_enabled": 1}, {"closed_curl_rad": 2.}, {"trial_clearance_m": .03},
    {"hold_clearance_m": .11}, {"close_minimum_s": 7.}, {"hold_tilt_deg": 20.},
    {"closure_seating_m": -.0001}, {"closure_seating_m": .030001},
    {"closure_seating_m": np.nan}, {"closure_seating_m": np.inf}, {"closure_seating_m": True},
])
def test_invalid_lift_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CrateLiftParameters(**kwargs)


@pytest.mark.parametrize("seating_m", [0., .03])
def test_closure_seating_accepts_both_finite_boundary_values(seating_m):
    assert CrateLiftParameters(closure_seating_m=seating_m).closure_seating_m == seating_m
