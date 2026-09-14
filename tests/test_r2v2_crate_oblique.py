"""Geometry and evidence gates for the isolated oblique-hand probe.

Synthetic gate cases verify rejection logic, not a physical pickup. The small
initialization checks retain real MuJoCo collision bodies and hand inertias.
"""

from common.path_config import PROJECT_ROOT

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_hand_preview import _hand_geoms, _local_part_clouds, _world_vertices
from common.r2v2_crate_lift import CrateLiftExperiment, CrateLiftParameters
from common.r2v2_crate_lift_scene import build_crate_lift_model
from common.r2v2_crate_oblique import ObliqueCrateExperiment, ObliqueParameters
from r2v2_description.model import SIDES, build_model_xml, load_config, urdf_hand_joints


@pytest.fixture
def experiment():
    return ObliqueCrateExperiment(ObliqueParameters(tip_up_deg=-20), keep_trace=False)


def synthetic_probe(phase="HOLD"):
    exp = ObliqueCrateExperiment.__new__(ObliqueCrateExperiment)
    exp.candidate = ObliqueParameters(tip_up_deg=-20)
    exp.active_sides = exp.candidate.active_sides
    exp.params = replace(CrateLiftParameters(), max_tilt_deg=45., hold_s=2.)
    exp.data = SimpleNamespace(time=0., qfrc_applied=np.zeros(1), xfrc_applied=np.zeros((1, 6)))
    exp.phase, exp.phase_start, exp.failure = phase, 0., None
    exp.current_metrics = {
        "clearance_m": .09, "table_bearing_contact": False,
        "crate_weight_N": 3.924, "grasp_slip_m": .003,
        "active_angular_slip_deg": {"left": 1.},
        "crate_linear_speed_m_s": .001, "crate_angular_speed_rad_s": .01,
        "crate_tilt_deg": 2.,
        "hands": {
            "left": {"vertical_force_N": 3.924, "has_bearing_finger_contact": True,
                     "finger_normal_force_N": 2., "contacts": [{"part": "index"}],
                     "T_wrist_crate": np.eye(4).tolist()},
            "right": {"vertical_force_N": 0., "has_bearing_finger_contact": False,
                      "finger_normal_force_N": 0., "contacts": [],
                      "T_wrist_crate": np.eye(4).tolist()},
        },
    }
    commands = {side: 0 for side in SIDES}
    exp.hands = SimpleNamespace(command=lambda side, value: commands.__setitem__(side, value))
    exp.test_commands = commands
    exp.stable_since = exp.bad_since = exp.baseline_relations = None
    exp.pickup_hold_s = exp.max_pickup_hold_s = 0.
    exp.level_hold_s = exp.max_level_hold_s = 0.
    exp.pickup_verified = False
    exp.hold_samples, exp.transitions = [], []
    exp.samples, exp.peaks, exp.layout, exp.hand_cfg = [], {}, {}, {}
    exp.crate_params = replace(load_crate_config(), width=.26)
    return exp


def test_probe_box_is_26_cm_without_changing_holes_mass_or_bilateral_default(experiment):
    exp = experiment
    original = load_crate_config()
    assert original.width == pytest.approx(.36)
    assert exp.crate_params == replace(original, width=.26)
    assert exp.crate_params.depth == pytest.approx(.24)
    assert exp.crate_params.height == pytest.approx(.16)
    assert exp.crate_params.handle_opening_width == pytest.approx(.120)
    assert exp.crate_params.handle_opening_height == pytest.approx(.055)
    crate = exp.model.body("cargo_crate")
    assert crate.mass[0] == pytest.approx(.4)
    assert np.all(crate.inertia > 0)
    baseline, _, _ = build_crate_lift_model()
    assert baseline.body("cargo_crate").mass[0] == pytest.approx(crate.mass[0])
    assert not np.allclose(baseline.body("cargo_crate").inertia, crate.inertia)
    matching, _, _ = build_crate_lift_model(crate_params=exp.crate_params)
    np.testing.assert_array_equal(crate.inertia, matching.body("cargo_crate").inertia)
    assert load_crate_config() == original


def test_real_hand_inertias_limits_and_wrist_only_welds_are_preserved(experiment):
    exp = experiment
    source = mujoco.MjModel.from_xml_string(build_model_xml(load_config(), fixture=True))
    for index in range(source.nbody):
        name = source.body(index).name
        if not name.startswith(("left_", "right_")) or name.endswith("_fixture"):
            continue
        actual = exp.model.body(name).id
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(exp.model, field)[actual],
                                       getattr(source, field)[index], atol=1e-14)
    assert exp.model.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert exp.model.body("cargo_crate").mocapid[0] == -1
    welds = np.flatnonzero(exp.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    assert len(welds) == 2
    crate = exp.model.body("cargo_crate").id
    assert crate not in exp.model.eq_obj1id[welds]
    assert crate not in exp.model.eq_obj2id[welds]
    assert not np.any(exp.data.xfrc_applied)
    assert not np.any(exp.data.qfrc_applied)
    assert exp.hand_cfg["servo"] == load_config()["servo"]
    for name in urdf_hand_joints():
        np.testing.assert_array_equal(exp.model.joint(name).range, source.joint(name).range)
    for side in SIDES:
        assert exp.hand_cfg["hands"][side]["open"][0] == pytest.approx(np.deg2rad(75))
        assert exp.model.body(f"{side}_hand_roll_link").mocapid[0] == -1


def test_negative_tip_up_points_down_and_normal_insertion_is_sixty_mm(experiment):
    exp, side = experiment, "left"
    geometry = exp.geometry[side]
    direction = geometry["insertion_direction_world"]
    np.testing.assert_allclose(direction, [0., -np.cos(np.deg2rad(20)), -np.sin(np.deg2rad(20))], atol=1e-12)
    assert np.linalg.det(geometry["rotation"]) == pytest.approx(1.)
    assert np.linalg.norm(geometry["quaternion"]) == pytest.approx(1.)
    inward = np.array([0., -1., 0.])
    parts = _local_part_clouds(exp.model, exp.scratch, side)
    measured = []
    for name in ("index", "middle", "ring", "pinky"):
        vertices = parts[name] @ geometry["rotation"].T + geometry["inserted"]
        depth = float((vertices @ inward).max()+exp.crate_params.width/2-exp.crate_params.wall_thickness)
        measured.append(depth)
        assert depth >= .060-1e-12
        assert geometry["per_finger_normal_insertion_m"][name] == pytest.approx(depth)
    assert min(measured) == pytest.approx(.060)


def test_all_initial_collision_geometry_is_25_mm_outside_and_right_hand_is_parked(experiment):
    exp = experiment
    all_points = np.concatenate([_world_vertices(exp.model, exp.scratch, geom)
                                 for geom in _hand_geoms(exp.model, "left")])
    clearance = float(all_points[:, 1].min()-exp.crate_params.width/2)
    assert clearance == pytest.approx(.025, abs=1e-12)
    assert exp.geometry["left"]["initial_all_hand_clearance_m"] == pytest.approx(clearance)
    right = exp.scratch.xpos[exp.model.body("right_hand_roll_link").id]
    crate = exp.scratch.xpos[exp.model.body("cargo_crate").id]
    assert np.linalg.norm(right-crate) > 1.
    assert exp.current_metrics["hands"]["right"]["contacts"] == []
    assert exp.current_metrics["hands"]["right"]["vertical_force_N"] == 0.
    assert exp.current_metrics["hands"]["left"]["contacts"] == []
    assert exp.current_metrics["hand_table_contacts"] == []
    assert exp.current_metrics["hand_floor_contacts"] == []
    assert exp.data.time == 0.


def test_insertion_moves_only_left_target_along_rotated_local_x_without_teleport(experiment):
    exp = experiment
    delta = exp.inserted_wrists["left"]-exp.initial_wrists["left"]
    direction = exp.geometry["left"]["rotation"][:, 0]
    np.testing.assert_allclose(delta/np.linalg.norm(delta), direction, atol=1e-12)
    original_qpos = exp.data.qpos.copy()
    right_target = exp.layout["mocap_ids"]["right"]
    for elapsed, fraction in ((0., 0.), (exp.params.insert_s/2, .5), (exp.params.insert_s, 1.)):
        exp.phase, exp.phase_start, exp.data.time = "INSERT", 0., elapsed
        exp._motion()
        expected = exp.initial_wrists["left"]+fraction*delta
        np.testing.assert_allclose(exp.data.mocap_pos[exp.layout["mocap_ids"]["left"]], expected, atol=1e-12)
        np.testing.assert_array_equal(exp.data.mocap_pos[right_target], exp.initial_wrists["right"])
        np.testing.assert_array_equal(exp.data.qpos, original_qpos)


@pytest.mark.parametrize("path,value", [
    (("clearance_m",), .0079),
    (("table_bearing_contact",), True),
    (("hands", "left", "has_bearing_finger_contact"), False),
    (("hands", "left", "vertical_force_N"), 3.13),
    (("hands", "left", "vertical_force_N"), -3.924),
    (("hands", "right", "contacts"), [{"part": "palm", "normal_force_N": 0.}]),
    (("grasp_slip_m",), None),
    (("grasp_slip_m",), .015),
    (("active_angular_slip_deg", "left"), 5.),
    (("crate_linear_speed_m_s",), .020),
    (("crate_angular_speed_rad_s",), .150),
])
def test_pickup_rejects_missing_physical_evidence_even_if_controller_is_closed(path, value):
    exp = synthetic_probe()
    exp.hands.state = "CLOSED"
    assert exp._supported_pickup()
    target = exp.current_metrics
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not exp._supported_pickup()


def test_pickup_requires_an_actual_angular_slip_measurement():
    exp = synthetic_probe()
    del exp.current_metrics["active_angular_slip_deg"]
    assert not exp._supported_pickup()


@pytest.mark.parametrize("measurements", [
    None, {}, {"right": 0.}, {"left": float("nan")}, {"left": float("inf")}, {"left": -.01},
])
def test_pickup_rejects_invalid_active_angular_slip_measurements(measurements):
    exp = synthetic_probe()
    exp.current_metrics["active_angular_slip_deg"] = measurements
    assert not exp._supported_pickup()


def test_relative_slip_uses_measured_wrist_and_crate_transforms(experiment):
    exp = experiment
    exp.baseline_relations = {side: np.asarray(exp.current_metrics["hands"][side]["T_wrist_crate"]).copy()
                              for side in SIDES}
    address = exp.model.joint("crate_free").qposadr[0]
    exp.data.qpos[address] += .006
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.array([0., 0., 1.]), np.deg2rad(10.))
    exp.data.qpos[address+3:address+7] = q
    measured = exp.sync()
    assert measured["grasp_slip_m"] == pytest.approx(.006, abs=1e-12)
    assert measured["active_angular_slip_deg"]["left"] == pytest.approx(10., abs=1e-10)


def test_closed_controller_without_finger_contact_times_out_and_does_not_lift():
    exp = synthetic_probe("CLOSE")
    exp.hands.state = "CLOSED"
    exp.current_metrics["hands"]["left"]["finger_normal_force_N"] = 0.
    exp.data.time = exp.params.close_timeout_s
    exp._gate()
    assert exp.phase == "FAILED"
    assert "contact" in exp.failure.lower()
    assert exp.baseline_relations is None


def test_closure_commands_only_the_active_hand():
    exp = synthetic_probe("READY")
    exp.enter("CLOSE")
    assert exp.test_commands == {"left": 1, "right": 0}


def test_complete_phase_is_not_success_without_verified_hold():
    exp = synthetic_probe("COMPLETE")
    exp.hands.state = "CLOSED"
    report = exp.report()
    assert report["experiment_completed"]
    assert not report["pickup_verified"]
    assert not report["lift_passed"]


def test_continuous_level_hold_reports_success_with_physical_evidence():
    exp = synthetic_probe("HOLD")
    for step in range(1, 201):
        exp.data.time = step*.01
        exp._gate()
    report = exp.report()
    assert report["experiment_completed"]
    assert report["pickup_verified"]
    assert report["lift_passed"]


def test_bilateral_probe_requires_angular_slip_for_both_active_hands():
    exp = synthetic_probe()
    exp.active_sides = ("left", "right")
    exp.current_metrics["hands"]["left"]["vertical_force_N"] = 1.962
    exp.current_metrics["hands"]["right"].update(
        vertical_force_N=1.962, has_bearing_finger_contact=True, contacts=[{"part": "index"}])
    assert not exp._supported_pickup()
    exp.current_metrics["active_angular_slip_deg"]["right"] = 1.
    assert exp._supported_pickup()
    exp.current_metrics["active_angular_slip_deg"]["right"] = 5.
    assert not exp._supported_pickup()


def test_supported_but_tilted_pickup_is_not_a_level_lift():
    exp = synthetic_probe("HOLD")
    exp.current_metrics["crate_tilt_deg"] = 20.
    for step in range(1, 201):
        exp.data.time = step*.01
        exp._gate()
    report = exp.report()
    assert report["experiment_completed"]
    assert report["pickup_verified"]
    assert not report["lift_passed"]
    assert exp.max_level_hold_s == 0.


def test_hold_requires_contiguous_success_and_failure_does_not_count_as_a_pass():
    exp = synthetic_probe("HOLD")
    for step in range(1, 201):
        exp.data.time = step*.01
        exp.current_metrics["table_bearing_contact"] = step == 90
        exp._gate()
    assert exp.report()["pickup_verified"]
    assert not exp.report()["lift_passed"]
    assert exp.max_level_hold_s < 1.2
    exp.failure = "diagnostic safety stop"
    exp.phase = "FAILED"
    assert not exp.report()["pickup_verified"]
    assert not exp.report()["lift_passed"]


@pytest.mark.parametrize("loss_at_step", [150, 200])
def test_late_loss_of_support_cannot_reuse_earlier_success_latches(loss_at_step):
    exp = synthetic_probe("HOLD")
    for step in range(1, 201):
        exp.data.time = step*.01
        exp.current_metrics["table_bearing_contact"] = step >= loss_at_step
        exp._gate()
    assert exp.phase == "COMPLETE"
    assert exp.pickup_verified  # Historical latch is diagnostic only.
    assert exp.max_pickup_hold_s >= 1.
    assert exp.pickup_hold_s == 0.
    assert not exp.report()["pickup_verified"]
    assert not exp.report()["lift_passed"]


def test_inactive_hand_contact_is_a_safety_stop(experiment, monkeypatch):
    exp = experiment
    monkeypatch.setattr(CrateLiftExperiment, "_safety", lambda self: None)
    exp.current_metrics["hands"]["right"]["contacts"] = [{"part": "index"}]
    exp._safety()
    assert exp.phase == "FAILED"
    assert "Parked hand" in exp.failure


def test_auxiliary_crate_force_is_rejected_by_inherited_safety(experiment):
    exp = experiment
    exp.data.xfrc_applied[exp.model.body("cargo_crate").id, 2] = 1.
    exp._safety()
    assert exp.phase == "FAILED"
    assert "applied force" in exp.failure


def test_probe_tilt_stop_does_not_relax_bilateral_baseline(experiment):
    exp = experiment
    assert exp.params.max_tilt_deg == 45.
    assert exp.params.hold_tilt_deg == 8.
    assert CrateLiftParameters().max_tilt_deg == 15.
    assert exp.params.max_penetration_m == CrateLiftParameters().max_penetration_m
    assert exp.params.max_joint_violation_rad == CrateLiftParameters().max_joint_violation_rad


@pytest.mark.parametrize("kwargs", [
    {"tip_up_deg": 46}, {"yaw_deg": float("nan")}, {"roll_deg": True},
    {"insertion_m": .01}, {"insertion_m": .081}, {"closed_curl_rad": 1.41},
    {"seating_m": .031}, {"active_sides": ("right",)},
])
def test_invalid_candidates_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ObliqueParameters(**kwargs)
