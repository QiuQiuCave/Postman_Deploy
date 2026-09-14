"""Top-entry scene contracts, not physical grasp-success acceptance tests."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from common.r2v2_cylinder_test import load_cylinder_profile
from common.r2v2_top_grasp_scene import TopGraspCandidate, build_top_grasp_model, top_wrist_rotation
from r2v2_description.model import SIDES, build_model_xml, initialize_hands, load_config


def test_real_dynamic_hands_and_only_wrist_tracking_welds():
    model, cfg, layout = build_top_grasp_model(candidate=TopGraspCandidate(depth_m=0))
    source = mujoco.MjModel.from_xml_string(build_model_xml(load_config(), fixture=True))
    assert (model.nq, model.nv, model.nu, model.nmocap, model.neq) == (43, 40, 12, 2, 12)
    assert np.count_nonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT) == 10
    welds = np.flatnonzero(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    assert len(welds) == 2
    for weld in welds:
        names = {model.body(int(model.eq_obj1id[weld])).name,
                 model.body(int(model.eq_obj2id[weld])).name}
        assert any(names == {f"{side}_hand_roll_link", f"{side}_wrist_fixture"} for side in SIDES)
    for body in range(1, source.nbody):
        name = source.body(body).name
        if name.endswith("_fixture"):
            continue
        actual = model.body(name).id
        assert model.body_mass[actual] == pytest.approx(source.body_mass[body])
        np.testing.assert_array_equal(model.body_inertia[actual], source.body_inertia[body])
        np.testing.assert_array_equal(model.body_ipos[actual], source.body_ipos[body])
    for joint in range(source.njnt):
        actual = model.joint(source.joint(joint).name).id
        np.testing.assert_array_equal(model.jnt_range[actual], source.jnt_range[joint])
    for geom in range(source.ngeom):
        name = source.geom(geom).name
        if name.endswith("_fixture_visual"):
            continue
        actual = model.geom(name).id
        for attribute in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
                          "geom_contype", "geom_conaffinity", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(model, attribute)[actual], getattr(source, attribute)[geom])
    for attribute in ("actuator_ctrlrange", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(model, attribute), getattr(source, attribute))
    assert cfg["simulation_dt"] == .001 and cfg["control_dt"] == .01
    assert layout["driver"] == "mocap_targets_with_dynamic_free_wrist_welds"


@pytest.mark.parametrize("profile_name,radius,height,mass", [
    ("baseline_40mm_100g", .020, .120, .100),
    ("sleek_330ml_approx_full", .029, .1454, .350),
])
def test_free_object_profile_and_can_skin_leave_physics_unchanged(profile_name, radius, height, mass):
    profile = load_cylinder_profile(profile_name)
    before = copy.deepcopy(profile)
    model, cfg, layout = build_top_grasp_model(profile, TopGraspCandidate(depth_m=0))
    assert profile == before
    body, collider = model.body("test_cylinder"), model.geom("cylinder_geom")
    assert model.joint("cylinder_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert body.mass == pytest.approx(mass)
    np.testing.assert_allclose(collider.size[:2], [radius, height/2])
    expected_inertia = [mass*(3*radius**2+height**2)/12]*2 + [mass*radius**2/2]
    np.testing.assert_allclose(body.inertia, expected_inertia, atol=1e-12)
    assert collider.contype[0] and collider.conaffinity[0]
    skin = [g for g in range(model.ngeom) if model.geom(g).name.startswith("r2v2_cola_can_")]
    assert skin and all(model.geom_contype[g] == model.geom_conaffinity[g] == 0 for g in skin)
    assert layout["cylinder_initial_position"][2] == pytest.approx(.4+height/2+.001)
    assert not model.body("tabletop").jntnum[0]
    assert model.body("tabletop").mocapid[0] == -1


@pytest.mark.parametrize("side", SIDES)
def test_top_frame_initial_clearance_and_static_diagnostic_not_runtime_reset(side):
    candidate = TopGraspCandidate(side=side, depth_m=0, yaw_deg=15, tilt_deg=10)
    model, cfg, layout = build_top_grasp_model(candidate=candidate)
    rotation = top_wrist_rotation(candidate)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert rotation[2, 0] < -.98
    np.testing.assert_allclose(layout["initial_wrist_positions"][side]-layout["grasp_wrist_position"], [0, 0, .1])
    parked = next(s for s in SIDES if s != side)
    assert np.linalg.norm(layout["initial_wrist_positions"][parked][:2]) > 1
    data = mujoco.MjData(model)
    initialize_hands(model, data, cfg)
    # Private FK inspection did not mutate the model's initial state.
    np.testing.assert_allclose(data.xpos[model.body(f"{side}_hand_roll_link").id], layout["initial_wrist_positions"][side])
    np.testing.assert_allclose(data.xpos[model.body("test_cylinder").id], layout["cylinder_initial_position"])
    assert not layout["geometry"]["initial_contacts"]["hand_cylinder"]
    assert not layout["geometry"]["initial_contacts"]["hand_table"]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("yaw", [-135., 0., 70.])
@pytest.mark.parametrize("tilt,spread", [(0., 0.), (60., 0.), (0., -60.), (30., 35.), (-45., 45.)])
def test_spread_tilt_preserves_proper_rotation_and_combined_downward_inclination(side, yaw, tilt, spread):
    candidate = TopGraspCandidate(side=side, yaw_deg=yaw, tilt_deg=tilt, spread_tilt_deg=spread)
    rotation = top_wrist_rotation(candidate)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    np.testing.assert_allclose(np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2], atol=1e-15)
    # Local +X is finger extension; its downward projection must depend on
    # both inclinations, not just the larger component or their sum.
    expected_projection = np.cos(np.deg2rad(tilt)) * np.cos(np.deg2rad(spread))
    assert -rotation[2, 0] == pytest.approx(expected_projection, abs=1e-15)
    assert -rotation[2, 0] >= .5 - 1e-12


@pytest.mark.parametrize("side,sign", [("left", 1.), ("right", -1.)])
def test_spread_inclines_across_fingers_not_in_original_flexion_plane(side, sign):
    angle = np.deg2rad(20.)
    flexion = top_wrist_rotation(TopGraspCandidate(side=side, tilt_deg=20.))[:, 0]
    spread = top_wrist_rotation(TopGraspCandidate(side=side, spread_tilt_deg=20.))[:, 0]
    np.testing.assert_allclose(flexion, [0., sign*np.sin(angle), -np.cos(angle)], atol=1e-15)
    np.testing.assert_allclose(spread, [-np.sin(angle), 0., -np.cos(angle)], atol=1e-15)
    assert not np.allclose(flexion, spread)


def test_spread_rotation_order_is_world_y_before_yaw_and_after_flexion():
    candidate = TopGraspCandidate(side="right", yaw_deg=31., tilt_deg=17., spread_tilt_deg=-23.)
    yaw, tilt, spread = np.deg2rad([31., -17., -23.])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                   [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    ry = np.array([[np.cos(spread), 0., np.sin(spread)], [0., 1., 0.],
                   [-np.sin(spread), 0., np.cos(spread)]])
    rx = np.array([[1., 0., 0.], [0., np.cos(tilt), -np.sin(tilt)],
                   [0., np.sin(tilt), np.cos(tilt)]])
    base = top_wrist_rotation(TopGraspCandidate())
    actual = top_wrist_rotation(candidate)
    np.testing.assert_allclose(actual, rz @ ry @ rx @ base, atol=1e-15)
    assert not np.allclose(actual, rz @ rx @ ry @ base)


@pytest.mark.parametrize("tilt,spread", [(45.01, 45.), (-45.01, -45.), (60., 1.), (1., -60.), (50., 40.)])
def test_individually_valid_tilts_with_combined_inclination_over_sixty_are_rejected(tilt, spread):
    assert abs(tilt) <= 60. and abs(spread) <= 60.
    with pytest.raises(ValueError, match="Combined inclination"):
        TopGraspCandidate(tilt_deg=tilt, spread_tilt_deg=spread)


@pytest.mark.parametrize("spread", [60.01, -60.01, float("nan"), float("inf"), True])
def test_invalid_spread_tilt_fails_closed(spread):
    with pytest.raises(ValueError):
        TopGraspCandidate(spread_tilt_deg=spread)


def test_default_candidate_has_zero_depth_and_nonoverlapping_pregrasp():
    candidate = TopGraspCandidate()
    assert candidate.depth_m == candidate.tilt_deg == candidate.spread_tilt_deg == candidate.yaw_deg == 0.
    _, _, layout = build_top_grasp_model(candidate=candidate)
    assert layout["geometry"]["valid_pregrasp"]
    assert not layout["geometry"]["initial_contacts"]["hand_cylinder"]
    assert not layout["geometry"]["initial_contacts"]["hand_table"]


def test_spread_only_reorients_wrist_and_preserves_original_hand_controls_and_mimics():
    original = copy.deepcopy(load_config())
    candidate = TopGraspCandidate(tilt_deg=10., spread_tilt_deg=20., lateral_m=.02)
    model, cfg, layout = build_top_grasp_model(candidate=candidate)
    source = mujoco.MjModel.from_xml_string(build_model_xml(original, fixture=True))
    assert cfg["hands"] == original["hands"]
    assert cfg["servo"] == original["servo"]
    assert cfg["trajectory"] == original["trajectory"]
    for side in SIDES:
        assert cfg["hands"][side]["open"][0] == pytest.approx(np.deg2rad(75.))
        assert cfg["hands"][side]["closed"][0] == pytest.approx(np.deg2rad(75.))
        assert cfg["hands"][side]["open"][1] == .04
    for attribute in ("actuator_ctrlrange", "actuator_forcerange", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(model, attribute), getattr(source, attribute))
    assert np.count_nonzero(model.eq_type == mujoco.mjtEq.mjEQ_JOINT) == 10
    for old in range(source.neq):
        if source.eq_type[old] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        actual = model.equality(source.equality(old).name).id
        for attribute in ("eq_type", "eq_data", "eq_solref", "eq_solimp", "eq_active0"):
            np.testing.assert_array_equal(getattr(model, attribute)[actual], getattr(source, attribute)[old])
        for attribute in ("eq_obj1id", "eq_obj2id"):
            actual_joint = model.joint(int(getattr(model, attribute)[actual])).name
            source_joint = source.joint(int(getattr(source, attribute)[old])).name
            assert actual_joint == source_joint
    assert layout["candidate"]["spread_tilt_deg"] == 20.
    assert load_config() == original


def test_invalid_nominal_depth_is_recorded_not_adjusted_to_make_candidate_pass():
    candidate = TopGraspCandidate(depth_m=.02)
    model, cfg, layout = build_top_grasp_model(candidate=candidate)
    geometry = layout["geometry"]
    assert not geometry["valid_pregrasp"]
    assert geometry["nominal_maximum_hand_object_penetration_m"] > .01
    assert geometry["nominal_maximum_hand_table_penetration_m"] > .005
    assert layout["candidate"]["depth_m"] == .02
    assert geometry["nominal_thumb_reference_world_m"][2] == pytest.approx(.4+.12-.02)


def test_precurl_is_candidate_local_and_does_not_raise_motor_limits():
    original = copy.deepcopy(load_config())
    candidate = TopGraspCandidate(depth_m=0, open_curl_rad=.4, lateral_m=.02)
    model, cfg, layout = build_top_grasp_model(candidate=candidate)
    assert load_config() == original
    assert cfg["hands"]["left"]["open"][2:] == [.4]*4
    assert cfg["hands"]["left"]["open"][:2] == original["hands"]["left"]["open"][:2]
    assert cfg["hands"]["left"]["closed"] == original["hands"]["left"]["closed"]
    assert cfg["hands"]["right"] == original["hands"]["right"]
    assert cfg["servo"] == original["servo"]
    assert cfg["trajectory"] == original["trajectory"]


@pytest.mark.parametrize("side", SIDES)
def test_raised_candidate_window_accepts_minus_eighty_mm_parameter_endpoint(side):
    # This only admits an upper grasp calibration candidate; no contact,
    # collision-free path, reachable grasp, or pickup success is inferred.
    candidate = TopGraspCandidate(side=side, depth_m=-.08, spread_tilt_deg=60.)
    assert candidate.depth_m == -.08
    assert candidate.spread_tilt_deg == 60.


@pytest.mark.parametrize("kwargs", [
    {"side": "both"}, {"depth_m": .041}, {"depth_m": -.081},
    {"yaw_deg": float("nan")}, {"tilt_deg": 61}, {"lateral_m": .05},
    {"open_curl_rad": .8}, {"open_curl_rad": True},
])
def test_bad_candidates_fail_closed(kwargs):
    with pytest.raises(ValueError):
        TopGraspCandidate(**kwargs)


def test_bad_model_parameters_and_start_collision_fail_closed():
    with pytest.raises(TypeError):
        build_top_grasp_model(candidate={})
    with pytest.raises(ValueError):
        build_top_grasp_model(table_height=.01)
    with pytest.raises(ValueError, match="Candidate start"):
        build_top_grasp_model(candidate=TopGraspCandidate(depth_m=.01, lateral_m=-.01))
