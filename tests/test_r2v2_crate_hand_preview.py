"""Static insertion diagnostics must not be confused with a lifting trial."""

from common.path_config import PROJECT_ROOT

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import CrateParameters
import common.r2v2_crate_hand_preview as hand_preview
from common.r2v2_crate_hand_preview import (
    _minimum_geometry_distance, build_hand_preview, sweep_hand_insertion,
)
from r2v2_description.model import build_model_xml, hand_names, load_config, urdf_hand_joints


@pytest.fixture(scope="module", params=["left", "right"])
def preview(request):
    return request.param, build_hand_preview(side=request.param)


def test_preview_is_fixed_wrist_free_crate_fk_only(preview, monkeypatch):
    def no_step(*args, **kwargs):
        raise AssertionError("A static preview must never integrate dynamics")
    monkeypatch.setattr(mujoco, "mj_step", no_step)
    side, (model, data, record) = preview
    # Also build after installing the guard, rather than testing only a fixture.
    guarded_model, guarded_data, _ = build_hand_preview(side=side)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (18, 17, 6, 5, 0)
    assert model.body(f"{side}_hand_roll_link").jntnum[0] == 0
    assert model.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert len([j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]) == 11
    np.testing.assert_array_equal(data.qvel, 0)
    np.testing.assert_array_equal(guarded_data.qpos, data.qpos)
    assert guarded_model.nq == model.nq
    assert data.time == guarded_data.time == record["time_s"] == 0
    assert record["scope"] == "static geometry only"
    assert record["simulation_steps"] == 0
    assert not record["grasp_success_evaluated"]
    assert record["crate_is_free_but_not_integrated"]


def test_palm_up_inward_pose_and_open_insertion_clearance(preview):
    side, (model, data, record) = preview
    rotation = np.array(record["wrist_transform_world"])[:3, :3]
    sign = 1 if side == "left" else -1
    np.testing.assert_allclose(rotation[:, 0], [0, -sign, 0], atol=1e-12)
    np.testing.assert_allclose(-sign*rotation[:, 1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)
    np.testing.assert_allclose(data.xpos[model.body("cargo_crate").id], [0, 0, 0], atol=1e-12)
    assert record["collision_free"]
    assert not record["hand_crate_contacts"]
    assert not record["hand_ground_contacts"]
    assert record["palm_to_outer_wall_m"] == pytest.approx(.008253115, abs=1e-8)
    assert record["thumb_to_outer_wall_m"] == pytest.approx(.05310553, abs=5e-8)
    assert record["four_finger_width_m"] == pytest.approx(.07833, abs=1e-5)
    assert record["four_finger_full_swept_thickness_m"] == pytest.approx(.04072, abs=1e-5)
    assert min(value["past_wall_inner_face_m"] for value in record["per_finger"].values()) == pytest.approx(.030)
    assert min(value["past_beam_inner_face_m"] for value in record["per_finger"].values()) == pytest.approx(.024)


def test_actual_hand_physics_and_mimic_relationships_are_preserved(preview):
    side, (model, data, _) = preview
    source = mujoco.MjModel.from_xml_string(build_model_xml(load_config(), fixture=True))
    for i in range(model.nbody):
        name = model.body(i).name
        if not name.startswith(side+"_"):
            continue
        original = source.body(name).id
        for attribute in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(model, attribute)[i], getattr(source, attribute)[original], atol=1e-14)
    for i in range(model.njnt):
        name = model.joint(i).name
        if not name.startswith(side+"_"):
            continue
        original = source.joint(name).id
        np.testing.assert_array_equal(model.jnt_range[i], source.jnt_range[original])
        dof, source_dof = model.jnt_dofadr[i], source.jnt_dofadr[original]
        for attribute in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(model, attribute)[dof], getattr(source, attribute)[source_dof])
    for i in range(model.ngeom):
        name = model.geom(i).name
        if not name.startswith(side+"_"):
            continue
        original = source.geom(name).id
        for attribute in ("geom_contype", "geom_conaffinity", "geom_friction", "geom_condim",
                          "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
            np.testing.assert_array_equal(getattr(model, attribute)[i], getattr(source, attribute)[original])
    for name, value in zip(hand_names(side), load_config()["hands"][side]["open"]):
        assert data.qpos[model.joint(name).qposadr[0]] == pytest.approx(value)
    for name, source_joint in urdf_hand_joints().items():
        mimic = source_joint.find("mimic")
        if name.startswith(side+"_") and mimic is not None:
            actual = data.qpos[model.joint(name).qposadr[0]]
            leader = data.qpos[model.joint(mimic.get("joint")).qposadr[0]]
            assert actual == pytest.approx(leader*float(mimic.get("multiplier", "1"))
                                          + float(mimic.get("offset", "0")))


def test_bounded_distance_has_real_positive_gap_and_consistent_witness(preview):
    _, (_, _, record) = preview
    assert record["minimum_hand_crate_distance_reliable"]
    assert record["minimum_hand_crate_distance_m"] == pytest.approx(.009700, abs=3e-7)
    closest = record["closest_hand_crate_pair"]
    assert closest["geoms"][0].endswith("middle_proximal_link_col_1")
    assert closest["geoms"][1].endswith("lower_wall")
    points = np.array(closest["points_world_m"])
    assert np.linalg.norm(points[1]-points[0]) == pytest.approx(closest["distance_m"], abs=1e-9)
    assert not record["unresolved_distance_pairs"]


def test_invalid_potentially_nearer_query_makes_minimum_unknown(monkeypatch):
    model, data, _ = build_hand_preview()
    hand = model.geom("left_middle_proximal_link_col_1").id
    crate = model.geom("crate_left_lower_wall").id
    def inconsistent_distance(model, data, geom1, geom2, radius, points):
        points[:] = [0, 0, 0, .2, 0, 0]
        return 0.0
    monkeypatch.setattr(mujoco, "mj_geomDistance", inconsistent_distance)
    record = _minimum_geometry_distance(model, data, [hand], [crate])
    assert record["minimum_hand_crate_distance_m"] is None
    assert not record["minimum_hand_crate_distance_reliable"]
    assert record["minimum_hand_crate_distance_lower_bound_m"] is None
    assert record["unresolved_distance_pairs"]


def test_distance_outside_radius_is_a_lower_bound_not_an_exact_gap():
    model, data, _ = build_hand_preview()
    record = _minimum_geometry_distance(model, data,
        [model.geom("left_middle_distal_link_col_3").id], [model.geom("crate_front_wall").id])
    assert record["minimum_hand_crate_distance_m"] is None
    assert not record["minimum_hand_crate_distance_reliable"]
    assert record["minimum_hand_crate_distance_lower_bound_m"] == .02
    assert record["closest_hand_crate_pair"] is None


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("curl, depth", [(.3, .0030807), (.5, .0110145)])
def test_curled_candidates_keep_wrist_pose_and_report_real_beam_penetration(side, curl, depth):
    _, _, opened = build_hand_preview(side=side)
    _, _, curled = build_hand_preview(side=side, curl_rad=curl)
    np.testing.assert_allclose(curled["wrist_transform_world"], opened["wrist_transform_world"], atol=1e-12)
    assert not curled["collision_free"]
    assert curled["maximum_hand_crate_penetration_m"] == pytest.approx(depth, abs=1e-6)
    assert all(f"crate_{side}_handle_beam" in contact["geoms"] for contact in curled["hand_crate_contacts"])
    assert not curled["grasp_success_evaluated"]
    if curl == .5:
        assert curled["per_finger"]["pinky"]["past_beam_inner_face_m"] < 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_hand_insertion_sweep_no_dynamics_and_no_hidden_collisions(side, monkeypatch):
    def no_step(*args, **kwargs):
        raise AssertionError("A sweep must use forward kinematics only")
    monkeypatch.setattr(mujoco, "mj_step", no_step)
    record = sweep_hand_insertion(side=side)
    assert record["scope"] == "static geometry only"
    assert record["simulation_steps"] == 0
    assert record["collision_free"]
    assert record["colliding_samples"] == 0
    assert len(record["samples"]) == 61
    assert record["samples"][0]["palm_to_outer_wall_m"] == pytest.approx(.060)
    assert record["samples"][-1]["palm_to_outer_wall_m"] == pytest.approx(.008253115, abs=1e-8)
    assert all(not row["hand_crate_contacts"] for row in record["samples"])
    for row in record["samples"]:
        if not row["minimum_hand_crate_distance_reliable"]:
            assert row["minimum_hand_crate_distance_m"] is None


def test_smaller_slot_cannot_silently_claim_clearance():
    _, _, record = build_hand_preview(replace(CrateParameters(), handle_opening_height=.020))
    assert not record["collision_free"]
    assert record["hand_crate_contacts"]
    assert record["maximum_hand_crate_penetration_m"] > 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_sweep_uses_private_model_and_never_changes_input_crate_or_hand_state(side, monkeypatch):
    model, data, record = build_hand_preview(side=side)
    original_body_pos = model.body_pos.copy()
    original_qpos, original_qvel = data.qpos.copy(), data.qvel.copy()
    original_xpos, original_xmat = data.xpos.copy(), data.xmat.copy()
    monkeypatch.setattr(hand_preview, "build_hand_preview", lambda *args, **kwargs: (model, data, record))
    sweep = sweep_hand_insertion(side=side, samples=7)
    assert sweep["collision_free"]
    np.testing.assert_array_equal(model.body_pos, original_body_pos)
    np.testing.assert_array_equal(data.qpos, original_qpos)
    np.testing.assert_array_equal(data.qvel, original_qvel)
    np.testing.assert_array_equal(data.xpos, original_xpos)
    np.testing.assert_array_equal(data.xmat, original_xmat)
    assert data.time == 0


@pytest.mark.parametrize("kwargs", [
    {"side": "third"}, {"insertion_m": np.nan}, {"insertion_m": .2},
    {"curl_rad": np.nan}, {"curl_rad": 5}, {"thumb_deg": 500},
])
def test_preview_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        build_hand_preview(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"samples": 1}, {"samples": 2.5}, {"start_palm_clearance_m": 0},
    {"start_palm_clearance_m": .001},
])
def test_sweep_rejects_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        sweep_hand_insertion(**kwargs)
