"""CPU-only profile/MJCF/placement checks; no renderer, policy or grasp rollout."""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import asdict
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_cylinder_test import CylinderParameters, load_cylinder_profile
from common.r2v2_reach_sim import load_reach_config
from common.r2v2_tabletop_demo import build_demo_scene_config, load_demo_config
from common.r2v2_tabletop_scene import build_tabletop_model, build_tabletop_xml, object_metrics


def test_omitted_and_explicit_baseline_retain_exact_parameters():
    expected = asdict(CylinderParameters())
    assert asdict(CylinderParameters.from_profile()) == expected
    assert asdict(CylinderParameters.from_profile("baseline_40mm_100g")) == expected
    assert load_demo_config()["cylinder_profile"]["profile_id"] == "baseline_40mm_100g"


def test_sleek_dimensions_do_not_imply_measured_mass_or_validated_grasp():
    profile = load_cylinder_profile("sleek_330ml_approx_full")
    p = CylinderParameters.from_profile(profile, side="right", grasp=False)
    assert 2*p.radius == pytest.approx(0.058)
    assert p.height == pytest.approx(0.1454)
    assert p.half_height == pytest.approx(0.0727)
    assert p.mass == pytest.approx(0.350)
    assert p.side == "right" and not p.grasp
    assert profile["nominal_capacity_ml"] == 330
    assert profile["geometry_source"] == "https://www.royalcanco.com/products-cans"
    assert profile["mass_provenance"] == "simulation_assumption_not_measured"
    assert "not a" in profile["mass_notes"] and "measured mass" in profile["mass_notes"]
    assert profile["grasp_calibration_status"] == "unvalidated"
    assert profile["simulation_only"] is True


@pytest.mark.parametrize("path", [
    "deploy_mujoco/config/cylinder_profiles/sleek_330ml_approx_full.yaml",
    PROJECT_ROOT / "deploy_mujoco/config/cylinder_profiles/sleek_330ml_approx_full.yaml",
])
def test_profile_paths_are_repo_relative_or_absolute(path):
    assert load_cylinder_profile(path) == load_cylinder_profile("sleek_330ml_approx_full")


def test_profile_resolution_does_not_mutate_input_or_share_state():
    profile = load_cylinder_profile("sleek_330ml_approx_full")
    old = copy.deepcopy(profile)
    resolved = load_cylinder_profile(profile)
    resolved["mass_kg"] = 0.5
    assert profile == old
    assert load_cylinder_profile("sleek_330ml_approx_full")["mass_kg"] == 0.35


@pytest.mark.parametrize("key,value", [
    ("radius_m", 0), ("height_m", -1), ("mass_kg", float("nan")),
    ("radius_m", float("inf")), ("height_m", True), ("mass_kg", "0.35"),
    ("nominal_capacity_ml", -330), ("schema_version", True), ("schema_version", 2),
    ("simulation_only", False), ("mass_provenance", ""), ("mass_notes", None),
    ("geometry_source", ""), ("grasp_calibration_status", "success"),
])
def test_invalid_profiles_fail_before_scene_compilation(key, value):
    profile = load_cylinder_profile("sleek_330ml_approx_full")
    profile[key] = value
    with pytest.raises(ValueError):
        CylinderParameters.from_profile(profile)


def test_missing_unknown_or_non_mapping_profile_is_rejected(tmp_path):
    profile = load_cylinder_profile()
    del profile["mass_kg"]
    with pytest.raises(ValueError, match="documented profile fields"):
        load_cylinder_profile(profile)
    profile = load_cylinder_profile()
    profile["mass_g"] = 100
    with pytest.raises(ValueError, match="documented profile fields"):
        load_cylinder_profile(profile)
    with pytest.raises(FileNotFoundError):
        load_cylinder_profile(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="profile source"):
        load_cylinder_profile([])


def test_changed_physics_cannot_keep_original_baseline_validation_label():
    profile = load_cylinder_profile()
    profile["radius_m"] = 0.029
    with pytest.raises(ValueError, match="baseline_verified applies only"):
        load_cylinder_profile(profile)
    with pytest.raises(ValueError, match="Put radius, height and mass"):
        CylinderParameters.from_profile("sleek_330ml_approx_full", mass=0.1)


@pytest.mark.parametrize("profile", ["baseline_40mm_100g", "sleek_330ml_approx_full"])
def test_demo_spawn_and_resting_center_follow_same_profile(profile):
    cfg = load_demo_config()
    cfg["cylinder_profile"] = profile
    old = copy.deepcopy(cfg)
    p = CylinderParameters.from_profile(profile)
    table_height = 1.13
    scene = build_demo_scene_config(cfg, table_height)
    assert cfg == old
    assert scene["cylinder_position_xyz"][:2] == cfg["cylinder_xy"]
    assert scene["cylinder_position_xyz"][2] == pytest.approx(
        table_height + p.half_height + cfg["cylinder_initial_clearance_m"])
    assert p.upright_center_height(table_height) == pytest.approx(table_height + p.half_height)
    assert p.upright_center_height(table_height, cfg["lift_m"]) == pytest.approx(
        table_height + p.half_height + cfg["lift_m"])
    assert scene["cylinder_profile"]["profile_id"] == profile


@pytest.fixture(scope="module")
def sleek_models():
    cfg = load_reach_config()
    scene = {"cylinder_profile": "sleek_330ml_approx_full"}
    return (build_tabletop_model(cfg, {**scene, "object_appearance": "orange_cylinder"})[0],
            build_tabletop_model(cfg, {**scene, "object_appearance": "cola_can"})[0])


def test_sleek_collision_mass_and_inertia_match_profile(sleek_models):
    p = CylinderParameters.from_profile("sleek_330ml_approx_full")
    expected_inertia = [p.mass*(3*p.radius**2+p.height**2)/12]*2 + [p.mass*p.radius**2/2]
    for model in sleek_models:
        body, geom = model.body("test_cylinder").id, model.geom("cylinder_geom").id
        np.testing.assert_array_equal(model.geom_size[geom, :2], [p.radius, p.half_height])
        assert model.body_mass[body] == pytest.approx(p.mass)
        np.testing.assert_allclose(model.body_inertia[body], expected_inertia, rtol=1e-12)
        assert model.jnt_type[model.joint("cylinder_free").id] == mujoco.mjtJoint.mjJNT_FREE
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        table = model.geom("tabletop_geom").id
        table_height = data.geom_xpos[table, 2] + model.geom_size[table, 2]
        assert object_metrics(model, data)["bottom_height_m"] - table_height == pytest.approx(0.001)


def test_scaled_can_visual_has_no_effect_on_physical_model(sleek_models):
    orange, can = sleek_models
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_type",
                  "jnt_range", "jnt_qposadr", "jnt_dofadr", "actuator_ctrlrange", "eq_data", "qpos0"):
        np.testing.assert_array_equal(getattr(orange, field), getattr(can, field), err_msg=field)
    for g in range(orange.ngeom):
        gid = can.geom(orange.geom(g).name).id
        for field in ("geom_size", "geom_friction", "geom_solref", "geom_solimp",
                      "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority"):
            np.testing.assert_array_equal(getattr(orange, field)[g], getattr(can, field)[gid], err_msg=field)
    visual = [g for g in range(can.ngeom) if can.geom(g).name.startswith("r2v2_cola_can_")]
    assert len(visual) == 8
    for g in visual:
        assert can.geom_contype[g] == can.geom_conaffinity[g] == 0
        assert can.geom_bodyid[g] == can.body("test_cylinder").id


@pytest.mark.parametrize("clearance", [-0.001, float("nan"), None, "0.001", True])
def test_invalid_clearance_rejected_before_xml_build(clearance):
    with pytest.raises(ValueError, match="cylinder_initial_clearance_m"):
        build_tabletop_xml(load_reach_config(), {"cylinder_initial_clearance_m": clearance})


def test_explicit_world_position_still_takes_precedence():
    cfg = {"cylinder_profile": "sleek_330ml_approx_full", "cylinder_position_xyz": [0.5, 0.2, 1.3]}
    root = ET.fromstring(build_tabletop_xml(load_reach_config(), cfg)[0])
    np.testing.assert_array_equal(np.fromstring(root.find('.//body[@name="test_cylinder"]').get("pos"), sep=" "),
                                  cfg["cylinder_position_xyz"])


def test_scene_default_profile_is_identical_to_explicit_baseline():
    cfg = load_reach_config()
    implicit, hands = build_tabletop_xml(cfg, {})
    explicit, explicit_hands = build_tabletop_xml(cfg, {"cylinder_profile": "baseline_40mm_100g"})
    assert implicit == explicit and hands == explicit_hands
