"""Recorded-motion scene preserves real robot/crate physics and true wrists."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_motion_replay_scene import build_motion_replay_model
from common.r2v2_crate_reach_scene import build_crate_reach_model
from common.r2v2_reach_sim import initialize_robot, load_reach_config
from r2v2_description.model import SIDES, build_model_xml


@pytest.fixture(scope="module")
def scene():
    cfg = load_reach_config("deploy_mujoco/config/r2v2_reach_wrist_v2.yaml")
    params = replace(load_crate_config(), width=.26)
    return (*build_motion_replay_model(cfg, params), cfg, params)


def test_real_free_robot_and_crate_without_extra_constraints(scene):
    model, hand_cfg, layout, _, _ = scene
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (64, 62, 40, 10, 0)
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 2
    assert model.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("cargo_crate").mocapid[0] == -1
    for name in ("test_cylinder", "left_wrist_fixture", "right_wrist_fixture"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) == -1
    for name in ("cylinder_free", "left_wrist_free", "right_wrist_free"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
    assert layout["endpoint_contract"] == "wrist_world_v2"
    assert layout["target_source"] == "recorded_crate_relative_wrist_motion"
    assert "inserted_wrist_transforms" not in layout

    source = mujoco.MjModel.from_xml_string(build_model_xml(hand_cfg, fixture=False))
    collections = (
        ("body", source.nbody, ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat")),
        ("joint", source.njnt, ("jnt_type", "jnt_range", "jnt_limited", "jnt_axis", "jnt_pos", "jnt_stiffness")),
        ("geom", source.ngeom, ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                              "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap")),
        ("actuator", source.nu, ("actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "actuator_gainprm", "actuator_biasprm")),
        ("equality", source.neq, ("eq_data", "eq_solref", "eq_solimp", "eq_type")),
    )
    for kind, count, fields in collections:
        for i in range(count):
            j = getattr(model, kind)(getattr(source, kind)(i).name).id
            for field in fields:
                np.testing.assert_array_equal(getattr(model, field)[j], getattr(source, field)[i])
    for i in range(source.njnt):
        j = model.joint(source.joint(i).name).id
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(model, field)[model.jnt_dofadr[j]],
                                          getattr(source, field)[source.jnt_dofadr[i]])


@pytest.mark.parametrize("side", SIDES)
def test_wrist_site_is_true_link_origin_and_axes_with_no_legacy_site(scene, side):
    model, hands, _, _, _ = scene
    site, body = model.site(f"{side}_wrist"), model.body(f"{side}_hand_roll_link")
    assert site.bodyid[0] == body.id
    np.testing.assert_array_equal(site.pos, [0., 0., 0.])
    np.testing.assert_array_equal(site.quat, [1., 0., 0., 0.])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp") == -1
    data = mujoco.MjData(model)
    initialize_robot(model, data, hands)
    for name, value in ((f"{side}_shoulder_roll_joint", .1), (f"{side}_hand_pitch_joint", -.2)):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.site_xpos[site.id], data.xpos[body.id])
    np.testing.assert_allclose(data.site_xmat[site.id], data.xmat[body.id], atol=1e-15)


def test_table_crate_geometry_and_contacts_match_existing_baseline(scene):
    model, _, layout, cfg, params = scene
    old, _, _ = build_crate_reach_model(cfg, params, table_top_m=layout["table_top_m"],
                                      crate_center_xy=layout["crate_center_xy"])
    assert old.ngeom == model.ngeom
    fields = ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
              "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap")
    for i in range(old.ngeom):
        j = model.geom(old.geom(i).name).id
        for field in fields:
            np.testing.assert_array_equal(getattr(model, field)[j], getattr(old, field)[i])
    np.testing.assert_array_equal(model.body("cargo_crate").pos, old.body("cargo_crate").pos)
    np.testing.assert_array_equal(model.body("cargo_crate").inertia, old.body("cargo_crate").inertia)
    np.testing.assert_allclose(model.geom("crate_bottom").size, [.12, .13, .0025], atol=1e-14)
    np.testing.assert_allclose(layout["crate_initial_position"], [.38, 0, layout["table_top_m"]+.001], atol=1e-14)
    assert model.body("cargo_crate").mass[0] == pytest.approx(.4)
    for side in SIDES:
        front, back = model.geom(f"crate_{side}_front_post"), model.geom(f"crate_{side}_back_post")
        assert front.pos[0]-front.size[0]-(back.pos[0]+back.size[0]) == pytest.approx(.12)
        assert 2*front.size[2] == pytest.approx(.055)
    assert model.opt.enableflags == old.opt.enableflags
    assert model.opt.disableflags == old.opt.disableflags


@pytest.mark.parametrize("contract", [None, "legacy_tcp_v1", "wrist_world_v3"])
def test_missing_or_non_v2_contract_rejected_before_scene_build(scene, contract):
    cfg = copy.deepcopy(scene[3])
    cfg["endpoint_contract"] = contract
    with pytest.raises(ValueError, match="wrist_world_v2"):
        build_motion_replay_model(cfg, scene[4])


@pytest.mark.parametrize("kwargs", [
    {"table_top_m": .02}, {"table_center_xy": [.4, 0]}, {"crate_center_xy": [.65, .3]},
    {"table_half_size": [.22, .35, 0]}, {"crate_center_xy": [np.nan, 0]},
])
def test_unsupported_table_layout_is_rejected(scene, kwargs):
    with pytest.raises(ValueError):
        build_motion_replay_model(scene[3], scene[4], **kwargs)


def test_config_inputs_remain_unchanged(scene):
    cfg = copy.deepcopy(scene[3])
    original = copy.deepcopy(cfg)
    xy = np.array([.38, 0.])
    _, hands, layout = build_motion_replay_model(cfg, scene[4], crate_center_xy=xy)
    assert cfg == original
    layout["crate_center_xy"][:] = 0
    np.testing.assert_array_equal(xy, [.38, 0.])
    hands["hands"]["left"]["open"][0] = 0
    assert scene[1]["hands"]["left"]["open"][0] != 0
