"""Integrity tests for the real-object scene, not claims of dynamic grasp success."""

import copy

import mujoco
import numpy as np
import pytest

from common.r2v2_front_drop_scene import build_front_drop_collision_diagnostic, build_front_drop_model
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_sim import initialize_robot
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, build_model, initialize_hands


REACH_CFG = {"simulation_dt": .001, "hand_dt": .01}
SCENE = {
    "tabletop_height_m": .98, "table_center_world_m": [.545, .255, .96],
    "table_half_size_m": [.195, .365, .02], "can_initial_world_m": [.4, .13, 1.041],
    "can_initial_quaternion_wxyz": [1, 0, 0, 0],
    "can_dimensions_diameter_height_mass": [.04, .12, .1],
    "crate_base_world_m": [.45, .44, .98], "crate_quaternion_wxyz": [1, 0, 0, 0],
    "crate_dimensions_depth_width_height_m": [.24, .36, .16],
    "crate_wall_thickness_m": .004, "crate_bottom_thickness_m": .005,
    "crate_mass_kg": .4, "crate_handle_opening_width_height_m": [.12, .055],
    "extra_provenance": {"kept": [1, 2, 3]},
}


@pytest.fixture(scope="module", params=["air", "contact"])
def scene(request):
    return build_front_drop_model(REACH_CFG, SCENE, request.param)


def test_robot_physics_preserved_exactly(scene):
    model, hand_cfg, _ = scene
    original = build_model(hand_cfg)
    for field in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                  "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(model, field)[:original.nbody], getattr(original, field))
    for field in ("jnt_type", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited", "jnt_stiffness"):
        np.testing.assert_array_equal(getattr(model, field)[:original.njnt], getattr(original, field))
    for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(model, field)[:original.nv], getattr(original, field))
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                  "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap"):
        np.testing.assert_array_equal(getattr(model, field)[:original.ngeom], getattr(original, field))
    for field in ("actuator_trnid", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
                  "actuator_gainprm", "actuator_biasprm", "eq_type", "eq_data", "eq_solref", "eq_solimp",
                  "exclude_signature"):
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    assert model.nmocap == 0
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert model.opt.timestep == .001


def test_scene_dimensions_and_true_wrist_sites(scene):
    model, hand_cfg, layout = scene
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body("test_cylinder").id], [.4, .13, 1.041])
    np.testing.assert_allclose(data.xpos[model.body("cargo_crate").id], [.45, .44, .98])
    np.testing.assert_allclose(model.geom("cylinder_geom").size[:2], [.02, .06])
    assert model.body("test_cylinder").mass[0] == pytest.approx(.1)
    assert model.body("cargo_crate").mass[0] == pytest.approx(.4)
    np.testing.assert_allclose(model.geom("crate_bottom").size, [.12, .18, .0025])
    np.testing.assert_allclose(layout["crate_interior_bounds_world_m"]["min"], [.334, .264, .985])
    np.testing.assert_allclose(layout["crate_interior_bounds_world_m"]["max"], [.566, .616, 1.14])
    np.testing.assert_allclose(layout["crate_opening_bounds_world_m"]["min"], [.334, .270, 1.14])
    for side in SIDES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp") == -1
        site = model.site(f"{side}_wrist").id
        assert model.site_bodyid[site] == model.body(f"{side}_hand_roll_link").id
        np.testing.assert_array_equal(model.site_pos[site], [0, 0, 0])
        np.testing.assert_array_equal(model.site_quat[site], [1, 0, 0, 0])
        assert np.rad2deg(hand_cfg["hands"][side]["open"][0]) == pytest.approx(75)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    assert model.body("tabletop").jntnum[0] == 0
    assert layout["scene"] == SCENE


def test_contact_props_free_air_props_hidden_and_contact_disabled(scene):
    model, _, layout = scene
    contact = layout["mode"] == "contact"
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == (3 if contact else 1)
    for name in ("cylinder_free", "crate_free"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert (jid >= 0) == contact
        if contact:
            assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
    for body_name in ("tabletop", "test_cylinder", "cargo_crate"):
        body_id = model.body(body_name).id
        geoms = np.flatnonzero(model.geom_bodyid == body_id)
        if contact:
            assert np.any(model.geom_contype[geoms])
        else:
            assert not model.geom_contype[geoms].any()
            assert not model.geom_conaffinity[geoms].any()
            assert not model.geom_rgba[geoms, 3].any()
            assert np.all(model.geom_matid[geoms] == -1)
    assert model.geom("floor").contype[0] != 0


def test_body_and_hand_channels_disjoint(scene):
    model, cfg, _ = scene
    data = mujoco.MjData(model)
    initialize_hands(model, data, cfg)
    hands = DualHandControl(model, data, cfg)
    body = JointMap.create(model, BODY_JOINTS)
    hand_ids = np.concatenate([h.actuators for h in hands.maps.values()])
    assert not set(hand_ids) & set(body.actuators)
    assert set(hand_ids) | set(body.actuators) == set(range(40))
    data.ctrl[body.actuators] = np.arange(28)
    before = data.ctrl[body.actuators].copy()
    hands.command("left", 1)
    hands.update()
    hands.apply(data)
    np.testing.assert_array_equal(data.ctrl[body.actuators], before)


def test_builder_does_not_mutate_inputs_or_share_layout():
    raw, cfg = copy.deepcopy(SCENE), copy.deepcopy(REACH_CFG)
    _, _, layout = build_front_drop_model(cfg, raw, "air")
    assert raw == SCENE and cfg == REACH_CFG
    layout["scene"]["extra_provenance"]["kept"][0] = 99
    layout["table_center_xyz"][0] = 99
    assert raw == SCENE


@pytest.mark.parametrize("key,value", [
    ("tabletop_height_m", .99), ("can_initial_quaternion_wxyz", [0, 0, 0, 0]),
    ("crate_quaternion_wxyz", [1, 0, np.nan, 0]),
    ("crate_dimensions_depth_width_height_m", [.24, -.36, .16]),
    ("can_dimensions_diameter_height_mass", [True, .12, .1]),
])
def test_invalid_geometry_rejected(key, value):
    raw = copy.deepcopy(SCENE)
    raw[key] = value
    with pytest.raises(ValueError):
        build_front_drop_model(REACH_CFG, raw)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        build_front_drop_model(REACH_CFG, SCENE, "welded")


def test_collision_diagnostic_detects_planted_hand_table_overlap_without_affecting_air():
    air, cfg, _ = build_front_drop_model(REACH_CFG, SCENE, "air")
    diagnostic = build_front_drop_collision_diagnostic(REACH_CFG, SCENE)
    assert (diagnostic.nq, diagnostic.nv, diagnostic.nu) == (air.nq, air.nv, air.nu)
    for field in ("jnt_qposadr", "jnt_dofadr", "jnt_type", "geom_bodyid", "body_parentid"):
        np.testing.assert_array_equal(getattr(diagnostic, field), getattr(air, field))
    for body_name in ("tabletop", "cargo_crate", "test_cylinder"):
        body = air.body(body_name).id
        assert air.body_bvhnum[body] == 0
        assert air.body_contype[body] == air.body_conaffinity[body] == 0
        assert diagnostic.body_bvhnum[body] > 0
        assert diagnostic.body_contype[body] == diagnostic.body_conaffinity[body] == 1
    for geom in range(air.ngeom):
        assert diagnostic.geom(geom).name == air.geom(geom).name
        if air.geom(geom).name.startswith("r2v2_cola_can_"):
            assert diagnostic.geom_contype[geom] == diagnostic.geom_conaffinity[geom] == 0
    data = mujoco.MjData(air)
    initialize_robot(air, data, cfg)
    pinky = next(g for g in range(air.ngeom)
                 if air.geom(g).name.startswith("left_pinky_") and air.geom_contype[g])
    root = int(air.joint("floating_base_joint").qposadr[0])
    # Plant geometry only in this diagnostic unit test, never in a rollout.
    data.qpos[root:root + 3] += np.array([.43, .15, .965]) - data.geom_xpos[pinky]
    mujoco.mj_forward(air, data)
    before_qpos, before_ctrl = data.qpos.copy(), data.ctrl.copy()
    probe = mujoco.MjData(diagnostic)
    probe.qpos[:] = data.qpos
    mujoco.mj_forward(diagnostic, probe)
    table = air.geom("tabletop_geom").id
    def penetrates(contacts):
        return any(c.dist < 0 and set(map(int, c.geom)) == {pinky, table} for c in contacts)
    assert not penetrates(data.contact)
    assert penetrates(probe.contact)
    np.testing.assert_array_equal(data.qpos, before_qpos)
    np.testing.assert_array_equal(data.ctrl, before_ctrl)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
