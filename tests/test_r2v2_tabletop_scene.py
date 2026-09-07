"""Scene integrity and contact semantics, not an end-to-end grasp claim."""

from common.path_config import PROJECT_ROOT

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_cylinder_test import CylinderExperiment
from common.r2v2_reach_sim import TCP_OFFSETS
from common.r2v2_tabletop_scene import (
    DEFAULT_TABLE_CENTER, DEFAULT_TABLE_HALF_SIZE, build_tabletop_model,
    contact_metrics, object_metrics,
)
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, build_model


REACH_CFG = {"simulation_dt": 0.001, "hand_dt": 0.01}


@pytest.fixture(scope="module")
def scene():
    return build_tabletop_model(REACH_CFG, {})


def test_scene_preserves_every_existing_robot_physical_parameter(scene):
    model, hand_cfg = scene
    original = build_model(hand_cfg)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (64, 62, 40, 10, 0)
    assert model.njnt == original.njnt + 1
    assert model.nbody == original.nbody + 2  # One fixed table, one free object.
    assert model.ngeom == original.ngeom + 6  # Board, four legs, cylinder.
    assert model.nexclude == original.nexclude
    for key in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(model, key)[:original.nbody], getattr(original, key))
    for key in ("jnt_type", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited", "jnt_stiffness"):
        np.testing.assert_array_equal(getattr(model, key)[:original.njnt], getattr(original, key))
    for key in ("dof_damping", "dof_armature", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(model, key)[:original.nv], getattr(original, key))
    for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                "geom_condim", "geom_friction", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(model, key)[:original.ngeom], getattr(original, key))
    for key in ("actuator_trnid", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
                "actuator_gainprm", "actuator_biasprm", "eq_type", "eq_data", "exclude_signature"):
        np.testing.assert_array_equal(getattr(model, key), getattr(original, key))
    mapping = JointMap.create(model, BODY_JOINTS)
    assert mapping.actuators.tolist() == list(range(28))


def test_free_cylinder_fixed_table_and_virtual_tcp(scene):
    model, _ = scene
    assert model.joint("cylinder_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.body("test_cylinder").mocapid[0] == -1
    assert model.body("tabletop").mocapid[0] == -1
    assert model.body("tabletop").jntnum[0] == 0
    assert np.all(model.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert model.body("test_cylinder").id not in model.eq_obj1id
    assert model.body("test_cylinder").id not in model.eq_obj2id
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in SIDES:
        wrist = model.body(f"{side}_hand_roll_link").id
        site = model.site(f"{side}_tcp").id
        assert model.site_bodyid[site] == wrist
        np.testing.assert_array_equal(model.site_pos[site], TCP_OFFSETS[side])
        np.testing.assert_array_equal(model.site_quat[site], [1, 0, 0, 0])
    assert not data.xfrc_applied.any()
    assert not data.qfrc_applied.any()


def test_cylinder_retains_successful_baseline_contact_parameters(scene):
    model, _ = scene
    baseline = CylinderExperiment().model
    source, target = baseline.geom("cylinder_geom").id, model.geom("cylinder_geom").id
    for key in ("geom_type", "geom_size", "geom_friction", "geom_condim", "geom_priority", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(model, key)[target], getattr(baseline, key)[source])
    assert model.body("test_cylinder").mass[0] == pytest.approx(0.1)
    np.testing.assert_allclose(model.geom_size[target, :2], [0.02, 0.06])


def test_tabletop_enables_multiccd_without_changing_original_collision_options(scene):
    model, hand_cfg = scene
    original = build_model(hand_cfg)
    flag = int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    assert model.opt.enableflags & flag
    assert model.opt.enableflags == original.opt.enableflags | flag
    assert model.opt.disableflags == original.opt.disableflags
    for key in ("timestep", "integrator", "solver", "iterations", "tolerance", "cone", "impratio",
                "ccd_iterations", "ccd_tolerance", "gravity"):
        np.testing.assert_array_equal(getattr(model.opt, key), getattr(original.opt, key))


@pytest.fixture(scope="module")
def rest_regression_scene():
    # Layout and state below are the observed first FINAL_HOLD frame at
    # t=17.80 s in demo_probe_v4, recorded on 2026-09-07 with MuJoCo 3.3.7.
    # Values live in the test, not ignored artifacts, so clean clones reproduce
    # the former cylinder/box single-contact wobble with no robot or fingers.
    return build_tabletop_model(REACH_CFG, {
        "table_center_xyz": [0.555, 0.16, 1.0909189696536514],
        "table_half_size": [0.195, 0.26, 0.02],
        "cylinder_position_xyz": [0.4, 0.2, 1.1719189696536514],
    })[0]


def isolated_rest_model(source):
    """Extract actual table/cylinder parameters into a zero-actuator model."""
    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    text = lambda vector: " ".join(map(str, vector))
    for body_name, geom_name, geom_type, size_length in (
        ("tabletop", "tabletop_geom", "box", 3),
        ("test_cylinder", "cylinder_geom", "cylinder", 2),
    ):
        bid, gid = source.body(body_name).id, source.geom(geom_name).id
        body = ET.SubElement(world, "body", name=body_name, pos=text(source.body_pos[bid]),
                             quat=text(source.body_quat[bid]))
        if body_name == "test_cylinder":
            ET.SubElement(body, "freejoint", name="cylinder_free")
        geom = ET.SubElement(body, "geom", name=geom_name, type=geom_type,
                             size=text(source.geom_size[gid, :size_length]),
                             pos=text(source.geom_pos[gid]), quat=text(source.geom_quat[gid]),
                             friction=text(source.geom_friction[gid]),
                             condim=str(source.geom_condim[gid]), priority=str(source.geom_priority[gid]),
                             solref=text(source.geom_solref[gid]), solimp=text(source.geom_solimp[gid]))
        if body_name == "test_cylinder":
            geom.set("mass", str(source.body_mass[bid]))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for key in ("timestep", "integrator", "solver", "iterations", "tolerance", "cone", "impratio",
                "ccd_iterations", "ccd_tolerance", "enableflags", "disableflags"):
        setattr(model.opt, key, getattr(source.opt, key))
    model.opt.gravity[:] = source.opt.gravity
    for name in ("tabletop_geom", "cylinder_geom"):
        src, dst = source.geom(name).id, model.geom(name).id
        for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_condim", "geom_priority",
                    "geom_friction", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(model, key)[dst], getattr(source, key)[src])
    np.testing.assert_allclose(model.body("test_cylinder").inertia, source.body("test_cylinder").inertia)
    return model


@pytest.mark.parametrize("initial_state", ["measured_final_hold", "five_degree_tilt"])
@pytest.mark.parametrize("multiccd", [True, False])
def test_free_cylinder_settles_after_disturbance_only_with_surface_contact_manifold(
        rest_regression_scene, initial_state, multiccd):
    model = isolated_rest_model(rest_regression_scene)
    if not multiccd:
        # Negative control changes only the collision algorithm on this
        # private diagnostic model. Never modify the full scene or baseline.
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (7, 6, 0, 0, 0)
    data = mujoco.MjData(model)
    table_top = 1.1109189696536514
    if initial_state == "measured_final_hold":
        data.qpos[:] = [0.4028421590207662, 0.12057950010344695, 1.1713553987246508,
                       0.8082893990792104, -0.003312395241540352,
                       -0.01382563488738214, -0.5886137334397604]
        data.qvel[:3] = [0.028153217952838604, 0.0583422866505431, 0.009155373480385177]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        world_omega = [-1.1254882305374796, 0.6202531567518736, -0.7443172532791266]
        data.qvel[3:] = rotation.reshape(3, 3).T @ world_omega
    else:
        angle = np.deg2rad(5)
        extent = 0.06*np.cos(angle) + 0.02*np.sin(angle)
        data.qpos[:] = [0.4, 0.12, table_top+extent+0.001,
                       np.cos(angle/2), np.sin(angle/2), 0, 0]
    mujoco.mj_forward(model, data)
    table_before = data.xpos[model.body("tabletop").id].copy()
    tail_speed, tail_tilt, tail_position, tail_contacts = [], [], [], []
    cylinder = model.body("test_cylinder").id
    for _ in range(round(10/model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.time >= 9:
            tail_speed.append(float(np.linalg.norm(data.qvel[:3])))
            axis_z = data.xmat[cylinder].reshape(3, 3)[2, 2]
            tail_tilt.append(float(np.rad2deg(np.arccos(np.clip(axis_z, -1, 1)))))
            tail_position.append(data.qpos[:3].copy())
            tail_contacts.append(data.ncon)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    assert not data.warning.number.any()
    np.testing.assert_array_equal(data.xpos[model.body("tabletop").id], table_before)
    if multiccd:
        # Stronger than the demo's .01 m/s acceptance. All last-second samples
        # must be settled; checking just the final sample hides oscillations.
        assert max(tail_speed) < 0.001
        assert max(tail_tilt) < 0.02
        assert np.max(np.ptp(tail_position, axis=0)) < 1e-5
        assert min(tail_contacts) >= 3
    else:
        assert np.mean(tail_speed) > 0.02
        assert max(tail_contacts) == 1


def test_default_table_geometry_and_cylinder_clearance(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    center = np.array(DEFAULT_TABLE_CENTER)
    half = np.array(DEFAULT_TABLE_HALF_SIZE)
    table = model.geom("tabletop_geom").id
    np.testing.assert_array_equal(data.geom_xpos[table], center)
    np.testing.assert_array_equal(model.geom_size[table], half)
    metrics = object_metrics(model, data)
    assert metrics["bottom_height_m"]-(center[2]+half[2]) == pytest.approx(0.001)
    for index in range(4):
        leg = model.geom(f"tabletop_leg_{index}").id
        assert data.geom_xpos[leg, 2]-model.geom_size[leg, 2] == pytest.approx(0)
        assert data.geom_xpos[leg, 2]+model.geom_size[leg, 2] == pytest.approx(center[2]-half[2])


@pytest.mark.parametrize("angle", [0, 30, 90, 135, 180])
def test_object_bottom_height_uses_oriented_finite_cylinder_extent(scene, angle):
    model, _ = scene
    data = mujoco.MjData(model)
    address = model.joint("cylinder_free").qposadr[0]
    theta = np.deg2rad(angle)
    data.qpos[address+3:address+7] = [np.cos(theta/2), np.sin(theta/2), 0, 0]
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    metrics = object_metrics(model, data)
    extent = 0.06*abs(np.cos(theta)) + 0.02*abs(np.sin(theta))
    assert metrics["vertical_extent_m"] == pytest.approx(extent, abs=1e-9)
    assert metrics["bottom_height_m"] == pytest.approx(data.qpos[address+2]-extent, abs=1e-9)
    assert metrics["tilt_deg"] == pytest.approx(angle, abs=1e-6)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(metrics["T_world_cylinder"][:3, 3], metrics["position_m"])


def test_object_velocity_is_world_frame_and_metrics_are_read_only(scene):
    model, _ = scene
    data = mujoco.MjData(model)
    address = model.joint("cylinder_free").qposadr[0]
    velocity = model.joint("cylinder_free").dofadr[0]
    data.qpos[address+3:address+7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    data.qvel[velocity:velocity+6] = [1, 2, 3, 1, 0, 0]
    mujoco.mj_forward(model, data)
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    metrics = object_metrics(model, data)
    np.testing.assert_allclose(metrics["linear_velocity_mps"], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(metrics["angular_velocity_radps"], [0, 1, 0], atol=1e-12)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.ctrl, ctrl)


def contact_model():
    # A small diagnostic model exercises contact classification independently
    # of the full robot's reachability. It is never used in the rendered task.
    return mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.001"/>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 .1"/>
        <body name="base_link"><freejoint/>
          <inertial pos="0 0 .3" mass="10" diaginertia="1 1 1"/>
          <body name="left_hand_roll_link">
            <body name="left_thumb_distal_link" pos=".025 0 .30">
              <geom name="left_thumb_pad" type="sphere" size=".008" mass=".01"/>
            </body>
            <body name="left_index_distal_link" pos="-.025 0 .30">
              <geom name="left_index_pad" type="sphere" size=".008" mass=".01"/>
            </body>
          </body>
          <body name="left_shoulder_link" pos="0 .025 .30">
            <geom name="left_shoulder_collision" type="sphere" size=".008" mass=".01"/>
          </body>
          <body name="left_arm_link" pos=".10 0 .242">
            <geom name="left_arm_collision" type="sphere" size=".006" mass=".01"/>
          </body>
          <body name="right_hand_roll_link" pos="0 1 .4">
            <geom name="right_palm" type="sphere" size=".01" mass=".01"/>
          </body>
        </body>
        <body name="tabletop" pos="0 0 .15">
          <geom name="tabletop_geom" type="box" size=".2 .2 .09"/>
        </body>
        <body name="test_cylinder" pos="0 0 .30"><freejoint name="cylinder_free"/>
          <geom name="cylinder_geom" type="cylinder" size=".02 .06" mass=".1"
                friction="1 .005 .0001" condim="3" solref=".008 1"/>
        </body>
      </worldbody>
    </mujoco>''')


def test_contact_classification_does_not_count_arm_collisions_as_palm():
    model = contact_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = contact_metrics(model, data)
    assert result["opposed"]
    assert result["fingers_normal_force_N"]["thumb"] > 0.05
    assert result["fingers_normal_force_N"]["index"] > 0.05
    assert result["fingers_normal_force_N"]["palm"] == 0.0
    assert any(item["other_geom"] == "left_shoulder_collision" and item["part"] is None
               for item in result["contacts"])
    assert result["robot_table_contacts"]
    assert result["max_hand_object_penetration_m"] > 0.001
    right = contact_metrics(model, data, "right")
    assert not right["opposed"]
    assert all(force == 0 for force in right["fingers_normal_force_N"].values())


def test_robot_table_collision_is_not_object_support():
    model = contact_model()
    data = mujoco.MjData(model)
    address = model.joint("cylinder_free").qposadr[0]
    data.qpos[address:address+3] = [3, 3, 1]
    mujoco.mj_forward(model, data)
    result = contact_metrics(model, data)
    assert result["robot_table_contacts"]
    assert not result["table_contact"]
    assert not result["floor_contact"]
    assert not result["contacts"]
    np.testing.assert_array_equal(result["object_contact_force_world_N"], np.zeros(3))


def test_stationary_table_support_is_positive_world_force_on_cylinder():
    model = contact_model()
    data = mujoco.MjData(model)
    data.qpos[:3] = [3, 3, 1]  # Keep diagnostic robot away from the object.
    mujoco.mj_forward(model, data)
    table_position = data.xpos[model.body("tabletop").id].copy()
    for _ in range(100):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    result = contact_metrics(model, data)
    assert result["table_contact"] and not result["floor_contact"]
    assert not result["opposed"]
    assert not result["robot_table_contacts"]
    assert result["object_contact_force_world_N"][2] == pytest.approx(0.981, abs=0.01)
    np.testing.assert_array_equal(data.xpos[model.body("tabletop").id], table_position)
    assert not data.xfrc_applied.any()


def test_floor_contact_is_specific_to_object():
    model = contact_model()
    data = mujoco.MjData(model)
    address = model.joint("cylinder_free").qposadr[0]
    data.qpos[address:address+3] = [2, 0, 0.059]
    mujoco.mj_forward(model, data)
    result = contact_metrics(model, data)
    assert result["floor_contact"]
    assert not result["table_contact"]
    assert not result["opposed"]
    assert result["object_contact_force_world_N"][2] > 0


def test_build_does_not_mutate_configuration():
    reach_cfg, demo_cfg = dict(REACH_CFG), {"table_center_xyz": [0.5, 0.1, 1.0],
                                          "cylinder_quaternion_wxyz": [2, 0, 0, 0]}
    before_reach, before_demo = copy.deepcopy(reach_cfg), copy.deepcopy(demo_cfg)
    build_tabletop_model(reach_cfg, demo_cfg)
    assert reach_cfg == before_reach
    assert demo_cfg == before_demo


@pytest.mark.parametrize("cfg", [
    {"table_half_size": [0.1, -0.1, 0.02]},
    {"table_center_xyz": [0.5, 0, np.nan]},
    {"table_center_xyz": [0.5, 0, 0.01]},
    {"cylinder_position_xyz": [0, 0]},
    {"cylinder_quaternion_wxyz": [0, 0, 0, 0]},
])
def test_invalid_geometry_rejected(cfg):
    with pytest.raises(ValueError):
        build_tabletop_model(REACH_CFG, cfg)


def test_wrong_contact_rate_and_unknown_side_rejected(scene):
    with pytest.raises(ValueError, match="1 kHz"):
        build_tabletop_model({"simulation_dt": 0.002, "hand_dt": 0.01}, {})
    model, _ = scene
    with pytest.raises(ValueError, match="Unknown hand"):
        contact_metrics(model, mujoco.MjData(model), "not_a_hand")
