"""The can is a rendering skin, not a change to the free-object experiment."""

from common.path_config import PROJECT_ROOT

import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from PIL import Image

from common.r2v2_can_visual import add_can_visual
from common.r2v2_tabletop_scene import build_tabletop_model


REACH_CFG = {"simulation_dt": 0.001, "hand_dt": 0.01}
LABEL_PATH = PROJECT_ROOT / "r2v2_description/visuals/cola_can/label.png"


@pytest.fixture
def label(tmp_path):
    path = tmp_path / "label.png"
    Image.new("RGB", (8, 8), (210, 15, 20)).save(path)
    return path


def minimal_scene():
    root = ET.fromstring('''
      <mujoco>
        <option timestep="0.001"/>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 .1"/>
          <body name="test_cylinder" pos="0 0 .061">
            <freejoint name="cylinder_free"/>
            <geom name="cylinder_geom" type="cylinder" size=".02 .06"
                  mass=".1" rgba=".95 .45 .08 1" friction="1 .005 .0001"
                  condim="3" priority="1" solref=".008 1" solimp=".95 .99 .001"/>
          </body>
        </worldbody>
      </mujoco>''')
    return root, root.find('.//body[@name="test_cylinder"]')


@pytest.fixture(scope="module")
def models():
    # Deliberately require the production asset: missing packaging must fail,
    # rather than silently falling back to a differently textured test model.
    assert LABEL_PATH.is_file(), f"Missing checked-in can label: {LABEL_PATH}"
    orange, hands = build_tabletop_model(REACH_CFG, {})
    can, can_hands = build_tabletop_model(REACH_CFG, {"object_appearance": "cola_can"})
    assert hands == can_hands
    return orange, can


def test_helper_only_appends_assets_and_zero_mass_visuals(label):
    root, cylinder = minimal_scene()
    original_children = [ET.tostring(child) for child in cylinder]
    original_body_attributes = cylinder.attrib.copy()
    add_can_visual(root, cylinder, label)

    assert cylinder.attrib == original_body_attributes
    assert [ET.tostring(child) for child in list(cylinder)[:len(original_children)]] == original_children
    assert len(root.findall(".//body")) == 1
    assert len(root.findall(".//freejoint")) == 1
    assert root.find(".//inertial") is None
    visuals = cylinder.findall("geom")[1:]
    assert len(visuals) == 8  # Shell, two lids, two rims, aperture, tab, rivet.
    for geom in visuals:
        assert geom.get("name").startswith("r2v2_cola_can_")
        assert all(float(geom.get(key)) == 0 for key in ("mass", "density", "contype", "conaffinity"))
        assert geom.get("group") == "0"

    texture = root.find('./asset/texture[@name="r2v2_cola_can_label_texture"]')
    assert texture is not None
    assert texture.get("file") == str(label.resolve())
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert (model.nbody, model.nq, model.nv, model.nu) == (2, 7, 6, 0)
    assert model.body("test_cylinder").mass[0] == pytest.approx(0.1)
    np.testing.assert_allclose(model.body("test_cylinder").inertia, [1.3e-4, 1.3e-4, 2e-5], atol=1e-15)


def test_helper_rejects_missing_texture_without_partial_edits(tmp_path):
    root, cylinder = minimal_scene()
    before = ET.tostring(root)
    with pytest.raises(FileNotFoundError, match="texture is missing"):
        add_can_visual(root, cylinder, tmp_path / "not_present.png")
    assert ET.tostring(root) == before


def test_helper_rejects_duplicate_call_without_partial_edits(label):
    root, cylinder = minimal_scene()
    add_can_visual(root, cylinder, label)
    before = ET.tostring(root)
    with pytest.raises(ValueError, match="already been added"):
        add_can_visual(root, cylinder, label)
    assert ET.tostring(root) == before


@pytest.mark.parametrize("attributes", [
    {"type": "box"}, {"pos": ".001 0 0"}, {"quat": "1 0 0 0"},
    {"size": "0 .06"}, {"size": ".02 nan"},
])
def test_helper_rejects_incompatible_colliders(label, attributes):
    root, cylinder = minimal_scene()
    cylinder.find("geom").attrib.update(attributes)
    before = ET.tostring(root)
    with pytest.raises(ValueError):
        add_can_visual(root, cylinder, label)
    assert ET.tostring(root) == before


def test_compiled_physical_body_joint_actuator_and_constraint_arrays_are_identical(models):
    orange, can = models
    for key in ("nq", "nv", "nu", "nbody", "njnt", "neq", "nmocap", "nsite", "nM", "nD", "nB"):
        assert getattr(can, key) == getattr(orange, key), key
    assert can.ngeom == orange.ngeom + 8
    for key in (
        "body_parentid", "body_rootid", "body_pos", "body_quat", "body_mass", "body_inertia",
        "body_ipos", "body_iquat", "body_subtreemass", "body_invweight0", "body_gravcomp", "body_simple",
        "body_jntnum", "body_jntadr", "body_dofnum", "body_dofadr", "body_mocapid",
        "body_contype", "body_conaffinity", "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr",
        "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited", "jnt_stiffness", "jnt_margin",
        "jnt_solref", "jnt_solimp", "jnt_actfrclimited", "jnt_actfrcrange",
        "dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp",
        "dof_M0", "dof_invweight0", "qpos0", "qpos_spring", "actuator_trnid", "actuator_trntype",
        "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_dyntype",
        "actuator_gaintype", "actuator_biastype", "actuator_ctrllimited", "actuator_ctrlrange",
        "actuator_forcelimited", "actuator_forcerange", "eq_type", "eq_obj1id", "eq_obj2id",
        "eq_data", "eq_solref", "eq_solimp", "eq_active0", "exclude_signature",
    ):
        np.testing.assert_array_equal(getattr(can, key), getattr(orange, key), err_msg=key)
    for key in ("meanmass", "meaninertia"):
        assert getattr(can.stat, key) == getattr(orange.stat, key), key
    # Other automatic visual statistics (especially meansize) may change.
    for key in ("timestep", "integrator", "solver", "cone", "jacobian", "iterations", "tolerance",
                "ls_iterations", "ls_tolerance", "noslip_iterations", "noslip_tolerance",
                "enableflags", "disableflags", "gravity", "wind", "density", "viscosity"):
        np.testing.assert_array_equal(getattr(can.opt, key), getattr(orange.opt, key), err_msg=key)


def test_original_geom_ids_and_collision_parameters_are_preserved(models):
    orange, can = models
    for geom_id in range(orange.ngeom):
        assert can.geom(orange.geom(geom_id).name).id == geom_id
    for key in ("geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_contype",
                "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp",
                "geom_priority", "geom_margin", "geom_gap", "geom_rbound", "geom_aabb", "geom_group",
                "geom_matid", "geom_dataid"):
        np.testing.assert_array_equal(getattr(can, key)[:orange.ngeom], getattr(orange, key), err_msg=key)
    expected_rgba = orange.geom_rgba.copy()
    expected_rgba[orange.geom("cylinder_geom").id, 3] = 0
    np.testing.assert_array_equal(can.geom_rgba[:orange.ngeom], expected_rgba)


def test_visual_geoms_share_existing_body_and_have_no_contacts(models):
    orange, can = models
    appended = np.arange(orange.ngeom, can.ngeom)
    assert len(appended) == 8
    np.testing.assert_array_equal(can.geom_bodyid[appended], can.body("test_cylinder").id)
    assert not can.geom_contype[appended].any()
    assert not can.geom_conaffinity[appended].any()
    assert can.body("test_cylinder").mass[0] == pytest.approx(0.1)
    data = mujoco.MjData(can)
    mujoco.mj_forward(can, data)
    assert not np.isin(data.contact.geom, appended).any()


@pytest.mark.parametrize("profile", ["baseline_40mm_100g", "sleek_330ml_approx_full"])
def test_visual_skin_fits_inside_original_collision_envelope(models, profile):
    if profile == "baseline_40mm_100g":
        orange, can = models
    else:
        orange, _ = build_tabletop_model(REACH_CFG, {"cylinder_profile": profile})
        can, _ = build_tabletop_model(REACH_CFG, {"cylinder_profile": profile, "object_appearance": "cola_can"})
    data = mujoco.MjData(can)
    mujoco.mj_forward(can, data)
    cylinder = can.body("test_cylinder").id
    center = data.xpos[cylinder]
    rotation = data.xmat[cylinder].reshape(3, 3)
    radius, half_height = can.geom("cylinder_geom").size[:2]
    for geom in range(orange.ngeom, can.ngeom):
        geom_type = can.geom_type[geom]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = can.geom_dataid[geom]
            start, count = can.mesh_vertadr[mesh], can.mesh_vertnum[mesh]
            world_vertices = (can.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T
                              + data.geom_xpos[geom])
            local = (world_vertices-center) @ rotation
            assert np.max(np.linalg.norm(local[:, :2], axis=1)) <= radius + 1e-8
            assert np.max(np.abs(local[:, 2])) <= half_height + 1e-8
        else:
            position = can.geom_pos[geom]
            size = can.geom_size[geom]
            if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                radial_extent, vertical_extent = size[:2]
            else:
                assert geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
                radial_extent, vertical_extent = max(size[:2]), size[2]
            assert np.linalg.norm(position[:2]) + radial_extent <= radius + 1e-8
            assert abs(position[2]) + vertical_extent <= half_height + 1e-8


def test_identical_controls_produce_bitwise_identical_dynamics(models):
    orange, can = models
    data = [mujoco.MjData(model) for model in models]
    for model, state in zip(models, data):
        # Separate the unactuated robot from the free cylinder so this is a
        # controlled dynamics/contact regression, not a grasp-success claim.
        state.qpos[:2] = [3, 3]
        address = model.joint("cylinder_free").qposadr[0]
        angle = np.deg2rad(5)
        state.qpos[address+3:address+7] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
        mujoco.mj_forward(model, state)
    for key in ("qpos", "qvel", "qacc", "qM", "qfrc_bias", "qfrc_passive", "qfrc_constraint"):
        np.testing.assert_array_equal(getattr(data[0], key), getattr(data[1], key), err_msg=key)
    for step in range(1000):
        control = 0.01*np.sin(step*0.01 + np.arange(orange.nu))
        for model, state in zip(models, data):
            state.ctrl[:] = control
            mujoco.mj_step(model, state)
        for key in ("qpos", "qvel", "qacc", "qfrc_constraint", "qfrc_actuator"):
            np.testing.assert_array_equal(getattr(data[0], key), getattr(data[1], key),
                                          err_msg=f"{key} at step {step}")
        np.testing.assert_array_equal(data[0].contact.geom, data[1].contact.geom)
        np.testing.assert_array_equal(data[0].contact.dist, data[1].contact.dist)
    np.testing.assert_array_equal(data[0].warning.number, data[1].warning.number)


def test_default_and_explicit_orange_appearance_are_identical(models):
    orange, _ = models
    explicit, _ = build_tabletop_model(REACH_CFG, {"object_appearance": "orange_cylinder"})
    assert explicit.ngeom == orange.ngeom
    for key in ("body_mass", "body_inertia", "geom_rgba", "geom_group", "geom_size", "geom_matid"):
        np.testing.assert_array_equal(getattr(explicit, key), getattr(orange, key), err_msg=key)


def test_tabletop_rejects_unknown_appearance():
    with pytest.raises(ValueError, match="appearance"):
        build_tabletop_model(REACH_CFG, {"object_appearance": "unknown_object"})


def test_tabletop_does_not_mutate_input_configuration(models):
    reach_cfg = dict(REACH_CFG)
    appearance = {"object_appearance": "cola_can"}
    before = copy.deepcopy((reach_cfg, appearance))
    build_tabletop_model(reach_cfg, appearance)
    assert (reach_cfg, appearance) == before
