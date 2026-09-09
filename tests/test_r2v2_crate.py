"""Physical asset checks for the open crate, not evidence of lifting success."""

from common.path_config import PROJECT_ROOT

import copy
from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import CrateParameters, add_crate, load_crate_config


def _root():
    return ET.fromstring("""
        <mujoco model="crate_asset_test">
          <compiler angle="radian"/>
          <option timestep="0.001" integrator="implicitfast">
            <flag multiccd="enable"/>
          </option>
          <worldbody/>
        </mujoco>
    """)


def _compile(root):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _model(params=None, *, free=True, position=(0, 0, 0), quaternion=(1, 0, 0, 0)):
    root = _root()
    add_crate(root, params=params, free=free, position=position, quaternion=quaternion)
    return _compile(root)


@pytest.fixture(scope="module")
def crate():
    return _model()


def test_default_dimensions_and_clear_opening_contract():
    params = CrateParameters()
    assert (params.depth, params.width, params.height) == pytest.approx((0.24, 0.36, 0.16))
    assert params.bottom_thickness == pytest.approx(0.005)
    assert params.wall_thickness == pytest.approx(0.004)
    assert params.mass == pytest.approx(0.4)
    assert params.handle_opening_width == pytest.approx(0.120)
    assert params.handle_opening_height == pytest.approx(0.055)
    assert params.handle_opening_bottom == pytest.approx(0.085)
    assert params.handle_opening_top == pytest.approx(0.140)
    assert params.handle_beam_thickness == pytest.approx(0.010)
    assert params.handle_beam_height == pytest.approx(0.020)
    assert params.handle_rounding_radius == pytest.approx(0.002)
    assert params.inner_depth == pytest.approx(0.232)
    assert params.inner_width == pytest.approx(0.352)
    assert load_crate_config() == params


@pytest.mark.parametrize("changes", [
    {"depth": 0}, {"depth": np.nan}, {"width": -0.1}, {"width": np.inf},
    {"height": 0}, {"height": np.nan},
    {"bottom_thickness": 0}, {"bottom_thickness": 0.1},
    {"wall_thickness": 0}, {"wall_thickness": 0.11},
    {"handle_opening_width": 0}, {"handle_opening_width": 0.24},
    {"handle_opening_height": 0}, {"handle_opening_height": 0.14},
    {"handle_beam_height": 0}, {"handle_beam_height": 0.11},
    {"handle_beam_thickness": 0}, {"handle_beam_thickness": 0.19},
    {"handle_rounding_radius": -0.001}, {"handle_rounding_radius": 0.006},
    {"mass": 0}, {"mass": -0.4}, {"mass": np.nan}, {"mass": np.inf},
    {"friction": (1, 0.1)}, {"friction": (1, -0.01, 0.0001)},
    {"condim": 2}, {"rgba": (0.1, 0.2, 0.3)}, {"rgba": (0.1, 0.2, 0.3, 2)},
])
def test_invalid_crate_parameters_fail_before_compilation(changes):
    with pytest.raises((ValueError, TypeError)):
        CrateParameters(**changes)


def test_positive_custom_mass_loads_without_changing_defaults(tmp_path):
    config = tmp_path / "crate.yaml"
    config.write_text("mass: 0.75\n")
    loaded = load_crate_config(config)
    assert isinstance(loaded, CrateParameters)
    assert loaded.mass == pytest.approx(0.75)
    assert loaded.depth == pytest.approx(0.24)
    assert CrateParameters().mass == pytest.approx(0.4)


def test_unknown_configuration_key_is_not_silently_ignored(tmp_path):
    config = tmp_path / "crate.yaml"
    config.write_text("masss: 0.75\n")
    with pytest.raises((ValueError, TypeError)):
        load_crate_config(config)


@pytest.mark.parametrize("kwargs", [
    {"position": (0, 0)}, {"position": (0, np.nan, 0)},
    {"quaternion": (0, 0, 0, 0)}, {"quaternion": (1, 0, 0)},
    {"quaternion": (1, 0, np.inf, 0)},
])
def test_bad_placement_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        add_crate(_root(), **kwargs)


def test_names_sites_and_genuinely_free_body(crate):
    model, _ = crate
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (7, 6, 0, 0, 0)
    assert model.nbody == 2
    body = model.body("cargo_crate").id
    assert model.joint("crate_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_mocapid[body] == -1
    assert model.body_jntnum[body] == 1
    expected_geoms = {"crate_bottom", "crate_front_wall", "crate_back_wall"}
    for side in ("left", "right"):
        expected_geoms.update(f"crate_{side}_{part}" for part in (
            "lower_wall", "front_post", "back_post", "handle_beam"
        ))
    assert {model.geom(index).name for index in range(model.ngeom)} == expected_geoms
    assert model.ngeom == 11
    assert np.all(model.geom_bodyid == body)
    assert np.all(model.geom_contype != 0)
    assert np.all(model.geom_conaffinity != 0)
    for side, sign in (("left", 1), ("right", -1)):
        site = model.site(f"crate_{side}_handle").id
        assert model.site_bodyid[site] == body
        np.testing.assert_allclose(model.site_pos[site], [0, sign * 0.178, 0.1125], atol=1e-12)
    for name in ("crate_floor", "crate_placement"):
        site = model.site(name).id
        assert model.site_bodyid[site] == body
        np.testing.assert_allclose(model.site_pos[site], [0, 0, 0.005], atol=1e-12)


def test_requested_placement_uses_bottom_center_and_world_quaternion():
    angle = np.deg2rad(60)
    quaternion = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    position = np.array([0.40, -0.15, 0.75])
    model, data = _model(position=position, quaternion=quaternion)
    body = model.body("cargo_crate").id
    np.testing.assert_allclose(data.xpos[body], position, atol=1e-12)
    np.testing.assert_allclose(data.xquat[body], quaternion, atol=1e-12)
    bottom = model.geom("crate_bottom").id
    assert data.geom_xpos[bottom, 2] - model.geom_size[bottom, 2] == pytest.approx(position[2])
    rotation = data.xmat[body].reshape(3, 3)
    for side, sign in (("left", 1), ("right", -1)):
        site = model.site(f"crate_{side}_handle").id
        expected = position + rotation @ np.array([0, sign * 0.178, 0.1125])
        np.testing.assert_allclose(data.site_xpos[site], expected, atol=1e-12)


def test_mass_is_distributed_once_and_inertia_is_physical(crate):
    root = _root()
    body_xml = add_crate(root)
    assert body_xml.tag == "body" and body_xml.get("name") == "cargo_crate"
    assert body_xml.find("inertial") is None
    geoms = body_xml.findall("geom")
    masses = [float(geom.get("mass")) for geom in geoms]
    assert len(masses) == 11 and min(masses) > 0
    assert sum(masses) == pytest.approx(0.4, abs=1e-12)
    model, _ = crate
    body = model.body("cargo_crate").id
    assert model.body_mass[body] == pytest.approx(0.4, abs=1e-12)
    assert np.sum(model.body_mass) == pytest.approx(0.4, abs=1e-12)
    principal = model.body_inertia[body]
    assert np.all(np.isfinite(principal)) and np.all(principal > 1e-6)
    assert np.max(principal) < np.sum(principal) - np.max(principal)
    assert np.max(principal) < 0.4 * (0.24**2 + 0.36**2 + 0.16**2) / 4
    np.testing.assert_allclose(model.body_ipos[body, :2], [0, 0], atol=1e-10)
    assert 0.005 < model.body_ipos[body, 2] < 0.080


@pytest.mark.parametrize("mass", [0.2, 0.8, 1.0])
def test_custom_mass_scales_inertia_once_and_preserves_center_of_mass(crate, mass):
    reference, _ = crate
    model, _ = _model(replace(CrateParameters(), mass=mass))
    old = reference.body("cargo_crate").id
    new = model.body("cargo_crate").id
    assert model.body_mass[new] == pytest.approx(mass, abs=1e-12)
    np.testing.assert_allclose(model.body_ipos[new], reference.body_ipos[old], atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[new], reference.body_inertia[old] * mass / 0.4,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(model.geom_size, reference.geom_size, rtol=0, atol=0)


def _ray(model, data, origin, direction):
    hit = np.full(1, -1, dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.asarray(origin, dtype=float),
                            np.asarray(direction, dtype=float), None, 1, -1, hit)
    return float(distance), int(hit[0])


@pytest.mark.parametrize("side", [-1, 1])
def test_compiled_openings_keep_twelve_by_five_point_five_cm_clearance(crate, side):
    model, data = crate
    origin = [0., side*.178, .1125]
    for direction, expected_distance, part in (
        ([1, 0, 0], .060, "front_post"),
        ([-1, 0, 0], .060, "back_post"),
        ([0, 0, 1], .0275, "handle_beam"),
        ([0, 0, -1], .0275, "lower_wall"),
    ):
        distance, geom = _ray(model, data, origin, direction)
        assert distance == pytest.approx(expected_distance, abs=1e-8)
        label = "left" if side == 1 else "right"
        assert geom == model.geom(f"crate_{label}_{part}").id


@pytest.mark.parametrize("x,y", [(0, 0), (0.04, 0.05), (-0.04, -0.05)])
def test_open_top_and_empty_cavity_ray_hits_only_the_bottom(crate, x, y):
    model, data = crate
    distance, geom = _ray(model, data, [x, y, 0.20], [0, 0, -1])
    assert geom == model.geom("crate_bottom").id
    assert distance == pytest.approx(0.195, abs=1e-8)
    distance, geom = _ray(model, data, [x, y, 0.02], [0, 0, 1])
    assert (distance, geom) == (-1.0, -1)


@pytest.mark.parametrize("x", [-0.050, 0.0, 0.050])
@pytest.mark.parametrize("z", [0.092, 0.1125, 0.133])
@pytest.mark.parametrize("side", [-1, 1])
def test_both_handle_openings_have_no_hidden_collision_face(crate, x, z, side):
    model, data = crate
    distance, geom = _ray(model, data, [x, side * 0.22, z], [0, -side, 0])
    assert (distance, geom) == (-1.0, -1)


@pytest.fixture(scope="module")
def probe_model():
    root = _root()
    add_crate(root, free=False)
    probe = ET.SubElement(root.find("worldbody"), "body", name="probe", pos="0 0 .2")
    ET.SubElement(probe, "freejoint", name="probe_free")
    ET.SubElement(probe, "geom", name="probe_geom", type="sphere", size=".003", mass=".001")
    return _compile(root)[0]


def _probe_contacts(model, position):
    data = mujoco.MjData(model)
    address = model.joint("probe_free").qposadr[0]
    data.qpos[address:address + 3] = position
    mujoco.mj_forward(model, data)
    probe = model.geom("probe_geom").id
    # MuJoCo contact objects are views into MjData's native memory. Return
    # owned IDs, not dangling views after this local data object is destroyed.
    contacts = [tuple(map(int, contact.geom)) for contact in data.contact
                if probe in contact.geom and contact.dist <= 0]
    return contacts


@pytest.mark.parametrize("x,z", [(-0.05, 0.092), (0, 0.1125), (0.05, 0.133)])
def test_small_physical_probe_can_pass_through_clear_handles(probe_model, x, z):
    # Includes the thin sidewall centerlines explicitly, so discretization
    # cannot skip over an accidental solid plate in a nominal handle hole.
    for y in (-0.20, -0.18, -0.178, -0.17, 0, 0.17, 0.178, 0.18, 0.20):
        assert not _probe_contacts(probe_model, [x, y, z]), (x, y, z)
    for point in ([0, 0, 0.02], [0.04, 0.05, 0.08], [0, 0, 0.12]):
        assert not _probe_contacts(probe_model, point)


@pytest.mark.parametrize("position,expected_name", [
    ([0, 0, 0.004], "crate_bottom"),
    ([0.118, 0, 0.05], "crate_front_wall"),
    ([-0.118, 0, 0.05], "crate_back_wall"),
    ([0, 0.178, 0.015], "crate_left_lower_wall"),
    ([0, -0.178, 0.015], "crate_right_lower_wall"),
    ([0.09, 0.178, 0.1125], "crate_left_front_post"),
    ([-0.09, 0.178, 0.1125], "crate_left_back_post"),
    ([0.09, -0.178, 0.1125], "crate_right_front_post"),
    ([-0.09, -0.178, 0.1125], "crate_right_back_post"),
    # Intersect the convex beam's outer surface from outside. A sphere
    # initialized completely inside a convex mesh is not a physical approach.
    ([0, 0.181, 0.150], "crate_left_handle_beam"),
    ([0, -0.181, 0.150], "crate_right_handle_beam"),
])
def test_bottom_walls_and_handle_frames_have_real_collision(probe_model, position, expected_name):
    expected = probe_model.geom(expected_name).id
    assert any(expected in contact for contact in _probe_contacts(probe_model, position))


def test_add_crate_does_not_mutate_existing_bodies_controls_or_options():
    root = _root()
    root.find("worldbody").append(ET.fromstring("""
        <body name="unrelated_body" pos="3 3 .6">
          <joint name="unrelated_joint" axis="0 1 0" damping=".4" armature=".05"/>
          <geom name="unrelated_geom" type="sphere" size=".03" mass=".23" friction=".6 .003 .0001"/>
        </body>
    """))
    actuators = ET.SubElement(root, "actuator")
    ET.SubElement(actuators, "motor", name="unrelated_motor", joint="unrelated_joint", gear="2")
    original, _ = _compile(copy.deepcopy(root))
    untouched = {tag: ET.tostring(root.find(tag)) for tag in ("compiler", "option", "actuator")}
    untouched_body = ET.tostring(root.find("worldbody/body"))
    add_crate(root, position=(0.4, 0.2, 0.75))
    for tag, before in untouched.items():
        assert ET.tostring(root.find(tag)) == before
    assert ET.tostring(root.find("worldbody/body")) == untouched_body
    model, _ = _compile(root)
    assert model.nu == original.nu and model.nq == original.nq + 7
    for key in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
        np.testing.assert_array_equal(getattr(model, key)[:original.nbody], getattr(original, key))
    for key in ("actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm"):
        np.testing.assert_array_equal(getattr(model, key), getattr(original, key))


def test_duplicate_crate_is_rejected_instead_of_overwriting_existing_scene():
    root = _root()
    add_crate(root)
    with pytest.raises((ValueError, TypeError)):
        add_crate(root)


def test_static_fixture_has_no_free_joint_or_fake_constraint():
    model, _ = _model(free=False)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (0, 0, 0, 0, 0)
    assert model.body("cargo_crate").jntnum[0] == 0


def test_zero_rounding_uses_box_beams_without_changing_mass_or_hole_clearance():
    model, data = _model(replace(CrateParameters(), handle_rounding_radius=0))
    assert model.nmesh == 0
    assert model.body("cargo_crate").mass[0] == pytest.approx(0.4, abs=1e-12)
    for side in ("left", "right"):
        assert model.geom(f"crate_{side}_handle_beam").type == mujoco.mjtGeom.mjGEOM_BOX
    assert _ray(model, data, [0, 0.22, 0.1125], [0, -1, 0]) == (-1.0, -1)


def test_failed_insertion_leaves_input_xml_unchanged():
    root = _root()
    before = ET.tostring(root)
    with pytest.raises(ValueError):
        add_crate(root, quaternion=(0, 0, 0, 0))
    assert ET.tostring(root) == before


@pytest.mark.parametrize("initial_tilt_degrees", [0.0, 3.0])
def test_free_crate_naturally_settles_on_static_table_without_support_forces(initial_tilt_degrees):
    root = _root()
    world = root.find("worldbody")
    table = ET.SubElement(world, "body", name="static_table", pos="0 0 .73")
    ET.SubElement(table, "geom", name="table_geom", type="box", size=".35 .35 .02", friction="1 .005 .0001")
    angle = np.deg2rad(initial_tilt_degrees)
    quaternion = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    add_crate(root, position=(0.03, -0.02, 0.790), quaternion=quaternion)
    model, data = _compile(root)
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (7, 6, 0, 0, 0)
    body = model.body("cargo_crate").id
    table_id = model.body("static_table").id
    table_pose = data.xpos[table_id].copy()
    initial_height = float(data.xpos[body, 2])
    tail_positions, tail_speed, tail_angular_speed, tail_tilt, tail_contacts = [], [], [], [], []
    for _ in range(2500):
        mujoco.mj_step(model, data)
        if data.time >= 2.0:
            tail_positions.append(data.qpos[:3].copy())
            tail_speed.append(float(np.linalg.norm(data.qvel[:3])))
            tail_angular_speed.append(float(np.linalg.norm(data.qvel[3:6])))
            tail_tilt.append(float(np.rad2deg(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1)))))
            tail_contacts.append(data.ncon)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not data.warning.number.any()
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    assert data.qpos[2] < initial_height - 0.02
    assert data.qpos[2] == pytest.approx(0.75, abs=0.001)
    assert max(tail_speed) < 0.005
    assert max(tail_angular_speed) < 0.02
    assert max(tail_tilt) < 0.5
    assert np.max(np.ptp(np.asarray(tail_positions), axis=0)) < 0.0005
    assert min(tail_contacts) > 0
    np.testing.assert_array_equal(data.xpos[table_id], table_pose)
