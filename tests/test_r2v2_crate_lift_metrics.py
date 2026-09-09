"""Forces, support classification and rotated bounds must be passive/physical."""

from common.path_config import PROJECT_ROOT

import itertools
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate_lift_metrics import _contact_force_on_geom, measure_crate_lift
from common.r2v2_crate import load_crate_config
from common.r2v2_crate_lift_scene import build_crate_lift_model
from r2v2_description.model import initialize_hands


@pytest.fixture(scope="module")
def scene():
    return build_crate_lift_model()


def new_data(scene):
    model, cfg, _ = scene
    data = mujoco.MjData(model)
    initialize_hands(model, data, cfg)
    mujoco.mj_forward(model, data)
    return data


def test_force_uses_frame_transpose_and_force_target_sign(monkeypatch):
    # This orthogonal frame is deliberately non-symmetric: frame @ force
    # would return a different answer from frame.T @ force.
    frame = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    contact = SimpleNamespace(geom1=3, geom2=4, frame=frame.ravel())
    data = SimpleNamespace(contact=[contact])
    def local_force(model, data, index, output):
        output[:] = [5, 2, 3, 0, 0, 0]
    monkeypatch.setattr(mujoco, "mj_contactForce", local_force)
    raw, first = _contact_force_on_geom(None, data, 0, 3)
    _, second = _contact_force_on_geom(None, data, 0, 4)
    np.testing.assert_array_equal(raw[:3], [5, 2, 3])
    np.testing.assert_array_equal(second, [2, 3, 5])
    np.testing.assert_array_equal(first, [-2, -3, -5])
    with pytest.raises(ValueError, match="not part"):
        _contact_force_on_geom(None, data, 0, 9)


def test_measurement_is_read_only_and_json_ready(scene, monkeypatch):
    model, _, layout = scene
    data = new_data(scene)
    arrays = ("qpos", "qvel", "qacc", "ctrl", "mocap_pos", "mocap_quat",
              "xpos", "xquat", "xmat", "efc_force", "qfrc_constraint")
    before = {name: getattr(data, name).copy() for name in arrays}
    masses, inertias, body_positions = model.body_mass.copy(), model.body_inertia.copy(), model.body_pos.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Metrics may not forward, step, or reset live data")
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_resetData", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, forbidden)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    assert data.time == result["time_s"] == 0
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(data, name), value)
    np.testing.assert_array_equal(model.body_mass, masses)
    np.testing.assert_array_equal(model.body_inertia, inertias)
    np.testing.assert_array_equal(model.body_pos, body_positions)
    json.dumps(result, allow_nan=False)
    assert result["finite_state"]
    assert not result["grasp_success_evaluated"]
    assert result["clearance_m"] == pytest.approx(.001)
    assert not result["table_contact"]


@pytest.mark.parametrize("angle", [0., .5, 1.3, np.pi])
def test_bottom_height_checks_all_rotated_collision_vertices(scene, angle):
    model, _, layout = scene
    data = new_data(scene)
    qpos = model.joint("crate_free").qposadr[0]
    data.qpos[qpos:qpos+3] = [.05, -.03, 1.0]
    quaternion = np.empty(4)
    axis = np.array([1., 2., 0.])/np.sqrt(5)
    mujoco.mju_axisAngle2Quat(quaternion, axis, angle)
    data.qpos[qpos+3:qpos+7] = quaternion
    mujoco.mj_forward(model, data)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    vertices = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != model.body("cargo_crate").id:
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            local = np.array(list(itertools.product((-1, 1), repeat=3)))*model.geom_size[geom]
        else:
            mesh = model.geom_dataid[geom]
            address, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[address:address+count]
        vertices.append(local @ data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom])
    expected = float(np.concatenate(vertices)[:, 2].min())
    assert result["bottom_height_m"] == pytest.approx(expected, abs=1e-12)
    assert result["clearance_m"] == pytest.approx(expected-layout["table_top_m"], abs=1e-12)
    assert result["crate_tilt_rad"] == pytest.approx(angle, abs=1e-12)
    if angle == np.pi:
        # Inverted crate's upper beams, not its original bottom plate, are lowest.
        assert result["bottom_height_m"] == pytest.approx(1.0-load_crate_config().height, abs=1e-8)


@pytest.mark.parametrize("surface", ["table", "floor"])
def test_real_solved_support_pushes_upward_but_is_not_a_grasp(scene, surface):
    model, _, layout = scene
    data = new_data(scene)
    qpos = model.joint("crate_free").qposadr[0]
    data.qpos[qpos+2] = (layout["table_top_m"] if surface == "table" else 0)-.0001
    mujoco.mj_forward(model, data)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    assert result[f"{surface}_contact"]
    assert not result[f"{'floor' if surface == 'table' else 'table'}_contact"]
    assert result[f"{surface}_vertical_force_N"] > 0
    contacts = result[f"{surface}_contacts"]
    assert contacts
    assert sum(contact["normal_force_N"] for contact in contacts) == pytest.approx(result[f"{surface}_vertical_force_N"])
    np.testing.assert_allclose(result["crate_contact_force_world_N"], result[f"{surface}_force_on_crate_world_N"])
    for side in ("left", "right"):
        assert result["hands"][side]["finger_vertical_force_N"] == 0
        assert not result["hands"][side]["has_bearing_finger_contact"]
    assert not result["grasp_success_evaluated"]


def test_real_finger_beam_contacts_are_not_mistaken_for_normal_bearing_force(scene):
    # Keep this known, deliberately intersecting 30 mm / .30 rad snapshot:
    # at the production 60 mm depth those fingers do not touch the beam.
    scene = build_crate_lift_model(insertion_m=.030)
    model, _, layout = scene
    data = new_data(scene)
    data.qpos[model.joint("crate_free").qposadr[0]+2] = layout["table_top_m"]
    # Deliberately penetrating static candidate tests force decomposition; it
    # is not a proposed grasp trajectory or a successful physical trial.
    for side in ("left", "right"):
        data.mocap_pos[layout["mocap_ids"][side]] = layout["inserted_wrist_positions"][side]
        # Test-only static snapshot: explicitly place the independent dynamic
        # wrist, since moving its target no longer teleports the actual hand.
        # Production never resets wrist or crate freejoint positions this way.
        address = model.joint(f"{side}_wrist_free").qposadr[0]
        data.qpos[address:address+3] = layout["inserted_wrist_positions"][side]
        for finger in ("index", "middle", "ring", "pinky"):
            data.qpos[model.joint(f"{side}_{finger}_proximal_joint").qposadr[0]] = .3
            data.qpos[model.joint(f"{side}_{finger}_distal_joint").qposadr[0]] = .3*1.155
    mujoco.mj_forward(model, data)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    for side in ("left", "right"):
        hand = result["hands"][side]
        # Dynamic-wrist compliance changes this deliberately invalid
        # snapshot's force, including possibly its vertical sign. Never
        # retain the old mocap-parented model's positive-load assumption.
        assert hand["finger_normal_force_N"] > 0
        assert not np.isclose(hand["finger_normal_force_N"], hand["finger_vertical_force_N"])
        upward = [contact for contact in hand["contacts"]
                  if contact["part"] in ("index", "middle", "ring", "pinky")
                  and contact["handle_beam_contact"] and contact["normal_force_N"] > 1e-8
                  and contact["vertical_force_N"] > 1e-8]
        assert hand["has_bearing_finger_contact"] == bool(upward)
        assert hand["bearing_finger_contacts"] == upward
        assert hand["parts"]["pinky"]["normal_force_N"] == hand["finger_normal_force_N"]
        assert hand["parts"]["thumb"]["normal_force_N"] == 0
        assert hand["parts"]["palm"]["normal_force_N"] == 0
        np.testing.assert_allclose(np.array(hand["T_world_wrist"]) @ np.array(hand["T_wrist_crate"]),
                                   result["T_world_crate"], atol=1e-12)
        assert all(contact["part"] == "pinky" for contact in hand["contacts"])
    total_hand = np.array(result["hands"]["left"]["force_on_crate_world_N"])+result["hands"]["right"]["force_on_crate_world_N"]
    np.testing.assert_allclose(total_hand, result["crate_contact_force_world_N"], atol=1e-9)
    assert result["max_hand_crate_penetration_m"] > .003
    assert not result["grasp_success_evaluated"]


def test_hand_table_and_self_contacts_are_separate_from_crate_support(scene):
    model, _, layout = scene
    data = new_data(scene)
    shift = data.mocap_pos[layout["mocap_ids"]["left"], 2]-(layout["table_top_m"]-.014)
    data.mocap_pos[layout["mocap_ids"]["left"], 2] -= shift
    # Deliberate test-only collision snapshot, with target and actual wrist
    # co-located to avoid introducing an unrelated weld tracking error.
    data.qpos[model.joint("left_wrist_free").qposadr[0]+2] -= shift
    mujoco.mj_forward(model, data)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    assert result["hand_table_contacts"]
    assert result["max_hand_table_penetration_m"] > 0
    assert result["hand_self_contacts"]
    assert result["max_hand_self_penetration_m"] > 0
    assert not result["table_contact"]  # Its unintegrated free crate is still 1 mm above the table.
    assert not result["floor_contact"]


def test_world_velocity_uses_crate_frame_origin_not_com(scene):
    model, _, layout = scene
    data = new_data(scene)
    joint = model.joint("crate_free")
    qpos, dof = joint.qposadr[0], joint.dofadr[0]
    mujoco.mju_axisAngle2Quat(data.qpos[qpos+3:qpos+7], np.array([1., 0, 0]), .6)
    data.qpos[qpos+2] = 1
    data.qvel[dof:dof+6] = [1, 2, 3, .5, -.2, .3]
    mujoco.mj_forward(model, data)
    result = measure_crate_lift(model, data, layout["table_top_m"])
    rotation = np.array(result["T_world_crate"])[:3, :3]
    np.testing.assert_allclose(result["crate_linear_velocity_world_m_s"], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(result["crate_angular_velocity_world_rad_s"], rotation @ [.5, -.2, .3], atol=1e-12)


def test_invalid_table_height_rejected(scene):
    model, _, _ = scene
    with pytest.raises(ValueError, match="finite"):
        measure_crate_lift(model, new_data(scene), np.nan)
