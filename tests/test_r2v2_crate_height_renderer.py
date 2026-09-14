"""Pure rendering checks; no policy or synthetic grasp success is involved."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from tools.compose_r2v2_crate_height_comparison import _sample, SOURCE_CROP, _validate_scene_variant
from deploy_mujoco.r2v2_crate_height_sweep import (
    PANEL_WIDTH, PANEL_HEIGHT, HEADER_HEIGHT, FRAME_HEIGHT,
    _box_edges, _virtual_prop_edges, _add_virtual_prop_outlines,
)


def _trial():
    return {
        "start_s": 4., "end_s": 4.1, "duration_s": .1,
        "times": [4., 4.03, 4.06, 4.09, 4.1, 4.1],
        "trace_times": [4., 4.05, 4.1],
        "trace": [{"phase": "PREPARE"}, {"phase": "TURN"}, {"phase": "FAILED"}],
    }


def test_comparison_uses_elapsed_physics_time_not_absolute_reset_time():
    index, sample, frozen, simulation_time = _sample(_trial(), .065)
    assert index == 2
    assert sample["phase"] == "TURN"
    assert not frozen
    assert simulation_time == 4.065


def test_comparison_freezes_failed_trial_without_extrapolation():
    index, sample, frozen, simulation_time = _sample(_trial(), 99.)
    assert index == 5
    assert sample["phase"] == "FAILED"
    assert frozen
    assert simulation_time == 4.1


def test_camera_crop_excludes_source_hud_and_preserves_image_size():
    x0, y0, x1, y1 = SOURCE_CROP
    assert x0 == PANEL_WIDTH
    assert x1 == 2 * PANEL_WIDTH
    assert y0 == HEADER_HEIGHT
    assert y1 - y0 == PANEL_HEIGHT
    assert y1 < FRAME_HEIGHT


def test_comparison_preserves_legacy_physical_report_defaults():
    assert _validate_scene_variant([{"report": {}}] * 5) == (0., False)


def test_comparison_accepts_one_matching_virtual_displacement():
    trial = {"report": {"x_offset_m": -.05, "virtual_props": True}}
    assert _validate_scene_variant([trial] * 5) == (-.05, True)


def test_comparison_accepts_canonical_displacement_and_rejects_conflicting_alias():
    trial = {"report": {"delta_x_m": -.05, "virtual_props": True}}
    assert _validate_scene_variant([trial] * 5) == (-.05, True)
    trial["report"]["x_offset_m"] = -.04
    with pytest.raises(ValueError, match="Conflicting"):
        _validate_scene_variant([trial] * 5)


@pytest.mark.parametrize("changed", [
    {"x_offset_m": -.04, "virtual_props": True},
    {"x_offset_m": -.05, "virtual_props": False},
    {"x_offset_m": float("nan"), "virtual_props": True},
    {"x_offset_m": -.05, "virtual_props": "yes"},
])
def test_comparison_rejects_mixed_or_invalid_variants(changed):
    trial = {"report": {"x_offset_m": -.05, "virtual_props": True}}
    with pytest.raises(ValueError):
        _validate_scene_variant([trial] * 4 + [{"report": changed}])


def test_oriented_box_has_twelve_edges_and_no_diagonals():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    center = np.array([.3, -.2, 1.])
    edges = _box_edges(center, rotation, [.1, .2, .3])
    assert len(edges) == 12
    vertices = np.array([point for edge in edges for point in edge])
    np.testing.assert_allclose(vertices.min(axis=0), center - [.2, .1, .3])
    np.testing.assert_allclose(vertices.max(axis=0), center + [.2, .1, .3])
    np.testing.assert_allclose(sorted(np.linalg.norm(b - a) for a, b in edges),
                               sorted([.2, .4, .6] * 4))


def _wireframe_fixture():
    from common.r2v2_crate import CrateParameters, add_crate

    root = ET.fromstring('''<mujoco><worldbody>
      <body name="tabletop" pos=".43 0 .8">
        <geom type="box" size=".22 .35 .02"/>
        <geom type="box" pos=".2 .3 -.41" size=".012 .012 .39"/>
      </body>
      <body name="robot"><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
    add_crate(root, CrateParameters(width=.26), position=(.33, 0., .82), free=False)
    for body in (root.find('.//body[@name="tabletop"]'), root.find('.//body[@name="cargo_crate"]')):
        for geom in body.findall("geom"):
            geom.set("rgba", "0 0 0 0")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model=model, scratch=data, virtual_props=True)


def test_wireframe_covers_table_parts_and_all_crate_solids_without_mutation():
    exp = _wireframe_fixture()
    before = {key: getattr(exp.scratch, key).copy() for key in ("qpos", "qvel", "geom_xpos", "geom_xmat")}
    edges = _virtual_prop_edges(exp)
    assert len(edges) == (2 + 11) * 12
    points = np.array([point for a, b, color in edges for point in (a, b)])
    assert np.all(np.isfinite(points))
    # Both rounded beam meshes are present, reaching the crate's real top.
    crate_points = np.array([point for a, b, color in edges if color[0] == .25 for point in (a, b)])
    np.testing.assert_allclose(crate_points.min(axis=0), [.21, -.13, .82], atol=1e-7)
    np.testing.assert_allclose(crate_points.max(axis=0), [.45, .13, .98], atol=1e-7)
    for key, values in before.items():
        np.testing.assert_array_equal(getattr(exp.scratch, key), values)
    scene = mujoco.MjvScene(exp.model, maxgeom=300)
    _add_virtual_prop_outlines(scene, exp)
    assert scene.ngeom == len(edges)
    assert all(scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_LINE for i in range(scene.ngeom))
    assert all(scene.geoms[i].category == mujoco.mjtCatBit.mjCAT_DECOR for i in range(scene.ngeom))
    assert exp.model.ngeom == 14
    assert exp.model.nq == 0
    exp.virtual_props = False
    assert _virtual_prop_edges(exp) == []


def test_wireframe_does_not_silently_drop_geometry_at_scene_capacity():
    exp = _wireframe_fixture()
    scene = mujoco.MjvScene(exp.model, maxgeom=1)
    with pytest.raises(RuntimeError, match="capacity"):
        _add_virtual_prop_outlines(scene, exp)
