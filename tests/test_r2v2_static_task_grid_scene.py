"""Delivery2 capacity, independent rigid bodies, per-can support and export."""

import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_static_task_grid_scene import (
    build_scene_xml, grid_layout, initial_checks, load_scene_config as load_current_config, settle_and_validate, validate_config,
)
from common.r2v2_static_task_scene import export_scene, build_scene_xml as single_xml
from common.r2v2_static_task_scene import load_scene_config as load_single_config


def load_scene_config():
    """Keep regression coverage for the previously delivered 35-can layout."""
    cfg = load_single_config()
    cfg["cargo_grid"] = dict(rows=5, columns=7, gap_m=.005, minimum_wall_clearance_m=.006)
    return validate_config(cfg)


@pytest.fixture(scope="module")
def scene():
    cfg = load_scene_config()
    xml = build_scene_xml(cfg)
    return cfg, xml, mujoco.MjModel.from_xml_string(xml)


def test_capacity_and_shared_visuals_with_independent_dynamics(scene):
    cfg, _, model = scene
    layout = grid_layout(cfg)
    assert layout["count"] == 35
    assert layout["max_rows_columns_with_current_gaps"] == [5, 7]
    np.testing.assert_allclose(layout["footprint_xy_m"], [.22, .31])
    np.testing.assert_allclose(layout["floor_wall_clearance_xy_m"], [.006, .021])
    np.testing.assert_allclose(layout["safe_envelope_clearance_xy_m"], [.006, .015])
    assert (model.nq, model.nv, model.njnt, model.nu, model.neq, model.nmocap) == (252, 216, 36, 0, 0, 0)
    assert np.all(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    for item in layout["cans"]:
        body = model.body(item["body"])
        assert body.mass[0] == pytest.approx(.1)
        assert body.jntnum[0] == 1
        assert model.geom(item["geom"]).contype[0] == 1
        assert model.geom(item["geom"]).conaffinity[0] == 1
    base = copy.deepcopy(cfg)
    base.pop("cargo_grid")
    original = mujoco.MjModel.from_xml_string(single_xml(base))
    assert (model.nmesh, model.ntex, model.nmat) == (original.nmesh, original.ntex, original.nmat)
    assert len({model.geom(g).name for g in range(model.ngeom)}) == model.ngeom
    data = mujoco.MjData(model)
    initial = initial_checks(model, data, cfg, layout)
    assert initial["passed"], initial
    assert initial["can_pair_count"] == 595
    assert initial["minimum_can_surface_gap_m"] == pytest.approx(.005)


@pytest.mark.parametrize("change", ["extra_row", "extra_column", "oversized_can", "zero_gap", "shift_grid", "through_floor", "above_rim"])
def test_unsafe_grid_configurations_rejected(change):
    cfg = load_scene_config()
    if change == "extra_row":
        cfg["cargo_grid"]["rows"] += 1
    elif change == "extra_column":
        cfg["cargo_grid"]["columns"] += 1
    elif change == "oversized_can":
        cfg["can"]["radius"] = .025
    elif change == "zero_gap":
        cfg["cargo_grid"]["gap_m"] = 0
    elif change == "shift_grid":
        cfg["can"]["position_xyz"][0] += .01
    elif change == "through_floor":
        cfg["can"]["position_xyz"][2] -= .01
    else:
        cfg["can"]["position_xyz"][2] += .05
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_coincident_cans_fail_initial_checks(scene):
    cfg, _, model = scene
    data = mujoco.MjData(model)
    layout = grid_layout(cfg)
    a, b = [model.joint(item["joint"]).qposadr[0] for item in layout["cans"][:2]]
    data.qpos[b:b + 3] = data.qpos[a:a + 3]
    result = initial_checks(model, data, cfg, layout)
    assert not result["passed"]
    assert not result["checks"]["configured_can_gaps"]


def test_every_can_is_supported_and_relocated_export_repeats(scene, tmp_path, monkeypatch):
    cfg, xml, model = scene
    before = copy.deepcopy(cfg)
    data, report = settle_and_validate(model, cfg)
    assert report["passed"], report["checks"]
    mean_force = report["stable_window"]["mean_support_force_N"]
    assert mean_force[0] == pytest.approx(3.9 * 9.81, rel=.005)
    np.testing.assert_allclose(mean_force[1:], np.full(35, .981), rtol=.005)
    assert report["max_grid_xy_deviation_m"] < .0005
    assert not report["unexpected_contacts"]
    source = tmp_path / "export"
    source.mkdir()
    scene_path = export_scene(xml, source)
    for item in ET.parse(scene_path).getroot().findall('.//*[@file]'):
        assert not Path(item.get("file")).is_absolute()
        assert (source / item.get("file")).is_file()
    destination = tmp_path / "relocated"
    shutil.move(str(source), destination)
    monkeypatch.chdir(tmp_path)
    other = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    fresh, fresh_report = settle_and_validate(other, cfg)
    assert fresh_report["passed"], fresh_report["checks"]
    np.testing.assert_allclose(fresh.qpos, data.qpos, rtol=0, atol=1e-9)
    np.testing.assert_allclose(fresh.qvel, data.qvel, rtol=0, atol=1e-9)
    assert cfg == before


def test_three_can_shelf_scene_support_and_portable_reload(tmp_path):
    cfg = load_current_config()
    layout = grid_layout(cfg)
    assert layout["count"] == 3
    np.testing.assert_allclose([c["position_xyz"] for c in layout["cans"]],
                               [[.615, .4, 1.1754], [.615, .5, 1.1754], [.615, .6, 1.1754]])
    assert layout["surface_gap_m"] == pytest.approx(.06)
    xml = build_scene_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    assert (model.nq, model.nv, model.njnt, model.nu) == (28, 24, 4, 0)
    assert model.body("shelf").jntnum[0] == 0
    np.testing.assert_allclose(model.geom("shelf_level_3_deck").size, [.65, .24, .035])
    data, report = settle_and_validate(model, cfg)
    assert report["passed"], report
    assert report["initial"]["checks"]["fixed_collision_shelf"]
    np.testing.assert_allclose(report["stable_window"]["mean_support_force_N"],
                               np.array([.7, .1, .1, .1]) * 9.81, rtol=.005)
    source = tmp_path / "source"
    source.mkdir()
    export_scene(xml, source)
    moved = tmp_path / "moved"
    shutil.move(source, moved)
    for item in ET.parse(moved / "scene.xml").getroot().findall('.//*[@file]'):
        assert not Path(item.get("file")).is_absolute()
        assert (moved / item.get("file")).is_file()
    fresh, result = settle_and_validate(mujoco.MjModel.from_xml_path(str(moved / "scene.xml")), cfg)
    assert result["passed"], result
    np.testing.assert_allclose(fresh.qpos, data.qpos, rtol=0, atol=1e-9)
    np.testing.assert_allclose(fresh.qvel, data.qvel, rtol=0, atol=1e-9)


@pytest.mark.parametrize("position", [[.65, -2.5, 0], [0, 0, 0]])
def test_shelf_cannot_intersect_table_or_robot(position):
    cfg = load_current_config()
    cfg["shelf"]["position_xyz"] = position
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data, report = settle_and_validate(model, cfg)
    assert not report["passed"]
    assert not report["initial"]["checks"]["no_initial_scene_penetration"]
    assert data.time == 0
