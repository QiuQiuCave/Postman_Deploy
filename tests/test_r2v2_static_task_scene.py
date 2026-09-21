"""Passive contact acceptance and relocatable scene export, without policies."""

import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_static_task_scene import (
    build_scene_xml, export_scene, initial_checks, load_scene_config, settle_and_validate,
)
from r2v2_description.model import build_model


@pytest.fixture(scope="module")
def scene():
    cfg = load_scene_config()
    xml = build_scene_xml(cfg)
    return cfg, xml, mujoco.MjModel.from_xml_string(xml)


def test_full_static_robot_and_two_free_objects(scene):
    cfg, xml, model = scene
    original = build_model()
    assert (model.nq, model.nv, model.nu, model.neq, model.nmocap) == (14, 12, 0, 0, 0)
    for body in range(1, original.nbody):
        name = original.body(body).name
        assert model.body(name).jntnum[0] == 0
    # Geometry is retained by name even if compiler body indexing changes.
    for i in range(original.ngeom):
        target = model.geom(original.geom(i).name).id
        for key in ("geom_type", "geom_size", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(model, key)[target], getattr(original, key)[i])
    assert model.body("cargo_crate").mass[0] == pytest.approx(.4)
    assert model.body("test_cylinder").mass[0] == pytest.approx(.1)
    assert model.nmesh >= original.nmesh
    data = mujoco.MjData(model)
    assert initial_checks(model, data, cfg)["passed"]
    for name in ("pickup_table", "dropoff_table"):
        board = model.geom(name + "_top").id
        top = data.geom_xpos[board, 2] + model.geom_size[board, 2]
        assert top == pytest.approx(1.11)
        for i in range(4):
            g = model.geom(f"{name}_leg_{i}").id
            assert data.geom_xpos[g, 2] - model.geom_size[g, 2] == pytest.approx(0)
            assert data.geom_xpos[g, 2] + model.geom_size[g, 2] == pytest.approx(1.07)


@pytest.mark.parametrize("case", ["overlapping_static_tables", "can_through_bottom", "can_outside_crate", "crate_overhang", "robot_inside_table"])
def test_bad_layout_is_rejected_before_simulation(case):
    cfg = load_scene_config()
    if case == "overlapping_static_tables":
        cfg["tables"]["dropoff_table"] = cfg["tables"]["pickup_table"].copy()
    elif case == "can_through_bottom":
        cfg["can"]["position_xyz"][2] -= .01
    elif case == "can_outside_crate":
        cfg["can"]["position_xyz"][0] += .4
    elif case == "crate_overhang":
        cfg["crate"]["position_xyz"][0] += .09
        cfg["can"]["position_xyz"][0] += .09
    else:
        cfg["robot"]["position_xyz"][:2] = [.65, .5]
    model = mujoco.MjModel.from_xml_string(build_scene_xml(cfg))
    data, report = settle_and_validate(model, cfg)
    assert not report["passed"]
    assert not report["initial"]["passed"]
    assert data.time == 0


def test_passive_stack_and_relocated_xml_reproduce(scene, tmp_path, monkeypatch):
    cfg, xml, model = scene
    before = copy.deepcopy(cfg)
    data, report = settle_and_validate(model, cfg)
    assert report["passed"], report
    assert report["checks"]["supported_by_contact"]
    assert report["max_contact_penetration_m"] < .0005
    np.testing.assert_allclose(report["stable_window"]["mean_support_force_N"], [4.905, .981], rtol=.005)
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    source = tmp_path / "export"
    source.mkdir()
    path = export_scene(xml, source)
    for asset in ET.parse(path).getroot().findall(".//*[@file]"):
        relative = Path(asset.get("file"))
        assert not relative.is_absolute()
        assert (source / relative).is_file()
    relocated = tmp_path / "another_location"
    shutil.move(str(source), relocated)
    monkeypatch.chdir(tmp_path)
    reloaded = mujoco.MjModel.from_xml_path(str(relocated / "scene.xml"))
    second, fresh_report = settle_and_validate(reloaded, cfg)
    assert fresh_report["passed"], fresh_report
    np.testing.assert_allclose(second.qpos, data.qpos, atol=1e-9, rtol=0)
    np.testing.assert_allclose(second.qvel, data.qvel, atol=1e-9, rtol=0)
    assert cfg == before


@pytest.mark.parametrize("change", ["typo", "negative_mass", "nonfinite_position", "unequal_table_heights"])
def test_invalid_configuration_fails_before_compilation(change):
    cfg = load_scene_config()
    if change == "typo":
        cfg["can"]["radius_typo"] = .02
    elif change == "negative_mass":
        cfg["can"]["mass"] = -1
    elif change == "nonfinite_position":
        cfg["can"]["position_xyz"][0] = float("nan")
    else:
        cfg["tables"]["dropoff_table"][2] += .1
    with pytest.raises(ValueError):
        build_scene_xml(cfg)
