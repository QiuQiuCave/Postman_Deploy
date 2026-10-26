"""Independent geometry variant, mass accounting, and native export checks."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import CrateParameters, add_crate
from common.r2v2_vertical_grip_crate import (
    VerticalGripCrateParameters, add_vertical_grip_crate, load_vertical_grip_crate_config,
)
from tools.render_r2v2_vertical_grip_crate import standalone


def test_same_shell_dimensions_with_two_vertical_grips_and_four_rods():
    p = load_vertical_grip_crate_config()
    assert (p.width, p.depth, p.height) == (.36, .24, .16)
    assert p.overall_width == pytest.approx(.52)
    assert p.unobstructed_grip_span == pytest.approx(.096)
    root = ET.Element("mujoco")
    body = add_vertical_grip_crate(root, p)
    assert len(body.findall("geom")) == 11
    assert len(body.findall("freejoint")) == 1
    assert not body.findall('.//geom[@type="mesh"]')
    for side, sign in (("left", 1), ("right", -1)):
        wall = body.find(f'geom[@name="crate_{side}_wall"]')
        assert wall is not None
        bar = body.find(f'geom[@name="crate_{side}_vertical_grip"]')
        end = np.fromstring(bar.get("fromto"), sep=" ").reshape(2, 3)
        np.testing.assert_allclose(end[:, :2], [[0., sign*.24]]*2)
        assert end[1, 2]-end[0, 2] == pytest.approx(.12)
        assert abs(end[0, 1])-.02-p.width/2 == pytest.approx(.04)
        for name in ("lower", "upper"):
            rod = body.find(f'geom[@name="crate_{side}_{name}_connector"]')
            ends = np.fromstring(rod.get("fromto"), sep=" ").reshape(2, 3)
            assert ends[0, 2] == ends[1, 2]
            assert abs(ends[0, 1]) < p.width/2
            assert ends[1, 1] == end[0, 1]


def test_export_compiles_and_settles_without_support_or_external_assets():
    p = VerticalGripCrateParameters()
    model, data, xml = standalone(p)
    assert not ET.fromstring(xml).findall('.//*[@file]')
    assert (model.nq, model.nv, model.nu, model.neq) == (7, 6, 0, 0)
    assert model.body("cargo_crate").mass[0] == pytest.approx(.4)
    assert np.all(model.body("cargo_crate").inertia > 0)
    np.testing.assert_allclose(model.body("cargo_crate").ipos[:2], 0., atol=1e-12)
    assert np.linalg.norm(data.qvel) < 1e-6
    assert abs(data.body("cargo_crate").xpos[2]) < 1e-4
    assert not data.warning.number.any()
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()


def test_old_hole_crate_still_unchanged_and_variants_cannot_overlap_names():
    root = ET.Element("mujoco")
    old = add_crate(root, CrateParameters())
    assert old.find('geom[@name="crate_left_handle_beam"]') is not None
    assert old.find('geom[@name="crate_left_vertical_grip"]') is None
    with pytest.raises(ValueError, match="already exists"):
        add_vertical_grip_crate(root)


@pytest.mark.parametrize("field,value", [
    ("grip_length", .2), ("wall_grip_clearance", -.01),
    ("connector_diameter", .08), ("mass", float("nan")),
    ("grip_assembly_mass_fraction", 1.), ("wall_thickness", .2),
])
def test_reject_invalid_geometry(field, value):
    with pytest.raises(ValueError):
        replace(VerticalGripCrateParameters(), **{field: value})
