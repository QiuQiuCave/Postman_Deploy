"""The exported crate is usable without policy weights or robot mesh paths."""

from common.path_config import PROJECT_ROOT

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.r2v2_crate import CrateParameters
from common.r2v2_tabletop_scene import build_tabletop_model, build_tabletop_xml
from tools.render_r2v2_crate_preview import standalone


def test_standalone_export_is_self_contained_and_naturally_settles():
    model, data, report, xml = standalone(CrateParameters())
    root = ET.fromstring(xml)
    assert not root.findall('.//*[@file]')
    exported = mujoco.MjModel.from_xml_string(xml)
    assert (exported.nq, exported.nv, exported.nu, exported.neq) == (7, 6, 0, 0)
    assert exported.body("cargo_crate").jntnum[0] == 1
    assert report["contact_count"] >= 3
    assert not data.warning.number.any()
    assert np.linalg.norm(data.qvel) < 1e-6
    assert abs(data.body("cargo_crate").xpos[2]) < 1e-4
    assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
    np.testing.assert_array_equal(exported.body_mass, model.body_mass)
    np.testing.assert_array_equal(exported.body_inertia, model.body_inertia)


def test_reusable_tabletop_xml_does_not_enable_crate_or_change_the_default_model():
    cfg = {"simulation_dt": .001, "hand_dt": .01}
    xml, hands = build_tabletop_xml(cfg, {})
    model, old_hands = build_tabletop_model(cfg, {})
    compiled = mujoco.MjModel.from_xml_string(xml)
    assert hands == old_hands
    assert 'name="cargo_crate"' not in xml
    for key in ("body_mass", "body_inertia", "geom_size", "geom_pos", "geom_friction",
                "jnt_range", "actuator_ctrlrange", "eq_data"):
        np.testing.assert_array_equal(getattr(compiled, key), getattr(model, key))
