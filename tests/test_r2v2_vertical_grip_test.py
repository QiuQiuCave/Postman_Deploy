import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_crate_lift_metrics import _geom_minimum_z
from common.r2v2_vertical_grip_test import GripTestParameters, VerticalGripExperiment, build_scene


def test_independent_dynamic_wrists_and_unassisted_crate():
    m, cfg, layout, xml = build_scene(GripTestParameters())
    root = ET.fromstring(xml)
    assert (m.nq, m.nv, m.nu, m.neq, m.nmocap) == (43, 40, 12, 12, 2)
    for weld in root.findall("equality/weld"):
        assert "wrist" in weld.get("body2")
        assert "cargo_crate" not in (weld.get("body1"), weld.get("body2"))
    assert m.body("cargo_crate").mass[0] == pytest.approx(.4)
    for side, sign in (("left", 1), ("right", -1)):
        np.testing.assert_allclose(layout["target"][side], [-.145, sign*.275, .48])
        np.testing.assert_allclose(layout["initial"][side], [-.145, sign*.355, .48])
        assert cfg["hands"][side]["open"][0] == pytest.approx(np.deg2rad(75))


def test_initial_geometry_is_not_intersecting_crate_or_table():
    exp = VerticalGripExperiment()
    assert exp.metrics["hand_crate_penetration_m"] == 0
    assert exp.metrics["hand_table_penetration_m"] == 0
    assert not exp.data.xfrc_applied.any()
    assert not exp.data.qfrc_applied.any()
    assert not exp.report()["passed"]


@pytest.mark.parametrize("time,travel,lift", [(0, 0, 0), (3.5, 1, 0), (8.5, 1, .02),
                                            (13., 1, .1), (19.5, 1, 0), (24.5, 0, 0)])
def test_motion_endpoints_do_not_move_the_free_crate(time, travel, lift):
    exp = VerticalGripExperiment()
    q = exp.data.qpos.copy()
    exp.data.time = time
    exp.motion()
    for side in ("left", "right"):
        a, b = exp.layout["initial"][side], exp.layout["target"][side]
        np.testing.assert_allclose(exp.data.mocap_pos[exp.mocap[side]], a+travel*(b-a)+[0,0,lift])
    np.testing.assert_array_equal(q, exp.data.qpos)


@pytest.mark.parametrize("angle", [0., .4, np.pi/2, np.pi])
def test_cylinder_bottom_matches_sampled_surface(angle):
    q = [np.cos(angle/2), np.sin(angle/2), 0, 0]
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom name="bar" type="cylinder" '
        f'pos="0 0 1" quat="{" ".join(map(str,q))}" size=".02 .06"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    theta = np.linspace(0, 2*np.pi, 4097)
    points = np.concatenate([np.column_stack((.02*np.cos(theta), .02*np.sin(theta), np.full_like(theta,z))) for z in (-.06,.06)])
    expected = np.min((points@d.geom_xmat[0].reshape(3,3).T+d.geom_xpos[0])[:,2])
    assert _geom_minimum_z(m,d,0) == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("values", [{"yaw_deg": float("nan")}, {"grasp": 1}, {"approach_m": 0}, {"wrist_z_offset_m": .1}])
def test_invalid_parameters(values):
    with pytest.raises(ValueError):
        GripTestParameters(**values)
