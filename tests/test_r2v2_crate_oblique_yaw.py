"""Combined down-tilt/yaw conventions; initialization is not grasp evidence."""
import numpy as np
import pytest

from common.r2v2_crate_oblique import ObliqueCrateExperiment, ObliqueParameters


@pytest.mark.parametrize("yaw", [-30., -20., -10., 0., 10., 20., 30.])
def test_combined_yaw_is_mirrored_without_losing_down_tilt_or_insertion_depth(yaw):
    exp = ObliqueCrateExperiment(ObliqueParameters(
        tip_up_deg=-20., yaw_deg=yaw, seating_m=.03, active_sides=("left", "right")),
        keep_trace=False)
    c, s = np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20))
    for side, sign in (("left", 1), ("right", -1)):
        g = exp.geometry[side]
        expected = [c*np.sin(np.deg2rad(yaw)), -sign*c*np.cos(np.deg2rad(yaw)), -s]
        np.testing.assert_allclose(g["insertion_direction_world"], expected, atol=1e-12)
        travel = g["inserted"]-g["initial"]
        np.testing.assert_allclose(travel/np.linalg.norm(travel), expected, atol=1e-12)
        np.testing.assert_allclose(g["rotation"].T @ g["rotation"], np.eye(3), atol=1e-12)
        assert min(g["per_finger_normal_insertion_m"].values()) == pytest.approx(.060)
        assert g["initial_all_hand_clearance_m"] == pytest.approx(.025)
    assert exp.crate_params.width == pytest.approx(.26)
    assert not exp.report()["pickup_verified"]
