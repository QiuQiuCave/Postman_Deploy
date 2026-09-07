from common.path_config import PROJECT_ROOT

import mujoco
import numpy as np
import pytest

from common.r2v2_cylinder_test import CylinderExperiment, CylinderParameters


def test_object_is_free_and_only_platform_is_mocap():
    exp = CylinderExperiment()
    m = exp.model
    assert m.body("test_cylinder").mocapid[0] == -1
    assert m.joint("cylinder_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert m.neq == 10
    assert np.all(m.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    assert m.nu == 12
    assert np.count_nonzero(m.body_mocapid >= 0) == 1
    assert not exp.data.xfrc_applied.any()


@pytest.mark.parametrize("side", ["left", "right"])
def test_nominal_supported_close_then_unsupported_hold_and_release(side):
    exp = CylinderExperiment(CylinderParameters(side=side))
    for _ in range(round(exp.params.duration / exp.model.opt.timestep)):
        exp.step()
    report = exp.report()
    assert report["grasp_passed"], report
    assert report["hold_displacement_m"] < 0.005
    assert report["opposing_contact_fraction"] > 0.99
    assert report["max_hand_penetration_m"] < 0.001
    assert report["max_hand_self_penetration_m"] < 0.001
    # A released cylinder may land on the withdrawn platform (18 cm below),
    # not necessarily on the floor. Check separation, not one landing site.
    assert report["release_drop_m"] > 0.08
    assert all(not sample["fingers_normal_force_N"] for sample in exp.samples[-100:])
    assert exp.samples[-1]["support_contact"] or exp.samples[-1]["floor_contact"]
    assert not exp.data.xfrc_applied.any()


def test_open_hand_negative_control_is_not_a_grasp():
    exp = CylinderExperiment(CylinderParameters(grasp=False))
    for _ in range(round(exp.params.duration / exp.model.opt.timestep)):
        exp.step()
    report = exp.report()
    assert not report["grasp_passed"]
    assert report["opposing_contact_fraction"] == 0
    assert report["hold_min_height_m"] < 0.30


@pytest.mark.parametrize("params", [CylinderParameters(radius=-0.01),
                                    CylinderParameters(mass=float("nan")),
                                    CylinderParameters(withdraw_end=4.0)])
def test_invalid_experiment_parameters(params):
    with pytest.raises(ValueError):
        CylinderExperiment(params)
