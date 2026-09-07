from common.path_config import PROJECT_ROOT

import hashlib
import json

import numpy as np
import pytest

from common.r2v2_cylinder_test import CylinderExperiment, CylinderParameters
from common.r2v2_grasp_recording import GraspRecorder, relative_transform


@pytest.mark.parametrize("side, y", [("left", -0.035), ("right", 0.035)])
def test_initial_recorded_hand_object_transform(side, y):
    recorder = GraspRecorder(CylinderExperiment(CylinderParameters(side=side)))
    arrays = recorder.arrays()
    assert arrays["time_s"].tolist() == [0]
    assert arrays["reference_q_rad"].shape == (1, 6)
    assert arrays["measured_q_rad"].shape == (1, 11)
    np.testing.assert_allclose(arrays["T_wrist_cylinder"][0, :3, 3], [0.145, y, 0], atol=1e-12)
    np.testing.assert_allclose(arrays["T_wrist_cylinder"][0, :3, :3], np.eye(3), atol=1e-12)
    assert len(set(arrays["measured_joint_names"])) == 11


def test_relative_transform_uses_parent_rotation_not_just_translation():
    parent, child = np.eye(4), np.eye(4)
    parent[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    parent[:3, 3], child[:3, 3] = [1, 2, 3], [1, 3, 3]
    relative = relative_transform(parent, child)
    np.testing.assert_allclose(relative[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(parent @ relative, child)


def test_recorder_uses_current_qpos_without_mutating_live_derived_state():
    exp = CylinderExperiment()
    recorder = GraspRecorder(exp)
    free = exp.model.joint("cylinder_free").qposadr[0]
    exp.data.qpos[free:free+3] += [0.01, 0.02, 0.03]
    exp.data.qpos[free+3:free+7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    exp.data.time = 0.01
    old_xpos, old_xmat = exp.data.xpos.copy(), exp.data.xmat.copy()
    recorder.capture()
    row = recorder.rows[-1]
    np.testing.assert_allclose(row["T_wrist_cylinder"][:3, 3], [0.155, -0.015, 0.03], atol=1e-12)
    np.testing.assert_allclose(row["T_wrist_cylinder"][:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_array_equal(exp.data.xpos, old_xpos)
    np.testing.assert_array_equal(exp.data.xmat, old_xmat)
    with pytest.raises(ValueError, match="strictly increase"):
        recorder.capture()


def test_recording_is_passive_and_archive_loads_without_pickle(tmp_path):
    recorded, baseline = CylinderExperiment(), CylinderExperiment()
    recorder = GraspRecorder(recorded)
    for _ in range(1100):
        recorded.step()
        baseline.step()
        if recorded.steps % recorded.decimation == 0:
            recorder.capture()
        np.testing.assert_array_equal(recorded.data.qpos, baseline.data.qpos)
        np.testing.assert_array_equal(recorded.data.qvel, baseline.data.qvel)
        np.testing.assert_array_equal(recorded.data.ctrl, baseline.data.ctrl)
    metadata = recorder.save(tmp_path, {"completed": False, "scope": "short unit test"})
    path = tmp_path / "grasp_trajectory.npz"
    assert metadata["trajectory_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["measured_q_rad"].shape == (111, 11)
        np.testing.assert_allclose(np.diff(arrays["time_s"]), 0.01, atol=1e-12)
        np.testing.assert_allclose(arrays["T_world_wrist"] @ arrays["T_wrist_cylinder"],
                                   arrays["T_world_cylinder"], atol=1e-12)
        np.testing.assert_allclose(arrays["reference_q_rad"][:, 0], np.deg2rad(75), atol=1e-12)
        for name in arrays.files:
            if arrays[name].dtype.kind == "f":
                assert np.all(np.isfinite(arrays[name])), name
    assert set(metadata["keyframes"]) == {"initial", "pre_close"}
    assert json.loads((tmp_path / "grasp_record.json").read_text())["samples"] == 111
    with pytest.raises(FileExistsError):
        recorder.save(tmp_path, {})


@pytest.mark.parametrize("side", ["left", "right"])
def test_archived_thumb75_baseline_is_complete_and_consistent(side):
    root = PROJECT_ROOT / "reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g" / side
    metadata = json.loads((root / "grasp_record.json").read_text())
    path = root / metadata["trajectory_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["trajectory_sha256"]
    assert metadata["validation"]["grasp_passed"]
    assert metadata["samples"] == 1501
    with np.load(path, allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["time_s"], np.arange(1501) * 0.01, atol=1e-10)
        assert arrays["reference_q_rad"].shape == (1501, 6)
        assert arrays["measured_q_rad"].shape == (1501, 11)
        np.testing.assert_allclose(arrays["T_world_wrist"] @ arrays["T_wrist_cylinder"],
                                   arrays["T_world_cylinder"], atol=1e-12)
        np.testing.assert_allclose(arrays["reference_q_rad"][:, 0], np.deg2rad(75), atol=1e-12)
        changes = np.flatnonzero(np.diff(arrays["command"])) + 1
        np.testing.assert_allclose(arrays["time_s"][changes], [0.51, 12.01], atol=1e-10)
        assert arrays["command"][changes].tolist() == [1, 0]
        assert arrays["reference_joint_names"].tolist() == metadata["reference_joint_names"]
        assert arrays["measured_joint_names"].tolist() == metadata["measured_joint_names"]
        for frame in metadata["keyframes"].values():
            np.testing.assert_allclose(arrays["T_wrist_cylinder"][frame["index"], :3, 3],
                                       frame["cylinder_in_wrist"]["position_m"])
        for name in arrays.files:
            if arrays[name].dtype.kind == "f":
                assert np.all(np.isfinite(arrays[name])), name
