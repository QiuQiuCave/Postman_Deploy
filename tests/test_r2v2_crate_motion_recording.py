import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from common.r2v2_crate_motion_recording import (
    FINGER_KEYS, SIDES, TRANSFORM_KEYS, interpolate_transform, load_crate_motion,
    pose_from_transform, record_crate_motion, slerp_wxyz, transform_from_pose,
)


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    turns = [{"time_s": 0., "state": "READY"}, {"time_s": .01, "state": "CLOSE"},
             {"time_s": .02, "state": "COMPLETE"}]
    layout = {"wrist_quaternions": {side: [np.sqrt(.5), 0., 0., np.sqrt(.5)] for side in SIDES}}
    report = {
        "lift_passed": True, "pickup_verified": True, "experiment_completed": True,
        "phase": "COMPLETE", "failure": None, "runtime_error": None,
        "crate_welds": 0, "crate_runtime_resets": 0, "no_external_crate_forces": True,
        "max_pickup_hold_s": 2., "max_level_hold_s": 2., "criteria": {},
        "candidate": {"active_sides": ["left", "right"], "tip_up_deg": -20., "yaw_deg": 15.},
        "parameters": {"grasp_enabled": True}, "crate_parameters": {"width": .26},
        "hand_configuration": {"control_dt": .01, "simulation_dt": .001, "channels": list("abcdef")},
        "layout": layout, "transitions": turns,
        "final_metrics": {"clearance_m": .1, "crate_tilt_deg": 2., "grasp_slip_m": .0003,
            "table_vertical_force_N": 0., "table_bearing_contact": False,
            "crate_linear_speed_m_s": 0., "crate_angular_speed_rad_s": 0.},
    }
    rows = []
    for i, when in enumerate([0., .01, .02]):
        crate = transform_from_pose([1., 2., 3.+i*.01], [np.sqrt(.5), 0., 0., np.sqrt(.5)])
        wrists, hands = {}, {}
        for index, side in enumerate(SIDES):
            wrist = crate @ transform_from_pose([.1, index*.2, .3+i*.01], [1., 0., 0., 0.])
            wrists[side] = {"T_world_wrist": wrist.tolist()}
            hands[side] = {"command": int(i == 2), "wrist_target_position_m": (wrist[:3, 3]+[0., 0., .001]).tolist(),
                          "q_rad": [i*.1]*6, "q_ref_rad": [i*.2]*6, "torque_Nm": [i*.3]*6}
        rows.append({"time_s": when, "phase": "READY" if i < 2 else "CLOSE",
                     "metrics": {"T_world_crate": crate.tolist(), "hands": wrists}, "hands": hands})
    rows.append(copy.deepcopy(rows[-1]))
    rows[-1]["phase"] = "COMPLETE"
    for name, data in (("report.json", report), ("trace.json", rows), ("transitions.json", turns)):
        (directory / name).write_text(json.dumps(data))
    return directory


def test_export_is_lossless_unique_and_pickle_free(source, tmp_path):
    manifest_path = record_crate_motion(source, tmp_path / "out")
    motion = load_crate_motion(manifest_path.parent)
    m, a = motion.manifest, motion.arrays
    assert m["source_samples"] == 4 and m["samples"] == 3 and m["duplicate_timestamp_rows_removed"] == 1
    np.testing.assert_array_equal(a["time_s"], [0., .01, .02])
    assert a["phase"].tolist() == ["READY", "READY", "COMPLETE"]
    assert a["phase"].dtype.kind == "U" and a["hand_command"].dtype == np.int8
    assert m["transitions"] == json.loads((source / "transitions.json").read_text())
    for name, entry in m["source_files"].items():
        assert entry["sha256"] == hashlib.sha256((source / name).read_bytes()).hexdigest()
    raw = json.loads((source / "trace.json").read_text())
    for i in range(3):
        np.testing.assert_array_equal(a["T_world_crate"][i], raw[i]["metrics"]["T_world_crate"])
        for side_index, side in enumerate(SIDES):
            np.testing.assert_array_equal(a["T_world_wrist"][i, side_index], raw[i]["metrics"]["hands"][side]["T_world_wrist"])
    with np.load(manifest_path.parent / m["trajectory_file"], allow_pickle=False) as archive:
        for value in archive.values():
            assert value.dtype.kind != "O"
    for key in (*TRANSFORM_KEYS, *FINGER_KEYS):
        assert a[key].dtype == np.float64
    assert not any("velocity" in key for key in a)
    with pytest.raises(FileExistsError):
        record_crate_motion(source, manifest_path.parent)


def test_world_anchor_crate_reconstruction_and_real_lift(source, tmp_path):
    motion = load_crate_motion(record_crate_motion(source, tmp_path / "out"))
    a = motion.arrays
    np.testing.assert_allclose(a["T_world_crate"][:, None] @ a["T_crate_wrist"], a["T_world_wrist"], atol=1e-12)
    np.testing.assert_allclose(a["T_anchor_crate"][:, None] @ a["T_crate_wrist_target"], a["T_anchor_wrist_target"], atol=1e-12)
    new_anchor = transform_from_pose([.4, .1, 1.], [1., 0., 0., 0.])
    for i, when in enumerate(a["time_s"]):
        np.testing.assert_allclose(motion.world_wrist_targets(when, new_anchor), new_anchor @ a["T_anchor_wrist_target"][i])
        np.testing.assert_allclose(motion.world_wrist_targets(when, new_anchor, False), new_anchor @ a["T_anchor_wrist"][i])
    lift = motion.world_wrist_targets(.02, new_anchor)[:, 2, 3] - motion.world_wrist_targets(0., new_anchor)[:, 2, 3]
    np.testing.assert_allclose(lift, .04)
    relative_only_lift = a["T_crate_wrist_target"][-1, :, 2, 3]-a["T_crate_wrist_target"][0, :, 2, 3]
    np.testing.assert_allclose(relative_only_lift, .02)


def test_exact_commands_are_not_one_sample_late(source, tmp_path):
    motion = load_crate_motion(record_crate_motion(source, tmp_path / "out"))
    before, at = motion.sample(.01-1e-8), motion.sample(.01)
    assert before["phase"] == "READY" and at["phase"] == "CLOSE"
    assert before["hand_command"].tolist() == [0, 0]
    assert at["hand_command"].tolist() == [1, 1]
    assert at["hand_command_preceding_step"].tolist() == [0, 0]
    midpoint = motion.sample(.015)
    np.testing.assert_allclose(midpoint["hand_reference_q_rad"], .3)
    np.testing.assert_allclose(midpoint["hand_measured_q_rad"], .15)
    np.testing.assert_allclose(midpoint["hand_torque_Nm"], .3)
    assert motion.sample(-1.)["time_s"] == 0.
    assert motion.sample(4.)["time_s"] == .02
    assert motion.sample(.02)["phase"] == "COMPLETE"
    with pytest.raises(ValueError):
        motion.sample(float("nan"))


@pytest.mark.parametrize("quat", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                  [-.5, .5, -.5, .5], [.01, .2, .5, .9]])
def test_pose_tuple_and_wxyz_roundtrip(quat):
    t = transform_from_pose([.1, -.2, .3], quat)
    pose = pose_from_transform(t)
    assert isinstance(pose, tuple) and len(pose) == 2
    assert pose[0].shape == (3,) and pose[1].shape == (4,)
    np.testing.assert_allclose(transform_from_pose(*pose), t, atol=1e-14)
    pose[0][0] = 999
    assert t[0, 3] == .1


def test_interpolation_uses_shortest_rotation_and_linear_position():
    q1 = [np.cos(np.deg2rad(85)), 0, 0, np.sin(np.deg2rad(85))]
    q2 = [np.cos(np.deg2rad(-85)), 0, 0, np.sin(np.deg2rad(-85))]
    first = transform_from_pose([0, 0, 0], q1)
    second = transform_from_pose([2, 4, 6], q2)
    middle = interpolate_transform(first, second, .5)
    np.testing.assert_allclose(middle[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(middle[:3, :3], np.diag([-1, -1, 1]), atol=1e-14)
    np.testing.assert_allclose(slerp_wxyz(q1, -np.asarray(q1), .5), q1, atol=1e-14)
    with pytest.raises(ValueError):
        interpolate_transform(first, second, 2.)


@pytest.mark.parametrize("field,value", [("lift_passed", False), ("pickup_verified", False),
    ("experiment_completed", False), ("failure", "failed"), ("crate_welds", 1),
    ("crate_runtime_resets", 1), ("no_external_crate_forces", False)])
def test_source_must_be_verified_free_crate_lift(source, tmp_path, field, value):
    report = json.loads((source / "report.json").read_text())
    report[field] = value
    (source / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        record_crate_motion(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_or_unordered_samples_rejected(source, tmp_path):
    rows = json.loads((source / "trace.json").read_text())
    rows.pop(1)
    (source / "trace.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="complete"):
        record_crate_motion(source, tmp_path / "out")


def test_integrity_mismatch_rejected(source, tmp_path):
    manifest = record_crate_motion(source, tmp_path / "out")
    archive = manifest.parent / "trajectory.npz"
    archive.write_bytes(archive.read_bytes()+b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        load_crate_motion(manifest)


def test_real_archived_down20_yaw15_trajectory_when_present():
    archive = Path(__file__).resolve().parents[1] / "reference_motion_bank/r2v2_crate/down20_yaw15_60mm/manifest.json"
    if not archive.is_file():
        pytest.skip("Real fixture recording has not been exported yet")
    motion = load_crate_motion(archive)
    assert motion.manifest["samples"] == 1285
    assert motion.manifest["source_samples"] == 1286
    assert motion.manifest["candidate"]["tip_up_deg"] == -20.
    assert motion.manifest["candidate"]["yaw_deg"] == 15.
    assert motion.manifest["candidate"]["insertion_m"] == .06
    np.testing.assert_allclose(motion.arrays["time_s"], np.arange(1285)*.01, atol=2e-12)
    assert motion.manifest["validation"]["final_metrics"]["clearance_m"] > .1
    assert motion.manifest["validation"]["final_metrics"]["table_vertical_force_N"] == 0.
    close = next(event["time_s"] for event in motion.manifest["transitions"] if event["state"] == "CLOSE")
    assert motion.sample(close)["hand_command"].tolist() == [1, 1]
    assert motion.sample(close)["hand_command_preceding_step"].tolist() == [0, 0]
