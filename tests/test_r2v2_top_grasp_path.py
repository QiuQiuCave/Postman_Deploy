"""Target calibration/relocation tests, not a whole-body reachability claim."""
import copy
import json

import numpy as np
import pytest

from common.r2v2_top_grasp_path import (
    DEFAULT_CALIBRATION_PATH, PHASES, calibration_sha256,
    build_calibration_from_artifacts,
    load_top_grasp_calibration, quaternion_wxyz, relocate_top_grasp_path,
    scene_candidates, verify_source_artifacts, world_path_for_scene,
)


def from_quaternion(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def test_bundled_calibration_is_portable_verified_and_correct_scope():
    data = load_top_grasp_calibration()
    assert DEFAULT_CALIBRATION_PATH.is_file()
    assert data["content_sha256"] == calibration_sha256(data)
    assert data["source_success"]
    assert data["object_profile"]["radius_m"] == .02
    assert data["object_profile"]["height_m"] == .12
    assert data["object_profile"]["mass_kg"] == .1
    assert not data["is_whole_body_validation"]
    assert not data["is_collision_free_certificate"]
    assert data["hand_candidate"]["spread_tilt_deg"] == 60.
    assert tuple(row["name"] for row in data["waypoints"]) == PHASES
    assert data["waypoint_pose_frame"] == "initial_object_not_current_object"
    assert set(data["source_artifacts"]) == {"report.json", "trace.json", "targets.json"}


def test_source_scene_reconstructs_target_not_actual_object_replay():
    data = load_top_grasp_calibration()
    path = relocate_top_grasp_path(data["source_initial_object_pose"], 0., data)
    assert np.allclose(path[0]["position_m"], [.1216387440701162, .035, .7005871454813921])
    assert np.allclose(path[2]["position_m"], [.1216387440701162, .035, .6005871455184108])
    for row, source in zip(path, data["waypoints"]):
        assert np.allclose(row["T_world_wrist_goal"],
                           np.asarray(data["source_initial_object_pose"]) @
                           np.asarray(source["T_initial_object_wrist_goal"]))
        assert not any(key in row for key in ("qpos", "qvel", "object_qpos"))
    # Stable command endpoints, not observed wrist following-error samples.
    assert path[3]["source_phase"] == "VERIFY"
    assert path[3]["source_phase_occurrence"] == 0
    assert path[4]["source_phase"] == "VERIFY"
    assert path[4]["source_phase_occurrence"] == 1
    assert path[7]["source_phase"] == "PLACE_SETTLE"


def test_short_approach_is_labelled_derived_and_separate_from_grasp():
    path = world_path_for_scene(.8, [.32, .18], 180.)
    assert path[1]["derived_waypoint"]
    assert not path[2]["derived_waypoint"]
    np.testing.assert_allclose(path[0]["position_m"]-path[2]["position_m"], [0., 0., .1], atol=1e-8)
    np.testing.assert_allclose(path[1]["position_m"]-path[2]["position_m"], [0., 0., .03], atol=1e-8)
    np.testing.assert_allclose(path[1]["T_world_wrist_goal"][:3, :3], path[2]["T_world_wrist_goal"][:3, :3])


def test_upright_and_placement_never_claim_contact_feedback():
    path = world_path_for_scene(.8, [.32, .18], 180.)
    for index in (4, 7):
        assert path[index]["feedback_required"]
        assert path[index]["contact_feedback_required"]
        assert "Nominal unloaded reference only" in path[index]["meaning"]
    assert path[2]["nominal_hand_command"] == 1
    assert path[7]["nominal_hand_command"] == 1  # Table contact gate precedes release.
    assert path[8]["nominal_hand_command"] == 0


@pytest.mark.parametrize("yaw", [-180., -90., 0., 90., 180., 270.])
def test_azimuth_preserves_object_relative_geometry_and_vertical(yaw):
    zero = world_path_for_scene(.8, [.32, .18], 0.)
    path = world_path_for_scene(.8, [.32, .18], yaw)
    anchor = path[0]["fixed_initial_object_pose"]
    angle = np.deg2rad(yaw)
    rz = np.array([[np.cos(angle), -np.sin(angle), 0.],
                   [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    for row, old in zip(path, zero):
        np.testing.assert_allclose(row["position_m"]-anchor[:3, 3], rz @ (old["position_m"]-anchor[:3, 3]), atol=1e-9)
        np.testing.assert_allclose(row["T_world_wrist_goal"][:3, :3], rz @ old["T_world_wrist_goal"][:3, :3], atol=1e-9)
        np.testing.assert_allclose(from_quaternion(row["quaternion_wxyz"]), row["T_world_wrist_goal"][:3, :3], atol=1e-9)
        assert np.isclose(np.linalg.norm(row["quaternion_wxyz"]), 1.)


def test_scene_table_height_and_xy_relocate_all_goals_equally():
    a = world_path_for_scene(.8, [.32, .18], 180.)
    b = world_path_for_scene(.9, [.38, .2], 180.)
    for p, q in zip(a, b):
        np.testing.assert_allclose(q["position_m"]-p["position_m"], [.06, .02, .1], atol=1e-9)
        np.testing.assert_allclose(q["quaternion_wxyz"], p["quaternion_wxyz"], atol=1e-9)


def test_path_goals_cannot_chase_later_object_state_or_alias_each_other():
    data = load_top_grasp_calibration()
    initial = np.asarray(data["source_initial_object_pose"])
    path = relocate_top_grasp_path(initial, 180., data)
    baseline = copy.deepcopy(path)
    initial[:3, 3] += [1., 1., 1.]
    data["waypoints"][0]["T_initial_object_wrist_goal"][0][3] += 3.
    for row, original in zip(path, baseline):
        np.testing.assert_array_equal(row["T_world_wrist_goal"], original["T_world_wrist_goal"])
        np.testing.assert_array_equal(row["fixed_initial_object_pose"], original["fixed_initial_object_pose"])
    path[0]["T_world_wrist_goal"][:] = 0.
    np.testing.assert_array_equal(path[1]["T_world_wrist_goal"], baseline[1]["T_world_wrist_goal"])


def test_transport_rotates_with_entire_path_not_with_chasing_object():
    for yaw, expected in [(180., [-.1, 0., 0.]), (90., [0., .1, 0.]), (-90., [0., -.1, 0.])]:
        path = world_path_for_scene(.8, [.32, .18], yaw)
        np.testing.assert_allclose(path[6]["position_m"]-path[5]["position_m"], expected, atol=1e-9)


@pytest.mark.parametrize("transform", [np.eye(3), np.zeros((4, 4)), np.full((4, 4), np.nan),
    np.diag([1., 1., -1., 1.]), np.diag([2., 1., 1., 1.])])
def test_invalid_initial_rigid_transforms_rejected(transform):
    with pytest.raises(ValueError):
        relocate_top_grasp_path(transform)


def test_tilted_can_rejected_without_silently_rotating_about_wrong_axis():
    transform = np.eye(4)
    transform[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    with pytest.raises(ValueError, match="upright"):
        relocate_top_grasp_path(transform)


@pytest.mark.parametrize("height,xy,yaw", [(float('nan'), [.3, .2], 0.),
    (-.8, [.3, .2], 0.), (.8, [1.], 0.), (.8, [float('nan'), .2], 0.),
    (.8, [.3, .2], float('inf'))])
def test_bad_scene_values_rejected(height, xy, yaw):
    with pytest.raises(ValueError):
        world_path_for_scene(height, xy, yaw)


def test_tampered_calibration_rejected_even_when_only_metadata_changed(tmp_path):
    data = load_top_grasp_calibration()
    data["source_success"] = False
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA256"):
        load_top_grasp_calibration(path)
    with pytest.raises(ValueError, match="SHA256"):
        relocate_top_grasp_path(np.eye(4), calibration=data)


def test_missing_provenance_source_raises_in_strict_optional_verification(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_source_artifacts(load_top_grasp_calibration(), tmp_path)


@pytest.mark.parametrize("mutation", ["order", "side", "pose", "feedback", "kind"])
def test_rehashed_invalid_semantics_still_rejected(tmp_path, mutation):
    data = load_top_grasp_calibration()
    if mutation == "order":
        data["waypoints"] = data["waypoints"][::-1]
    elif mutation == "side":
        data["side"] = "right"
    elif mutation == "pose":
        data["waypoints"][0]["T_initial_object_wrist_goal"][0][0] = 3.
    elif mutation == "feedback":
        data["waypoints"][4]["contact_feedback_required"] = False
    else:
        data["reference_kind"] = "actual_qpos_replay"
    data["content_sha256"] = calibration_sha256(data)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_top_grasp_calibration(path)


def test_geometric_candidates_have_no_ik_or_collision_success_claims():
    scenes = scene_candidates()
    assert len(scenes) == 18
    assert {scene["table_height_m"] for scene in scenes} == {.8, .9, 1.}
    assert {scene["yaw_deg"] for scene in scenes} == {-90., 90., 180.}
    for row in scenes:
        assert not row["reachability_verified"]
        assert not row["collision_clearance_verified"]
        assert np.all(np.asarray(row["wrist_max_xyz_m"]) >= row["wrist_min_xyz_m"])


def source_artifacts(directory):
    """Small synthetically reconstructed source for extraction failure tests."""
    calibration = load_top_grasp_calibration()
    anchor = np.asarray(calibration["source_initial_object_pose"])
    report = dict(success=True, grasp_verified=True, release_commanded=True,
                  object_free=True, object_pose_replay=False, object_weld=False,
                  profile=calibration["object_profile"], candidate=calibration["hand_candidate"],
                  layout=dict(cylinder_initial_position=anchor[:3, 3].tolist(), table_top_m=.4),
                  place_target_xy_m=[.1, 0.], scope="Synthetic extraction unit test, not physics evidence")
    targets = []
    for row in calibration["waypoints"]:
        goal = (anchor @ np.asarray(row["source_T_initial_object_wrist_goal"])).tolist()
        for _ in range(2):
            targets.append(dict(time_s=len(targets)*.01, phase=row["source_phase"],
                                command=row["nominal_hand_command"], T_world_wrist_goal=goal))
        if row["name"] == "probe":
            # Keep VERIFY plateaus distinct, as they are in the original.
            targets.append(dict(time_s=len(targets)*.01, phase="UPRIGHT", command=1,
                                T_world_wrist_goal=goal))
    trace = [dict(phase="READY", metrics=dict(T_world_object=anchor.tolist()), qpos=[12345.]),
             dict(phase="COMPLETE", metrics={}, qpos=[67890.])]
    content = {"report.json": report, "targets.json": targets, "trace.json": trace}
    for name, value in content.items():
        (directory / name).write_text(json.dumps(value))
    return content


def test_source_extractor_uses_commands_checks_digests_and_ignores_qpos(tmp_path):
    source_artifacts(tmp_path)
    data = build_calibration_from_artifacts(tmp_path)
    original = load_top_grasp_calibration()
    assert verify_source_artifacts(data, tmp_path)
    for row, reference in zip(data["waypoints"], original["waypoints"]):
        np.testing.assert_allclose(row["T_initial_object_wrist_goal"], reference["T_initial_object_wrist_goal"])
    assert "12345" not in json.dumps(data)
    assert "67890" not in json.dumps(data)
    (tmp_path / "trace.json").write_text("[]")
    with pytest.raises(ValueError, match="Source artifact SHA256 mismatch: trace.json"):
        verify_source_artifacts(data, tmp_path)


@pytest.mark.parametrize("failure", ["not_success", "replay", "wrong_anchor", "unfinished", "nonplateau"])
def test_source_extractor_rejects_invalid_fixture_evidence(tmp_path, failure):
    content = source_artifacts(tmp_path)
    if failure == "not_success":
        content["report.json"]["success"] = False
    elif failure == "replay":
        content["report.json"]["object_pose_replay"] = True
    elif failure == "wrong_anchor":
        content["report.json"]["layout"]["cylinder_initial_position"][2] += .1
    elif failure == "unfinished":
        content["trace.json"][-1]["phase"] = "FAILED"
    else:
        content["targets.json"][0] = copy.deepcopy(content["targets.json"][0])
        content["targets.json"][0]["T_world_wrist_goal"][0][3] += .1
    for name, value in content.items():
        (tmp_path / name).write_text(json.dumps(value))
    with pytest.raises(ValueError):
        build_calibration_from_artifacts(tmp_path)


@pytest.mark.parametrize("rotation", [np.eye(3), np.diag([1., -1., -1.]),
                                     np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])])
def test_quaternion_is_correct_at_pi_rotations(rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    q = quaternion_wxyz(pose)
    assert q[0] >= 0.
    np.testing.assert_allclose(from_quaternion(q), rotation, atol=1e-9)
