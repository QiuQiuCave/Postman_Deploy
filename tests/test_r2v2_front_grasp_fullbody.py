"""Frontal-grasp planning/control contracts, not successful rollout evidence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.path_config import PROJECT_ROOT
from common.r2v2_front_grasp_path import (
    front_world_path_for_scene, load_front_grasp_calibration,
)
from common.r2v2_top_grasp_fullbody import (
    TopGraspFullbodyExperiment, build_fullbody_model, fingerprint, load_config,
)
from r2v2_description.model import BODY_JOINTS, JointMap, SIDES, build_model, hand_names


RECORD = PROJECT_ROOT / "reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/left/grasp_record.json"
FRONT_CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_front_grasp_fullbody.json"


def scene_config(**changes):
    result = dict(
        grasp_style="front", calibration_path=str(RECORD),
        reach_config=str(PROJECT_ROOT / "deploy_mujoco/config/r2v2_reach_wrist_v2.yaml"),
        parity_report="unused-in-scene-only-tests.json",
        table_height_m=.98, can_xy_m=[.40, .13], yaw_deg=0.,
        table_center_xy_m=[.555, .16], table_half_size_m=[.195, .26, .02],
    )
    result.update(changes)
    return result


def path(**changes):
    arguments = dict(table_height_m=.98, can_xy_m=[.40, .13], yaw_deg=0.)
    arguments.update(changes)
    return {row["name"].upper(): row for row in front_world_path_for_scene(**arguments)}


def test_front_is_an_explicit_opt_in_and_does_not_replace_the_upper_baseline():
    assert load_config()["grasp_style"] == "upper"
    assert load_config(scene_config())["grasp_style"] == "front"
    cfg = load_config(FRONT_CONFIG)
    assert cfg["grasp_style"] == "front" and cfg["yaw_deg"] == 0.
    assert cfg["calibration_path"] == str(RECORD)
    assert cfg["profile"] == "baseline_40mm_100g"


@pytest.mark.parametrize("value", ["unknown", "", True, None, 1])
def test_unknown_grasp_style_cannot_silently_fall_back_to_upper(value):
    with pytest.raises(ValueError):
        load_config(scene_config(grasp_style=value))


def test_front_style_and_calibration_change_the_air_evidence_fingerprint():
    front = scene_config()
    assert fingerprint(front) != fingerprint({**front, "grasp_style": "upper"})
    assert fingerprint(front) != fingerprint({**front, "calibration_path": str(
        PROJECT_ROOT / "deploy_mujoco/config/r2v2_top_grasp_calibration.json")})
    assert fingerprint(front) == fingerprint({**front, "air_evidence": "elsewhere.json"})


@pytest.mark.parametrize("yaw", [-180., -90., .1, 90., 180., float("nan"), True])
def test_front_mode_cannot_reintroduce_the_rejected_wrist_flip_via_yaw(yaw):
    with pytest.raises(ValueError):
        load_config(scene_config(yaw_deg=yaw))
    with pytest.raises(ValueError):
        path(yaw_deg=yaw)


def test_frontal_grasp_uses_the_recorded_initial_relation_not_the_held_relation():
    record = json.loads(RECORD.read_text())
    initial = record["keyframes"]["initial"]["cylinder_in_wrist"]
    np.testing.assert_allclose(initial["position_m"], [.145, -.035, 0.], atol=1e-12)
    np.testing.assert_allclose(initial["quaternion_wxyz"], [1., 0., 0., 0.], atol=1e-12)
    load_front_grasp_calibration(RECORD)
    target = path()["GRASP"]["T_world_wrist_goal"]
    relation = np.eye(4)
    relation[:3, 3] = initial["position_m"]
    cylinder = np.eye(4)
    cylinder[:3, 3] = [.40, .13, .98 + .06 + .001]
    np.testing.assert_allclose(target @ relation, cylinder, atol=1e-12)


@pytest.mark.parametrize("mutation", ["side", "wrist", "radius", "height", "mass", "offset", "rotation"])
def test_wrong_archived_grasp_cannot_be_used_as_front_calibration(tmp_path, mutation):
    record = json.loads(RECORD.read_text())
    if mutation == "side":
        record["side"] = "right"
    elif mutation == "wrist":
        record["wrist_body"] = "left_palm_link"
    elif mutation in ("radius", "height", "mass"):
        record["experiment_parameters"][mutation] *= 1.1
    elif mutation == "offset":
        record["keyframes"]["initial"]["cylinder_in_wrist"]["position_m"][0] += .01
    else:
        record["keyframes"]["initial"]["cylinder_in_wrist"]["quaternion_wxyz"] = [0., 1., 0., 0.]
    target = tmp_path / "grasp_record.json"
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_front_grasp_calibration(target)


def test_all_nominal_front_waypoints_keep_the_wrist_upright_without_flip():
    targets = path()
    assert tuple(targets) == ("HOVER", "APPROACH", "GRASP", "PROBE", "UPRIGHT",
                              "LIFT", "TRANSLATE", "PLACE", "RETREAT")
    for row in targets.values():
        transform = row["T_world_wrist_goal"]
        np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(row["quaternion_wxyz"], [1., 0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(targets["UPRIGHT"]["T_world_wrist_goal"],
                               targets["PROBE"]["T_world_wrist_goal"])


def test_front_approaches_along_palm_normal_not_longitudinal_finger_axis():
    targets = path()
    xyz = {name: row["T_world_wrist_goal"][:3, 3] for name, row in targets.items()}
    np.testing.assert_allclose(xyz["HOVER"] - xyz["GRASP"], [0., .05, .04], atol=1e-12)
    np.testing.assert_allclose(xyz["APPROACH"] - xyz["GRASP"], [0., .05, 0.], atol=1e-12)
    np.testing.assert_allclose(xyz["PROBE"] - xyz["GRASP"], [0., 0., .02], atol=1e-12)
    np.testing.assert_allclose(xyz["LIFT"] - xyz["GRASP"], [0., 0., .08], atol=1e-12)
    np.testing.assert_allclose(xyz["TRANSLATE"] - xyz["LIFT"], [0., -.10, 0.], atol=1e-12)
    np.testing.assert_allclose(xyz["PLACE"] - xyz["GRASP"], [0., -.10, -.001], atol=1e-12)
    np.testing.assert_allclose(xyz["RETREAT"] - xyz["PLACE"], [-.04, .12, .04], atol=1e-12)


def test_front_path_relocation_translates_every_stage_without_chasing_live_object():
    original = path()
    relocated = path(table_height_m=1.03, can_xy_m=[.44, .11])
    for name in original:
        np.testing.assert_allclose(relocated[name]["T_world_wrist_goal"][:3, 3]
                                   - original[name]["T_world_wrist_goal"][:3, 3],
                                   [.04, -.02, .05], atol=1e-12)
        np.testing.assert_allclose(relocated[name]["T_world_wrist_goal"][:3, :3],
                                   original[name]["T_world_wrist_goal"][:3, :3])


def test_nominal_feedback_waypoints_do_not_claim_verified_physical_grasp():
    calibration = load_front_grasp_calibration()
    assert calibration["initial_relation_is_calibrated"] is True
    assert calibration["transport_path_is_recorded"] is False
    assert calibration["is_whole_body_validation"] is False
    assert calibration["is_collision_free_certificate"] is False
    for row in path().values():
        assert row["derived_waypoint"] is True
        assert row["source_recorded_transport"] is False
    for name in ("UPRIGHT", "PLACE"):
        assert path()[name]["feedback_required"] is True


@pytest.fixture(scope="module", params=["air", "contact"])
def scene(request):
    return request.param, build_fullbody_model(scene_config(), request.param)


def test_front_scene_preserves_robot_and_independent_body_hand_channels(scene):
    _, (model, hands, _) = scene
    original = build_model(hands)
    for field in ("jnt_range", "jnt_type", "jnt_axis"):
        np.testing.assert_array_equal(getattr(model, field)[:original.njnt], getattr(original, field))
    for field in ("body_mass", "body_inertia"):
        np.testing.assert_array_equal(getattr(model, field)[:original.nbody], getattr(original, field))
    for field in ("geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(model, field)[:original.ngeom], getattr(original, field))
    np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)
    assert model.nmocap == 0 and not np.any(model.eq_type == mujoco.mjtEq.mjEQ_WELD)
    body = set(JointMap.create(model, BODY_JOINTS).actuators)
    hand = {side: set(JointMap.create(model, hand_names(side)).actuators) for side in SIDES}
    assert len(body) == 28 and len(hand["left"]) == len(hand["right"]) == 6
    assert not body & (hand["left"] | hand["right"])
    assert not hand["left"] & hand["right"]


def test_front_scene_air_is_noncontact_while_contact_keeps_a_real_free_can(scene):
    mode, (model, _, _) = scene
    assert model.body("test_cylinder").mass[0] == pytest.approx(.1)
    np.testing.assert_allclose(model.geom("cylinder_geom").size[:2], [.02, .06])
    if mode == "air":
        assert model.body("test_cylinder").jntnum[0] == 0
        for name in ("tabletop_geom", "cylinder_geom"):
            geom = model.geom(name).id
            assert model.geom_contype[geom] == model.geom_conaffinity[geom] == 0
    else:
        assert model.joint("cylinder_free").type == mujoco.mjtJoint.mjJNT_FREE
        assert model.geom("cylinder_geom").contype[0] != 0


def supervisor(phase="GRASP", mode="contact"):
    """Synthetic state only: these tests must never be reported as a rollout."""
    exp = TopGraspFullbodyExperiment.__new__(TopGraspFullbodyExperiment)
    exp.config, exp.mode = load_config(scene_config()), mode
    exp.data = SimpleNamespace(time=0., qvel=np.zeros(6))
    exp.phase, exp.phase_start = phase, 0.
    exp.failure = exp.failure_phase = None
    exp.completed = exp.grasp_verified = exp.release_commanded = False
    exp.stable_since = exp.bad_since = None
    exp.transitions, exp.targets, exp.stage_results = [], [], []
    exp.safety_violations_seen, exp.hand_commands, exp.policy_commands = [], [], []
    exp.feedback_target_checks = []
    exp.stage_collision = False
    exp.stage_peak = dict(position_m=0., orientation_deg=0., right_position_m=0., right_orientation_deg=0.)
    exp.path_targets = {name: row["T_world_wrist_goal"].copy() for name, row in path().items()}
    exp.active_goal = exp.path_targets["GRASP"].copy()
    exp.goal_wrist_transforms = {side: exp.active_goal.copy() for side in SIDES}
    controllers = {side: SimpleNamespace(command=0) for side in SIDES}

    def command(side, value):
        controllers[side].command = value
        exp.hand_commands.append((side, value))

    exp.hands = SimpleNamespace(controllers=controllers, command=command)
    exp.policy = SimpleNamespace(set_wrist_target_world=lambda *args: exp.policy_commands.append(args))
    exp.current_metrics = dict(wrist_errors={side: dict(position_m=.004, orientation_deg=2.,
        linear_speed_mps=.01) for side in SIDES}, opposed_contact=False)
    return exp


def test_front_air_remains_open_through_every_grasp_related_waypoint():
    exp = supervisor("HOVER", "air")
    for name in exp.path_targets:
        assert exp.phase == name
        exp.data.time = exp.phase_start
        exp._gate()
        exp.data.time += .31
        exp._gate()
    assert exp.phase == "COMPLETE"
    assert not exp.hand_commands and not exp.grasp_verified and not exp.release_commanded


def test_front_contact_cannot_close_just_because_the_nominal_waypoint_is_grasp():
    exp = supervisor()
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .03
    for time in (0., .5):
        exp.data.time = time
        exp._gate()
    assert exp.phase == "GRASP" and not exp.hand_commands
    exp.current_metrics["wrist_errors"]["left"]["position_m"] = .004
    for time in (.6, .91):
        exp.data.time = time
        exp._gate()
    assert exp.phase == "CLOSE" and exp.hand_commands == [("left", 1)]
    assert not exp.grasp_verified


def test_front_contact_retreat_clears_fingers_laterally_and_backward_after_release():
    exp = supervisor("OPEN")
    exp.weight_N, exp.place_xy = .981, np.array([.40, .03])
    exp.active_goal = exp.path_targets["PLACE"].copy()
    before = exp.active_goal.copy()
    exp.current_metrics.update(finite_state=True, table_contact=True, table_vertical_force_N=.981,
        floor_contact=False, object_tilt_deg=0., object_linear_speed_mps=0.,
        object_angular_speed_radps=0., object_position_m=[.40, .03, 1.04],
        active_hand_object_contact_count=0)
    for time in (2.5, 2.81):
        exp.data.time = time
        exp._gate()
    assert exp.phase == "RETREAT"
    np.testing.assert_allclose(exp.active_goal[:3, 3] - before[:3, 3], [-.04, .12, .04], atol=1e-12)
    assert not exp.hand_commands


def test_front_placement_uses_measured_held_relation_not_nominal_initial_relation():
    exp = supervisor("TRANSLATE")
    exp.weight_N, exp.place_xy, exp.table_height = .981, np.array([.40, .03]), .98
    relation = np.eye(4)
    angle = np.deg2rad(2.)
    relation[:3, :3] = [[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]]
    relation[:3, 3] = [.142, -.035, .001]
    exp.current_metrics.update(finite_state=True, opposed_contact=True,
        clearance_m=.08, table_contact=False, floor_contact=False,
        hand_table_contact_count=0, parked_hand_contact_count=0,
        hand_vertical_force_N=.981, object_tilt_deg=2., object_linear_speed_mps=0.,
        object_angular_speed_radps=0., T_wrist_object=relation.tolist())
    for time in (0., 1.01):
        exp.data.time = time
        exp._gate()
    assert exp.phase == "PLACE" and exp.feedback_target_checks[-1]["accepted"]
    desired = np.eye(4)
    desired[:3, 3] = [.40, .03, 1.04]
    np.testing.assert_allclose(exp.active_goal @ relation, desired, atol=1e-12)
    assert not np.allclose(exp.active_goal, exp.path_targets["PLACE"])
    assert not exp.hand_commands and not exp.release_commanded
