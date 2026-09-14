"""Contracts for the frozen-policy height screen and new-scene state restore.

Pure contract tests always run. Two real-model initialization checks additionally
use the locally pinned 3500 artifacts when present, without advancing physics or
changing training. Missing local experiment data is an explicit skip.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate_height_path import HeightPath
from common.r2v2_crate_height_state import initialize_from_common
from common.r2v2_crate_motion_recording import transform_from_pose
from common.r2v2_crate_height_sweep import HeightSweepExperiment


ARTIFACT_ROOT = Path("/root/autodl-tmp/Postman_Deploy")
PREPARED = ARTIFACT_ROOT / "crate_height_sweep_20260911/preparation/prepared_state.json"
CONFIG = ARTIFACT_ROOT / "simple_dual_reach_20260911/policy/reach_wrist_v2.yaml"
PARITY = ARTIFACT_ROOT / "simple_dual_reach_20260911/parity/report.json"


@pytest.fixture(scope="module")
def prepared_trials():
    if not all(path.is_file() for path in (PREPARED, CONFIG, PARITY)):
        pytest.skip("Pinned 3500 parity + prepared-state artifacts are not installed locally")
    return tuple(HeightSweepExperiment(CONFIG, PARITY, delta, PREPARED) for delta in (0., -.20))


@pytest.mark.parametrize("time_s,steps", [(0.001, 0), (0., 1), (1., 20), (float("nan"), 0)])
def test_prepared_state_never_injected_during_runtime(time_s, steps):
    exp = SimpleNamespace(data=SimpleNamespace(time=time_s), steps=steps)
    with pytest.raises(ValueError, match="ongoing trial"):
        initialize_from_common(exp, None)


@pytest.mark.parametrize("delta", [.01, -.025, -.25, float("nan")])
def test_only_five_preregistered_heights(delta):
    with pytest.raises(ValueError, match="five preregistered"):
        HeightSweepExperiment("unused", "unused", delta, "unused")


def test_common_prepared_state_is_required():
    with pytest.raises(ValueError, match="SAME policy-prepared"):
        HeightSweepExperiment("unused", "unused", -.10)


def test_two_height_models_start_with_identical_robot_and_policy_state(prepared_trials):
    first, second = prepared_trials
    assert first.prepared_state_info["sha256"] == second.prepared_state_info["sha256"]
    assert first.parity_evidence["onnx_sha256"] == second.parity_evidence["onnx_sha256"]
    assert first.parity_evidence["checkpoint_sha256"] == second.parity_evidence["checkpoint_sha256"]
    assert first.data.time == second.data.time == 0.
    assert first.steps == second.steps
    assert first.phase == second.phase == "START_HOLD"
    for first_joint in range(first.model.njnt):
        name = first.model.joint(first_joint).name
        if name == "crate_free":
            continue
        second_joint = second.model.joint(name).id
        free = first.model.jnt_type[first_joint] == mujoco.mjtJoint.mjJNT_FREE
        nq, nv = (7, 6) if free else (1, 1)
        qa, qb = first.model.jnt_qposadr[first_joint], second.model.jnt_qposadr[second_joint]
        va, vb = first.model.jnt_dofadr[first_joint], second.model.jnt_dofadr[second_joint]
        np.testing.assert_array_equal(first.data.qpos[qa:qa+nq], second.data.qpos[qb:qb+nq])
        np.testing.assert_array_equal(first.data.qvel[va:va+nv], second.data.qvel[vb:vb+nv])
    for name in ("last_action", "q_des", "last_torque", "last_observation"):
        np.testing.assert_array_equal(getattr(first.policy, name), getattr(second.policy, name))
    assert first.policy.histories.keys() == second.policy.histories.keys()
    for name in first.policy.histories:
        np.testing.assert_array_equal(first.policy.histories[name], second.policy.histories[name])
    for side in ("left", "right"):
        for field, value in asdict(first.policy.references[side]).items():
            np.testing.assert_array_equal(value, getattr(second.policy.references[side], field))
        np.testing.assert_array_equal(first.hands.controllers[side].reference.position,
                                      second.hands.controllers[side].reference.position)
        assert first.hands.controllers[side].command == second.hands.controllers[side].command == 0


def test_height_changes_only_scene_vertical_placement_not_dynamics(prepared_trials):
    first, second = prepared_trials
    assert first.initial_geometry_contacts == second.initial_geometry_contacts == []
    assert first._prop_contacts() == second._prop_contacts() == []
    assert not first.done and not second.done
    assert second.table_height-first.table_height == pytest.approx(-.20)
    for name in ("cargo_crate", "tabletop"):
        a = first.scratch.xpos[first.model.body(name).id]
        b = second.scratch.xpos[second.model.body(name).id]
        np.testing.assert_allclose(b-a, [0., 0., -.20], atol=1e-12)
    np.testing.assert_array_equal(first.model.jnt_range, second.model.jnt_range)
    # Shorter fixed table legs change their inferred (dynamically irrelevant)
    # mass. The articulated robot and free crate must remain exactly identical.
    dynamic_bodies = [body for body in range(first.model.nbody)
                      if first.model.body(body).name != "tabletop"]
    np.testing.assert_array_equal(first.model.body_mass[dynamic_bodies], second.model.body_mass[dynamic_bodies])
    np.testing.assert_array_equal(first.model.body_inertia[dynamic_bodies], second.model.body_inertia[dynamic_bodies])
    for exp in prepared_trials:
        assert exp.cfg["gates"]["max_joint_violation_rad"] == .01
        assert exp.cfg["gates"]["max_self_penetration_m"] == .002
        assert exp.model.opt.timestep == .001
        assert exp.model.nmocap == 0
        assert not np.any(exp.model.eq_type == mujoco.mjtEq.mjEQ_WELD)
        assert not np.any(exp.data.xfrc_applied) and not np.any(exp.data.qfrc_applied)


def test_source_close_is_logged_but_never_executed():
    exp = object.__new__(HeightSweepExperiment)
    exp.path = HeightPath(-.10)
    exp.segment_index = next(i for i, segment in enumerate(exp.path.segments) if segment["name"] == "CLOSE")
    exp.phase, exp.phase_start = "INSERT_SETTLE", 0.
    exp.data = SimpleNamespace(time=.5)
    exp.source_time_s = 0.
    exp.transitions, exp.targets = [], []
    commanded = []
    exp.hands = SimpleNamespace(command=lambda *args: commanded.append(args),
        controllers={side: SimpleNamespace(command=0) for side in ("left", "right")})
    goals = {}
    exp._set_goal = lambda side, target: goals.update({side: target})
    exp.enter("CLOSE")
    exp._stream_targets()
    assert not commanded
    assert np.array_equal(exp.targets[-1]["recorded_hand_command"], [1, 1])
    assert exp.targets[-1]["executed_hand_command"] == [0, 0]
    assert all(hand.command == 0 for hand in exp.hands.controllers.values())
    assert set(goals) == {"left", "right"}


def test_outside_path_continues_reference_not_biased_measured_wrist(monkeypatch):
    import common.r2v2_crate_height_sweep as sweep
    exp = object.__new__(HeightSweepExperiment)
    exp.phase, exp.delta_z = "START_HOLD", -.15
    exp.data, exp.scratch = SimpleNamespace(time=2.), object()
    exp.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=name))
    refs = {side: SimpleNamespace(position=np.array([.18, y, 1.13]),
                                 quaternion=np.array([1., 0., 0., 0.]))
            for side, y in (("left", .27), ("right", -.27))}
    exp.policy = SimpleNamespace(references=refs)
    actual = {f"{side}_hand_roll_link": transform_from_pose([.18, y, 1.13], [1., 0., 0., 0.])
              for side, y in (("left", .23), ("right", -.23))}
    actual["cargo_crate"] = transform_from_pose([.38, 0., 1.0109189696536514-.15], [1., 0., 0., 0.])
    monkeypatch.setattr(sweep, "body_transform", lambda data, body: actual[body].copy())
    monkeypatch.setattr(sweep, "inspect_approach_hand_sweep", lambda path: {"passed": True})
    entered = []
    exp._next_segment = lambda: entered.append("OUTSIDE")
    exp._gate()
    assert entered == ["OUTSIDE"]
    start = exp.path.sample("OUTSIDE", 0.)["T_world_wrist"]
    for index, side in enumerate(("left", "right")):
        expected = transform_from_pose(refs[side].position, refs[side].quaternion)
        np.testing.assert_allclose(start[index], expected, atol=1e-12)
        assert np.linalg.norm(start[index, :3, 3]-actual[f"{side}_hand_roll_link"][:3, 3]) > .039


@pytest.mark.parametrize("phase", ["FAILED", "COMPLETE"])
def test_terminal_trial_cannot_advance_physics_or_commands(monkeypatch, phase):
    exp = object.__new__(HeightSweepExperiment)
    exp.phase, exp.steps = phase, 10
    exp.data = SimpleNamespace(time=1.25)
    def forbidden(*args, **kwargs):
        pytest.fail("Terminal trial performed work after stop")
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    exp.sync = exp._safety = exp.record = exp._gate = forbidden
    exp.step()
    assert exp.data.time == 1.25 and exp.steps == 10


def test_report_never_claims_grasp_and_complete_is_not_reach_success(prepared_trials, monkeypatch):
    exp = prepared_trials[0]
    monkeypatch.setattr(exp, "phase", "COMPLETE")
    monkeypatch.setattr(exp, "segment_results", [dict(name="OUTSIDE", coarse_reach_passed=False)])
    report = exp.report()
    assert report["experiment_completed"] and report["path_sequence_completed"]
    assert not report["path_passed"]
    assert not report["lift_passed"] and not report["grasp_commanded"]
    assert "EMPTY-HAND" in report["scope"]
    assert "no grasp or loaded lift" in report["scope"]


def test_report_requires_complete_stage_evidence(prepared_trials, monkeypatch):
    exp = prepared_trials[0]
    monkeypatch.setattr(exp, "phase", "COMPLETE")
    monkeypatch.setattr(exp, "path", HeightPath(0.))
    monkeypatch.setattr(exp, "segment_results", [])
    assert not exp.report()["path_passed"], "Empty stage evidence must never vacuously pass"
    monkeypatch.setattr(exp, "segment_results", [dict(name="OUTSIDE", coarse_reach_passed=True)])
    assert not exp.report()["path_passed"], "One passing stage cannot certify the complete path"
