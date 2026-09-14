"""Virtual close-prop screens change scene targets, never robot physics.

Model/path tests need only repository assets. Prepared-state tests additionally
use the frozen local 3500 artifacts when installed. No rollout/video/training is
launched here; the short physics test checks only a static placeholder contract.
"""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_height_path import HeightPath
from common.r2v2_crate_height_sweep import HeightSweepExperiment
from common.r2v2_crate_motion_recording import transform_from_pose
from common.r2v2_crate_motion_replay_scene import build_motion_replay_model
from common.r2v2_reach_sim import foot_collision_ids, initialize_robot, load_reach_config


ARTIFACT_ROOT = Path("/root/autodl-tmp/Postman_Deploy")
PREPARED = ARTIFACT_ROOT / "crate_height_sweep_20260911/preparation/prepared_state.json"
CONFIG = ARTIFACT_ROOT / "simple_dual_reach_20260911/policy/reach_wrist_v2.yaml"
PARITY = ARTIFACT_ROOT / "simple_dual_reach_20260911/parity/report.json"


@pytest.fixture(scope="module")
def scenes():
    cfg = load_reach_config("deploy_mujoco/config/r2v2_reach_wrist_v2.yaml")
    params = replace(load_crate_config(), width=.26)
    physical = build_motion_replay_model(cfg, params)
    virtual = build_motion_replay_model(cfg, params, virtual_props=True,
        crate_center_xy=(.33, 0.), table_center_xy=(.43, 0.))
    return physical, virtual


def _prop_geom_ids(model):
    return [g for g in range(model.ngeom)
            if model.body(int(model.geom_bodyid[g])).name in ("tabletop", "cargo_crate")]


def test_virtual_props_static_invisible_noncolliding_and_default_still_physical(scenes):
    (physical, _, _), (virtual, _, layout) = scenes
    assert (physical.nq, physical.nv, physical.nu, physical.neq, physical.nmocap) == (64, 62, 40, 10, 0)
    assert (virtual.nq, virtual.nv, virtual.nu, virtual.neq, virtual.nmocap) == (57, 56, 40, 10, 0)
    assert np.count_nonzero(virtual.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert mujoco.mj_name2id(virtual, mujoco.mjtObj.mjOBJ_JOINT, "crate_free") == -1
    assert physical.joint("crate_free").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert np.all(virtual.eq_type == mujoco.mjtEq.mjEQ_JOINT)
    for name in ("tabletop", "cargo_crate"):
        assert virtual.body(name).mocapid[0] == -1
        assert virtual.body(name).jntnum[0] == 0
    props = _prop_geom_ids(virtual)
    assert props and {virtual.body(int(virtual.geom_bodyid[g])).name for g in props} == {"tabletop", "cargo_crate"}
    np.testing.assert_array_equal(virtual.geom_contype[props], 0)
    np.testing.assert_array_equal(virtual.geom_conaffinity[props], 0)
    np.testing.assert_array_equal(virtual.geom_rgba[props, 3], 0.)
    physical_props = [physical.geom(virtual.geom(g).name).id for g in props]
    assert np.all(physical.geom_contype[physical_props] != 0)
    assert np.all(physical.geom_rgba[physical_props, 3] > 0.)
    np.testing.assert_allclose(layout["crate_center_xy"], [.33, 0.], atol=1e-12)
    np.testing.assert_allclose(layout["table_center_xy"], [.43, 0.], atol=1e-12)


def test_robot_dynamics_collision_geometry_and_floor_unchanged(scenes):
    (physical, _, _), (virtual, _, _) = scenes
    excluded_bodies = {"tabletop", "cargo_crate"}
    groups = (
        ("body", physical.nbody, ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat")),
        ("joint", physical.njnt, ("jnt_type", "jnt_range", "jnt_limited", "jnt_axis", "jnt_pos", "jnt_stiffness")),
        ("geom", physical.ngeom, ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
            "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap")),
        ("actuator", physical.nu, ("actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "actuator_gainprm", "actuator_biasprm")),
        ("equality", physical.neq, ("eq_data", "eq_solref", "eq_solimp", "eq_type")),
    )
    for kind, count, fields in groups:
        for i in range(count):
            name = getattr(physical, kind)(i).name
            if kind == "body" and name in excluded_bodies:
                continue
            if kind == "joint" and name == "crate_free":
                continue
            if kind == "geom" and physical.body(int(physical.geom_bodyid[i])).name in excluded_bodies:
                continue
            j = getattr(virtual, kind)(name).id
            for field in fields:
                np.testing.assert_array_equal(getattr(physical, field)[i], getattr(virtual, field)[j],
                                               err_msg=f"{kind} {name}: {field}")
    for i in range(virtual.njnt):
        name = virtual.joint(i).name
        j = physical.joint(name).id
        ndof = 6 if virtual.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1
        va, pa = virtual.jnt_dofadr[i], physical.jnt_dofadr[j]
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(virtual, field)[va:va+ndof], getattr(physical, field)[pa:pa+ndof])
    assert virtual.opt.timestep == physical.opt.timestep == .001
    assert virtual.opt.disableflags == physical.opt.disableflags
    assert virtual.opt.enableflags == physical.opt.enableflags


def test_actual_robot_overlap_has_no_prop_contact_but_floor_contact_remains(scenes):
    _, (model, hands, _) = scenes
    data = mujoco.MjData(model)
    initialize_robot(model, data, hands)
    props = set(_prop_geom_ids(model))
    foot = foot_collision_ids(model)[0]
    crate_bottom = model.geom("crate_bottom").id
    base_q = model.joint("floating_base_joint").qposadr[0]
    # Deliberately put a physical robot sphere at the middle of a placeholder
    # box volume. mj_forward must create no robot/prop contact pair.
    data.qpos[base_q:base_q+3] += data.geom_xpos[crate_bottom]-data.geom_xpos[foot]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.geom_xpos[foot], data.geom_xpos[crate_bottom], atol=1e-12)
    assert not any(set(map(int, c.geom)) & props for c in data.contact)
    # Reset and move feet through the real floor: its contacts are not disabled
    # by the virtual scene mode, nor by a global collision-disable flag.
    data = mujoco.MjData(model)
    initialize_robot(model, data, hands)
    data.qpos[base_q+2] -= .005
    mujoco.mj_forward(model, data)
    floor, feet = model.geom("floor").id, set(foot_collision_ids(model))
    assert any(floor in set(map(int, c.geom)) and set(map(int, c.geom)) & feet and c.dist < 0
               for c in data.contact)


def test_virtual_crate_cannot_fall_through_virtual_table(scenes):
    _, (model, hands, _) = scenes
    data = mujoco.MjData(model)
    initialize_robot(model, data, hands)
    body = model.body("cargo_crate").id
    initial = data.xpos[body].copy()
    for _ in range(50):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.xpos[body], initial)
    assert data.time == pytest.approx(.05)


@pytest.mark.parametrize("delta_z", [0., -.05, -.10, -.15, -.20])
def test_closer_paths_translate_world_goals_preserve_relative_motion_and_start(delta_z):
    old = HeightPath(delta_z)
    close = HeightPath(delta_z, delta_x=-.05)
    translation = np.array([-.05, 0., 0.])
    np.testing.assert_allclose(close.world_crate_pose[:3, 3]-old.world_crate_pose[:3, 3], translation, atol=1e-12)
    np.testing.assert_array_equal(close.sample("OUTSIDE", 0.)["T_world_wrist"], old.sample("OUTSIDE", 0.)["T_world_wrist"])
    for segment in old.segments:
        name, duration = segment["name"], segment["duration_s"]
        for fraction in (0., .5, 1.):
            a, b = old.sample(name, fraction*duration), close.sample(name, fraction*duration)
            # Start stays at the identical current policy reference; only its
            # destination is shifted. All subsequent motion is rigidly shifted.
            scale = fraction*fraction*(3.-2.*fraction) if name == "OUTSIDE" else 1.
            np.testing.assert_allclose(b["T_world_wrist"][:, :3, 3]-a["T_world_wrist"][:, :3, 3],
                np.broadcast_to(scale*translation, (2, 3)), atol=1e-12)
            np.testing.assert_array_equal(b["T_world_wrist"][:, :3, :3], a["T_world_wrist"][:, :3, :3])
            np.testing.assert_array_equal(b["hand_command"], a["hand_command"])
            assert b["source_time_s"] == a["source_time_s"]
            if name != "OUTSIDE":
                np.testing.assert_allclose(np.linalg.inv(b["T_world_crate"])@b["T_world_wrist"],
                                          np.linalg.inv(a["T_world_crate"])@a["T_world_wrist"], atol=1e-12)


def test_explicit_world_crate_pose_is_not_shifted_twice():
    original = HeightPath(-.15)
    moved_pose = original.world_crate_pose.copy()
    moved_pose[0, 3] -= .05
    moved = HeightPath(-.15, delta_x=-.05, world_crate_pose=moved_pose)
    auto = HeightPath(-.15, delta_x=-.05)
    for segment in moved.segments:
        for elapsed in (0., segment["duration_s"]):
            np.testing.assert_allclose(moved.sample(segment["name"], elapsed)["T_world_wrist"],
                auto.sample(segment["name"], elapsed)["T_world_wrist"], atol=1e-12)


@pytest.fixture(scope="module")
def prepared_trials():
    if not all(path.is_file() for path in (PREPARED, CONFIG, PARITY)):
        pytest.skip("Pinned 3500 parity + prepared-state artifacts not installed locally")
    return (HeightSweepExperiment(CONFIG, PARITY, -.15, PREPARED),
            HeightSweepExperiment(CONFIG, PARITY, -.15, PREPARED, delta_x=-.05, virtual_props=True))


def test_virtual_closer_trial_preserves_frozen_robot_policy_and_open_hands(prepared_trials):
    physical, virtual = prepared_trials
    assert physical.prepared_state_info["sha256"] == virtual.prepared_state_info["sha256"]
    assert physical.parity_evidence == virtual.parity_evidence
    assert physical.data.time == virtual.data.time == 0.
    assert physical.phase == virtual.phase == "START_HOLD"
    assert not physical.done and not virtual.done
    assert virtual.initial_geometry_contacts == []
    for i in range(virtual.model.njnt):
        j = physical.model.joint(virtual.model.joint(i).name).id
        nq, nv = (7, 6) if virtual.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
        vq, pq = virtual.model.jnt_qposadr[i], physical.model.jnt_qposadr[j]
        vv, pv = virtual.model.jnt_dofadr[i], physical.model.jnt_dofadr[j]
        np.testing.assert_array_equal(virtual.data.qpos[vq:vq+nq], physical.data.qpos[pq:pq+nq])
        np.testing.assert_array_equal(virtual.data.qvel[vv:vv+nv], physical.data.qvel[pv:pv+nv])
        np.testing.assert_array_equal(virtual.data.qacc_warmstart[vv:vv+nv], physical.data.qacc_warmstart[pv:pv+nv])
    for name in ("last_action", "q_des", "last_torque", "last_observation"):
        np.testing.assert_array_equal(getattr(virtual.policy, name), getattr(physical.policy, name))
    np.testing.assert_array_equal(virtual.data.ctrl, physical.data.ctrl)
    for name in physical.policy.histories:
        np.testing.assert_array_equal(virtual.policy.histories[name], physical.policy.histories[name])
    for side in ("left", "right"):
        for field, value in asdict(physical.policy.references[side]).items():
            np.testing.assert_array_equal(value, getattr(virtual.policy.references[side], field))
        np.testing.assert_array_equal(physical.hands.controllers[side].reference.position,
                                      virtual.hands.controllers[side].reference.position)
        assert physical.hands.controllers[side].command == virtual.hands.controllers[side].command == 0
    assert not np.any(virtual.data.xfrc_applied) and not np.any(virtual.data.qfrc_applied)
    assert virtual.cfg["gates"]["max_joint_violation_rad"] == .01
    assert virtual.cfg["gates"]["max_self_penetration_m"] == .002


def test_virtual_geometry_warning_does_not_prevent_path_start(monkeypatch):
    import common.r2v2_crate_height_sweep as sweep
    exp = object.__new__(HeightSweepExperiment)
    exp.phase, exp.delta_z, exp.delta_x, exp.virtual_props = "START_HOLD", -.15, -.05, True
    exp.data, exp.scratch = SimpleNamespace(time=2.), object()
    exp.model = SimpleNamespace(body=lambda name: SimpleNamespace(id=name))
    refs = {side: SimpleNamespace(position=np.array([.18, y, 1.13]), quaternion=np.array([1., 0., 0., 0.]))
            for side, y in (("left", .27), ("right", -.27))}
    exp.policy = SimpleNamespace(references=refs)
    actual = {f"{side}_hand_roll_link": transform_from_pose(ref.position, ref.quaternion)
              for side, ref in refs.items()}
    actual["cargo_crate"] = transform_from_pose([.33, 0., 1.0109189696536514-.15], [1., 0., 0., 0.])
    monkeypatch.setattr(sweep, "body_transform", lambda data, body: actual[body].copy())
    monkeypatch.setattr(sweep, "inspect_approach_hand_sweep", lambda path: {"passed": False})
    entered = []
    exp._next_segment = lambda: entered.append("OUTSIDE")
    exp.fail = lambda *args: pytest.fail("Virtual prop geometry must not stop a path trial")
    exp._gate()
    assert entered == ["OUTSIDE"]
    assert exp.approach_geometry["passed"] is False
    np.testing.assert_allclose(exp.path.world_crate_pose[:3, 3], [.33, 0., 1.0109189696536514-.15], atol=1e-12)


def test_virtual_report_cannot_claim_physical_grasp(prepared_trials):
    exp = prepared_trials[1]
    report = exp.report()
    assert not report["grasp_commanded"] and not report["lift_passed"]
    assert not report["path_passed"]
    assert "virtual" in report["scope"].lower()
    assert report["delta_x_m"] == -.05
    metrics = report["final_metrics"]
    assert metrics["prop_contact_physics_evaluated"] is False
    assert not metrics["grasp_success_evaluated"]
    assert metrics["max_hand_crate_penetration_m"] == 0.
    for hand in metrics["hands"].values():
        assert hand["vertical_force_N"] == 0.


@pytest.mark.parametrize("violation", ["body_joint", "nonfoot_floor"])
def test_virtual_mode_still_stops_actual_robot_hazards(prepared_trials, violation):
    # Fresh state keeps the shared prepared-state evidence immutable. This
    # injects a bad diagnostic state, not a policy trajectory or video frame.
    exp = HeightSweepExperiment(CONFIG, PARITY, -.15, PREPARED, delta_x=-.05, virtual_props=True)
    if violation == "body_joint":
        joint = exp.model.joint("left_ankle_roll_joint")
        exp.data.qpos[joint.qposadr[0]] = joint.range[1]+.03
        expected = "body joint limit"
    else:
        address = exp.model.joint("floating_base_joint").qposadr[0]
        exp.data.qpos[address+2] -= .8
        expected = "non-foot ground contact"
    exp.sync()
    assert exp._prop_contacts() == []
    exp._safety()
    assert exp.done and exp.phase == "FAILED"
    assert expected in exp.failure
