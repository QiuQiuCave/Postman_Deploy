from common.path_config import PROJECT_ROOT

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_reach_sim import (
    TCP_OFFSETS, build_reach_model, foot_collision_ids, initialize_robot,
    load_reach_config, require_parity, rotation_from_rpy_deg, ReachCompatibilityExperiment,
)
from r2v2_description.model import build_model


def test_wrist_scene_adds_only_zero_offset_wrist_sites_preserving_dynamics():
    cfg = load_reach_config()
    cfg["endpoint_contract"] = "wrist_world_v2"
    model, hands = build_reach_model(cfg)
    baseline = build_model(hands)
    assert (model.nq, model.nv, model.nu, model.neq) == (57, 56, 40, 10)
    np.testing.assert_array_equal(model.body_mass, baseline.body_mass)
    np.testing.assert_array_equal(model.body_inertia, baseline.body_inertia)
    np.testing.assert_array_equal(model.jnt_range, baseline.jnt_range)
    for side in ("left", "right"):
        site = model.site(f"{side}_wrist")
        assert site.bodyid[0] == model.body(f"{side}_hand_roll_link").id
        np.testing.assert_array_equal(site.pos, np.zeros(3))
        np.testing.assert_array_equal(site.quat, [1, 0, 0, 0])
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp") == -1


def test_unknown_scene_contract_rejected():
    cfg = load_reach_config()
    cfg["endpoint_contract"] = "wrong_version"
    with pytest.raises(ValueError, match="Unknown Reach endpoint_contract"):
        build_reach_model(cfg)


def test_legacy_parity_cannot_unlock_wrist_scene(tmp_path):
    path = tmp_path / "legacy_report.json"
    path.write_text(json.dumps({"passed": True}))
    cfg = load_reach_config()
    cfg["endpoint_contract"] = "wrist_world_v2"
    with pytest.raises(ValueError, match="different endpoint contract"):
        require_parity(path, cfg)


def test_new_robot_only_adds_virtual_tcp_and_preserves_source_dynamics():
    cfg = load_reach_config()
    m, hands = build_reach_model(cfg)
    original = build_model(hands)
    assert (m.nq, m.nv, m.nu, m.neq) == (57, 56, 40, 10)
    assert m.nsite == original.nsite + 2
    assert m.body("base_link").mocapid[0] == -1
    assert m.joint("floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_range", "jnt_axis",
                 "dof_damping", "dof_armature", "dof_frictionloss", "geom_contype", "geom_conaffinity",
                 "geom_friction", "geom_condim", "geom_solref", "geom_solimp"):
        np.testing.assert_array_equal(getattr(m, name), getattr(original, name))
    for side, offset in TCP_OFFSETS.items():
        site = m.site(f"{side}_tcp")
        assert site.bodyid[0] == m.body(f"{side}_hand_roll_link").id
        np.testing.assert_array_equal(site.pos, offset)


def test_ground_initialization_uses_new_feet_not_training_home_height():
    m, hands = build_reach_model(load_reach_config())
    d = mujoco.MjData(m)
    height = initialize_robot(m, d, hands)
    assert height == pytest.approx(0.8171516130818831)
    feet = foot_collision_ids(m)
    assert len(feet) == 28
    assert min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in feet) == pytest.approx(0.001)
    assert max([-float(c.dist) for c in d.contact] or [0]) < 0.001
    assert not d.xfrc_applied.any()


def test_parity_failure_blocks_new_robot_trial(tmp_path):
    path = tmp_path / "failure.json"
    path.write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="has not passed"):
        require_parity(path, load_reach_config())


def test_rpy_debug_input_is_extrinsic_xyz():
    np.testing.assert_allclose(rotation_from_rpy_deg([0, 0, 90]),
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(rotation_from_rpy_deg([90, 90, 0]),
                               [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], atol=1e-12)
    with pytest.raises(ValueError, match="finite"):
        rotation_from_rpy_deg([0, float("nan"), 0])


def gate_for_unit_test():
    # Test transition criteria independently of a proprietary policy file.
    exp = ReachCompatibilityExperiment.__new__(ReachCompatibilityExperiment)
    exp.cfg = load_reach_config()
    exp.data = SimpleNamespace(time=0.0)
    exp.scratch = SimpleNamespace(time=0.0, xpos=np.array([[0, 0, 0.82]]), xmat=np.eye(3).reshape(1, 9))
    exp.base = 0
    exp.phase, exp.phase_start = "AIR_GRASP", 0.0
    exp.stable_since = exp.inactive_bad_since = None
    exp.stable_time = exp.inactive_bad_time = 0.0
    exp.peaks = {"base_tilt_deg": 0.0, "inactive_wrist_position_error_m": 0.0,
                 "inactive_orientation_error_deg": 0.0}
    exp.right_lock = None
    exp.failure = exp.failure_phase = None
    exp.transitions = []
    exp.record = lambda: None
    exp.next_air_target = lambda: setattr(exp, "phase", "PASSED")
    errors = {s: {"position_m": 0.0, "orientation_deg": 0.0, "linear_speed_mps": 0.0,
                  "wrist_position_m": 0.0, "wrist_linear_speed_mps": 0.0} for s in ("left", "right")}
    exp.errors = lambda side: errors[side].copy()
    return exp, errors


def tick_gate(exp, time):
    exp.data.time = exp.scratch.time = time
    exp.update_gate()


def test_stable_duration_means_elapsed_time_not_sample_count():
    exp, _ = gate_for_unit_test()
    for i in range(15):  # 15 samples span 0.28 seconds, not 0.30.
        tick_gate(exp, i*0.02)
        assert exp.phase == "AIR_GRASP"
    tick_gate(exp, 0.30)
    assert exp.phase == "PASSED"


def test_failed_stability_sample_resets_elapsed_timer():
    exp, e = gate_for_unit_test()
    for time in (0, 0.1, 0.2):
        tick_gate(exp, time)
    e["left"]["orientation_deg"] = 4
    tick_gate(exp, 0.25)
    e["left"]["orientation_deg"] = 0
    for time in (0.3, 0.4, 0.5):
        tick_gate(exp, time)
        assert exp.phase == "AIR_GRASP"
    tick_gate(exp, 0.6)
    assert exp.phase == "PASSED"


def test_wrist_error_blocks_even_when_virtual_tcp_is_at_target():
    exp, e = gate_for_unit_test()
    e["left"]["wrist_position_m"] = 0.006
    for time in (0, 0.3, 1, 9.98):
        tick_gate(exp, time)
        assert exp.phase == "AIR_GRASP"
    tick_gate(exp, 10 - 1e-10)
    assert exp.phase == "FAILED"
    assert exp.failure_phase == "AIR_GRASP"
    assert exp.transitions[-1]["exit_errors"]["left"]["wrist_position_m"] == 0.006


def test_inactive_hand_gate_checks_wrist_and_full_violation_duration():
    exp, e = gate_for_unit_test()
    exp.phase = "STANDING_CHECK"
    exp.right_lock = True
    e["right"]["wrist_position_m"] = 0.025
    for i in range(15):
        tick_gate(exp, i*0.02)
        assert exp.phase == "STANDING_CHECK"
    tick_gate(exp, 0.30)
    assert exp.phase == "FAILED"
    assert "Inactive hand" in exp.failure


def test_safety_failure_between_record_ticks_uses_current_state(monkeypatch):
    exp, _ = gate_for_unit_test()
    exp.steps = 1001  # Neither a policy tick nor a hand/record tick.
    exp.data.time, exp.scratch.time = 1.001, 1.0
    exp.data.ctrl = np.zeros(28)
    exp.body_map = SimpleNamespace(actuators=np.arange(28))
    exp.model = SimpleNamespace(actuator_ctrlrange=np.tile([-1.0, 1.0], (28, 1)))
    exp.policy = exp.hands = SimpleNamespace(apply=lambda data: None)
    exp.saturation_count = exp.torque_count = 0
    exp.safety = lambda: "Injected non-foot ground contact"
    exp.sync = lambda: setattr(exp.scratch, "time", exp.data.time)
    exp.errors = lambda side: {"sample_time": exp.scratch.time}
    records = []
    exp.record = lambda: records.append((exp.scratch.time, exp.phase))
    monkeypatch.setattr(mujoco, "mj_step", lambda model, data: setattr(data, "time", 1.002))

    exp.step()

    assert exp.phase == "FAILED"
    assert exp.transitions[-1]["time_s"] == pytest.approx(1.002)
    assert exp.transitions[-1]["exit_errors"]["left"]["sample_time"] == pytest.approx(1.002)
    assert records and all(time == pytest.approx(1.002) for time, phase in records)
