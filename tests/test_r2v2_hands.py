from common.path_config import PROJECT_ROOT

import copy

import mujoco
import numpy as np
import pytest
import yaml

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.r2v2_hand_control import BinaryHandController, DualHandControl
from common.r2v2_hand_test import DEMO_DURATION, HandExperiment
from common.utils import FSMCommand
from FSM.R2V2FSM import R2V2FSM
from r2v2_description.model import (
    BODY_JOINTS, SIDES, JointMap, build_model, hand_names,
    initialize_hands, load_config, urdf_hand_joints,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def models(cfg):
    return {fixture: build_model(cfg, fixture=fixture) for fixture in (False, True)}


def test_thumb_stays_preopposed_during_binary_commands(cfg):
    for side in SIDES:
        target = np.deg2rad(75)
        assert cfg["hands"][side]["open"][0] == pytest.approx(target)
        assert cfg["hands"][side]["closed"][0] == pytest.approx(target)
        controller = BinaryHandController(cfg, side, cfg["hands"][side]["open"])
        for command in (1, 0):
            controller.set_command(command)
            for _ in range(300):
                reference = controller.step()
                assert reference.position[0] == pytest.approx(target)
                assert reference.velocity[0] == pytest.approx(0)


def test_full_robot_and_fixture_dimensions(models):
    full, fixture = models[False], models[True]
    assert (full.nq, full.nv, full.nu, full.neq) == (57, 56, 40, 10)
    assert (fixture.nq, fixture.nv, fixture.nu, fixture.neq) == (22, 22, 12, 10)
    assert full.joint("floating_base_joint").type == mujoco.mjtJoint.mjJNT_FREE
    assert full.nexclude == fixture.nexclude == 10


def test_body_mapping_skips_interleaved_fingers(models):
    m = models[False]
    mapping = JointMap.create(m, BODY_JOINTS)
    np.testing.assert_array_equal(mapping.actuators, np.arange(28))
    # Right shoulder follows the eleven left-hand joints in qpos, not in ctrl.
    assert mapping.qpos[21] == 7 + 21 + 11
    assert mapping.dofs[21] == 6 + 21 + 11
    assert len(mapping.names) == 28
    hand_maps = [JointMap.create(m, hand_names(s)) for s in SIDES]
    assert not set(mapping.qpos) & set(np.concatenate([h.qpos for h in hand_maps]))
    with pytest.raises(ValueError, match="Missing joint"):
        JointMap.create(m, ["not_a_joint"])
    with pytest.raises(ValueError, match="Duplicate"):
        JointMap.create(m, [BODY_JOINTS[0], BODY_JOINTS[0]])


def test_couplings_match_source_urdf(models):
    m = models[True]
    for name, joint in urdf_hand_joints().items():
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        eq = m.equality(f"couple_{name}")
        assert eq.obj1id == m.joint(name).id
        assert eq.obj2id == m.joint(mimic.get("joint")).id
        np.testing.assert_allclose(eq.data[:5], [float(mimic.get("offset", "0")),
                                               float(mimic.get("multiplier", "1")), 0, 0, 0])


@pytest.mark.parametrize("pose", ["open", "closed"])
def test_fixture_preserves_hand_kinematics(models, cfg, pose):
    custom = copy.deepcopy(cfg)
    for side in SIDES:
        custom["hands"][side]["open"] = cfg["hands"][side][pose]
    data = {}
    for fixed, model in models.items():
        data[fixed] = mujoco.MjData(model)
        initialize_hands(model, data[fixed], custom)
    for side in SIDES:
        for part in ("thumb_distal", "index_distal", "middle_distal", "ring_distal", "pinky_distal"):
            local_positions, local_rotations = [], []
            for fixed, model in models.items():
                d = data[fixed]
                wrist = d.body(f"{side}_hand_roll_link")
                finger = d.body(f"{side}_{part}_link")
                rot = wrist.xmat.reshape(3, 3)
                local_positions.append(rot.T @ (finger.xpos - wrist.xpos))
                local_rotations.append(rot.T @ finger.xmat.reshape(3, 3))
            np.testing.assert_allclose(*local_positions, atol=1e-10)
            np.testing.assert_allclose(*local_rotations, atol=1e-10)


def test_hand_control_does_not_write_body_outputs(models, cfg):
    m = models[False]
    d = mujoco.MjData(m)
    initialize_hands(m, d, cfg)
    body = JointMap.create(m, BODY_JOINTS)
    d.ctrl[body.actuators] = np.linspace(-0.1, 0.1, 28)
    before = d.ctrl[body.actuators].copy()
    control = DualHandControl(m, d, cfg)
    control.command("left", 1)
    control.update()
    control.apply(d)
    np.testing.assert_array_equal(d.ctrl[body.actuators], before)
    assert control.controllers["right"].command == 0
    assert control.controllers["left"].command == 1


def test_existing_fsm_keeps_28_dimensional_contract():
    state, output = StateAndCmd(28), PolicyOutput(28)
    fsm = R2V2FSM(state, output, np.zeros(28))
    for command in (FSMCommand.PASSIVE, FSMCommand.POS_RESET):
        fsm.request(command)
        fsm.run()
        fsm.run()
        assert output.actions.shape == output.kps.shape == output.kds.shape == (28,)
    assert fsm.state_name == "r2v2_hold_pose"


def test_repeated_binary_command_is_idempotent(cfg):
    a, b = [BinaryHandController(cfg, "left", cfg["hands"]["left"]["open"]) for _ in range(2)]
    a.set_command(1)
    b.set_command(1)
    for _ in range(300):
        assert not b.set_command(1)
        np.testing.assert_array_equal(a.step().position, b.step().position)
    assert a.command_changes == b.command_changes == 1
    assert a.finished and b.finished


def test_mid_motion_reversal_keeps_reference_state(cfg):
    c = BinaryHandController(cfg, "left", cfg["hands"]["left"]["open"])
    previous_acc = np.zeros(6)
    for step in range(600):
        if step in (0, 55, 110, 165):
            before = np.r_[c.input.current_position, c.input.current_velocity, c.input.current_acceleration]
            c.set_command(1 - c.command)
            np.testing.assert_array_equal(before, np.r_[c.input.current_position,
                                                       c.input.current_velocity, c.input.current_acceleration])
        ref = c.step()
        for x, limit in ((ref.velocity, "max_velocity"), (ref.acceleration, "max_acceleration"),
                         ((ref.acceleration - previous_acc) / c.dt, "max_jerk")):
            assert np.all(np.abs(x) <= np.asarray(cfg["trajectory"][limit]) + 1e-6)
        previous_acc = ref.acceleration.copy()
    assert c.finished


@pytest.mark.parametrize("value", [-1, 2, 0.5, "1", None, float("nan")])
def test_reject_invalid_commands(cfg, value):
    c = BinaryHandController(cfg, "left", cfg["hands"]["left"]["open"])
    with pytest.raises(ValueError):
        c.set_command(value)


def test_reject_config_that_exceeds_coupled_velocity(cfg, tmp_path):
    changed = copy.deepcopy(cfg)
    # Proximal limit is 2.2685, but its 1.155x distal must meet that limit too.
    changed["trajectory"]["max_velocity"][2] = 2.1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="coupled velocity"):
        load_config(path)


def test_complete_empty_hand_physics_demo(cfg):
    exp = HandExperiment(cfg)
    for _ in range(round(DEMO_DURATION / cfg["simulation_dt"])):
        exp.step()
    report = exp.report()
    assert report["passed"], report["checks"]
    assert len(report["events"]) == 13
    assert report["final_states"] == {"left": "OPEN", "right": "OPEN"}
    assert report["command_changes"] == {"left": 10, "right": 8}
