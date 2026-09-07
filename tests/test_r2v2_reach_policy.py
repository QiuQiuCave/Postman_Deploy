"""Policy contract tests without requiring a bundled checkpoint or Torch."""

from common.path_config import PROJECT_ROOT

from dataclasses import asdict
from types import SimpleNamespace
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from common.r2v2_reach_policy import (
    ACTION_DIM, ANGULAR_ACCELERATION_LIMIT, ANGULAR_SPEED_LIMIT, HISTORY_LENGTH,
    LINEAR_ACCELERATION_LIMIT, LINEAR_SPEED_LIMIT, POLICY_DT, ReachPolicy,
    ReachReference, TCP_WRIST_POSITIONS, TERM_DIMS, TRAINING_EFFORT_LIMITS,
    _box_plus, _quat_inv, _quat_mul, _quaternion, _rotation_vector,
)
from r2v2_description.model import (
    BODY_JOINTS, JointMap, SIDES, build_model_xml, hand_names, initialize_hands,
    load_config,
)


def metadata():
    return {
        "joint_names": ",".join(BODY_JOINTS),
        "observation_names": ",".join(TERM_DIMS),
        "command_names": "twist,base_pose,reach,reach_right",
        "joint_stiffness": ",".join(map(str, [100, 100, 100, 150, 30, 30]*2
                                               + [300, 300] + [100, 100, 100, 40, 40, 40, 40]*2)),
        "joint_damping": ",".join(map(str, [5, 5, 5, 7, 3, 3]*2
                                             + [3, 3] + [5, 5, 5, 2, 2, 2, 2]*2)),
        "default_joint_pos": ",".join(map(str, [-0.1, 0, 0, 0.3, -0.2, 0]*2 + [0]*16)),
        "action_scale": "0.25",
    }


@pytest.fixture(scope="module")
def model():
    root = ET.fromstring(build_model_xml(load_config()))
    for side in SIDES:
        parent = root.find(f'.//body[@name="{side}_hand_roll_link"]')
        ET.SubElement(parent, "site", name=f"{side}_tcp",
                      pos=" ".join(map(str, TCP_WRIST_POSITIONS[side])))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.fixture
def policy_factory(model, tmp_path, monkeypatch):
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"ONNX session is stubbed by this unit test")

    def make(meta=None, input_shape=None, output_shape=None, output=None):
        session = SimpleNamespace()
        session.get_inputs = lambda: [SimpleNamespace(name="obs", shape=input_shape or [1, 1460], type="tensor(float)")]
        session.get_outputs = lambda: [SimpleNamespace(name="actions", shape=output_shape or [1, 28], type="tensor(float)")]
        session.get_modelmeta = lambda: SimpleNamespace(custom_metadata_map=meta if meta is not None else metadata())
        session.output = np.zeros((1, ACTION_DIM), dtype=np.float32) if output is None else np.asarray(output)

        def run(names, feed):
            assert names == ["actions"]
            assert feed["obs"].dtype == np.float32
            assert feed["obs"].shape == (1, 1460)
            session.last_feed = feed["obs"].copy()
            return [session.output.copy()]

        session.run = run
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
            SessionOptions=SimpleNamespace,
            InferenceSession=lambda *args, **kwargs: session,
        ))
        data = mujoco.MjData(model)
        initialize_hands(model, data, load_config())
        controller = ReachPolicy(model, data, path)
        return controller, data

    return make


def term_slices():
    offset = 0
    result = {}
    for name, dim in TERM_DIMS.items():
        result[name] = slice(offset, offset+dim*HISTORY_LENGTH)
        offset += dim*HISTORY_LENGTH
    return result


def test_onnx_and_named_joint_contract(policy_factory):
    policy, data = policy_factory()
    assert len(TERM_DIMS) == 24
    assert sum(TERM_DIMS.values()) == 146
    assert policy.last_observation.shape == (1460,)
    assert policy.body_map.names == BODY_JOINTS
    assert policy.body_map.qpos[21] == 39  # Skip eleven interleaved left fingers.
    assert policy.body_map.actuators.tolist() == list(range(28))
    np.testing.assert_array_equal(policy.torque_upper, TRAINING_EFFORT_LIMITS)
    np.testing.assert_array_equal(policy.torque_lower, -TRAINING_EFFORT_LIMITS)
    assert policy._direct_torque


@pytest.mark.parametrize("key,value", [
    ("joint_names", ",".join(reversed(BODY_JOINTS))),
    ("observation_names", "wrong"),
    ("command_names", "twist,base_pose,reach"),
    ("action_scale", "1.0"),
    ("joint_stiffness", ",".join(["nan"]*28)),
    ("joint_damping", ",".join(["-1"]*28)),
    ("default_joint_pos", "0,0"),
])
def test_bad_metadata_rejected(policy_factory, key, value):
    meta = metadata()
    meta[key] = value
    with pytest.raises(ValueError):
        policy_factory(meta=meta)


@pytest.mark.parametrize("shapes", [
    {"input_shape": [1, 1220]}, {"output_shape": [1, 14]},
])
def test_wrong_policy_dimension_rejected(policy_factory, shapes):
    with pytest.raises(ValueError, match="Expected dual-arm ONNX"):
        policy_factory(**shapes)


def test_reset_backfills_per_term_without_changing_physics(policy_factory):
    policy, data = policy_factory()
    qpos, qvel, ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
    policy.last_action[:] = 2.0
    policy.reset(data)
    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.ctrl, ctrl)
    np.testing.assert_array_equal(policy.last_action, np.zeros(28))
    terms = policy.observation_terms(data)
    for name, sl in term_slices().items():
        history = policy.last_observation[sl].reshape(10, TERM_DIMS[name])
        np.testing.assert_array_equal(history, np.repeat(terms[name][None], 10, axis=0))
    for side in SIDES:
        ref = policy.references[side]
        p, q = policy.tcp_pose(data, side)
        np.testing.assert_array_equal(ref.position, p)
        np.testing.assert_array_equal(ref.goal_quaternion, q)


def test_history_is_term_major_oldest_first_and_snapshot_is_read_only(policy_factory):
    policy, data = policy_factory()
    slices = term_slices()
    base_q = data.qpos[policy.body_map.qpos].copy()
    for index in range(12):
        data.qpos[policy.body_map.qpos] = base_q + index*0.01
        policy.last_action[:] = index
        mujoco.mj_forward(policy.model, data)
        observed = policy.observe(data)
    previous = policy.last_observation.copy()
    histories = {key: value.copy() for key, value in policy.histories.items()}
    data.qpos[policy.body_map.qpos] += 0.7
    mujoco.mj_forward(policy.model, data)
    snapshot = policy.observe(data, advance_reference=False)
    for key, value in histories.items():
        np.testing.assert_array_equal(policy.histories[key], value)
    np.testing.assert_array_equal(policy.last_observation, previous)
    assert not np.array_equal(snapshot, previous)
    np.testing.assert_array_equal(observed[slices["actions"]].reshape(10, 28)[:, 0], np.arange(2, 12))
    expected = base_q-policy.default_joint_pos + np.arange(2, 12)[:, None]*0.01
    np.testing.assert_allclose(observed[slices["joint_pos"]].reshape(10, 28), expected, atol=1e-7)


def test_imu_uses_full_base_origin_frame_but_position_error_uses_yaw(policy_factory):
    policy, data = policy_factory()
    q_roll = np.array([np.cos(0.15), np.sin(0.15), 0, 0])
    q_pitch = np.array([np.cos(-0.2), 0, np.sin(-0.2), 0])
    q_yaw = np.array([np.cos(0.5), 0, 0, np.sin(0.5)])
    data.qpos[3:7] = _quat_mul(q_yaw, _quat_mul(q_pitch, q_roll))
    data.qvel[:6] = [0.3, -0.4, 0.5, 0.2, 0.4, 0.6]
    mujoco.mj_forward(policy.model, data)
    ref = policy.references["left"]
    p, q = policy.tcp_pose(data, "left")
    ref.position = p + [0.1, 0.2, 0.3]
    ref.goal_position = p + [0.3, 0.2, 0.1]
    ref.quaternion = _quat_mul(q, q_roll)
    ref.goal_quaternion = _quat_mul(q, q_pitch)
    terms = policy.observation_terms(data)
    rotation = data.xmat[policy.base_id].reshape(3, 3)
    np.testing.assert_allclose(terms["base_lin_vel"], rotation.T @ data.qvel[:3], atol=1e-7)
    np.testing.assert_allclose(terms["base_ang_vel"], data.qvel[3:6], atol=1e-7)
    np.testing.assert_allclose(terms["projected_gravity"], rotation.T @ [0, 0, -1], atol=1e-7)
    yaw_inv = np.array([[np.cos(1), np.sin(1), 0], [-np.sin(1), np.cos(1), 0], [0, 0, 1]])
    np.testing.assert_allclose(terms["reach_position_error"], yaw_inv @ [0.1, 0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(terms["reach_orientation_error"], q_roll, atol=1e-7)
    np.testing.assert_allclose(terms["final_reach_orientation_error"], q_pitch, atol=1e-7)


def test_world_goal_and_repeated_commands_do_not_reset_references(policy_factory):
    policy, data = policy_factory()
    goal = policy.references["left"].position + [0.2, 0.1, 0.15]
    quat = _quaternion([1, 0.1, 0.2, 0.3])
    policy.set_target_world("left", goal, quat)
    policy.observe(data)
    before = asdict(policy.references["left"])
    policy.set_target_world("left", goal.copy(), -quat)
    for key, value in before.items():
        np.testing.assert_allclose(getattr(policy.references["left"], key), value)
    data.qpos[:3] += [1, 2, 3]
    mujoco.mj_forward(policy.model, data)
    policy.observe(data)
    np.testing.assert_array_equal(policy.references["left"].goal_position, goal)
    np.testing.assert_allclose(policy.references["left"].goal_quaternion, quat)


def test_follow_current_retains_history_and_previous_actions(policy_factory):
    policy, data = policy_factory()
    policy.last_action[:] = 0.4
    previous = policy.last_observation.copy()
    data.qpos[policy.body_map.qpos[14]] += 0.2
    mujoco.mj_forward(policy.model, data)
    policy.follow_current(data)
    np.testing.assert_array_equal(policy.last_observation, previous)
    np.testing.assert_allclose(policy.last_action, 0.4)
    for side in SIDES:
        ref = policy.references[side]
        p, q = policy.tcp_pose(data, side)
        np.testing.assert_array_equal(ref.position, p)
        np.testing.assert_array_equal(ref.goal_position, p)
        np.testing.assert_array_equal(ref.quaternion, q)
        np.testing.assert_array_equal(ref.linear_velocity, np.zeros(3))


def test_action_scaling_no_extra_normalization_or_raw_clipping(policy_factory):
    output = np.linspace(-5, 5, 28, dtype=np.float32)[None]
    policy, data = policy_factory(output=output)
    q_des = policy.act(data)
    np.testing.assert_array_equal(policy.session.last_feed[0], policy.last_observation)
    np.testing.assert_array_equal(policy.last_action, output[0])
    np.testing.assert_allclose(q_des, policy.default_joint_pos + 0.25*output[0])
    # ONNX output > 1 is deliberately preserved. Physical effort still clips.
    assert np.max(policy.last_action) == 5.0
    policy.act(data)
    actual = policy.last_observation[term_slices()["actions"]].reshape(10, 28)[-1]
    np.testing.assert_array_equal(actual, output[0])


def test_body_pd_clips_effort_and_leaves_hands_and_model_untouched(policy_factory):
    policy, data = policy_factory()
    hand_acts = np.concatenate([JointMap.create(policy.model, hand_names(side)).actuators for side in SIDES])
    data.ctrl[hand_acts] = np.linspace(-0.1, 0.1, 12)
    before = data.ctrl[hand_acts].copy()
    ranges = policy.model.jnt_range.copy()
    policy.q_des = data.qpos[policy.body_map.qpos] + np.linspace(-10, 10, 28)
    expected = np.clip(policy.kp*(policy.q_des-data.qpos[policy.body_map.qpos])
                       - policy.kd*data.qvel[policy.body_map.dofs], policy.torque_lower, policy.torque_upper)
    torque = policy.apply(data)
    np.testing.assert_allclose(torque, expected)
    np.testing.assert_allclose(data.ctrl[policy.body_map.actuators], expected)
    np.testing.assert_array_equal(data.ctrl[hand_acts], before)
    np.testing.assert_array_equal(policy.model.jnt_range, ranges)
    assert np.max(np.abs(torque)) > 100


@pytest.mark.parametrize("output", [np.full((1, 28), np.nan), np.zeros((28,))])
def test_invalid_policy_outputs_fail_before_writing_controls(policy_factory, output):
    policy, data = policy_factory(output=output)
    before = data.ctrl.copy()
    with pytest.raises(RuntimeError, match="ONNX returned"):
        policy.act(data)
    np.testing.assert_array_equal(data.ctrl, before)


def test_reference_first_step_and_reversal_match_training_equations():
    ref = ReachReference(np.zeros(3), np.array([1., 0, 0, 0]), np.zeros(3), np.zeros(3),
                         np.array([1., 0, 0]), np.array([np.cos(0.5), 0, 0, np.sin(0.5)]))
    ref.step()
    np.testing.assert_allclose(ref.linear_velocity, [0.016, 0, 0], atol=1e-12)
    np.testing.assert_allclose(ref.position, [0.00032, 0, 0], atol=1e-12)
    np.testing.assert_allclose(ref.angular_velocity, [0, 0, 0.04], atol=1e-12)
    np.testing.assert_allclose(ref.quaternion, [np.cos(0.0004), 0, 0, np.sin(0.0004)], atol=1e-12)
    for _ in range(25):
        ref.step()
    previous_v = ref.linear_velocity.copy()
    previous_w = ref.angular_velocity.copy()
    assert np.linalg.norm(previous_v) == pytest.approx(LINEAR_SPEED_LIMIT)
    assert np.linalg.norm(previous_w) == pytest.approx(ANGULAR_SPEED_LIMIT)
    ref.goal_position = np.array([-1., 0, 0])
    ref.goal_quaternion = np.array([np.cos(0.5), 0, 0, -np.sin(0.5)])
    ref.step()
    assert np.linalg.norm(ref.linear_velocity-previous_v) == pytest.approx(LINEAR_ACCELERATION_LIMIT*POLICY_DT)
    assert np.linalg.norm(ref.angular_velocity-previous_w) == pytest.approx(ANGULAR_ACCELERATION_LIMIT*POLICY_DT)
    assert ref.linear_velocity[0] > 0  # No instantaneous reversal.
    assert ref.angular_velocity[2] > 0


def test_reference_shortest_rotation_world_left_multiplication_and_final_snap():
    initial = _quaternion([0.8, 0.2, -0.1, 0.3])
    world_delta = np.array([np.cos(0.15), np.sin(0.15), 0, 0])
    target = -_quat_mul(world_delta, initial)
    ref = ReachReference(np.zeros(3), initial.copy(), np.zeros(3), np.zeros(3),
                         np.array([1e-8, 0, 0]), target.copy())
    ref.step()
    np.testing.assert_array_equal(ref.position, ref.goal_position)
    np.testing.assert_array_equal(ref.linear_velocity, np.zeros(3))
    np.testing.assert_allclose(ref.angular_velocity, [0.04, 0, 0], atol=1e-12)
    expected = _quat_mul([np.cos(0.0004), np.sin(0.0004), 0, 0], initial)
    np.testing.assert_allclose(ref.quaternion, expected, atol=1e-12)
    for _ in range(250):
        ref.step()
    np.testing.assert_allclose(ref.quaternion, -target, atol=1e-10)
    np.testing.assert_array_equal(ref.angular_velocity, np.zeros(3))


@pytest.mark.parametrize("side,position,quaternion", [
    ("wrong", [0, 0, 0], [1, 0, 0, 0]),
    ("left", [0, np.nan, 0], [1, 0, 0, 0]),
    ("left", [0, 0, 0], [0, 0, 0, 0]),
])
def test_invalid_goals_rejected(policy_factory, side, position, quaternion):
    policy, data = policy_factory()
    with pytest.raises(ValueError):
        policy.set_target_world(side, position, quaternion)
