"""Small contract checks; real inference parity is a separate artifact."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from common.r2v2_loco_policy import LocoPolicy, LOCO_JOINTS, TERM_DIMS
from r2v2_description.model import build_model_xml, load_config, JointMap, BODY_JOINTS, hand_names


@pytest.fixture
def instance(monkeypatch):
    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def get_modelmeta(self):
            return SimpleNamespace(custom_metadata_map={
                'joint_names': ','.join(LOCO_JOINTS),
                'observation_names': ','.join(TERM_DIMS),
                'default_joint_pos': ','.join(['0']*14),
                'joint_stiffness': ','.join(['100']*14),
                'joint_damping': ','.join(['5']*14), 'action_scale': '.25'})

        def get_inputs(self):
            return [SimpleNamespace(shape=[1,560])]

        def get_outputs(self):
            return [SimpleNamespace(shape=[1,14])]

        def run(self, names, inputs):
            return [np.zeros((1,14),dtype=np.float32)]

    monkeypatch.setattr('common.r2v2_loco_policy.ort.InferenceSession',Session)
    model=mujoco.MjModel.from_xml_string(build_model_xml(load_config()))
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    return model,data,LocoPolicy(model,data,'unused.onnx')


def test_no_arm_or_finger_observation(instance):
    model,data,policy=instance
    before=policy.terms(data)
    other=JointMap.create(model,BODY_JOINTS[14:]+hand_names('left')+hand_names('right'))
    data.qpos[other.qpos]+=.02
    data.qvel[other.dofs]=.13
    mujoco.mj_forward(model,data)
    for key,value in before.items():
        np.testing.assert_allclose(policy.terms(data)[key],value,atol=1e-7)


def test_history_is_term_major_oldest_first(instance):
    _,data,policy=instance
    policy.command[:]=[.1,.2,.3]
    obs=policy.observe(data)
    offset=sum(list(TERM_DIMS.values())[:6])*10
    history=obs[offset:offset+30].reshape(10,3)
    np.testing.assert_array_equal(history[:-1],0)
    np.testing.assert_allclose(history[-1],[.1,.2,.3])


def test_act_does_not_write_simulator_channels(instance):
    _,data,policy=instance
    data.ctrl[:]=.123
    before=data.ctrl.copy()
    assert policy.act(data).shape==(14,)
    np.testing.assert_array_equal(data.ctrl,before)


def test_handoff_initializes_previous_action_from_executed_target(instance):
    _,data,policy=instance
    policy.reset(data,policy.default+.05)
    np.testing.assert_allclose(policy.last_action,.2)
    np.testing.assert_allclose(policy.histories['actions'],.2)
