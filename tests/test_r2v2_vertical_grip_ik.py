"""Recorded-fixture transfer and full-robot preservation checks; not dynamics."""
import copy
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from r2v2_description.model import BODY_JOINTS, build_model, load_config
from tools.check_r2v2_vertical_grip_ik import (
    DEFAULT_SOURCE, DEFAULT_SCENE, RecordedGrip, VerticalGripScreen,
    mesh_box_separation_bound, checked_distance,
)


@pytest.fixture(scope='module')
def recording():
    if not (DEFAULT_SOURCE/'trajectory.npz').exists() or not DEFAULT_SCENE.exists():
        pytest.skip('Local archived fixture/scene evidence is not installed')
    return RecordedGrip(DEFAULT_SOURCE)


@pytest.fixture(scope='module')
def screen(recording):
    return VerticalGripScreen(recording,json.loads(DEFAULT_SCENE.read_text()))


def test_full_robot_has_original_ranges_inertias_and_only_mimics(screen):
    source=build_model(load_config(),fixture=False)
    model=screen.model
    assert (model.nq,model.nv,model.nu,model.neq,model.nmocap)==(64,62,40,10,0)
    assert np.all(model.eq_type==mujoco.mjtEq.mjEQ_JOINT)
    for name in BODY_JOINTS:
        np.testing.assert_array_equal(model.joint(name).range,source.joint(name).range)
    for i in range(1,source.nbody):
        name=source.body(i).name
        np.testing.assert_array_equal(model.body(name).mass,source.body(i).mass)
        np.testing.assert_array_equal(model.body(name).inertia,source.body(i).inertia)
    assert not ET.fromstring(screen.xml).findall('equality/weld')


def test_preserves_beverage_table_and_shell_world_pose(screen):
    scene=screen.scene
    np.testing.assert_allclose(screen.model.body('tabletop').pos,scene['table_center_world_m'])
    np.testing.assert_allclose(screen.model.geom('lift_table_geom').size,scene['table_half_size_m'])
    np.testing.assert_allclose(screen.origin,[.12,.44,0.])
    assert scene['tabletop_height_m']==.98
    assert screen.model.body('cargo_crate').mass[0]==pytest.approx(.4)


@pytest.mark.parametrize('time',[0.,3.49,6.49,12.5,24.5])
def test_relative_wrist_and_finger_trajectory_transfers_exactly(screen,time):
    record=screen.recording
    source_crate,source_wrists,source_fingers,_=record.sample(time,np.eye(4))
    crate,wrists=screen.targets(time)
    np.testing.assert_allclose(np.linalg.inv(crate)@wrists,np.linalg.inv(source_crate)@source_wrists,atol=1e-12)
    np.testing.assert_array_equal(screen.data.qpos[screen.hand_qadr],source_fingers)
    q=screen.data.qpos[screen.crate_qadr:screen.crate_qadr+7]
    np.testing.assert_allclose(q[:3],crate[:3,3])
    np.testing.assert_allclose(Rotation.from_quat(q[3:][[1,2,3,0]]).as_matrix(),crate[:3,:3],atol=1e-12)


def test_scene_dimension_mismatch_is_rejected(recording):
    scene=json.loads(DEFAULT_SCENE.read_text())
    scene=copy.deepcopy(scene)
    scene['scene']['crate_dimensions_depth_width_height_m'][1]=.26
    with pytest.raises(ValueError,match='dimensions'):
        VerticalGripScreen(recording,scene)


def test_source_sampling_rejects_extrapolation(recording):
    with pytest.raises(ValueError,match='outside'):
        recording.sample(-1,np.eye(4))


def test_separating_plane_corrects_only_proven_false_zero(monkeypatch):
    vertices=' '.join(str(v) for p in __import__('itertools').product((-.01,.01),repeat=3) for v in p)
    m=mujoco.MjModel.from_xml_string(f'<mujoco><asset><mesh name="hull" vertex="{vertices}"/></asset>'
        '<worldbody><geom name="a" type="mesh" mesh="hull" pos="0 0 .05"/>'
        '<geom name="b" type="box" size=".1 .1 .02"/></worldbody></mujoco>')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert mesh_box_separation_bound(m,d,0,1)==pytest.approx(.02,abs=1e-8)
    monkeypatch.setattr(mujoco,'mj_geomDistance',lambda *a:0.)
    assert checked_distance(m,d,0,1)==pytest.approx(.02,abs=1e-8)
    m.geom_pos[0,2]=.015;mujoco.mj_forward(m,d)
    assert mesh_box_separation_bound(m,d,0,1)==0.
    assert checked_distance(m,d,0,1)==0.  # Real overlap must never be erased.
