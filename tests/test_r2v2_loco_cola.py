import numpy as np
from types import SimpleNamespace

from common.r2v2_front_drop import pose
from common.r2v2_loco_cola import grasp_target, interpolate_pose, LocoColaExperiment
from common.r2v2_loco_cola import destination_layout
from common.r2v2_loco_cola import unload_height
from common.r2v2_loco_cola import fingers_clear
import pytest


def test_grasp_transform_is_wrist_not_can_center():
    can=pose([.45,.13,1.04],[1.,0,0,0])
    relation=pose([.145,-.035,0],[1.,0,0,0])
    wrist=grasp_target(can,relation)
    np.testing.assert_allclose(wrist[:3,3],[.305,.165,1.04])
    np.testing.assert_allclose(wrist@relation,can)


def test_interpolation_shortest_arc_and_no_mutation():
    start=(np.zeros(3),np.array([1.,0,0,0]))
    end=(np.ones(3),np.array([-1.,0,0,0]))
    p,q=interpolate_pose(start,end,.15)
    np.testing.assert_allclose(p,.15)
    np.testing.assert_allclose(q,[1,0,0,0])
    np.testing.assert_array_equal(end[1],[-1,0,0,0])


def test_stable_requires_consecutive_samples():
    exp=object.__new__(LocoColaExperiment);exp.stable_s=0.
    for _ in range(14): assert not exp.stable(True,.3)
    assert not exp.stable(False,.3)
    for _ in range(14): assert not exp.stable(True,.3)
    assert exp.stable(True,.3)


def test_grasp_pose_gate_requires_speed_and_orientation():
    exp=object.__new__(LocoColaExperiment)
    good=dict(position_error_m=.004,orientation_error_deg=2.,speed_mps=.01)
    row=dict(wrists={'left':good})
    assert exp.pose_good(row,.005,3.)
    good['speed_mps']=.03
    assert not exp.pose_good(row,.005,3.)
    good.update(speed_mps=.01,orientation_error_deg=4.)
    assert not exp.pose_good(row,.005,3.)


def test_grasp_slip_reference_waits_for_settling_and_never_relatches():
    exp=object.__new__(LocoColaExperiment)
    exp.phase='PROBE'; exp.phase_start=0.; exp.stable_s=0.
    exp.data=SimpleNamespace(time=1.); exp.config={'stage_timeout_s':10.}
    exp.baseline_relation=None; exp.baseline_time=None
    good=dict(position_error_m=.003,orientation_error_deg=1.,speed_mps=.01)
    row=dict(velocity_body=[0,0,0],angular_velocity_body=[0,0,0],wrists={'left':good},
             contacts=dict(opposed=True,table_contact=False,floor_contact=False,robot_table_contacts=[]),
             bottom_clearance_m=.02,hand_vertical_force_N=.98,object_speed_mps=.1,
             object_angular_speed_radps=.1,relative=np.eye(4).tolist(),slip_m=None,slip_deg=None)
    exp.gate(row)
    assert exp.baseline_relation is None
    row['object_speed_mps']=.01
    exp.gate(row)
    np.testing.assert_array_equal(exp.baseline_relation,np.eye(4))
    row['relative'][0][3]=.03
    exp.gate(row)
    np.testing.assert_array_equal(exp.baseline_relation,np.eye(4))


def test_explicit_destination_and_edge_rejection():
    cfg=dict(table_center_xyz=[.6,.2,.96],table_half_size=[.2,.16,.02],cylinder_position_xyz=[.45,.13,1.04],
             destination_table_center_xyz=[-.45,.4,.96],destination_table_half_size=[.2,.12,.02],
             destination_can_position_xyz=[-.55,.37,1.04])
    center,half,target=destination_layout(cfg)
    np.testing.assert_allclose(target,[-.55,.37,1.04])
    np.testing.assert_allclose(center,[-.45,.4,.96])
    cfg['destination_can_position_xyz']=[-.65,.37,1.04]
    with pytest.raises(ValueError,match='fit'): destination_layout(cfg)


def test_backward_route_does_not_command_lateral_or_turn():
    exp=object.__new__(LocoColaExperiment)
    exp.phase='CARRY_BACK';exp.phase_start=0.;exp.data=SimpleNamespace(time=1.)
    exp.scratch=SimpleNamespace(xpos=np.array([[-.8,0,.8]]));exp.base=0
    exp.loco_start=np.array([0.,0.,.8]);exp.loco=SimpleNamespace(command=np.zeros(3))
    exp.config=dict(carry_distance_m=1.,stage_timeout_s=10.)
    exp.stop_after='place';entered=[];exp.enter=entered.append
    row=dict(velocity_body=[-.2,0,0],angular_velocity_body=[0,0,0],contacts=dict(opposed=False),bottom_clearance_m=.1)
    exp.gate(row)
    np.testing.assert_array_equal(exp.loco.command,[-.2,0,0]);assert entered==[]
    exp.scratch.xpos[0,0]=-1.01
    exp.gate(row);assert entered==['STOP_AT_B']


def test_contact_unloading_moves_up_under_excess_pressure_and_is_bounded():
    assert unload_height(1.,1.,8.)>1.
    assert unload_height(1.,1.,0.)<1.
    assert unload_height(1.,1.,.981)==1.
    assert unload_height(1.02,1.,100.)==1.02
    assert unload_height(.99,1.,0.)==.99


def test_place_requires_destination_support_position_tilt_and_settling():
    exp=object.__new__(LocoColaExperiment)
    exp.config=dict(table_center_xyz=[.6,.2,.96],table_half_size=[.2,.16,.02],cylinder_position_xyz=[.45,.13,1.04],
             destination_table_center_xyz=[-.45,.4,.96],destination_table_half_size=[.2,.12,.02],
             destination_can_position_xyz=[-.55,.37,1.04])
    row=dict(contacts=dict(destination_table_contact=True,floor_contact=False),object_position_world=[-.55,.37,1.04],
             object_tilt_deg=2.,object_speed_mps=.001,object_angular_speed_radps=.01)
    assert exp.placed(row)
    for key,value in [('object_tilt_deg',11.),('object_speed_mps',.03),('object_angular_speed_radps',.2),
                      ('object_position_world',[-.55,.40,1.04])]:
        bad={**row,key:value};assert not exp.placed(bad)
    row['contacts']['destination_table_contact']=False
    assert not exp.placed(row)


def test_open_can_withdraw_from_palm_contact_but_not_finger_contact():
    contacts=dict(contacts=[dict(active=True,part='palm'),dict(active=True,part=None)])
    assert fingers_clear(contacts)
    contacts['contacts'].append(dict(active=True,part='thumb'))
    assert not fingers_clear(contacts)
    contacts['contacts'][-1]['active']=False
    assert fingers_clear(contacts)
