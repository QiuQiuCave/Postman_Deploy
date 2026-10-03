"""Frontal wrist goals from the archived 75-degree-thumb cylinder calibration.

Only the initial wrist/object relation is calibrated by the fixed-wrist test.
The approach/carry/place path below is new planning, NOT recorded whole-body
success or a collision-free certificate. No measured joint trajectory is used.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from common.r2v2_top_grasp_path import PHASES, quaternion_wxyz


DEFAULT_FRONT_CALIBRATION_PATH = (Path(__file__).resolve().parents[1] /
    'reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/left/grasp_record.json')
FRONT_RETREAT_OFFSET_M = np.array([-.04, .12, .04])


def load_front_grasp_calibration(path=None):
    """Read the initial relation, never the settled/grasped relation or qpos."""
    source = Path(path or DEFAULT_FRONT_CALIBRATION_PATH)
    raw = source.read_bytes()
    record = json.loads(raw)
    if (record.get('schema_version') != 1 or record.get('side') != 'left'
            or record.get('wrist_body') != 'left_hand_roll_link'
            or record.get('validation', {}).get('grasp_passed') is not True):
        raise ValueError('Expected the verified left-hand frontal wrist calibration')
    params = record['experiment_parameters']
    if any(not np.isclose(params[key], value, atol=1e-12, rtol=0.)
           for key, value in (('radius', .02), ('height', .12), ('mass', .1))):
        raise ValueError('Frontal calibration requires the original 40 mm / 120 mm / 100 g object')
    initial = record['keyframes']['initial']
    relation = initial['cylinder_in_wrist']
    p, q = np.asarray(relation['position_m']), np.asarray(relation['quaternion_wxyz'])
    if (initial.get('index') != 0 or initial.get('time_s') != 0.
            or initial.get('command_during_preceding_step') != 0
            or p.shape != (3,) or q.shape != (4,)
            or not np.allclose(p, [.145, -.035, 0.], atol=1e-10, rtol=0.)
            or not np.allclose(q, [1., 0., 0., 0.], atol=1e-10, rtol=0.)):
        raise ValueError('Frontal grasp must use the archived initial wrist/cylinder relation')
    transform = np.eye(4)
    transform[:3, 3] = p
    return dict(schema_version=1, grasp_style='front', side='left',
        source_record_path=str(source.resolve()), source_record_sha256=hashlib.sha256(raw).hexdigest(),
        source_scope=record['scope'], source_grasp_verified=True,
        T_wrist_cylinder_initial=transform.tolist(),
        hand_configuration=copy.deepcopy(record['hand_configuration']),
        initial_relation_is_calibrated=True, transport_path_is_recorded=False,
        is_whole_body_validation=False, is_collision_free_certificate=False)


def front_world_path_for_scene(table_height_m, can_xy_m, yaw_deg=0., calibration=None):
    """Fixed-anchor world-wrist path: lateral palm-normal approach, no flip.

    The initial cylinder is 1 mm above the table, like the real scene. PLACE
    removes this initial drop gap. UPRIGHT/PLACE in CONTACT must be computed
    from actual wrist/object measurements; these are only nominal AIR poses.
    This first frontal variant intentionally permits zero world yaw only.
    """
    calibration = load_front_grasp_calibration() if calibration is None else copy.deepcopy(calibration)
    xy = np.asarray(can_xy_m, dtype=float)
    if (xy.shape != (2,) or not np.isfinite(xy).all()
            or isinstance(table_height_m, bool) or not np.isfinite(table_height_m) or table_height_m <= 0.
            or isinstance(yaw_deg, bool) or not np.isfinite(yaw_deg) or abs(yaw_deg) > 1e-12):
        raise ValueError('Frontal path requires finite table/XY and yaw_deg=0 (no wrist flip)')
    relation = np.asarray(calibration['T_wrist_cylinder_initial'], dtype=float)
    expected = np.eye(4); expected[:3, 3] = [.145, -.035, 0.]
    if (calibration.get('grasp_style') != 'front' or relation.shape != (4, 4)
            or not np.allclose(relation, expected, atol=1e-10, rtol=0.)):
        raise ValueError('Unexpected frontal wrist/object calibration')
    anchor = np.eye(4); anchor[:3, 3] = [*xy, float(table_height_m)+.061]
    grasp = anchor @ np.linalg.inv(relation)
    offsets = (
        [0., .05, .04],  # HOVER alias: exterior alignment, not an upper grasp.
        [0., .05, 0.],  # Palm faces -Y; approach from +Y, not finger-axis -X.
        [0., 0., 0.],
        [0., 0., .02],
        [0., 0., .02],  # Nominal already-upright probe.
        [0., 0., .08],
        [0., -.10, .08],
        [0., -.10, -.001],
        (np.array([0., -.10, -.001])+FRONT_RETREAT_OFFSET_M).tolist(),
    )
    commands = (0, 0, 1, 1, 1, 1, 1, 1, 0)
    result = []
    for name, offset, command in zip(PHASES, offsets, commands):
        goal = grasp.copy(); goal[:3, 3] += offset
        feedback = name in ('upright', 'place')
        result.append(dict(name=name, T_world_wrist_goal=goal,
            position_m=goal[:3, 3].copy(), quaternion_wxyz=quaternion_wxyz(goal),
            T_initial_object_wrist_goal=np.linalg.inv(anchor) @ goal,
            fixed_initial_object_pose=anchor.copy(), nominal_hand_command=command,
            contact_feedback_required=feedback, feedback_required=feedback,
            derived_waypoint=True, source_recorded_transport=False,
            meaning='Planned frontal goal from archived initial relation; actual state gates required'))
    return result


__all__ = ['DEFAULT_FRONT_CALIBRATION_PATH', 'FRONT_RETREAT_OFFSET_M',
           'load_front_grasp_calibration', 'front_world_path_for_scene']
