"""Rigid-pose and physical-contact geometry used by the native controller."""

import numpy as np
import mujoco


def blend(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def rotation_error_deg(a, b):
    return float(
        np.rad2deg(np.arccos(np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)))
    )


def pose_quaternion(transform):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(transform[:3, :3]).ravel())
    return result


def interpolate_pose(start, end, fraction):
    x = blend(fraction)
    result = np.eye(4)
    result[:3, 3] = (1.0 - x) * start[:3, 3] + x * end[:3, 3]
    q1, q2 = (pose_quaternion(start), pose_quaternion(end))
    if q1 @ q2 < 0:
        q2 = -q2
    dot = np.clip(q1 @ q2, -1.0, 1.0)
    if dot > 0.9995:
        q = (1.0 - x) * q1 + x * q2
        q /= np.linalg.norm(q)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1.0 - x) * angle) * q1 + np.sin(x * angle) * q2) / np.sin(angle)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q)
    result[:3, :3] = matrix.reshape(3, 3)
    return result


def contact_opposition_angle(thumb_normals, finger_normals):
    """Largest cross-group angle of normals acting on the object, in degrees.

    Directional opposition is a useful filter, not a force-closure proof.
    Both sets must use the same frame and the same force-on-object convention.
    """
    groups = []
    for values in (thumb_normals, finger_normals):
        normals = np.asarray(values, dtype=float)
        if normals.shape == (0,):
            normals = normals.reshape(0, 3)
        if (
            normals.ndim != 2
            or normals.shape[1] != 3
            or (not np.all(np.isfinite(normals)))
        ):
            raise ValueError("Contact normals must be finite arrays of shape (N, 3)")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError("Contact normals must be nonzero")
        groups.append(normals / lengths[:, None])
    if any((len(group) == 0 for group in groups)):
        return None
    return float(
        np.rad2deg(np.arccos(np.clip(np.min(groups[0] @ groups[1].T), -1.0, 1.0)))
    )


def unsupported_load_conditions(metrics, weight_N):
    """Real, stationary load support before any optional upright correction."""
    m = metrics
    return bool(
        m["finite_state"]
        and m["opposed_contact"]
        and (m["clearance_m"] >= 0.008)
        and (not m["table_contact"])
        and (not m["floor_contact"])
        and (m["hand_table_contact_count"] == 0)
        and (m["parked_hand_contact_count"] == 0)
        and (m["hand_vertical_force_N"] >= 0.8 * weight_N)
        and (m["object_tilt_deg"] <= 30.0)
        and (m["object_linear_speed_mps"] < 0.02)
        and (m["object_angular_speed_radps"] < 0.15)
    )
