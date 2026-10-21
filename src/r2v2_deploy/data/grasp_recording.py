"""Rigid-body transforms shared by contact measurements and rendering."""

import numpy as np
import mujoco

CONTACT_PARTS = ("palm", "thumb", "index", "middle", "ring", "pinky")


def body_transform(data, body_id):
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def relative_transform(parent, child):
    """T_parent_child: child coordinates -> parent coordinates."""
    result = np.eye(4)
    result[:3, :3] = parent[:3, :3].T @ child[:3, :3]
    result[:3, 3] = parent[:3, :3].T @ (child[:3, 3] - parent[:3, 3])
    return result


def transform_pose(transform):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(transform[:3, :3]).ravel())
    if quat[0] < 0:
        quat *= -1
    return {"position_m": transform[:3, 3].tolist(), "quaternion_wxyz": quat.tolist()}
