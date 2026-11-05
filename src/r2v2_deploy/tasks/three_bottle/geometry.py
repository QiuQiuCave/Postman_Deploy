"""Rigid transforms, bounded IK, and reference interpolation."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from common.r2v2_loco_cola import interpolate_pose as interpolate_pose
from r2v2_deploy.control.reach_policy import ReachPolicy as NativeReachPolicy
import hashlib as hashlib
import json as json
import mujoco as mujoco
from .resources import resource


def axis_rotation(axis, target, max_angle=None):
    axis = axis / np.linalg.norm(axis)
    target = target / np.linalg.norm(target)
    cross = np.cross(axis, target)
    n = np.linalg.norm(cross)
    if n < 1e-10:
        return np.eye(3)
    angle = float(np.arctan2(n, np.clip(axis @ target, -1.0, 1.0)))
    if max_angle is not None:
        angle = min(angle, max_angle)
    (x, y, z) = cross / n
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def smoothstep(x):
    f = float(np.clip(x, 0.0, 1.0))
    return f * f * (3.0 - 2.0 * f)


def rotation_z(yaw):
    (c, s) = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def mix_transform(a, b, fraction):
    (q0, q1) = (np.empty(4), np.empty(4))
    mujoco.mju_mat2Quat(q0, a[:3, :3].ravel())
    mujoco.mju_mat2Quat(q1, b[:3, :3].ravel())
    (pos, quat) = interpolate_pose((a[:3, 3], q0), (b[:3, 3], q1), fraction)
    out = np.eye(4)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    (out[:3, :3], out[:3, 3]) = (rotation.reshape(3, 3), pos)
    return out


def rotation_error(target, actual):
    inv = actual.copy()
    inv[1:] *= -1
    err = np.empty(4)
    mujoco.mju_mulQuat(err, target, inv)
    if err[0] < 0:
        err = -err
    n = np.linalg.norm(err[1:])
    return np.zeros(3) if n < 1e-10 else err[1:] * (2 * np.arctan2(n, err[0]) / n)


class Inventory50Hz(list):

    def append(self, value):
        if int(round(value["time_s"] / 0.01)) % 2 == 0:
            super().append(value)


class SampledLog(list):
    """Decimate history only; [-1] always remains the actual newest sample."""

    def __init__(self, stride=10):
        super().__init__()
        self.stride = stride
        self.count = 0
        self.extra = False

    def append(self, value):
        if self.extra and self:
            super().pop()
        self.count += 1
        super().append(value)
        self.extra = self.count % self.stride != 0


def bottle_bottom_clearance(m, d, side, bottle="test_cylinder"):
    b = m.body(bottle).id
    axis = d.xmat[b].reshape(3, 3)[:, 2]
    center = d.xpos[b]
    points = collision_hand_points(m, d, side)
    extent = 0.06 * abs(axis[2]) + 0.02 * np.sqrt(max(0.0, 1.0 - axis[2] ** 2))
    bottom = float(center[2] - extent)
    return dict(
        hand_lowest_world_z_m=float(np.min(points[:, 2])),
        bottle_lowest_world_z_m=bottom,
        hand_above_bottle_bottom_m=float(np.min(points[:, 2]) - bottom),
        hand_above_bottle_bottom_along_axis_m=float(
            np.min((points - center) @ axis) + 0.06
        ),
        collision_surface_points=len(points),
    )


def matrix_targets(values, R, t):
    out = np.tile(np.eye(4), (2, 1, 1))
    for i, v in enumerate(values):
        r = np.empty(9)
        mujoco.mju_quat2Mat(r, v[3:])
        out[i, :3, :3] = R @ r.reshape(3, 3)
        out[i, :3, 3] = R @ v[:3] + t
    return out


def bounded_step(J, error, regularization, posture, lower, upper):
    """Small active-set bounded linear least squares, no SciPy dependency."""
    matrix = np.vstack((J, regularization * np.eye(J.shape[1])))
    target = np.r_[error, regularization * posture]
    step = np.zeros(J.shape[1])
    free = np.ones(J.shape[1], dtype=bool)
    for _ in range(J.shape[1] + 1):
        ids = np.flatnonzero(free)
        if not len(ids):
            break
        step[ids] = np.linalg.lstsq(
            matrix[:, ids], target - matrix[:, ~free] @ step[~free], rcond=None
        )[0]
        violation = np.maximum(lower - step, step - upper)
        bad = ids[violation[ids] > 1e-12]
        if not len(bad):
            break
        k = bad[np.argmax(violation[bad])]
        step[k] = np.clip(step[k], lower[k], upper[k])
        free[k] = False
    return np.clip(step, lower, upper)


class WristReferencePolicy(NativeReachPolicy):
    INPUT_DIM = 1461
    INPUT_NAMES = (*NativeReachPolicy.INPUT_NAMES, "left_finger_closure_progress")

    def __init__(self, *args, **kwargs):
        self.closure_provider = lambda: 0.0
        NativeReachPolicy.__init__(self, *args, **kwargs)

    def observe(self, data, advance_reference=True, *, update_history=None):
        body = super().observe(data, advance_reference, update_history=update_history)
        condition = float(self.closure_provider())
        if not np.isfinite(condition) or not 0 <= condition <= 1:
            raise ValueError("Invalid finger reference progress")
        self.last_observation = np.r_[body, np.float32(condition)].astype(np.float32)
        return self.last_observation.copy()


def loaded_loco_tests_wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def turn_command(error, elapsed, maximum=0.2, minimum=0.0):
    rate = np.clip(0.8 * error, -maximum, maximum)
    if abs(error) > np.deg2rad(3.0):
        rate = np.sign(error) * max(abs(rate), minimum)
    return float(rate * min(1.0, max(0.0, elapsed / 0.6)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenPath:

    def __init__(self):
        directory = resource("@skills/dual_wrist_vertical_grip_stop4/path_snapshot")
        self.manifest = json.loads((directory / "manifest.json").read_text())
        archive = directory / self.manifest["trajectory_file"]
        if digest(archive) != self.manifest["trajectory_sha256"]:
            raise ValueError("Original desired-path hash differs")
        if self.manifest["scene_profile"] != "vertical_grip_crate_36_v1":
            raise ValueError("Not the published vertical-grip scene")
        with np.load(archive, allow_pickle=False) as data:
            self.time = data["time_s"].copy()
            self.targets = data["T_world_wrist_goal"].copy()
            self.phases = data["phase"].copy()
        self.duration = 15.4 + float(self.time[-1]) + 2.0
        self.home = np.array(
            [[0.246, 0.19698, 1.087902013081883], [0.246, -0.19643, 1.087902013081883]]
        )

    def sample(self, time_s):
        if time_s < 0.4 - 1e-09:
            return ("STAND", None)
        if time_s < 15.4 - 1e-09:
            fraction = np.clip((time_s - 5.4) / 8, 0, 1)
            fraction = fraction * fraction * (3 - 2 * fraction)
            result = self.targets[0].copy()
            for i in range(2):
                q = np.empty(4)
                mujoco.mju_mat2Quat(q, self.targets[0, i, :3, :3].ravel())
                (p, q) = interpolate_pose(
                    (self.home[i], np.array([1.0, 0, 0, 0])),
                    (self.targets[0, i, :3, 3], q),
                    fraction,
                )
                r = np.empty(9)
                mujoco.mju_quat2Mat(r, q)
                (result[i, :3, 3], result[i, :3, :3]) = (p, r.reshape(3, 3))
            return (
                (
                    "HOME"
                    if time_s < 5.4
                    else "SAFE_WAIT" if time_s < 13.4 else "START_HOLD"
                ),
                result,
            )
        clock = np.clip(time_s - 15.4, 0, self.time[-1])
        upper = int(
            np.clip(
                np.searchsorted(self.time, clock, side="right"), 1, len(self.time) - 1
            )
        )
        lower = upper - 1
        weight = (clock - self.time[lower]) / (self.time[upper] - self.time[lower])
        result = self.targets[lower].copy()
        for i in range(2):
            (q0, q1) = (np.empty(4), np.empty(4))
            mujoco.mju_mat2Quat(q0, self.targets[lower, i, :3, :3].ravel())
            mujoco.mju_mat2Quat(q1, self.targets[upper, i, :3, :3].ravel())
            (p, q) = interpolate_pose(
                (self.targets[lower, i, :3, 3], q0),
                (self.targets[upper, i, :3, 3], q1),
                weight,
            )
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, q)
            (result[i, :3, 3], result[i, :3, :3]) = (p, rotation.reshape(3, 3))
        index = int(
            np.clip(
                np.searchsorted(self.time, clock + 1e-09, side="right") - 1,
                0,
                len(self.time) - 1,
            )
        )
        return (str(self.phases[index]), result)


def collision_hand_points(m, d, side):
    root = m.body(side + "_hand_roll_link").id
    points = []
    for g in range(m.ngeom):
        if not descendant(m, int(m.geom_bodyid[g]), root) or not (
            m.geom_contype[g] or m.geom_conaffinity[g]
        ):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        kind = int(m.geom_type[g])
        size = m.geom_size[g]
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(m.geom_dataid[g])
            a = int(m.mesh_vertadr[mesh])
            n = int(m.mesh_vertnum[mesh])
            local = m.mesh_vert[a : a + n]
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            local = np.array(
                [
                    [x, y, z]
                    for x in [-size[0], size[0]]
                    for y in [-size[1], size[1]]
                    for z in [-size[2], size[2]]
                ]
            )
        else:
            raise ValueError((m.geom(g).name, kind))
        points.append(local @ R.T + p)
    return np.vstack(points)


def descendant(m, b, ancestor):
    while b > 0:
        if b == ancestor:
            return True
        b = int(m.body_parentid[b])
    return False
