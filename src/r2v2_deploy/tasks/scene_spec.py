"""Hash-validated scene specification and native initialization settings."""

import copy
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from r2v2_deploy.simulation.reach_sim import resolve_asset

SCOPE = "tabletop_to_crate_drop_v1"

PHASES = (
    "STAND",
    "HOME",
    "SAFE_OUT",
    "APPROACH",
    "GRASP",
    "PROBE",
    "LIFT_CLEAR",
    "TRANSFER_ABOVE",
    "DROP_RELEASE",
    "RETREAT_HIGH",
    "RETURN_HOME",
)

DEFAULT_CONFIG = dict(
    schema_version=1,
    reach_config=None,
    parity_report=None,
    path_manifest=None,
    air_evidence=None,
    stage_timeout_s=10.0,
    startup_mode="hold_home",
    delivery_background=False,
    pickup_tray_height_m=0.0,
)


def canonical_digest(document):
    return hashlib.sha256(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class FrontDropPath:
    """NumPy mirror of the archived target interpolation, not a qpos record."""

    def __init__(self, filename):
        self.document = json.loads(Path(filename).read_text())
        d = self.document
        self.sha256 = canonical_digest(
            {k: v for k, v in d.items() if k != "content_sha256"}
        )
        if (
            d.get("content_sha256") != self.sha256
            or d.get("schema_version") != 1
            or d.get("endpoint_contract") != "wrist_world_v2"
            or (d.get("path_contract") != "front_manipulation_path_v1")
            or (d.get("task_scope") != SCOPE)
            or (d.get("geometry_validation", {}).get("passed") is not True)
        ):
            raise ValueError("Invalid certified front-drop path contract/hash")
        self.phases = d["phases"]
        if tuple((p["name"] for p in self.phases)) != PHASES:
            raise ValueError("Unexpected front-drop phase sequence")
        self.indices = {p["name"]: i for i, p in enumerate(self.phases)}
        self.positions = np.asarray([p["position_m"] for p in self.phases], dtype=float)
        self.quaternions = np.asarray(
            [p["quaternion_wxyz"] for p in self.phases], dtype=float
        )
        if (
            self.positions.shape != (len(PHASES), 2, 3)
            or self.quaternions.shape != (len(PHASES), 2, 4)
            or (not np.isfinite(self.positions).all())
            or (not np.isfinite(self.quaternions).all())
            or (
                not np.allclose(
                    np.linalg.norm(self.quaternions, axis=-1), 1.0, atol=1e-06
                )
            )
        ):
            raise ValueError("Invalid wrist target arrays")
        if not np.allclose(
            self.positions[:, 1], self.positions[0, 1], atol=1e-09, rtol=0
        ):
            raise ValueError("Inactive wrist must remain world-fixed")
        if not np.allclose(
            np.abs(self.quaternions[:, 1] @ self.quaternions[0, 1]),
            1.0,
            atol=1e-09,
            rtol=0,
        ):
            raise ValueError("Inactive wrist orientation must remain world-fixed")
        for p in self.phases:
            if (
                not 0.0 <= p["motion_s"] <= p["duration_s"] <= 10.0
                or p["duration_s"] <= 0.0
            ):
                raise ValueError("Invalid path duration")
        self.scene = copy.deepcopy(d["provenance"]["scene"])
        scene = self.scene
        can = pose(scene["can_initial_world_m"], scene["can_initial_quaternion_wxyz"])
        relative = pose(
            scene["grasp_can_in_wrist_position_m"],
            scene["grasp_can_in_wrist_quaternion_wxyz"],
        )
        expected = can @ np.linalg.inv(relative)
        if not np.allclose(expected, self.target("GRASP")[0], atol=1e-07):
            raise ValueError(
                "Archived grasp goal differs from wrist/object calibration"
            )

    def target(self, name):
        i = self.indices[name]
        return np.array(
            [pose(p, q) for p, q in zip(self.positions[i], self.quaternions[i])]
        )

    def sample_phase(self, name, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("Expected nonnegative finite path time")
        i = self.indices[name]
        duration = self.phases[i]["motion_s"]
        f = 1.0 if duration == 0.0 else float(np.clip(elapsed / duration, 0.0, 1.0))
        f = f * f * (3.0 - 2.0 * f)
        start = max(0, i - 1)
        position = (1.0 - f) * self.positions[start] + f * self.positions[i]
        result = []
        for p, q1, q2 in zip(position, self.quaternions[start], self.quaternions[i]):
            if q1 @ q2 < 0.0:
                q2 = -q2
            dot = float(np.clip(q1 @ q2, -1.0, 1.0))
            if dot > 0.9995:
                q = (1.0 - f) * q1 + f * q2
            else:
                angle = np.arccos(dot)
                q = (np.sin((1.0 - f) * angle) * q1 + np.sin(f * angle) * q2) / np.sin(
                    angle
                )
            result.append(pose(p, q / np.linalg.norm(q)))
        return np.asarray(result)


def pose(position, quaternion):
    result = np.eye(4)
    result[:3, 3] = position
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    result[:3, :3] = rotation.reshape(3, 3)
    return result


def load_config(source=None):
    values = (
        {}
        if source is None
        else json.loads(Path(source).read_text())
        if isinstance(source, (str, Path))
        else copy.deepcopy(source)
    )
    if not isinstance(values, dict) or set(values) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown front-drop configuration fields")
    result = {**DEFAULT_CONFIG, **values}
    from r2v2_deploy.configuration import resolve_archived_path

    for key in ("reach_config", "parity_report", "path_manifest", "air_evidence"):
        if result[key] is not None:
            result[key] = str(resolve_archived_path(result[key]))
    if type(result["delivery_background"]) is not bool:
        raise ValueError("delivery_background must be boolean")
    height = result["pickup_tray_height_m"]
    if (
        isinstance(height, bool)
        or not isinstance(height, (float, int))
        or (not np.isfinite(height))
        or (not 0.0 <= height <= 0.08)
        or (height and (not result["delivery_background"]))
    ):
        raise ValueError(
            "Pickup tray requires delivery background and height in [0,.08]"
        )
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValueError("Expected front-drop schema_version 1")
    if result["startup_mode"] not in ("follow_current", "hold_home"):
        raise ValueError("Expected explicit follow_current or hold_home startup")
    timeout = result["stage_timeout_s"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or (not np.isfinite(timeout))
        or (not 0.0 < timeout <= 10.0)
    ):
        raise ValueError("Stage deadline must be positive and at most 10s")
    for key in ("reach_config", "parity_report", "path_manifest"):
        result[key] = str(resolve_asset(result[key]))
    if result["air_evidence"] is not None:
        result["air_evidence"] = str(resolve_asset(result["air_evidence"]))
    return result


def cylinder_envelope_in_crate(
    object_transform,
    crate_transform,
    radius,
    half_height,
    dimensions,
    wall,
    opening_bounds=None,
):
    """Exact solid-cylinder support along crate axes, including held tilt."""
    local = np.linalg.inv(crate_transform) @ object_transform
    axial = np.clip(local[:3, 2], -1.0, 1.0)
    extent = half_height * np.abs(axial) + radius * np.sqrt(
        np.maximum(0.0, 1.0 - axial**2)
    )
    low, high = (local[:3, 3] - extent, local[:3, 3] + extent)
    inner = np.asarray(dimensions[:2]) / 2.0 - wall
    xy_margin = float(np.min(np.r_[low[:2] + inner, inner - high[:2]]))
    if opening_bounds is None:
        opening = inner.copy()
        opening[1] = dimensions[1] / 2.0 - max(wall, 0.01)
        opening_bounds = dict(min=-opening, max=opening)
    opening_margin = float(
        np.min(
            np.r_[
                low[:2] - np.asarray(opening_bounds["min"])[:2],
                np.asarray(opening_bounds["max"])[:2] - high[:2],
            ]
        )
    )
    return dict(
        local_center_m=local[:3, 3].tolist(),
        local_min_m=low.tolist(),
        local_max_m=high.tolist(),
        opening_xy_margin_m=opening_margin,
        interior_xy_margin_m=xy_margin,
        bottom_above_rim_m=float(low[2] - dimensions[2]),
    )
