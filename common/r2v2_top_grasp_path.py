"""Portable world-wrist goals from the verified 100 g upper-grasp fixture run.

This contains target poses, never robot or object qpos replay. The initial object
frame is frozen when a path is relocated. In contact experiments the recorded
upright and placement poses are *not* feedback: recompute them from the actually
held wrist/object relation, and advance phases using actual state gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_CALIBRATION = (Path(__file__).resolve().parents[1] / "deploy_mujoco" /
                       "config" / "r2v2_top_grasp_calibration.json")
DEFAULT_CALIBRATION_PATH = DEFAULT_CALIBRATION
PHASES = ("hover", "approach", "grasp", "probe", "upright", "lift", "translate",
          "place", "retreat")
# Stable plateau phases preserve the exact reached *command* endpoint rather
# than the last pre-end sample of a quintic interpolation. VERIFY occurs twice.
SOURCE_PHASES = (
    ("hover", "READY", 0, 0),
    ("approach", "SETTLE", 0, 0),
    ("grasp", "CLOSE", 0, 1),
    ("probe", "VERIFY", 0, 1),
    ("upright", "VERIFY", 1, 1),
    ("lift", "HOLD_LIFT", 0, 1),
    ("translate", "HOLD_MOVED", 0, 1),
    ("place", "PLACE_SETTLE", 0, 1),
    ("retreat", "FINAL_HOLD", 0, 0),
)


def _transform(value, label="pose"):
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite 4 x 4 rigid transform")
    r = result[:3, :3]
    if (not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-9, rtol=0.)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-7, rtol=0.)
            or not np.isclose(np.linalg.det(r), 1., atol=1e-7, rtol=0.)):
        raise ValueError(f"{label} must have a proper orthonormal rotation")
    return result.copy()


def _yaw_transform(yaw_deg):
    if not np.isfinite(yaw_deg):
        raise ValueError("yaw_deg must be finite")
    angle = np.deg2rad(float(yaw_deg))
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    return result


def quaternion_wxyz(transform):
    """Deterministic unit quaternion (w >= 0), without simulation dependencies."""
    m = _transform(transform)[:3, :3]
    # Symmetric eigenproblem is well conditioned even at pi rotations.
    xx, yx, zx = m[:, 0]
    xy, yy, zy = m[:, 1]
    xz, yz, zz = m[:, 2]
    k = np.array([[xx-yy-zz, yx+xy, zx+xz, zy-yz],
                  [yx+xy, yy-xx-zz, zy+yz, xz-zx],
                  [zx+xz, zy+yz, zz-xx-yy, yx-xy],
                  [zy-yz, xz-zx, yx-xy, xx+yy+zz]]) / 3.
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    if q[0] < 0.:
        q = -q
    return q


def calibration_sha256(calibration):
    """Digest all portable metadata/goals, excluding the digest field itself."""
    payload = {key: value for key, value in calibration.items() if key != "content_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_calibration(calibration, verify):
    if calibration.get("schema_version") != 1:
        raise ValueError("Unsupported top-grasp calibration schema")
    if verify and calibration.get("content_sha256") != calibration_sha256(calibration):
        raise ValueError("Top-grasp calibration content SHA256 mismatch")
    if calibration.get("side") != "left" or calibration.get("object_profile_id") != "baseline_40mm_100g":
        raise ValueError("This verified calibration supports only the left-hand 100 g baseline")
    if calibration.get("reference_kind") != "commanded_wrist_goals_anchored_to_initial_object":
        raise ValueError("Calibration must contain commanded wrist targets, not actual pose replay")
    initial = _transform(calibration["source_initial_object_pose"], "source initial object pose")
    if not np.allclose(initial[:3, 2], [0., 0., 1.], atol=1e-7, rtol=0.):
        raise ValueError("The calibrated source object must be upright")
    phases = calibration.get("waypoints", [])
    if tuple(row.get("name") for row in phases) != PHASES:
        raise ValueError("Calibration has missing, repeated or reordered phases")
    for row in phases:
        _transform(row["T_initial_object_wrist_goal"], row["name"])
        if row.get("nominal_hand_command") not in (0, 1):
            raise ValueError("Nominal hand commands must be binary")
        if row["name"] in ("upright", "place") and not row.get("contact_feedback_required"):
            raise ValueError("Upright/place references must require actual contact feedback")


def load_top_grasp_calibration(path=None, verify=True):
    """Read a self-contained, digest-checked calibration; raw recordings optional."""
    result = json.loads(Path(path or DEFAULT_CALIBRATION).read_text(encoding="utf-8"))
    _validate_calibration(result, verify=verify)
    return result


def verify_source_artifacts(calibration, directory):
    """Optional strict provenance check against the three original recordings."""
    _validate_calibration(calibration, verify=True)
    directory = Path(directory)
    for filename in ("report.json", "targets.json", "trace.json"):
        expected = calibration["source_artifacts"][filename]["sha256"]
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source artifact SHA256 mismatch: {filename}")
    return True


def build_calibration_from_artifacts(directory):
    """Extract compact phase endpoints from the original successful fixture run.

    No files are written. Callers may serialize the returned dictionary. The
    full trace is used only to check the initial object anchor and successful
    physical terminal state, never to derive robot joint targets.
    """
    directory = Path(directory)
    raw = {name: (directory / name).read_bytes()
           for name in ("report.json", "targets.json", "trace.json")}
    report, targets, trace = (json.loads(raw[name])
                              for name in ("report.json", "targets.json", "trace.json"))
    if not report.get("success") or not report.get("grasp_verified") or not report.get("release_commanded"):
        raise ValueError("Source run must have completed verified grasp and table release")
    if not report.get("object_free") or report.get("object_pose_replay") or report.get("object_weld"):
        raise ValueError("Source object must have been free, without pose replay or weld")
    if report["profile"]["profile_id"] != "baseline_40mm_100g" or report["candidate"]["side"] != "left":
        raise ValueError("Expected left-hand 100 g calibration")
    anchor = _transform(trace[0]["metrics"]["T_world_object"], "initial recorded object")
    if not np.allclose(anchor[:3, 3], report["layout"]["cylinder_initial_position"], atol=1e-8, rtol=0.):
        raise ValueError("Report and trace disagree about the initial object anchor")
    if trace[-1]["phase"] != "COMPLETE":
        raise ValueError("Source trace did not reach COMPLETE")
    blocks = {}
    previous = None
    for index, row in enumerate(targets):
        phase = row["phase"]
        if phase != previous:
            blocks.setdefault(phase, []).append([])
        blocks[phase][-1].append((index, row))
        previous = phase
    waypoints = []
    for name, phase, occurrence, command in SOURCE_PHASES:
        try:
            block = blocks[phase][occurrence]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Source is missing {name}/{phase} plateau {occurrence}") from exc
        index, row = block[-1]
        goal = _transform(row["T_world_wrist_goal"], f"source {name}")
        if not all(np.allclose(item["T_world_wrist_goal"], goal, atol=1e-10, rtol=0.)
                   for _, item in block):
            raise ValueError(f"Source {name} phase must be a constant goal plateau")
        if row["command"] != command:
            raise ValueError(f"Source {name} has inconsistent hand command")
        # Short final approach is a clearly labelled planning waypoint, 3 cm
        # above the calibrated grasp; it is not claimed to be a recorded phase.
        offset = np.array([0., 0., .03 if name == "approach" else 0.])
        source_goal = goal.copy()
        goal[:3, 3] += offset
        waypoints.append(dict(name=name, source_phase=phase, source_phase_occurrence=occurrence,
            source_target_index=index, source_time_s=float(row["time_s"]),
            source_T_initial_object_wrist_goal=(np.linalg.inv(anchor) @ source_goal).tolist(),
            derived_waypoint=name == "approach", source_world_offset_m=offset.tolist(),
            T_initial_object_wrist_goal=(np.linalg.inv(anchor) @ goal).tolist(),
            nominal_hand_command=command, contact_feedback_required=name in ("upright", "place"),
            meaning=("Nominal unloaded reference only; recompute from measured held wrist/object relation in contact"
                     if name in ("upright", "place") else
                     "Derived pre-approach 3 cm above recorded grasp target; advance only after actual state gate"
                     if name == "approach" else
                     "Recorded commanded wrist endpoint; advance only after actual state gate")))
    result = dict(schema_version=1, reference_kind="commanded_wrist_goals_anchored_to_initial_object",
        side="left", object_profile_id="baseline_40mm_100g", object_profile=report["profile"],
        hand_candidate=report["candidate"], source_run_name=directory.name,
        source_success=True, source_scope=report["scope"],
        source_artifacts={name: dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
                          for name, content in raw.items()},
        source_initial_object_pose=anchor.tolist(), source_table_top_m=report["layout"]["table_top_m"],
        source_initial_table_gap_m=float(anchor[2, 3]-report["layout"]["table_top_m"]-report["profile"]["height_m"]/2.),
        source_place_target_xy_m=report["place_target_xy_m"],
        waypoint_pose_frame="initial_object_not_current_object", quaternion_convention="wxyz",
        is_whole_body_validation=False, is_collision_free_certificate=False,
        waypoints=waypoints)
    result["content_sha256"] = calibration_sha256(result)
    _validate_calibration(result, verify=True)
    return result


def relocate_top_grasp_path(initial_object_pose, yaw_deg=180., calibration=None):
    """Freeze world goals at an *initial*, upright object pose and grasp azimuth.

    Rotation is around the world's vertical through the fixed initial object
    origin. Cylinder axial symmetry permits this grasp-azimuth search, but does
    not certify whole-body reachability or table/body collision clearance.
    """
    calibration = load_top_grasp_calibration() if calibration is None else copy.deepcopy(calibration)
    _validate_calibration(calibration, verify=True)
    initial = _transform(initial_object_pose, "initial world object pose")
    if not np.allclose(initial[:3, 2], [0., 0., 1.], atol=1e-7, rtol=0.):
        raise ValueError("Scene relocation expects an upright object, not a tilted can")
    anchor = initial.copy()
    anchor[:3, :3] = _yaw_transform(yaw_deg)[:3, :3] @ initial[:3, :3]
    result = []
    for source in calibration["waypoints"]:
        goal = anchor @ np.asarray(source["T_initial_object_wrist_goal"])
        row = copy.deepcopy(source)
        row.update(T_world_wrist_goal=goal, position_m=goal[:3, 3].copy(),
                   quaternion_wxyz=quaternion_wxyz(goal),
                   fixed_initial_object_pose=initial.copy(), azimuth_yaw_deg=float(yaw_deg),
                   feedback_required=source["contact_feedback_required"])
        result.append(row)
    return result


def world_path_for_scene(table_height_m, can_xy_m, yaw_deg=180., calibration=None):
    calibration = load_top_grasp_calibration() if calibration is None else calibration
    _validate_calibration(calibration, verify=True)
    xy = np.asarray(can_xy_m, dtype=float)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.isfinite(table_height_m) or table_height_m <= 0.:
        raise ValueError("Scene requires finite XY coordinates and a positive table height")
    initial = np.eye(4)
    initial[:3, 3] = [xy[0], xy[1], float(table_height_m)+calibration["object_profile"]["height_m"]/2.
                      + calibration["source_initial_table_gap_m"]]
    return relocate_top_grasp_path(initial, yaw_deg, calibration)


def scene_candidates(calibration=None):
    """Pure geometric screen only: 18 scenes, no IK or policy success claim."""
    calibration = load_top_grasp_calibration() if calibration is None else calibration
    result = []
    for height in (.8, .9, 1.):
        for xy in ((.32, .18), (.38, .18)):
            for yaw in (-90., 90., 180.):
                path = world_path_for_scene(height, xy, yaw, calibration)
                xyz = np.array([row["position_m"] for row in path])
                result.append(dict(table_height_m=height, can_xy_m=list(xy), yaw_deg=yaw,
                    wrist_min_xyz_m=xyz.min(axis=0).tolist(), wrist_max_xyz_m=xyz.max(axis=0).tolist(),
                    grasp_position_m=path[2]["position_m"].tolist(),
                    grasp_quaternion_wxyz=path[2]["quaternion_wxyz"].tolist(),
                    upright_quaternion_wxyz=path[4]["quaternion_wxyz"].tolist(),
                    reachability_verified=False, collision_clearance_verified=False))
    return result
