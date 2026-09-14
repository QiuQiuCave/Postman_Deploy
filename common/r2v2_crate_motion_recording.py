"""Portable crate-relative hand-fixture recordings, without a physics dependency.

T_A_B maps column coordinates from B to A; quaternions are WXYZ. Measured
trajectories and externally driven fixture goals are deliberately separate.
For a lift replay, freeze the new scene's initial crate frame as the anchor:
``T_world_anchor @ T_anchor_wrist_target(t)``. Multiplying a moving, measured
crate pose by a relative wrist pose alone does not command the crate to rise.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SIDES = ("left", "right")
TRANSFORM_KEYS = (
    "T_world_crate", "T_world_wrist", "T_crate_wrist", "T_anchor_crate",
    "T_anchor_wrist", "T_crate_wrist_target", "T_anchor_wrist_target",
)
FINGER_KEYS = ("hand_reference_q_rad", "hand_measured_q_rad", "hand_torque_Nm")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape[-2:] != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Expected finite SE3 transform(s) with shape (..., 4, 4)")
    rotation = transform[..., :3, :3]
    if (not np.allclose(transform[..., 3, :], [0., 0., 0., 1.], atol=1e-8, rtol=0)
            or not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0)):
        raise ValueError("Invalid SE3 homogeneous row or proper rotation")
    return transform


def transform_from_pose(position, quaternion_wxyz):
    position = np.asarray(position, dtype=np.float64)
    quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quat.shape != (4,) or not np.all(np.isfinite(np.r_[position, quat])):
        raise ValueError("Expected finite position (3,) and WXYZ quaternion (4,)")
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion cannot be zero")
    w, x, y, z = quat / norm
    transform = np.eye(4)
    transform[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    transform[:3, 3] = position
    return transform


def pose_from_transform(transform):
    """Return independent (position_m, quaternion_wxyz) arrays for one SE3."""
    transform = _validate_transform(transform)
    if transform.shape != (4, 4):
        raise ValueError("pose_from_transform expects one (4, 4) transform")
    r = transform[:3, :3]
    # Symmetric quaternion eigensystem remains well-conditioned at 180 degrees.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
        [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
        [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3.
    _, vectors = np.linalg.eigh(k)
    quat = vectors[:, -1][[3, 0, 1, 2]]
    if quat[0] < 0:
        quat *= -1
    return transform[:3, 3].copy(), quat


def slerp_wxyz(first, second, fraction):
    """Shortest-arc normalized quaternion interpolation; never Euler blending."""
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError("Interpolation fraction must lie in [0, 1]")
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("SLERP requires two WXYZ quaternions")
    if not np.all(np.isfinite(np.r_[first, second])) or min(np.linalg.norm(first), np.linalg.norm(second)) < 1e-12:
        raise ValueError("SLERP requires finite nonzero quaternions")
    first, second = first / np.linalg.norm(first), second / np.linalg.norm(second)
    dot = float(first @ second)
    if dot < 0:
        second, dot = -second, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        result = (1-fraction)*first + fraction*second
    else:
        angle = np.arccos(dot)
        result = (np.sin((1-fraction)*angle)*first + np.sin(fraction*angle)*second) / np.sin(angle)
    return result / np.linalg.norm(result)


def interpolate_transform(first, second, fraction):
    """Linear translation plus shortest-path SLERP; supports (..., 4, 4)."""
    first, second = _validate_transform(first), _validate_transform(second)
    if first.shape != second.shape:
        raise ValueError("Transform shapes differ")
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError("Interpolation fraction must lie in [0, 1]")
    if fraction == 0.:
        return first.copy()
    if fraction == 1.:
        return second.copy()
    result = np.empty_like(first)
    for a, b, out in zip(first.reshape(-1, 4, 4), second.reshape(-1, 4, 4), result.reshape(-1, 4, 4)):
        pa, qa = pose_from_transform(a)
        pb, qb = pose_from_transform(b)
        out[:] = transform_from_pose((1-fraction)*pa + fraction*pb, slerp_wxyz(qa, qb, fraction))
    return result


def _validate_arrays(arrays, sample_dt=None):
    required = {"time_s", "phase", "hand_command", *TRANSFORM_KEYS, *FINGER_KEYS}
    if set(arrays) != required:
        raise ValueError(f"Unexpected trajectory fields: missing={required-set(arrays)}, extra={set(arrays)-required}")
    times = arrays["time_s"]
    if times.ndim != 1 or len(times) < 2 or times.dtype != np.float64:
        raise ValueError("time_s must contain at least two float64 samples")
    if not np.all(np.isfinite(times)) or times[0] != 0. or not np.all(np.diff(times) > 0):
        raise ValueError("Sample times must start at zero and strictly increase")
    if sample_dt is not None and not np.allclose(np.diff(times), sample_dt, atol=1e-8, rtol=0):
        raise ValueError("Source trace is not complete at the declared control rate")
    n = len(times)
    if arrays["phase"].shape != (n,) or arrays["phase"].dtype.kind != "U":
        raise ValueError("phase must be a non-object Unicode vector")
    commands = arrays["hand_command"]
    if commands.shape != (n, 2) or commands.dtype != np.int8 or not np.all(np.isin(commands, [0, 1])):
        raise ValueError("hand_command must contain binary int8 values with shape (N, 2)")
    for key in TRANSFORM_KEYS:
        expected = (n, 4, 4) if key in ("T_world_crate", "T_anchor_crate") else (n, 2, 4, 4)
        if arrays[key].shape != expected or arrays[key].dtype != np.float64:
            raise ValueError(f"Wrong shape/dtype for {key}")
        _validate_transform(arrays[key])
    for key in FINGER_KEYS:
        if arrays[key].shape != (n, 2, 6) or arrays[key].dtype != np.float64 or not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Wrong shape/dtype or nonfinite values for {key}")
    anchor = arrays["T_world_crate"][0]
    for expected, actual in (
        (arrays["T_world_crate"][:, None] @ arrays["T_crate_wrist"], arrays["T_world_wrist"]),
        (anchor @ arrays["T_anchor_crate"], arrays["T_world_crate"]),
        (anchor @ arrays["T_anchor_wrist"], arrays["T_world_wrist"]),
        (arrays["T_anchor_crate"][:, None] @ arrays["T_crate_wrist_target"], arrays["T_anchor_wrist_target"]),
    ):
        if not np.allclose(expected, actual, atol=1e-10, rtol=0):
            raise ValueError("Stored world/relative transform reconstruction failed")


def _command_events(report, transitions, arrays):
    """Use audited fixture CLOSE entry semantics, not one-sample-late traces."""
    initial = arrays["hand_command"][0].copy()
    events = [{"time_s": 0., "hand_command": initial.tolist(), "source": "initial trace sample"}]
    commanded = initial.copy()
    for event in transitions:
        if event["state"] == "CLOSE":
            for side in report["candidate"]["active_sides"]:
                commanded[SIDES.index(side)] = int(report["parameters"]["grasp_enabled"])
            events.append({"time_s": event["time_s"], "hand_command": commanded.tolist(),
                           "source": "CLOSE entry immediately calls BinaryHandControl.command"})
    dt = float(report["hand_configuration"]["control_dt"])
    for time_s, observed in zip(arrays["time_s"], arrays["hand_command"]):
        preceding = max(0., float(time_s) - dt/2)
        selected = max(i for i, event in enumerate(events) if event["time_s"] <= preceding)
        if not np.array_equal(observed, events[selected]["hand_command"]):
            raise ValueError("Source binary command differs from audited CLOSE-entry/preceding-step convention")
    return events


def record_crate_motion(source_dir, output_dir):
    """Validate one successful fixture run and export NPZ + manifest; no overwrite."""
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Recording destination must be new or empty")
    paths = {name: source_dir / name for name in ("report.json", "trace.json", "transitions.json")}
    before = {name: sha256_file(path) for name, path in paths.items()}
    report, trace, transitions = (json.loads(paths[name].read_text()) for name in paths)
    if not (report.get("lift_passed") is True and report.get("pickup_verified") is True
            and report.get("experiment_completed") is True and report.get("phase") == "COMPLETE"
            and report.get("failure") is None and report.get("runtime_error") is None):
        raise ValueError("Source must be an actually completed, lift_passed and pickup_verified fixture run")
    if report.get("crate_welds") != 0 or report.get("crate_runtime_resets") != 0 or report.get("no_external_crate_forces") is not True:
        raise ValueError("Source must use a free crate without external forces or runtime resets")
    if report.get("transitions") != transitions or not transitions:
        raise ValueError("Exact source transition files disagree")
    if transitions[0] != {"time_s": 0., "state": "READY"} or transitions[-1]["state"] != "COMPLETE":
        raise ValueError("Incomplete source transition timeline")
    event_times = np.asarray([event["time_s"] for event in transitions], dtype=float)
    if not np.all(np.isfinite(event_times)) or not np.all(np.diff(event_times) >= 0):
        raise ValueError("Nonfinite or unordered transition times")
    rows = []
    for row in trace:
        when = float(row["time_s"])
        if not np.isfinite(when) or (rows and when < rows[-1]["time_s"]):
            raise ValueError("Source sample times must be finite and nondecreasing")
        if rows and when == rows[-1]["time_s"]:
            rows[-1] = row  # Terminal HOLD -> COMPLETE at identical physical time.
        else:
            rows.append(row)
    if len(rows) < 2 or rows[-1]["time_s"] != transitions[-1]["time_s"] or rows[-1]["phase"] != "COMPLETE":
        raise ValueError("Source trace is incomplete")
    arrays = {
        "time_s": np.asarray([row["time_s"] for row in rows], dtype=np.float64),
        "phase": np.asarray([row["phase"] for row in rows], dtype=str),
        "T_world_crate": np.asarray([row["metrics"]["T_world_crate"] for row in rows], dtype=np.float64),
        "T_world_wrist": np.asarray([[row["metrics"]["hands"][side]["T_world_wrist"] for side in SIDES]
                                      for row in rows], dtype=np.float64),
        "hand_command": np.asarray([[row["hands"][side]["command"] for side in SIDES] for row in rows], dtype=np.int8),
    }
    for target, source in zip(FINGER_KEYS, ("q_ref_rad", "q_rad", "torque_Nm")):
        arrays[target] = np.asarray([[row["hands"][side][source] for side in SIDES] for row in rows], dtype=np.float64)
    inverse_crate = np.linalg.inv(arrays["T_world_crate"])
    inverse_anchor = inverse_crate[0]
    arrays["T_crate_wrist"] = inverse_crate[:, None] @ arrays["T_world_wrist"]
    arrays["T_anchor_crate"] = inverse_anchor @ arrays["T_world_crate"]
    arrays["T_anchor_wrist"] = inverse_anchor @ arrays["T_world_wrist"]
    targets = np.asarray([[transform_from_pose(row["hands"][side]["wrist_target_position_m"],
        report["layout"]["wrist_quaternions"][side]) for side in SIDES] for row in rows])
    arrays["T_crate_wrist_target"] = inverse_crate[:, None] @ targets
    arrays["T_anchor_wrist_target"] = inverse_anchor @ targets
    dt = float(report["hand_configuration"]["control_dt"])
    _validate_arrays(arrays, sample_dt=dt)
    commands = _command_events(report, transitions, arrays)
    if before != {name: sha256_file(path) for name, path in paths.items()}:
        raise RuntimeError("Source files changed while exporting")
    final = report["final_metrics"]
    validation = {key: report[key] for key in ("lift_passed", "pickup_verified", "experiment_completed",
        "phase", "failure", "max_pickup_hold_s", "max_level_hold_s", "criteria", "crate_welds",
        "crate_runtime_resets", "no_external_crate_forces")}
    validation["final_metrics"] = {key: final[key] for key in ("clearance_m", "crate_tilt_deg", "grasp_slip_m",
        "table_vertical_force_N", "table_bearing_contact", "crate_linear_speed_m_s", "crate_angular_speed_rad_s")}
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "trajectory.npz"
    # Exclusive create still protects against another process choosing this path.
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = {
        "schema_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_file": archive.name, "trajectory_sha256": sha256_file(archive),
        "samples": len(rows), "source_samples": len(trace), "duplicate_timestamp_rows_removed": len(trace)-len(rows),
        "duration_s": float(arrays["time_s"][-1]), "sample_dt_s": dt,
        "physics_dt_s": report["hand_configuration"]["simulation_dt"],
        "sides": list(SIDES), "wrist_body_names": [f"{side}_hand_roll_link" for side in SIDES],
        "hand_channel_names": report["hand_configuration"]["channels"],
        "units": {"distance": "m", "angle": "rad", "time": "s", "torque": "Nm", "force": "N"},
        "transform_convention": "T_A_B maps B coordinates into A; column vectors; translation expressed in A",
        "quaternion_convention": "WXYZ scalar first, active rotation from local frame into parent frame",
        "anchor_definition": "source initial actual crate pose, T_world_crate[0], frozen for the whole trajectory",
        "T_world_anchor": arrays["T_world_crate"][0].tolist(),
        "sampling_convention": (
            "100 Hz complete source trace, no resampling. Repeated timestamps retain the LAST row; exact transitions "
            "are preserved separately. Sampled phase, command/reference/torque describe the preceding integration step; "
            "a boundary CLOSE command appears at the next row. The final COMPLETE row is an instantaneous event, not "
            "an extra physics step. Interpolation uses exact transitions and hand_command_events for commands/phases."),
        "pose_semantics": {
            "measured": "T_world_wrist/T_crate_wrist/T_anchor_wrist are actual dynamic wrist-link poses from source metrics",
            "target": "*_wrist_target are fixture mocap goals, built from recorded target translation and fixed layout quaternion",
            "replay": "T_world_anchor_new @ T_anchor_wrist_target(t); or measured T_anchor_wrist(t) when explicitly selected",
            "lift_warning": "current measured crate @ T_crate_wrist(t) alone cannot reproduce object lift; retain the recorded T_anchor_crate motion",
        },
        "missing_signals": ["finger velocities/accelerations and five mimic-joint states per hand were not in source; not synthesized"],
        "finger_torque_semantics": "source hands[side].torque_Nm: commanded actuator control, not reconstructed contact/actuator force",
        "scope": "successful free-crate, externally driven dynamic wrist-fixture experiment; not full-body reachability or deployment certification",
        "candidate": report["candidate"], "parameters": report["parameters"],
        "crate_parameters": report["crate_parameters"], "hand_configuration": report["hand_configuration"],
        "layout": report["layout"], "angle_convention": report.get("render_angle_convention"),
        "transitions": transitions, "hand_command_events": commands, "validation": validation,
        "source_files": {name: {"path": str(path), "sha256": before[name]} for name, path in paths.items()},
        "source_code_sha256": report.get("source_sha256", {}),
        "exporter_sha256": sha256_file(__file__), "mujoco_version": report.get("mujoco_version"),
        "array_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
    }
    destination = output_dir / "manifest.json"
    with destination.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    load_crate_motion(destination)  # Read back through the public validation path.
    return destination


@dataclass
class CrateMotion:
    manifest: dict
    arrays: dict

    @property
    def duration_s(self):
        return float(self.arrays["time_s"][-1])

    def sample(self, time_s):
        """Clamped pose interpolation; exact events override sampled phase/command.

        Raw preceding-step command remains in ``hand_command_preceding_step``.
        Finger reference/measured q are linearly interpolated; torque is held,
        not converted to an artificial velocity/force signal.
        """
        if not np.isfinite(time_s):
            raise ValueError("Sample time must be finite")
        time_s = float(np.clip(time_s, 0., self.duration_s))
        times = self.arrays["time_s"]
        before = max(0, int(np.searchsorted(times, time_s, side="right"))-1)
        after = min(before+1, len(times)-1)
        fraction = 0. if before == after else (time_s-times[before])/(times[after]-times[before])
        result = {"time_s": time_s, "sampled_phase": str(self.arrays["phase"][before])}
        for key in TRANSFORM_KEYS:
            result[key] = interpolate_transform(self.arrays[key][before], self.arrays[key][after], fraction)
        for key in FINGER_KEYS:
            result[key] = (self.arrays[key][before].copy() if key == "hand_torque_Nm" else
                (1-fraction)*self.arrays[key][before] + fraction*self.arrays[key][after])
        result["hand_command_preceding_step"] = self.arrays["hand_command"][before].copy()
        result["phase"] = next(event["state"] for event in reversed(self.manifest["transitions"]) if event["time_s"] <= time_s)
        result["hand_command"] = np.asarray(next(event["hand_command"] for event in
            reversed(self.manifest["hand_command_events"]) if event["time_s"] <= time_s), dtype=np.int8)
        return result

    def world_wrist_targets(self, time_s, T_world_anchor, use_fixture_targets=True):
        """Rigidly relocate the complete motion, preserving commanded lift."""
        anchor = _validate_transform(T_world_anchor)
        if anchor.shape != (4, 4):
            raise ValueError("T_world_anchor must be a single frozen SE3")
        key = "T_anchor_wrist_target" if use_fixture_targets else "T_anchor_wrist"
        return anchor @ self.sample(time_s)[key]


def load_crate_motion(path):
    """Load a self-contained recording and verify hash, schema, SE3 and timing."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("validation", {}).get("lift_passed") is not True:
        raise ValueError("Unsupported or unvalidated crate motion")
    filename = manifest["trajectory_file"]
    if Path(filename).name != filename:
        raise ValueError("Trajectory filename must be local to its manifest")
    archive = path.parent / filename
    if sha256_file(archive) != manifest["trajectory_sha256"]:
        raise ValueError("Trajectory SHA256 mismatch")
    with np.load(archive, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    _validate_arrays(arrays, sample_dt=manifest["sample_dt_s"])
    if len(arrays["time_s"]) != manifest["samples"] or arrays["time_s"][-1] != manifest["duration_s"]:
        raise ValueError("Manifest duration/sample count mismatch")
    for key, value in arrays.items():
        if manifest["array_schema"][key] != {"shape": list(value.shape), "dtype": str(value.dtype)}:
            raise ValueError(f"Manifest shape/dtype mismatch for {key}")
    return CrateMotion(manifest=manifest, arrays=arrays)
