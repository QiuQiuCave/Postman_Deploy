"""Frozen-source dual-wrist paths for a five-height crate screening experiment.

This module produces goals, never simulator states, IK commands or support
forces. Added approach stages keep open hands outside the crate before turning
the wrists. The measured fixture motion is relocated through a frozen initial
crate anchor, including the source crate's actual lift (not live-object chasing).
"""

from pathlib import Path

import numpy as np

from common.r2v2_crate_motion_recording import (
    interpolate_transform, load_crate_motion, transform_from_pose,
)


BASE_TABLE_TOP_M = 1.0109189696536514
HEIGHT_OFFSETS_M = (0., -.05, -.10, -.15, -.20)
CRATE_CENTER_XY = (.38, 0.)
DEFAULT_MOTION_PATH = (Path(__file__).resolve().parents[1]
    / "reference_motion_bank/r2v2_crate/down20_yaw15_60mm")
SIDES = ("left", "right")
ADDED_SEGMENTS = ("OUTSIDE", "TURN_WRISTS", "PREALIGN")


def _pair(value):
    if isinstance(value, dict):
        value = [value[side] for side in SIDES]
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2, 4, 4):
        raise ValueError("Expected bilateral world wrist transforms in left/right order")
    # Reuse the recording's full finite/proper-rotation checks.
    return interpolate_transform(pair, pair, 0.)


def default_start_wrists():
    """Requested outward warm-up goals; not assumed actual reachable states."""
    return np.array([transform_from_pose([.18, sign*.27, 1.13], [1., 0., 0., 0.])
                     for sign in (1, -1)])


class HeightPath:
    """Measured hand/crate trajectory plus explicit collision-aware preparation.

    ``sample(name, elapsed)`` returns T_world_wrist (2,4,4), T_world_crate
    (4,4), hand_command (2,), and source_time_s (None for added phases).
    Phase clocks are caller-owned: physical gates may pause a stage without
    restarting the policy's reference filter. All source binary events remain
    recorded separately; a deployment caller must gate actual closure.
    """

    def __init__(self, delta_z, start_wrists_world=None, *, motion_path=DEFAULT_MOTION_PATH,
                 world_crate_pose=None, outside_x=.18, outside_abs_y=.37,
                 outside_s=3., turn_s=4., prealign_s=3., delta_x=0.):
        if not np.isfinite(delta_z) or not -.25 <= delta_z <= .05:
            raise ValueError("Height offset must be finite and in [-0.25, +0.05] m")
        if not all(np.isfinite(value) and value > 0. for value in
                   (outside_abs_y, outside_s, turn_s, prealign_s)) or not np.isfinite(outside_x):
            raise ValueError("Invalid outside waypoint or positive phase duration")
        self.delta_z = float(delta_z)
        if not np.isfinite(delta_x) or not -.15 <= delta_x <= .15:
            raise ValueError("Horizontal offset must be finite and in [-0.15, +0.15] m")
        self.delta_x = float(delta_x)
        self.table_top_m = BASE_TABLE_TOP_M + self.delta_z
        self.motion_path = Path(motion_path).resolve()
        self.motion = load_crate_motion(self.motion_path)
        self.start_wrists_world = _pair(default_start_wrists()
            if start_wrists_world is None else start_wrists_world)
        self.boundaries = {item["state"]: float(item["time_s"])
                           for item in self.motion.manifest["transitions"]}
        if world_crate_pose is None:
            world_crate_pose = transform_from_pose([CRATE_CENTER_XY[0]+self.delta_x, CRATE_CENTER_XY[1], self.table_top_m],
                                                   [1., 0., 0., 0.])
        world_crate_pose = np.asarray(world_crate_pose, dtype=float)
        if world_crate_pose.shape != (4, 4):
            raise ValueError("world_crate_pose must be one SE3 transform")
        self.world_crate_pose = interpolate_transform(world_crate_pose, world_crate_pose, 0.)
        # Align to the settled source crate at INSERT entry, not the source's
        # initial 1 mm free-fall gap. The recorded READY settling remains visible.
        settled = self.motion.sample(self.boundaries["INSERT"])
        self.world_anchor = self.world_crate_pose @ np.linalg.inv(settled["T_anchor_crate"])
        ready = self.world_anchor @ self.motion.sample(0.)["T_anchor_wrist"]
        outside = self.start_wrists_world.copy()
        outside[:, :3, :3] = np.eye(3)
        outside[:, 0, 3] = outside_x+self.delta_x
        outside[:, 1, 3] = [outside_abs_y, -outside_abs_y]
        outside[:, 2, 3] = ready[:, 2, 3]
        turned = outside.copy()
        turned[:, :3, :3] = ready[:, :3, :3]
        self.outside_wrists_world = outside.copy()
        self.ready_wrists_world = ready.copy()
        self.segments = [
            dict(name="OUTSIDE", kind="approach", duration_s=float(outside_s),
                 start_world=self.start_wrists_world.copy(), target_world=outside.copy()),
            dict(name="TURN_WRISTS", kind="approach", duration_s=float(turn_s),
                 start_world=outside.copy(), target_world=turned.copy()),
            dict(name="PREALIGN", kind="approach", duration_s=float(prealign_s),
                 start_world=turned.copy(), target_world=ready.copy()),
        ]
        transitions = self.motion.manifest["transitions"]
        for first, last in zip(transitions[:-1], transitions[1:]):
            self.segments.append(dict(name=first["state"], kind="recorded",
                duration_s=float(last["time_s"]-first["time_s"]),
                source_start_s=float(first["time_s"]), source_end_s=float(last["time_s"])))
        self._by_name = {segment["name"]: segment for segment in self.segments}
        if len(self._by_name) != len(self.segments):
            raise ValueError("Source phases overlap added approach names")
        self.duration_s = sum(segment["duration_s"] for segment in self.segments)

    def sample(self, segment_name, elapsed):
        segment = self._by_name[segment_name]
        if not np.isfinite(elapsed) or elapsed < 0.:
            raise ValueError("Segment elapsed time must be finite and nonnegative")
        elapsed = min(float(elapsed), segment["duration_s"])
        if segment["kind"] == "approach":
            fraction = elapsed / segment["duration_s"]
            smooth = fraction*fraction*(3.-2.*fraction)
            wrists = interpolate_transform(segment["start_world"], segment["target_world"], smooth)
            return dict(T_world_wrist=wrists, T_world_crate=self.world_crate_pose.copy(),
                        hand_command=np.zeros(2, dtype=np.int8), source_time_s=None,
                        segment=segment_name)
        source_time = segment["source_start_s"] + elapsed
        sample = self.motion.sample(source_time)
        desired_crate = self.world_anchor @ sample["T_anchor_crate"]
        return dict(T_world_wrist=desired_crate @ sample["T_crate_wrist"],
                    T_world_crate=desired_crate, hand_command=sample["hand_command"].copy(),
                    source_time_s=source_time, segment=segment_name)

    def metadata(self):
        return dict(delta_z_m=self.delta_z, delta_x_m=self.delta_x, table_top_m=self.table_top_m,
            crate_center_xy_m=self.world_crate_pose[:2, 3].tolist(), outside_wrists_world=self.outside_wrists_world.tolist(),
            ready_wrists_world=self.ready_wrists_world.tolist(), world_anchor=self.world_anchor.tolist(),
            motion_manifest=str(self.motion_path / "manifest.json" if self.motion_path.is_dir() else self.motion_path),
            motion_sha256=self.motion.manifest["trajectory_sha256"],
            pose_source="measured_dynamic_fixture_wrist_link_not_mocap_goal",
            source_crate_motion_preserved=True,
            segments=[{k: v for k, v in segment.items() if k not in ("start_world", "target_world")}
                      for segment in self.segments],
            safety_scope="World targets only; no guarantee of body reachability or policy tracking")


def _aabb_distance(first_min, first_max, second_min, second_max):
    return float(np.linalg.norm(np.maximum(0., np.maximum(first_min-second_max, second_min-first_max))))


def inspect_approach_hand_sweep(path, intervals_per_segment=100):
    """Conservative continuous open-hand mesh clearance for added approach only.

    For every interval, transform all collision-mesh vertices at both ends and
    expand their AABB by the rotational arc sagitta. This covers the continuous
    translation + shortest-SLERP segment; cubic timing changes speed, not locus.
    The crate is conservatively a *solid* bounding box. No claims are made about
    forearms, torso, articulated body reachability, or post-insertion contact.
    """
    if not isinstance(intervals_per_segment, int) or intervals_per_segment < 2:
        raise ValueError("Require at least two sweep intervals per segment")
    import itertools
    import xml.etree.ElementTree as ET
    import mujoco
    from common.r2v2_crate_hand_preview import _one_hand_root, _set_hand_pose, _local_part_clouds

    crate_cfg = path.motion.manifest["crate_parameters"]
    crate_vertices = np.array(list(itertools.product(
        (-crate_cfg["depth"]/2, crate_cfg["depth"]/2),
        (-crate_cfg["width"]/2, crate_cfg["width"]/2), (0., crate_cfg["height"]))))
    crate_vertices = (crate_vertices @ path.world_crate_pose[:3, :3].T
                      + path.world_crate_pose[:3, 3])
    crate_min, crate_max = crate_vertices.min(0), crate_vertices.max(0)
    # Entire table (including legs) lies in this conservative support column.
    table_min = np.array([.26+path.delta_x, -.35, 0.])
    table_max = np.array([.70+path.delta_x, .35, path.table_top_m])
    records = []
    for side_index, side in enumerate(SIDES):
        root, _ = _one_hand_root(side)
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        _set_hand_pose(model, data, side, .03, 75.)
        vertices = np.concatenate(list(_local_part_clouds(model, data, side).values()))
        radius = float(np.linalg.norm(vertices, axis=1).max())
        for name in ADDED_SEGMENTS:
            duration = path._by_name[name]["duration_s"]
            minimum_crate = minimum_table = minimum_vertical = float("inf")
            previous = path.sample(name, 0.)["T_world_wrist"][side_index]
            previous_points = vertices @ previous[:3, :3].T + previous[:3, 3]
            for elapsed in np.linspace(0., duration, intervals_per_segment+1)[1:]:
                current = path.sample(name, elapsed)["T_world_wrist"][side_index]
                points = vertices @ current[:3, :3].T + current[:3, 3]
                angle = np.arccos(np.clip((np.trace(current[:3, :3] @ previous[:3, :3].T)-1.)/2., -1., 1.))
                padding = radius*(1.-np.cos(angle/2.)) + 1e-10
                low = np.minimum(points.min(0), previous_points.min(0))-padding
                high = np.maximum(points.max(0), previous_points.max(0))+padding
                minimum_crate = min(minimum_crate, _aabb_distance(low, high, crate_min, crate_max))
                minimum_table = min(minimum_table, _aabb_distance(low, high, table_min, table_max))
                minimum_vertical = min(minimum_vertical, float(low[2]-path.table_top_m))
                previous, previous_points = current, points
            records.append(dict(side=side, segment=name,
                minimum_crate_clearance_lower_bound_m=minimum_crate,
                minimum_table_clearance_lower_bound_m=minimum_table,
                minimum_tabletop_vertical_clearance_lower_bound_m=minimum_vertical,
                geometric_open_hand_clear=minimum_crate > 0. and minimum_table > 0.))
    return dict(scope="continuous_added_approach_open_hand_geometry_only",
        dynamic_grasp_or_fullbody_success_evaluated=False,
        body_parts="palm, thumb, index, middle, ring, pinky; no forearm or torso",
        method="collision mesh endpoint AABBs expanded by radius*(1-cos(interval_rotation/2)); solid crate bounding box and enclosing table column",
        intervals_per_segment=intervals_per_segment,
        passed=all(row["geometric_open_hand_clear"] for row in records), segments=records)


__all__ = ["HeightPath", "BASE_TABLE_TOP_M", "HEIGHT_OFFSETS_M", "DEFAULT_MOTION_PATH",
           "SIDES", "ADDED_SEGMENTS", "default_start_wrists", "inspect_approach_hand_sweep"]
