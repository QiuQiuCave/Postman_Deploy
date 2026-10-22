"""Validate the fixed successful 6DoF wrist program; no teacher/controller."""

import json
from pathlib import Path
import numpy as np
from r2v2_deploy.configuration import digest
from .scene_spec import canonical_digest

PROGRAM_ID = "conditional02-exit-placement-v1"
SUPPORT_PROGRAM_ID = "conditional02-exit-placement-v2"
SUPPORT_PROTOCOL = "table_support_release_v1"


def release_protocol(document):
    return document.get("handoff_protocol", "held_until_lower_endpoint_v1")


def validate_pause_variant(document):
    if document.get("execution_variant") is not None:
        raise ValueError("Only the fixed validated Reach program is supported")
    if "support_pause_config" in document or "source_program_sha256" in document:
        raise ValueError("Unexpected alternate program fields")
    return False


class ExitPlacementProgram:
    def __init__(self, path):
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        if d.get("schema_version") != 1 or d.get("task_id") not in (
            PROGRAM_ID,
            SUPPORT_PROGRAM_ID,
        ):
            raise ValueError("Unknown legacy-exit placement identity")
        if (d["task_id"] == SUPPORT_PROGRAM_ID) != (
            release_protocol(d) == SUPPORT_PROTOCOL
        ) or release_protocol(d) not in (
            SUPPORT_PROTOCOL,
            "held_until_lower_endpoint_v1",
        ):
            raise ValueError("Release protocol requires its own explicit task identity")
        if canonical_digest(
            {k: v for k, v in d.items() if k != "content_sha256"}
        ) != d.get("content_sha256"):
            raise ValueError("Placement program hash mismatch")
        if not d.get("geometry_passed"):
            raise ValueError(
                "Passing offline geometry required; not a dynamic certificate"
            )
        if d.get("geometry_profile", "strict_fk_1mm_v1") not in (
            "strict_fk_1mm_v1",
            "placement_intermediate_5mm_v1",
        ):
            raise ValueError("Unknown explicit offline geometry profile")
        waist = d.get("teacher_carry_waist_assist", False)
        if type(waist) is not bool or (
            waist and d.get("carry_orientation_mode") != "minimum_swing"
        ):
            raise ValueError("Unsupported teacher carry-waist program")
        prefix = d["plan"]["stages"][: d["pick_stage_count"]]
        if (
            prefix != d["original_pickup_prefix"]
            or canonical_digest(prefix) != d["original_pickup_prefix_sha256"]
        ):
            raise ValueError("Original conditional02 pickup prefix changed")
        if prefix[-1]["name"] != "TOP_RETREAT" or d["timing"] != dict(
            lift_motion_s=4.0, release_hold_s=2.5
        ):
            raise ValueError(
                "Original pickup timing or delivery release timing changed"
            )
        if d["contact_pick_exit_wrist_goal"] != prefix[-1]["target"]:
            raise ValueError("CONTACT lift endpoint differs from the old pickup")
        names = [s["name"] for s in d["plan"]["stages"]][d["pick_stage_count"] :]
        if names != [
            "DELIVERY_TRANSFER",
            "DELIVERY_UPRIGHT",
            "DELIVERY_LOWER",
            "DELIVERY_RETREAT",
        ]:
            raise ValueError("Incomplete placement suffix")
        for stage in d["plan"]["stages"]:
            poses = np.asarray(stage["target"])
            if (
                poses.shape != (2, 4, 4)
                or not np.isfinite(poses).all()
                or not 0 < stage["motion_s"] <= 10
            ):
                raise ValueError("Invalid bounded wrist stage")
            for p in poses:
                if (
                    not np.allclose(p[3], [0, 0, 0, 1])
                    or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-8)
                    or not np.isclose(np.linalg.det(p[:3, :3]), 1)
                ):
                    raise ValueError("Nonrigid wrist stage")
        self.document = d
        self.sha256 = digest(self.path)
        self.content_sha256 = d["content_sha256"]
        validate_pause_variant(d)
        right = d.get("right_carry_wrist_goals", {})
        right_flag = d.get("carry_right_follows_waist", False)
        if (
            type(right_flag) is not bool
            or bool(right) != right_flag
            or (right_flag and not waist)
        ):
            raise ValueError("Explicit paired carry reference identity required")
        if right:
            if set(right) != set(d["carry_wrist_goals"]):
                raise ValueError("Incomplete right carry reference")
            for name, values in right.items():
                poses = np.asarray(values)
                if (
                    poses.shape != (len(d["carry_wrist_goals"][name]), 4, 4)
                    or not np.isfinite(poses).all()
                    or not np.allclose(poses[:, 3], [0, 0, 0, 1])
                    or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1.0)
                    or not np.allclose(
                        poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3],
                        np.eye(3),
                        atol=1e-8,
                    )
                ):
                    raise ValueError("Invalid paired carry reference")
