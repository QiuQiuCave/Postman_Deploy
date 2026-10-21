"""Lossless wrist/finger reference projection and unchanged CONTACT validity."""

import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL = "contact_reach_neighborhood_v1"


class GoalTube:
    def __init__(self, manifest, program_sha256, dataset=None):
        from r2v2_deploy.configuration import digest

        path = Path(manifest)
        m = json.loads(path.read_text())
        if (
            m.get("schema") != "contact_reach_reference_v1"
            or m.get("program_sha256") != program_sha256
            or m.get("derivation") != "lossless_native_target_projection_v1"
        ):
            raise ValueError("Unknown/mismatched native Reach reference projection")
        data = path.parent / m["data_file"]
        if digest(data) != m["data_sha256"]:
            raise ValueError("Reach reference data hash mismatch")
        with np.load(data, allow_pickle=False) as z:
            for name in (
                "time_s",
                "wrist_target",
                "phase",
                "phase_elapsed_s",
                "hand_commands",
            ):
                array = z[name]
                if (
                    hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                    != m["array_sha256"][name]
                ):
                    raise ValueError("Reach reference array identity mismatch: " + name)
            self.times = z["time_s"].copy()
            self.targets = z["wrist_target"].copy()
            self.phases = z["phase"].copy()
            self.elapsed = z["phase_elapsed_s"].copy()
            self.commands = z["hand_commands"].copy()
        if (
            self.targets.shape != (len(self.times), 2, 7)
            or self.commands.shape != (len(self.times), 2)
            or not np.allclose(np.diff(self.times), 0.02, atol=1e-8)
            or not np.isfinite(self.targets).all()
        ):
            raise ValueError("Finite native PD50 wrist/finger stream required")
        self.manifest_sha256 = digest(path)
        self.data_sha256 = m["source_data_sha256"]
        self.compact_data_sha256 = m["data_sha256"]
        self.provenance = dict(
            manifest_sha256=self.manifest_sha256,
            data_sha256=self.data_sha256,
            compact_data_sha256=self.compact_data_sha256,
            program_sha256=program_sha256,
            source_group=m["source_group"],
            evaluation_scope="fixed_scene_nominal_not_independent_generalization",
            body_trajectory_used=False,
            expert_action_labels_used=False,
        )

    def index_for(self, phase, elapsed):
        ids = np.flatnonzero(self.phases == phase)
        if not len(ids):
            raise ValueError(
                "Snapshot phase absent from successful goal stream: " + phase
            )
        return int(ids[np.argmin(abs(self.elapsed[ids] - elapsed))])

    def sample_index(self, index):
        index = np.clip(index, 0, len(self.times) - 1)
        return self.targets[index], self.commands[index], self.phases[index]


def safety_conditions(row, control_loss_s):
    """Same scalar/tensor semantics; no pose/FSM/foot-anchor success gates."""
    return (
        ("Nonfinite state", ~row["finite_state"]),
        (
            "Body joint physical limit excess >0.05rad",
            row["body_joint_violation_rad"] > 0.05,
        ),
        (
            "Body tilt/height safety limit",
            (row["base_tilt_deg"] > 35) | (row["root_height_m"] < 0.55),
        ),
        ("Deep robot self-collision", row["robot_self_penetration_m"] > 0.005),
        (
            "Finger joint or mimic limit",
            (row["hand_joint_violation_rad"] > 0.01) | (row["mimic_error_rad"] > 0.015),
        ),
        ("Non-foot ground contact", row["nonfoot_ground_contact"]),
        (
            "Non-grasping robot part contacted object",
            row["parked_hand_contact"] | row["nonhand_object_contact"],
        ),
        (
            "Hand/object penetration exceeded3mm",
            row["hand_object_penetration_m"] > 0.003,
        ),
        (
            "Robot/table or crate penetration exceeded3mm",
            row["robot_prop_penetration_m"] > 0.003,
        ),
        (
            "Robot/table or crate contact force exceeded150N",
            row["robot_prop_force_N"] > 150.0,
        ),
        ("Object reached floor", row["floor_contact"]),
        ("Held segment lost real object control for0.2s", control_loss_s >= 0.2 - 1e-8),
    )


def source_clock(elapsed, mode):
    """0: target/hold; 1: adjacent native targets then hold; 2: normal start."""
    if mode == 0:
        return 0.0
    if mode == 1:
        return min(max(float(elapsed) - 0.4, 0.0), 3.2)
    if mode == 2:
        return float(elapsed)
    raise ValueError("Unknown Reach curriculum mode")
