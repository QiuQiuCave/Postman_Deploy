"""Strict source lineage checks for sealed native reset banks."""

import hashlib
import json
from pathlib import Path
import numpy as np
from r2v2_deploy.configuration import digest
from r2v2_deploy.data.asset_paths import bundle_path

NOMINAL_SCHEMA = "r2v2_delivery_nominal_learning_index_v1"
NOMINAL_PROTOCOL = "conditional02_seed0_relay_fit_v1"


def validate_nominal_learning_index(path, *, audit_physics=True):
    """Explicit new experiment assignment; never rewrite old validation files.

    All seed0 descendants become TRAIN together. There is no independent
    validation here, and neither nominal evaluation nor resampled fit errors
    may be reported as held-out/generalization performance. The old v1 index
    builder, seed guards and final-test exclusions remain unchanged.
    """
    RELAY_SOURCE = "conditional02_native_prefix_private_upper_teacher_suffix"
    d = json.loads(Path(path).read_text())
    if (
        d.get("schema") != NOMINAL_SCHEMA
        or d.get("state") != "sealed"
        or d.get("protocol") != NOMINAL_PROTOCOL
        or d.get("independent_validation") is not False
        or d.get("source_assignment") != "entire_nominal_seed0_family_to_train"
        or d.get("evaluation_scope")
        != "in_sample_nominal_closed_loop_not_generalization"
        or d.get("original_files_unchanged") is not True
        or d.get("reserved_test_seeds") != [13, 14, 15]
        or not 1 <= len(d.get("entries", [])) <= 16
    ):
        raise ValueError("Unknown explicit nominal-learning split protocol")
    hashes, fingerprints = set(), set()
    for row in d["entries"]:
        p = bundle_path(row["manifest"])
        if (
            row.get("split") != "train"
            or row.get("source_split") != "validation"
            or row.get("initial_seed") != 0
            or digest(p) != row["manifest_sha256"]
        ):
            raise ValueError("Nominal assignment changed source identity/split")
        m = json.loads(p.read_text())
        if (
            m.get("role") != "pd50-relay-teacher"
            or m.get("source_teacher") != RELAY_SOURCE
            or m.get("parent_student_checkpoint_sha256")
            != d["parent_checkpoint_sha256"]
        ):
            raise ValueError(
                "Only the qualified conditional02 relay source may be reassigned"
            )
        episodes = m["episodes"]
        if any(e["initial_seed"] != 0 or e["split"] != "validation" for e in episodes):
            raise ValueError(
                "Whole source family must retain its immutable original split"
            )
        contact = next(e for e in episodes if e["mode"] == "contact")
        if (
            contact["label"] != "full_success_demo"
            or row["data_sha256"] != contact["sha256"]
            or row["data_sha256"] in hashes
        ):
            raise ValueError("Duplicate or unqualified nominal demonstration")
        if audit_physics:
            raise ValueError(
                "Full native trajectory audit must use the immutable source data tooling; this loader only verifies sealed reset lineage"
            )
        fingerprint = initial_fingerprint(p.parent / contact["path"])
        if row["initial_fingerprint"] != fingerprint:
            raise ValueError("Actual nominal initial-state identity changed")
        fingerprints.add(fingerprint)
        hashes.add(row["data_sha256"])
    if len(fingerprints) != 1:
        raise ValueError(
            "This protocol is one nominal initial family, not unseen initials"
        )
    return dict(
        passed=True,
        independent_validation=False,
        independent_initials=1,
        contact_episodes=len(hashes),
        original_files_unchanged=True,
    )


def initial_fingerprint(path):
    with np.load(path, allow_pickle=False) as z:
        # Same robot root/body state, independent of AIR/CONTACT object dimensions.
        values = np.r_[z["qpos"][0, :7], z["body_q"][0], z["body_qdot"][0]]
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
