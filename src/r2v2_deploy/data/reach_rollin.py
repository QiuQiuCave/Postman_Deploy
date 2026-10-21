"""Validate native student reset references; never teacher actions or success labels."""

import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from r2v2_deploy.configuration import digest

SCHEMA = "contact_reach_student_rollin_v1"
KIND = "actual_native_student_reach_reset"


def load_student_references(path, program_sha256):
    path = Path(path)
    m = json.loads(path.read_text())
    if (
        m.get("schema") != SCHEMA
        or m.get("state") != "sealed"
        or m.get("split") != "train"
        or m.get("program_sha256") != program_sha256
        or m.get("student_success_claimed") is not False
        or m.get("body_action_supervision") is not False
        or not 8 <= len(m.get("snapshots", [])) <= 12
    ):
        raise ValueError("Unknown/unsealed actual student Reach reset bank")
    report = path.parent / "source_contact.json"
    if digest(report) != m["source_report_sha256"]:
        raise ValueError("Student source CONTACT evidence changed")
    if (
        json.loads(report.read_text()).get("bindings", {}).get("checkpoint_sha256")
        != m["source_checkpoint_sha256"]
    ):
        raise ValueError("Student source checkpoint identity differs")
    states = []
    for row in m["snapshots"]:
        raw = (path.parent / row["path"]).read_bytes()
        check = row["continuation"]
        if (
            hashlib.sha256(raw).hexdigest() != row["sha256"]
            or not np.isfinite(row["native_observation_max_error"])
            or row["native_observation_max_error"] > 2e-6
            or row.get("safety_legal_at_capture") is not True
            or not np.isfinite(row.get("actual_action_history_max_error", float("inf")))
            or row["actual_action_history_max_error"] > 1e-6
            or check.get("ticks", 0) < 10
            or any(
                not np.isfinite(check.get(k, float("inf"))) or check[k] > tol
                for k, tol in [("observation", 2e-6), ("integration", 1e-8)]
            )
        ):
            raise ValueError("Student reset observation/continuation audit differs")
        s = pickle.loads(raw)
        if (
            s.get("source_kind") != KIND
            or s.get("official_air_contact_evidence") is not False
            or s.get("program_sha256") != program_sha256
            or s.get("reach_source_data_sha256") != m["goal_data_sha256"]
            or s.get("reach_source_index") != row["source_index"]
            or s.get("source_policy_sha256") != m["source_policy_sha256"]
            or not np.isfinite(s["physics"]).all()
            or s["runtime"].get("failure")
            or s["runtime"].get("contact_air_verified")
            or not s.get("reach_program_state")
        ):
            raise ValueError(
                "Student state cannot grant success or change reference identity"
            )
        states.append(s)
    return states
