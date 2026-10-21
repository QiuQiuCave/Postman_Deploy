"""Load sealed, content-pinned reset banks; never perform teacher collection."""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from r2v2_deploy.configuration import digest

SCHEMA = "delivery_native_reset_references_v1"
EXIT_SCHEMA = "exit_placement_native_reset_references_v1"


def load_references(path, program_sha256):
    path = Path(os.path.expandvars(str(path)))
    m = json.loads(path.read_text())
    if m.get("schema") == "contact_reach_student_rollin_v1":
        from .reach_rollin import load_student_references

        return load_student_references(path, program_sha256)
    exit_task = m.get("schema") == EXIT_SCHEMA
    if (
        m.get("schema") not in (SCHEMA, EXIT_SCHEMA)
        or m.get("state") != "sealed"
        or m.get("split") != "train"
        or m.get("initial_seed") != (0 if exit_task else 1)
        or m.get("program_sha256") != program_sha256
        or m.get("student_success_claimed") is not False
    ):
        raise ValueError("Unknown/unvalidated training-only reference initial states")
    if exit_task:
        from r2v2_deploy.data.delivery_index import validate_nominal_learning_index
        from r2v2_deploy.data.asset_paths import bundle_path

        dataset = bundle_path(m["nominal_dataset"])
        if (
            digest(dataset) != m.get("nominal_dataset_sha256")
            or m.get("independent_validation") is not False
        ):
            raise ValueError("Nominal training reference lineage changed")
        validate_nominal_learning_index(dataset, audit_physics=False)
    proof = path.parent / m["teacher_manifest"]
    if digest(proof) != m["teacher_manifest_sha256"]:
        raise ValueError("Teacher success evidence changed")
    teacher = json.loads(proof.read_text())
    if not teacher.get("paired_air_contact_success"):
        raise ValueError("Native reference teacher lacks complete physical success")
    # Native audits and all nested report/data hashes were checked before seal.
    # Recheck the small reports/audits at each worker startup, not gigabytes of
    # immutable NPZ at every reset. Snapshot bytes are checked before unpickle.
    for row in m["proof_files"]:
        if digest(path.parent / row["path"]) != row["sha256"]:
            raise ValueError("Native teacher report/audit changed")
    states = []
    for row in m["snapshots"]:
        f = path.parent / row["path"]
        if (
            digest(f) != row["sha256"]
            or not np.isfinite(row["native_observation_max_error"])
            or row["native_observation_max_error"] > 2e-6
        ):
            raise ValueError("Native reference state/observation proof changed")
        with f.open("rb") as stream:
            state = pickle.load(stream)
        if m.get("continuation_contract") == "delivery_fsm_index_native_history_v2":
            check = row.get("continuation", {})
            if (
                state.get("continuation_contract") != m["continuation_contract"]
                or check.get("ticks", 0) < 10
                or not all(
                    np.isfinite(check.get(k, float("inf")))
                    and check.get(k, float("inf")) <= tol
                    for k, tol in (("observation", 2e-6), ("integration", 1e-8))
                )
            ):
                raise ValueError("Missing/invalid future continuation evidence")
        if (
            state.get("source_kind") != "successful_native_pd50_teacher_reference"
            or state.get("official_air_contact_evidence") is not False
            or state["runtime"].get("contact_air_verified")
        ):
            raise ValueError(
                "Teacher initial states cannot grant student success certificates"
            )
        states.append(state)
    if not states:
        raise ValueError("Empty reference initial state pool")
    return states
