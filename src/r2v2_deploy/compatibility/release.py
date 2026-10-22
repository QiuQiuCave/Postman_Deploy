"""Content-pinned runtime and explicit model migration; unknown pairs reject."""

import importlib.metadata
import json
from pathlib import Path
import onnx
from r2v2_deploy.configuration import digest

PACKAGE = Path(__file__).resolve().parents[1]


def execution_identity(profile, program_sha256):
    from r2v2_deploy.tasks.scene_spec import canonical_digest

    if profile != "native_contact_reach_v1":
        raise ValueError("Unknown native CONTACT execution profile")
    return dict(
        execution_profile=profile,
        execution_task_sha256=canonical_digest(
            dict(
                program_sha256=program_sha256,
                execution_profile=profile,
                runtime_profile_sha256=digest(
                    PACKAGE / "compatibility/runtime_profile.json"
                ),
            )
        ),
    )


def validate_runtime():
    profile = json.loads((PACKAGE / "compatibility/runtime_profile.json").read_text())
    if profile.get("schema") != "native_contact_reach_runtime_v1":
        raise ValueError("Unknown native runtime profile")
    for relative, expected in profile["source_files"].items():
        if digest(PACKAGE / relative) != expected:
            raise ValueError("Runtime implementation hash mismatch: " + relative)
    from r2v2_deploy.resources import PROJECT_ROOT

    for relative, expected in profile["resource_files"].items():
        if digest(PROJECT_ROOT / relative) != expected:
            raise ValueError("Runtime physical resource hash mismatch: " + relative)
    for name, expected in profile["runtime_packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Unvalidated runtime package: " + name)
    return profile


def validated_candidate(path, program, *, reach_finger_condition_protocol=None):
    from r2v2_deploy.control.exit_placement_policy import model_identity, TASK_ID
    from r2v2_deploy.tasks.contact_reach_program import EXECUTION, FINGER_CONDITION

    profile = validate_runtime()
    c = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    for key in ("checkpoint", "onnx", "parity"):
        value = Path(c[key])
        c[key] = str(value if value.is_absolute() else base / value)
    parity = json.loads(Path(c["parity"]).read_text())
    graph = onnx.load(c["onnx"])
    meta = {p.key: p.value for p in graph.metadata_props}
    required = dict(
        learning_protocol="contact_reach_neighborhood_v1",
        reach_execution_protocol=EXECUTION,
        reach_finger_condition_protocol=FINGER_CONDITION,
        actor_architecture="conditional_complementary_open_features_v1",
    )
    if reach_finger_condition_protocol != FINGER_CONDITION:
        raise ValueError("Explicit continuous hand-reference condition required")
    if any(
        row.get(key) != value
        for row in (c, parity, meta)
        for key, value in required.items()
    ):
        raise ValueError("Native candidate execution/architecture protocol mismatch")
    if any(meta.get(k) != v for k, v in model_identity().items()):
        raise ValueError("Candidate does not match this runtime implementation")
    identity = execution_identity("native_contact_reach_v1", program.sha256)
    if any(row.get(k) != v for row in (c, parity, meta) for k, v in identity.items()):
        raise ValueError("Native CONTACT execution identity mismatch")
    if (
        not parity.get("passed")
        or parity.get("task") != TASK_ID
        or parity.get("actor_input_dim") != 1461
        or parity.get("checkpoint_sha256") != digest(c["checkpoint"])
        or parity.get("onnx_sha256") != digest(c["onnx"])
        or parity.get("delivery_program_sha256") != program.sha256
        or meta.get("delivery_program_sha256") != program.sha256
        or parity.get("delivery_adapter_sha256")
        != model_identity()["delivery_adapter_sha256"]
        or meta.get("actor_implementation_sha256")
        != parity.get("actor_implementation_sha256")
    ):
        raise ValueError("Candidate model/parity/program/source identity mismatch")
    if c.get("migration_record"):
        record_path = Path(c["migration_record"])
        if not record_path.is_absolute():
            record_path = base / record_path
        record = json.loads(record_path.read_text())
        import hashlib

        if (
            digest(record_path) != c.get("migration_record_sha256")
            or record.get("source_checkpoint_sha256")
            != profile["source_checkpoint_sha256"]
            or digest(c["checkpoint"]) != record["source_checkpoint_sha256"]
            or record.get("source_onnx_sha256") != profile["source_onnx_sha256"]
            or record.get("target_onnx_sha256") != digest(c["onnx"])
            or record.get("target_runtime_profile_sha256")
            != model_identity()["delivery_core_profile_sha256"]
            or record.get("optimizer_updates") != 0
            or hashlib.sha256(graph.graph.SerializeToString()).hexdigest()
            != profile["source_graph_sha256"]
            or not record.get("graph_and_initializer_bytes_unchanged")
        ):
            raise ValueError("Unknown or altered model migration")
    return c
