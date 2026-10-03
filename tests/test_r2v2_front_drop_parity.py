"""Provenance must distinguish frontal drop from identically shaped actors."""
import copy
import json

import numpy as np
import pytest

from tools.verify_r2v2_front_drop_policy import (
    CONTRACT, canonical_digest, sha256, streamed_target, validate_front_bindings,
)


def inputs(tmp_path):
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"final actor")
    manifest = {k: CONTRACT[k] for k in ("endpoint_contract", "path_contract", "task_scope")}
    manifest["content_sha256"] = canonical_digest(manifest)
    path = tmp_path/"path.json"
    path.write_text(json.dumps(manifest))
    run = dict(**CONTRACT, path_sha256=manifest["content_sha256"], asset_manifest_sha256="actual-assets")
    state = dict(infos={**CONTRACT, "front_manipulation_run": run,
        "front_manipulation_curriculum": dict(schema_version=4, path_contract=CONTRACT["path_contract"],
            contract=dict(path_sha256=manifest["content_sha256"], task_scope=CONTRACT["task_scope"]))})
    metadata = dict(**CONTRACT, checkpoint_sha256=sha256(checkpoint), path_sha256=manifest["content_sha256"])
    return state, metadata, checkpoint, path


def test_exact_front_bindings_pass(tmp_path):
    report = validate_front_bindings(*inputs(tmp_path), asset_manifest_sha256="actual-assets")
    assert report["passed"] and report["native_asset_verified"]
    assert report["task_scope"] == "tabletop_to_crate_drop_v1"


@pytest.mark.parametrize("location,key,value", [
    ("checkpoint", "task_id", "old-task"),
    ("checkpoint", "task_scope", "extraction"),
    ("run", "path_contract", "wrist_payload_path_v2"),
    ("run", "path_sha256", "different"),
    ("run", "asset_manifest_sha256", "changed-assets"),
    ("onnx", "quaternion_order", "xyzw"),
    ("onnx", "checkpoint_sha256", "another-checkpoint"),
    ("onnx", "endpoint_contract", "legacy_tcp_v1"),
    ("onnx", "path_sha256", "changed-path"),
])
def test_contract_or_provenance_mismatch_rejected(tmp_path, location, key, value):
    args = inputs(tmp_path)
    target = {"checkpoint": args[0]["infos"], "run": args[0]["infos"]["front_manipulation_run"],
              "onnx": args[1]}[location]
    target[key] = value
    with pytest.raises(ValueError, match="Front"):
        validate_front_bindings(*args, asset_manifest_sha256="actual-assets")


def test_changed_manifest_rejected(tmp_path):
    args = inputs(tmp_path)
    manifest = json.loads(args[3].read_text())
    manifest["extra"] = "tampered"
    args[3].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path hash"):
        validate_front_bindings(*args, asset_manifest_sha256="actual-assets")


def test_changed_curriculum_rejected(tmp_path):
    args = inputs(tmp_path)
    args[0]["infos"]["front_manipulation_curriculum"]["contract"]["task_scope"] = "extract"
    with pytest.raises(ValueError, match="curriculum"):
        validate_front_bindings(*args, asset_manifest_sha256="actual-assets")


def test_cubic_targets_are_continuous_and_asymmetric():
    home = [.246, .19698, 1.0879]
    values = [streamed_target(home, 0, i, 240) for i in range(240)]
    position = np.asarray([p for p, q in values])
    angles = np.asarray([q for p, q in values])
    assert np.max(np.linalg.norm(np.diff(position, axis=0), axis=1)) < .0005
    assert np.max(np.linalg.norm(np.diff(angles, axis=0), axis=1)) < .001
    assert np.array_equal(position[0], home)
    left = streamed_target(home, 0, 239, 240)
    right = streamed_target(home, 1, 239, 240)
    assert left[0][1] > home[1] > right[0][1]
    assert left[1][2] > 0 > right[1][2]
