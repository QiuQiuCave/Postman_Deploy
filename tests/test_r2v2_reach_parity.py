"""Lightweight regression tests for the independent policy provenance verifier."""

import numpy as np
import pytest

from tools.verify_r2v2_reach_policy import (
    TASK, WRIST_TASK, PATH_TASK, PAYLOAD_TASK, PAYLOAD_PATH_CONTRACT,
    compare_weight_arrays, endpoint_contract_for_task, validate_payload_bindings,
    resolve_training_scene, seed_reset_history_from_current_state, sha256,
)


def test_parity_task_selects_explicit_endpoint_semantics():
    assert endpoint_contract_for_task(TASK) == "legacy_tcp_v1"
    assert endpoint_contract_for_task(WRIST_TASK) == "wrist_world_v2"
    assert endpoint_contract_for_task(PATH_TASK) == "wrist_world_v2"
    assert endpoint_contract_for_task(PAYLOAD_TASK) == "wrist_world_v2"
    with pytest.raises(ValueError, match="Unsupported parity task"):
        endpoint_contract_for_task("unknown")


@pytest.mark.parametrize("task", [WRIST_TASK, PATH_TASK, PAYLOAD_TASK])
def test_v2_native_scene_uses_new_source_meshes_not_old_body_assets(tmp_path, task):
    xml = tmp_path / "scene.xml"
    xml.write_text('<mujoco><compiler/><asset/><worldbody/></mujoco>')
    import xml.etree.ElementTree as ET
    result = ET.fromstring(resolve_training_scene(xml, tmp_path / "AMO_R2", task))
    assert result.find("compiler").get("meshdir").endswith("r2v2_description/source/r2v2_with_hand/meshes")


def _weights():
    state = {
        "obs_normalizer._mean": np.array([[0.25, -0.2]], dtype=np.float32),
        "obs_normalizer._std": np.array([[0.5, 0.9]], dtype=np.float32),
        "mlp.0.weight": np.array([[0.1, 0.2]], dtype=np.float32),
        "mlp.0.bias": np.array([0.3], dtype=np.float32),
    }
    initializers = {key: value.copy() for key, value in state.items() if key != "obs_normalizer._std"}
    initializers["onnx::Div_24"] = state["obs_normalizer._std"] + 0.01
    return initializers, state


def test_actor_weights_and_folded_normalizer_must_match_exactly():
    initializers, state = _weights()
    report = compare_weight_arrays(initializers, state, 0.01)
    assert report["passed"]
    assert report["normalization_divisor_max_abs_error"] == 0
    assert report["tensor_max_abs_error"]["mlp.0.weight"] == 0


@pytest.mark.parametrize("name", ["mlp.0.weight", "mlp.0.bias", "obs_normalizer._mean", "onnx::Div_24"])
def test_changed_actor_or_normalizer_is_rejected(name):
    initializers, state = _weights()
    initializers[name].flat[0] += 1e-3
    assert not compare_weight_arrays(initializers, state, 0.01)["passed"]


def test_missing_weight_is_rejected():
    initializers, state = _weights()
    del initializers["mlp.0.bias"]
    assert not compare_weight_arrays(initializers, state, 0.01)["passed"]


def test_missing_normalization_epsilon_is_rejected():
    initializers, state = _weights()
    initializers["onnx::Div_24"] = state["obs_normalizer._std"].copy()
    assert not compare_weight_arrays(initializers, state, 0.01)["passed"]


def test_reset_history_is_rebuilt_from_deployment_state_after_reference_restore():
    class Adapter:
        def __init__(self):
            self.reference_velocity = np.array([0., 0., 0.])
            self.histories = {"reference_velocity": np.zeros((10, 3))}

        def observation_terms(self, data):
            assert data == "independent_physics_state"
            return {"reference_velocity": self.reference_velocity}

    adapter = Adapter()
    adapter.reference_velocity[:] = [-.04, .01, .02]
    seed_reset_history_from_current_state(adapter, "independent_physics_state")
    assert np.array_equal(adapter.histories["reference_velocity"],
                          np.tile([-.04, .01, .02], (10, 1)))
    adapter.reference_velocity[:] = 0.
    assert np.array_equal(adapter.histories["reference_velocity"][0], [-.04, .01, .02])


def _payload_inputs(tmp_path):
    import json
    checkpoint = tmp_path / "model_5999.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    archive = tmp_path / "planned_path.npz"
    archive.write_bytes(b"frozen path")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(trajectory_file=archive.name,
        trajectory_sha256=sha256(archive), payload_training=dict(contract=PAYLOAD_PATH_CONTRACT))))
    contract = dict(task_id=PAYLOAD_TASK, path_contract=PAYLOAD_PATH_CONTRACT,
        endpoint_contract="wrist_world_v2", quaternion_order="wxyz",
        speed_reference_point="wrist", endpoint_body_names="left_hand_roll_link,right_hand_roll_link")
    run = dict(**contract, trajectory_sha256=sha256(archive),
        path_manifest_sha256=sha256(manifest), asset_manifest_sha256="assets")
    state = dict(infos=dict(**contract, wrist_payload_run=run))
    metadata = dict(**contract, checkpoint_sha256=sha256(checkpoint),
        path_trajectory_sha256=sha256(archive))
    return state, metadata, checkpoint, manifest


def test_payload_native_bindings_include_new_task_path_and_real_asset_hash(tmp_path):
    report = validate_payload_bindings(*_payload_inputs(tmp_path), asset_manifest_sha256="assets")
    assert report["passed"] and report["native_asset_verified"]
    assert report["task_id"] == PAYLOAD_TASK
    assert report["path_contract"] == PAYLOAD_PATH_CONTRACT


@pytest.mark.parametrize("location,key,value", [
    ("checkpoint", "task_id", PATH_TASK),
    ("checkpoint", "path_contract", "crate_wrist_path_v1"),
    ("run", "task_id", PATH_TASK),
    ("onnx", "task_id", PATH_TASK),
    ("onnx", "quaternion_order", "xyzw"),
    ("onnx", "checkpoint_sha256", "wrong"),
    ("onnx", "path_trajectory_sha256", "wrong"),
    ("run", "trajectory_sha256", "wrong"),
    ("run", "path_manifest_sha256", "wrong"),
    ("run", "asset_manifest_sha256", "wrong"),
])
def test_payload_rejects_same_shape_but_wrong_provenance(tmp_path, location, key, value):
    args = _payload_inputs(tmp_path)
    target = {"checkpoint": args[0]["infos"], "run": args[0]["infos"]["wrist_payload_run"],
              "onnx": args[1]}[location]
    target[key] = value
    with pytest.raises(ValueError, match="Payload"):
        validate_payload_bindings(*args, asset_manifest_sha256="assets")


def test_payload_rejects_changed_trajectory_bytes(tmp_path):
    args = _payload_inputs(tmp_path)
    (tmp_path / "planned_path.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        validate_payload_bindings(*args, asset_manifest_sha256="assets")
