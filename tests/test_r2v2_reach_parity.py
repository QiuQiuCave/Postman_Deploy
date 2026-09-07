"""Lightweight regression tests for the independent policy provenance verifier."""

import numpy as np
import pytest

from tools.verify_r2v2_reach_policy import compare_weight_arrays


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
