"""Verify the published handoff files and ONNX contracts; never starts a robot.

Optional --torch-parity compares synthetic observations with the frozen actor
and normalizer in each trusted checkpoint. This is NOT observation-builder,
physics, safety, grasp, or locomotion acceptance testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

DEFAULT = Path(__file__).resolve().parents[1] / "policy/r2v2_skills/20261026"


def checked_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-bundle file: {relative}")
    return path


def verify(root, torch_parity=False):
    manifest = json.loads((root / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = checked_file(root, relative)
        if (path.stat().st_size != expected["bytes"] or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]):
            raise ValueError(f"File integrity failed: {relative}")
    reports = {}
    for name, spec in manifest["models"].items():
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(checked_file(root, spec["onnx"])),
            sess_options=options, providers=["CPUExecutionProvider"])
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        if inp.shape != spec["input_shape"] or out.shape != spec["output_shape"]:
            raise ValueError(f"Wrong interface: {name}")
        meta = session.get_modelmeta().custom_metadata_map
        for key, expected in spec["required_metadata"].items():
            if meta.get(key) != expected:
                raise ValueError(f"Wrong {name} metadata: {key}")
        if meta.get("checkpoint_sha256") not in (
                None, manifest["files"][spec["checkpoint"]]["sha256"]):
            raise ValueError(f"Wrong ONNX/checkpoint binding: {name}")
        actual = session.run([out.name], {inp.name: np.zeros(inp.shape, np.float32)})[0]
        if actual.shape != tuple(out.shape) or not np.isfinite(actual).all():
            raise ValueError(f"Invalid ONNX result: {name}")
        reports[name] = dict(passed=True, status=spec["status"],
                             input_shape=inp.shape, output_shape=out.shape)
        if torch_parity:
            # Optional training-only dependency; not needed by the deploy runtime.
            import torch
            from rsl_rl.modules.normalization import EmpiricalNormalization

            torch.set_num_threads(1)
            source = torch.load(checked_file(root, spec["checkpoint"]),
                                map_location="cpu", weights_only=False)
            actor = source["actor_state_dict"]
            if source["iter"] != spec["checkpoint_iteration"]:
                raise ValueError(f"Wrong checkpoint iteration: {name}")
            mean, std = actor["obs_normalizer._mean"], actor["obs_normalizer._std"]
            eps = float(EmpiricalNormalization(1).eps)
            rng = np.random.default_rng(20261026)
            max_error = 0.0
            # Include zero and samples around the saved observation distribution.
            for i in range(33):
                raw = (torch.zeros_like(mean) if i == 0 else mean + std *
                       torch.from_numpy(rng.normal(size=inp.shape).astype(np.float32)))
                with torch.inference_mode():
                    expected = (raw - mean) / (std + eps)
                    for layer in (0, 2, 4, 6):
                        expected = torch.nn.functional.linear(expected,
                            actor[f"mlp.{layer}.weight"], actor[f"mlp.{layer}.bias"])
                        if layer != 6:
                            expected = torch.nn.functional.elu(expected)
                result = session.run([out.name], {inp.name: raw.numpy()})[0]
                error = float(np.max(np.abs(result - expected.numpy())))
                if not np.isfinite(error) or error > 1e-4:
                    raise ValueError(f"Actor/ONNX mismatch: {name}: {error}")
                max_error = max(max_error, error)
            reports[name].update(synthetic_samples=33,
                max_action_abs_error=max_error, normalization_epsilon=eps)
    return dict(passed=True, checked_files=len(manifest["files"]), models=reports,
                scope="File integrity and inference only; not physical acceptance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    parser.add_argument("--torch-parity", action="store_true",
                        help="Only load trusted .pt files; requires training dependencies")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.torch_parity), indent=2))


if __name__ == "__main__":
    main()
