"""Numerical parity check: policy.onnx vs policy.pt for the loco_new policy.

Run:
    python tools/loco_new_parity_test.py

Standalone — no MuJoCo, no FSM wiring. Confirms the ONNX export is
numerically equivalent to the TorchScript export on a spread of synthetic
inputs covering zeros, noise, and in-distribution obs vectors.
"""
import os
import numpy as np
import torch
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(HERE, "..", "policy", "loco_new", "model", "policy_loco_new.pt")
ONNX_PATH = (
    "/home/qiuziyu/code/postman/upper_lower/logs/rsl_rl/g1_loco/"
    "2026-04-19_16-24-28_cliped_with_lin/exported/policy.onnx"
)
OBS_DIM = 99
ACT_DIM = 29
# FP32 drift scales ~linearly with input magnitude; rtol covers that,
# atol keeps small-signal cases honest. Both must be violated to fail.
RTOL = 1e-4
ATOL = 2e-5


def synth_realistic_obs(rng):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0:3]   = rng.uniform(-0.3, 0.3, 3)                                 # base_lin_vel
    obs[3:6]   = rng.uniform(-0.1, 0.1, 3)                                 # base_ang_vel·0.2
    obs[6:9]   = np.array([0.0, 0.0, -1.0]) + rng.uniform(-0.05, 0.05, 3)  # projected gravity
    obs[9:12]  = rng.uniform(-0.2, 0.2, 3)                                 # velocity_commands
    obs[12:41] = rng.uniform(-0.2, 0.2, 29)                                # joint_pos_rel
    obs[41:70] = rng.uniform(-0.1, 0.1, 29)                                # joint_vel·0.05
    obs[70:99] = rng.uniform(-1.0, 1.0, 29)                                # last_action
    return obs.reshape(1, -1).astype(np.float32)


def main():
    print(f"loading  .pt  : {PT_PATH}")
    pt_model = torch.jit.load(PT_PATH)
    pt_model.eval()

    print(f"loading .onnx : {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_info = sess.get_inputs()[0]
    out_info = sess.get_outputs()[0]
    print(f"ONNX input  : name={in_info.name}  shape={in_info.shape}  dtype={in_info.type}")
    print(f"ONNX output : name={out_info.name}  shape={out_info.shape}  dtype={out_info.type}")

    rng = np.random.default_rng(42)
    cases = [
        ("zeros",          np.zeros((1, OBS_DIM), dtype=np.float32)),
        ("ones",           np.ones((1, OBS_DIM), dtype=np.float32)),
        ("small_gauss",    (rng.standard_normal((1, OBS_DIM)) * 0.1).astype(np.float32)),
        ("medium_gauss",   (rng.standard_normal((1, OBS_DIM)) * 1.0).astype(np.float32)),
        ("large_gauss",    (rng.standard_normal((1, OBS_DIM)) * 5.0).astype(np.float32)),
        ("realistic_1",    synth_realistic_obs(rng)),
        ("realistic_2",    synth_realistic_obs(rng)),
        ("realistic_3",    synth_realistic_obs(rng)),
    ]

    print()
    print(f"{'case':16s}  {'max|err|':>10s}  {'mean|err|':>10s}  {'max|a|':>8s}  result")
    print("-" * 62)
    failures = 0
    for name, obs in cases:
        with torch.inference_mode():
            a_pt = pt_model(torch.from_numpy(obs)).detach().numpy().squeeze(0)
        a_onnx = sess.run([out_info.name], {in_info.name: obs})[0].squeeze(0)
        assert a_pt.shape == (ACT_DIM,) and a_onnx.shape == (ACT_DIM,), \
            f"unexpected output shape: pt={a_pt.shape}  onnx={a_onnx.shape}"
        max_err = float(np.max(np.abs(a_pt - a_onnx)))
        mean_err = float(np.mean(np.abs(a_pt - a_onnx)))
        max_mag = float(np.max(np.abs(a_pt)))
        ok = np.allclose(a_pt, a_onnx, rtol=RTOL, atol=ATOL)
        print(f"{name:16s}  {max_err:10.3e}  {mean_err:10.3e}  {max_mag:8.2f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures += 1

    print("-" * 62)
    if failures:
        print(f"FAILED: {failures}/{len(cases)} cases exceeded rtol={RTOL:.0e} atol={ATOL:.0e}")
        raise SystemExit(1)
    print(f"PASSED: all {len(cases)} cases within rtol={RTOL:.0e} atol={ATOL:.0e}")
    print("ONNX export is numerically equivalent to TorchScript.")


if __name__ == "__main__":
    main()
